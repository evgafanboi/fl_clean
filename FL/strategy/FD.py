from __future__ import annotations

import time
from typing import Dict

import numpy as np
from ..backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext
from ..poison_utils import parse_poison_config, poisonedfl_unified_weights, poisonedfl_unified_weights
from .base import DistillationStrategy
from ._checkpoint import save_mid_round, load_mid_round, clear_mid_round
from .common import create_model, create_private_dataset


def generate_per_class_logits(model_wrapper, X_path: str, y_path: str, num_classes: int):
    if hasattr(model_wrapper, '_accumulated_logits') and hasattr(model_wrapper, '_accumulated_counts'):
        class_logits_sum = model_wrapper._accumulated_logits
        class_counts = model_wrapper._accumulated_counts

        per_class_logits = {}
        for class_id in range(num_classes):
            count = class_counts[class_id]
            if count > 0:
                per_class_logits[class_id] = class_logits_sum[class_id] / count

        class_counts_dict = {i: int(class_counts[i]) for i in range(num_classes)}

        delattr(model_wrapper, '_accumulated_logits')
        delattr(model_wrapper, '_accumulated_counts')

        return per_class_logits, class_counts_dict

    print("  WARNING: No accumulated logits found, using final model state")
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, 'get_logits_model') else model_wrapper

    X_mmap = np.load(X_path, mmap_mode='r')
    y_mmap = np.load(y_path, mmap_mode='r')
    total_samples = X_mmap.shape[0]

    class_logits_sum = np.zeros((num_classes, num_classes), dtype=np.float32)
    class_counts = np.zeros(num_classes, dtype=np.int32)

    chunk_size = 10000
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
        y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.int32)

        logits_chunk = logits_model.predict(X_chunk, batch_size=8192, verbose=0)

        np.add.at(class_logits_sum, y_chunk, logits_chunk)
        np.add.at(class_counts, y_chunk, 1)

        del X_chunk, y_chunk, logits_chunk

    per_class_logits = {}
    for class_id in range(num_classes):
        count = class_counts[class_id]
        if count > 0:
            per_class_logits[class_id] = class_logits_sum[class_id] / count

    class_counts_dict = {i: int(class_counts[i]) for i in range(num_classes)}

    return per_class_logits, class_counts_dict


def _local_training_with_distillation_pt(
    model_wrapper, dataloader, global_logits, num_classes, epochs, gamma, batch_size, client_class_counts=None,
):
    import torch
    import torch.nn.functional as F
    from ..backend import get_torch_device
    dev = get_torch_device()
    net = model_wrapper.nn
    opt = model_wrapper.optimizer
    net.to(dev)

    class_logits_sum = np.zeros((num_classes, num_classes), dtype=np.float32)
    class_counts = np.zeros(num_classes, dtype=np.int32)

    if len(global_logits) == 0:
        print("    No global logits available - using standard CE training")
        model_wrapper.fit(dataloader, epochs=epochs, verbose=1)
        logits_model = model_wrapper.get_logits_model()
        for X_b, y_b in dataloader:
            batch_logits = logits_model.predict(X_b.cpu().numpy(), batch_size=batch_size, verbose=0)
            y_labels = y_b.argmax(dim=1).numpy() if y_b.ndim > 1 else y_b.numpy()
            np.add.at(class_logits_sum, y_labels, batch_logits)
            np.add.at(class_counts, y_labels, 1)
        model_wrapper._accumulated_logits = class_logits_sum
        model_wrapper._accumulated_counts = class_counts
        return model_wrapper

    gl_tensor = torch.zeros(num_classes, num_classes, device=dev)
    has_gl = torch.zeros(num_classes, dtype=torch.bool, device=dev)
    for cid, logits in global_logits.items():
        gl_tensor[cid] = torch.from_numpy(np.array(logits, dtype=np.float32)).to(dev)
        has_gl[cid] = True

    net.train()
    for epoch in range(epochs):
        total_loss, batches = 0.0, 0
        for X_b, y_b in dataloader:
            X_b = X_b.to(dev, non_blocking=True)
            y_b = y_b.to(dev, non_blocking=True)
            y_labels = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
            logits = net(X_b, return_logits=True)
            ce = F.cross_entropy(logits, y_labels)
            batch_has = has_gl[y_labels]
            distill = torch.tensor(0.0, device=dev)
            if batch_has.any():
                distill = gamma * F.l1_loss(logits[batch_has], gl_tensor[y_labels][batch_has])
            loss = ce + distill
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                np.add.at(class_logits_sum, y_labels.cpu().numpy(), logits.detach().cpu().numpy())
                np.add.at(class_counts, y_labels.cpu().numpy(), 1)
            total_loss += loss.item()
            batches += 1
        print(f"    epoch {epoch + 1}/{epochs}  loss={total_loss / max(batches, 1):.4f}")

    model_wrapper._accumulated_logits = class_logits_sum
    model_wrapper._accumulated_counts = class_counts
    return model_wrapper


def local_training_with_distillation(
    model_wrapper,
    private_dataset: tf.data.Dataset,
    global_logits: Dict[int, np.ndarray],
    num_classes: int,
    epochs: int,
    gamma: float,
    batch_size: int,
    client_class_counts: Dict[int, int] | None = None,
):
    if not _use_tf():
        return _local_training_with_distillation_pt(
            model_wrapper, private_dataset, global_logits, num_classes, epochs, gamma, batch_size, client_class_counts,
        )
    setup_start = time.time()
    keras_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper

    if hasattr(model_wrapper, 'get_logits_model'):
        logits_model = model_wrapper.get_logits_model()
    else:
        logits_model = keras_model

    print(f"  [LOCAL TRAINING] Training with distillation for {epochs} epochs...")
    print(f"    Global logits available for {len(global_logits)}/{num_classes} classes")
    print(f"    Setup time: {time.time() - setup_start:.3f}s")

    class_logits_sum = np.zeros((num_classes, num_classes), dtype=np.float32)
    class_counts = np.zeros(num_classes, dtype=np.int32)

    if len(global_logits) == 0:
        print("    No global logits available - using standard CE training")
        history = model_wrapper.fit(private_dataset, epochs=epochs, verbose=1)
        if 'loss' in history.history:
            final_loss = history.history['loss'][-1]
            print(f"    Loss: {final_loss:.4f}")

        for X_batch, y_batch in private_dataset:
            batch_logits = logits_model(X_batch, training=False).numpy()
            y_labels = np.argmax(y_batch.numpy(), axis=1)
            np.add.at(class_logits_sum, y_labels, batch_logits)
            np.add.at(class_counts, y_labels, 1)

        model_wrapper._accumulated_logits = class_logits_sum
        model_wrapper._accumulated_counts = class_counts
        return model_wrapper

    if client_class_counts is not None:
        owned_counts = sum(1 for count in client_class_counts.values() if count > 0)
        print(f"    Client owns {owned_counts} classes")

    global_logits_tensor = np.zeros((num_classes, num_classes), dtype=np.float32)
    has_global_logits = np.zeros(num_classes, dtype=bool)

    for class_id, logits in global_logits.items():
        global_logits_tensor[class_id] = logits
        has_global_logits[class_id] = True

    global_logits_tensor = tf.constant(global_logits_tensor, dtype=tf.float32)
    has_global_logits_tensor = tf.constant(has_global_logits, dtype=tf.bool)

    logits_layer = keras_model.get_layer('logits')
    dual_output_model = tf.keras.Model(inputs=keras_model.input, outputs=[logits_layer.output, keras_model.output])

    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    mae_loss = tf.keras.losses.MeanAbsoluteError()

    if hasattr(keras_model, 'optimizer') and keras_model.optimizer is not None:
        optimizer = keras_model.optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        keras_model.optimizer = optimizer

    if not hasattr(model_wrapper, "_fd_train_step"):
        @tf.function
        def train_step(X_batch, y_batch, global_logits_tensor, has_global_logits_tensor, gamma_val):
            with tf.GradientTape() as tape:
                student_logits, predictions = dual_output_model(X_batch, training=True)
                y_labels = tf.argmax(y_batch, axis=1)

                ce = ce_loss_fn(y_batch, predictions)

                batch_global_logits = tf.gather(global_logits_tensor, y_labels)
                batch_has_global = tf.gather(has_global_logits_tensor, y_labels)

                valid_student = tf.boolean_mask(student_logits, batch_has_global)
                valid_global = tf.boolean_mask(batch_global_logits, batch_has_global)

                if tf.shape(valid_student)[0] > 0:
                    distill = gamma_val * mae_loss(valid_global, valid_student)
                else:
                    distill = 0.0

                loss = ce + distill

            gradients = tape.gradient(loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return loss, ce, distill, student_logits, y_labels

        model_wrapper._fd_train_step = train_step

    train_step = model_wrapper._fd_train_step

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_distill_loss = 0.0
        batches = 0

        print(f"    Epoch {epoch + 1}/{epochs}", end='', flush=True)

        for X_batch, y_batch in private_dataset:
            loss, ce, distill, batch_logits, batch_labels = train_step(
                X_batch,
                y_batch,
                global_logits_tensor,
                has_global_logits_tensor,
                gamma,
            )
            batch_logits_np = batch_logits.numpy()
            batch_labels_np = batch_labels.numpy()
            np.add.at(class_logits_sum, batch_labels_np, batch_logits_np)
            np.add.at(class_counts, batch_labels_np, 1)

            epoch_loss += float(loss)
            epoch_ce_loss += float(ce)
            epoch_distill_loss += float(distill)
            batches += 1

        epoch_time = time.time() - epoch_start
        if batches > 0:
            print(
                f" - Loss: {epoch_loss / batches:.4f} (CE: {epoch_ce_loss / batches:.4f}, Distill: {epoch_distill_loss / batches:.4f})"
                f" [{epoch_time:.2f}s]"
            )
        else:
            print(" - No batches processed")

    model_wrapper._accumulated_logits = class_logits_sum
    model_wrapper._accumulated_counts = class_counts
    return model_wrapper


class FederatedDistillation(DistillationStrategy):
    name = "FD"

    def extra_log_tokens(self) -> Dict[str, float]:
        return {"gamma": self.config.gamma}

    def setup(self, context: PipelineContext) -> None:
        print(f"{COLORS.OKGREEN}Using FULL test set{COLORS.ENDC}")

        for client_id, paths in enumerate(context.paths):
            context.add_client_state(client_id, None, paths)

        global_logits: Dict[int, Dict[int, np.ndarray]] = {
            state.client_id: {} for state in context.client_states
        }
        context.shared_state["global_logits"] = global_logits

        if not context.config.data_calc:
            print(f"\n{COLORS.OKCYAN}[PRE-CALCULATION] Calculating data composition once for all rounds{COLORS.ENDC}")
            class_counts_cache: Dict[int, Dict[int, int]] = {}
            for state in context.client_states:
                y_mmap = np.load(state.paths["train_y"], mmap_mode="r")
                y_array = np.array(y_mmap, dtype=np.int32)
                class_counts = np.zeros(context.num_classes, dtype=np.int32)
                np.add.at(class_counts, y_array, 1)
                class_counts_cache[state.client_id] = {
                    class_id: int(class_counts[class_id])
                    for class_id in range(context.num_classes)
                }
                del y_mmap, y_array
            context.shared_state["class_counts_cache"] = class_counts_cache
            print(f"  Data composition cached for {len(class_counts_cache)} clients")

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        global_logits_per_client: Dict[int, Dict[int, np.ndarray]] = context.shared_state.setdefault(
            "global_logits", {state.client_id: {} for state in context.client_states}
        )

        counts_cache: Dict[int, Dict[int, int]] | None = context.shared_state.get("class_counts_cache")

        print(f"\n{COLORS.OKCYAN}[STEP 1/3] Local training with distillation{COLORS.ENDC}")
        per_client_counts: Dict[int, Dict[int, int]] = {}
        all_client_logits: Dict[int, Dict[int, np.ndarray]] = {}
        all_client_counts_for_round: Dict[int, Dict[int, int]] = {}
        all_client_metrics = []
        round_metrics: Dict[int, Dict[str, float]] = {}
        pool = context.model_pool
        attack_type, poison_value, _ = parse_poison_config(getattr(config, "poison", None))

        _ckpt = getattr(config, "checkpoint", 0)
        first_client = 0
        if _ckpt:
            mid = load_mid_round(context, "fd", round_number)
            if mid is not None:
                first_client = mid["last_client_idx"] + 1
                all_client_logits = mid.get("all_client_logits", {})
                all_client_counts_for_round = mid.get("all_client_counts_for_round", {})
                all_client_metrics = mid.get("all_client_metrics", [])
                round_metrics = mid.get("round_metrics", {})
                print(f"{COLORS.OKGREEN}Resuming round {round_number} from client {first_client}{COLORS.ENDC}")

        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_client:
                continue
            print(f"\n{COLORS.BOLD}Client {state.client_id}{COLORS.ENDC}")

            if counts_cache is not None:
                client_counts = counts_cache[state.client_id]
            else:
                y_mmap = np.load(state.paths["train_y"], mmap_mode="r")
                y_array = np.array(y_mmap, dtype=np.int32)
                class_counts_arr = np.zeros(context.num_classes, dtype=np.int32)
                np.add.at(class_counts_arr, y_array, 1)
                client_counts = {
                    class_id: int(class_counts_arr[class_id])
                    for class_id in range(context.num_classes)
                }
                del y_mmap, y_array
            per_client_counts[state.client_id] = client_counts

            # Apply poisoning for poisoned clients
            poison_loader = context.poison_loader if state.client_id in context.poisoned_clients else None
            if poison_loader:
                print(f"  \u26a0\ufe0f  POISONED CLIENT - Labels will be flipped")

            private_dataset = create_private_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                context.input_dim,
                context.num_classes,
                config.batch_size,
                poison_loader=poison_loader,
            )

            global_logits = global_logits_per_client.get(state.client_id, {})

            model = pool.checkout(state.client_id)
            old_w = model.get_weights()
            model = local_training_with_distillation(
                model,
                private_dataset,
                global_logits,
                context.num_classes,
                config.epochs,
                config.gamma,
                config.batch_size,
                client_class_counts=client_counts,
            )
            new_w = model.get_weights()
            if state.client_id in context.poisoned_clients and attack_type == "gradient_scale":
                model.set_weights([o + poison_value * (n - o) for o, n in zip(old_w, new_w)])

            logits, counts = generate_per_class_logits(
                model,
                state.paths["train_X"],
                state.paths["train_y"],
                context.num_classes,
            )
            all_client_logits[state.client_id] = logits
            all_client_counts_for_round[state.client_id] = counts

            if state.client_id in context.poisoned_clients and attack_type == "gradient_scale":
                model.set_weights(new_w)
            pool.checkin(state.client_id, model)

            del private_dataset
            aggressive_memory_cleanup()

            # Per-client checkpoint
            if _ckpt and (
                client_idx == len(context.client_states) - 1
                or (client_idx + 1) % _ckpt == 0
            ):
                save_mid_round(context, "fd", {
                    "round": round_number,
                    "last_client_idx": client_idx,
                    "all_client_logits": all_client_logits,
                    "all_client_counts_for_round": all_client_counts_for_round,
                    "all_client_metrics": all_client_metrics,
                    "round_metrics": round_metrics,
                })

        # ---- PoisonedFL: unified weights post-loop, re-generate logits ----
        _pfl = getattr(context, 'poisoned_fl_state', None)
        if _pfl is not None and context.poisoned_clients:
            pool = context.model_pool
            byz_w = []
            for st in context.client_states:
                if st.client_id in context.poisoned_clients:
                    m = pool.checkout(st.client_id)
                    byz_w.append(m.get_weights())
                    pool.release(m)
            if byz_w:
                poisoned_w = poisonedfl_unified_weights(byz_w, _pfl)
                for st in context.client_states:
                    if st.client_id not in context.poisoned_clients:
                        continue
                    m = pool.checkout(st.client_id)
                    m.set_weights(poisoned_w)
                    logits, counts = generate_per_class_logits(m, st.paths["train_X"], st.paths["train_y"], context.num_classes)
                    all_client_logits[st.client_id] = logits
                    all_client_counts_for_round[st.client_id] = counts
                    pool.checkin(st.client_id, m)
                context.logger.info("Round %s | PoisonedFL | FD byzantine clients unified logits applied", round_number)

        print(f"\n{COLORS.OKCYAN}[STEP 2/3] Aggregating logits per client{COLORS.ENDC}")
        new_global_logits: Dict[int, Dict[int, np.ndarray]] = {}

        for client_id in all_client_logits:
            class_logits_aggregated: Dict[int, list[np.ndarray]] = {}
            class_counts_aggregated: Dict[int, list[int]] = {}

            for other_client_id, logits in all_client_logits.items():
                if other_client_id == client_id:
                    continue
                for class_id, class_logits in logits.items():
                    class_logits_aggregated.setdefault(class_id, []).append(class_logits)
                    class_counts_aggregated.setdefault(class_id, []).append(
                        all_client_counts_for_round[other_client_id].get(class_id, 0)
                    )

            weighted_logits: Dict[int, np.ndarray] = {}
            for class_id, logits_list in class_logits_aggregated.items():
                weights = np.array(class_counts_aggregated[class_id], dtype=np.float32)
                if logits_list and weights.sum() > 0:
                    stacked_logits = np.stack(logits_list, axis=0)
                    normalized_weights = weights / weights.sum()
                    weighted_logits[class_id] = np.average(stacked_logits, axis=0, weights=normalized_weights)

            new_global_logits[client_id] = weighted_logits

        context.shared_state["global_logits"] = new_global_logits

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = (
            context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        )
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        if _ckpt:
            clear_mid_round(context, "fd")

        return {}
