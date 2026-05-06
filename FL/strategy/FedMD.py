from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
from ..backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext
from .base import DistillationStrategy
from ..poison_utils import poisonedfl_apply_cached_weights, poisonedfl_store_round_weights, poisonedfl_unified_weights, poisonedfl_warmstart_weights, parse_poison_config
from ._checkpoint import save_mid_round, load_mid_round, clear_mid_round
from .common import (
    create_model,
    create_private_dataset,
    load_public_dataset_from_clients,
    numpy_from_dataset,
)

LOGITS_CACHE_DIR = os.path.join("temp_weights", "fedmd_logits")


def generate_public_logits_to_file(model_wrapper, public_features: np.ndarray, batch_size: int, output_path: str) -> tuple:
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else model_wrapper

    chunk_size = 500_000
    total_rows = 0
    num_classes = None

    with open(output_path, "wb") as fp:
        for start in range(0, len(public_features), chunk_size):
            chunk = public_features[start : start + chunk_size]
            logits = logits_model.predict(chunk, batch_size=batch_size, verbose=0)
            if num_classes is None:
                num_classes = logits.shape[1]
            fp.write(logits.astype(np.float32).tobytes())
            total_rows += logits.shape[0]
            del logits, chunk

    return output_path, (total_rows, num_classes)


def compute_consensus_from_files(logit_files: List[str], shape: tuple) -> np.ndarray:
    """Stream-average logits from multiple files without loading all into RAM."""
    n_clients = len(logit_files)
    n_samples, n_classes = shape
    consensus = np.zeros(shape, dtype=np.float32)

    chunk_rows = 100000
    row_bytes = n_classes * 4

    for fpath in logit_files:
        with open(fpath, "rb") as f:
            offset = 0
            while offset < n_samples:
                rows = min(chunk_rows, n_samples - offset)
                raw = f.read(rows * row_bytes)
                chunk = np.frombuffer(raw, dtype=np.float32).reshape(rows, n_classes)
                consensus[offset:offset + rows] += chunk
                offset += rows
                del chunk

    consensus /= n_clients
    return consensus


def _digest_phase_pt(model_wrapper, consensus_logits, public_features, batch_size, epochs):
    import torch
    from ..backend import get_torch_device
    dev = get_torch_device()
    net = model_wrapper.nn
    opt = model_wrapper.optimizer
    net.to(dev)
    net.train()
    n_samples = len(public_features)
    X_t = torch.from_numpy(public_features.astype(np.float32))
    C_t = torch.from_numpy(consensus_logits.astype(np.float32))
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss, batches = 0.0, 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:min(start + batch_size, n_samples)]
            X_b = X_t[idx].to(dev)
            C_b = C_t[idx].to(dev)
            student_logits = net(X_b, return_logits=True)
            loss = torch.mean((C_b - student_logits) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            batches += 1
        print(f"    DIGEST epoch {epoch + 1}/{epochs} - MSE {epoch_loss / max(batches, 1):.4f}")


def digest_phase(
    model_wrapper,
    consensus_logits: np.ndarray,
    public_features: np.ndarray,
    batch_size: int,
    epochs: int,
) -> None:
    if not _use_tf():
        return _digest_phase_pt(model_wrapper, consensus_logits, public_features, batch_size, epochs)
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else keras_model

    optimizer = keras_model.optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model.optimizer = optimizer

    # Cache a compiled train_step on the model wrapper so it isn't recompiled each round
    if not hasattr(model_wrapper, "_fedmd_train_step"):

        @tf.function
        def train_step(batch_X, batch_consensus):
            with tf.GradientTape() as tape:
                student_logits = logits_model(batch_X, training=True)
                loss = tf.reduce_mean(tf.square(batch_consensus - student_logits))
            gradients = tape.gradient(loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return loss

        model_wrapper._fedmd_train_step = train_step

    train_step = model_wrapper._fedmd_train_step

    n_samples = len(public_features)
    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]
            loss = train_step(public_features[idx], consensus_logits[idx])
            epoch_loss += float(loss)
            batches += 1

        if batches > 0:
            print(f"    DIGEST epoch {epoch + 1}/{epochs} - MSE {epoch_loss / batches:.4f}")


def revisit_phase(model_wrapper, private_dataset: tf.data.Dataset, epochs: int) -> None:
    print(f"    REVISIT training for {epochs} epochs")
    history = model_wrapper.fit(private_dataset, epochs=epochs, verbose=1)
    if hasattr(history, "history") and "loss" in history.history:
        print(f"      Final REVISIT loss: {history.history['loss'][-1]:.4f}")


class FedMD(DistillationStrategy):
    name = "FedMD"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.digest_epochs = 1
        self.transfer_epochs = config.epochs
        self.revisit_epochs = config.epochs

    def extra_log_tokens(self) -> Dict[str, float]:
        return {}

    def setup(self, context: PipelineContext) -> None:
        print(f"{COLORS.OKGREEN}Preparing FedMD with public data from client slices{COLORS.ENDC}")
        public_unlabeled_ds, total_public = load_public_dataset_from_clients(
            context.paths,
            batch_size=context.config.batch_size,
            num_classes=context.num_classes,
            shuffle=False,
            return_labels=False,
        )
        public_features = numpy_from_dataset(public_unlabeled_ds)
        del public_unlabeled_ds

        os.makedirs(LOGITS_CACHE_DIR, exist_ok=True)
        pub_path = os.path.join(LOGITS_CACHE_DIR, "public_features.npy")
        np.save(pub_path, public_features)
        del public_features

        public_labeled_ds, _ = load_public_dataset_from_clients(
            context.paths,
            batch_size=context.config.batch_size,
            num_classes=context.num_classes,
            shuffle=True,
            return_labels=True,
        )

        context.shared_state.update(
            {
                "public_features_path": pub_path,
                "public_sample_count": total_public,
                "public_dataset_labeled": public_labeled_ds,
            }
        )

        print(f"{COLORS.OKGREEN}Using FULL test set{COLORS.ENDC}")

        for client_id, paths in enumerate(context.paths):
            state = context.add_client_state(client_id, None, paths)
            model = context.model_pool.checkout(client_id)
            print(f"Client {state.client_id}: initial transfer learning")
            model.fit(public_labeled_ds, epochs=self.transfer_epochs, verbose=1)

            _attack_type, _, _ = parse_poison_config(getattr(context.config, "poison", None))
            _skip_private = client_id in context.poisoned_clients and (
                hasattr(context, 'poisoned_fl_state') and context.poisoned_fl_state is not None
            )
            if not _skip_private:
                private_dataset = create_private_dataset(
                    paths["train_X"],
                    paths["train_y"],
                    context.input_dim,
                    context.num_classes,
                    context.config.batch_size,
                )
                model.fit(private_dataset, epochs=self.revisit_epochs, verbose=1)
                context.model_pool.checkin(client_id, model)
                del private_dataset
            else:
                context.model_pool.checkin(client_id, model)
            aggressive_memory_cleanup()

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        public_features = np.load(context.shared_state["public_features_path"], mmap_mode="r")

        os.makedirs(LOGITS_CACHE_DIR, exist_ok=True)

        _ckpt = getattr(config, "checkpoint", 0)
        first_logit = 0
        first_digest = 0
        skip_logits = False
        saved_logit_files = None
        if _ckpt:
            mid = load_mid_round(context, "fedmd", round_number)
            if mid is not None:
                stage = mid.get("stage", "logits")
                last = mid["last_client_idx"]
                if stage == "logits":
                    first_logit = last + 1
                    saved_logit_files = mid.get("logit_files", [])
                elif stage == "digest":
                    skip_logits = True
                    first_digest = last + 1
                print(f"{COLORS.OKGREEN}Resuming round {round_number} stage={stage} from client {last + 1}{COLORS.ENDC}")

        print(f"\n{COLORS.OKCYAN}[STEP 1/3] Generating public logits{COLORS.ENDC}")
        logit_files = []
        logit_shape = None
        pool = context.model_pool

        if not skip_logits:
            # Recover already-generated logit files for skipped clients
            for skipped_idx in range(first_logit):
                fpath = os.path.join(LOGITS_CACHE_DIR, f"client_{context.client_states[skipped_idx].client_id}.bin")
                if os.path.exists(fpath):
                    logit_files.append(fpath)

            for client_idx, state in enumerate(context.client_states):
                if client_idx < first_logit:
                    continue
                _pfl_lgt = getattr(context, 'poisoned_fl_state', None)
                if _pfl_lgt is not None and state.client_id in context.poisoned_clients:
                    continue
                fpath = os.path.join(LOGITS_CACHE_DIR, f"client_{state.client_id}.bin")
                model = pool.checkout(state.client_id)
                _, shape = generate_public_logits_to_file(model, public_features, config.batch_size, fpath)
                pool.release(model)
                logit_files.append(fpath)
                logit_shape = shape
                print(f"  Client {state.client_id}: logits {shape} -> {fpath}")
                aggressive_memory_cleanup()

                if _ckpt and (
                    client_idx == len(context.client_states) - 1
                    or (client_idx + 1) % _ckpt == 0
                ):
                    save_mid_round(context, "fedmd", {
                        "round": round_number,
                        "stage": "logits",
                        "last_client_idx": client_idx,
                        "logit_files": logit_files,
                    })
        else:
            logit_files = saved_logit_files or []
            if logit_files:
                n_classes = context.num_classes
                row_bytes_test = os.path.getsize(logit_files[0])
                logit_shape = (row_bytes_test // (n_classes * 4), n_classes)

        print(f"\n{COLORS.OKCYAN}[STEP 2/3] Computing consensus logits{COLORS.ENDC}")
        consensus_logits = compute_consensus_from_files(logit_files, logit_shape)
        print(f"  Consensus shape: {consensus_logits.shape}")

        for fpath in logit_files:
            os.remove(fpath)

        print(f"\n{COLORS.OKCYAN}[STEP 3/3] Digest and revisit phases{COLORS.ENDC}")

        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_digest:
                continue
            _pfl_dgt = getattr(context, 'poisoned_fl_state', None)
            attack_type_dgt, _, _ = parse_poison_config(getattr(config, "poison", None))
            if state.client_id in context.poisoned_clients and _pfl_dgt is not None:
                context.logger.info("Round %s | Client %s [PoisonedFL] all stages skipped", round_number, state.client_id)
                continue
            print(f"\n{COLORS.BOLD}Client {state.client_id}{COLORS.ENDC}")
            model = pool.checkout(state.client_id)
            digest_phase(model, consensus_logits, public_features, config.batch_size, self.digest_epochs)

            _pfl = getattr(context, 'poisoned_fl_state', None)
            _skip_revisit = state.client_id in context.poisoned_clients and _pfl is not None
            if not _skip_revisit:
                private_dataset = create_private_dataset(
                    state.paths["train_X"],
                    state.paths["train_y"],
                    context.input_dim,
                    context.num_classes,
                    config.batch_size,
                )
                revisit_phase(model, private_dataset, self.revisit_epochs)
                del private_dataset
            else:
                context.logger.info("Round %s | Client %s [PoisonedFL] revisit skipped", round_number, state.client_id)
            pool.checkin(state.client_id, model)
            aggressive_memory_cleanup()

            if _ckpt and (
                client_idx == len(context.client_states) - 1
                or (client_idx + 1) % _ckpt == 0
            ):
                save_mid_round(context, "fedmd", {
                    "round": round_number,
                    "stage": "digest",
                    "last_client_idx": client_idx,
                })

        # ---- PoisonedFL: ghost model (digest only, no private data) ----
        _pfl = getattr(context, 'poisoned_fl_state', None)
        if _pfl is not None and context.poisoned_clients:
            retry_proxy = context.shared_state.get("poisonedfl_proxy_w")
            ghost = create_model(context.input_dim, context.num_classes,
                                 config.batch_size, model_type=config.model_type)
            ghost_start = poisonedfl_warmstart_weights(context.shared_state, _pfl, fallback=context.shared_state.get("init_w"))
            if ghost_start is not None:
                ghost.set_weights(ghost_start)
            digest_phase(ghost, consensus_logits, public_features, config.batch_size, self.digest_epochs)
            ghost_w = ghost.get_weights()
            del ghost
            poisoned_w = poisonedfl_unified_weights([ghost_w], _pfl)
            if (
                _pfl.last_hypothesis_success is False
                and _pfl.last_poisoned is not None
                and retry_proxy is not None
            ):
                context.logger.info("Round %s | PoisonedFL | H0 -> retry digest from previous unpoisoned proxy", round_number)
                ghost = create_model(context.input_dim, context.num_classes,
                                     config.batch_size, model_type=config.model_type)
                ghost.set_weights([w.copy() for w in retry_proxy])
                digest_phase(ghost, consensus_logits, public_features, config.batch_size, self.digest_epochs)
                ghost_w = ghost.get_weights()
                del ghost
                poisoned_w = poisonedfl_apply_cached_weights(ghost_w, _pfl, track_as_prev=True)
            poisonedfl_store_round_weights(context.shared_state, ghost_w, poisoned_w)
            for st in context.client_states:
                if st.client_id in context.poisoned_clients:
                    m = pool.checkout(st.client_id)
                    m.set_weights(poisoned_w)
                    pool.checkin(st.client_id, m)
            context.logger.info("Round %s | PoisonedFL | Ghost digest'd, injected into %d byzantine clients", round_number, len(context.poisoned_clients))
            del ghost_w, poisoned_w
            aggressive_memory_cleanup()

        del consensus_logits, public_features
        aggressive_memory_cleanup()

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        if _ckpt:
            clear_mid_round(context, "fedmd")

        return {}
