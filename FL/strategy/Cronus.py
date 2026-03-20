from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext, ModelPool, evaluate_model
from .base import DistillationStrategy
from .common import (
    create_model,
    load_public_dataset_from_clients,
    numpy_from_dataset,
)
from tqdm import tqdm
from .robust_filter import RobustFilter
from ..poison_utils import parse_poison_config, apply_gaussian_noise_scale
from models.dense_discri import create_discriminator

DISC_WEIGHTS_DIR = os.path.join("temp_weights", "cronus_disc_weights")

CRONUS_CACHE_DIR = os.path.join("temp_weights", "cronus_cache")


def _ensure_cache_dir():
    os.makedirs(CRONUS_CACHE_DIR, exist_ok=True)


def _prediction_path(client_id: int, round_number: int) -> str:
    return os.path.join(CRONUS_CACHE_DIR, f"r{round_number}_c{client_id}_preds.bin")


def _base_public_path() -> str:
    return os.path.join(CRONUS_CACHE_DIR, "public_features.npy")


def _public_features_path(round_number: int) -> str:
    return os.path.join(CRONUS_CACHE_DIR, f"r{round_number}_public_X.npy")


def _pseudo_labels_path(round_number: int) -> str:
    return os.path.join(CRONUS_CACHE_DIR, f"r{round_number}_pseudo_y.npy")


def predict_to_file(
    classify_model,
    discri_model,
    X_open: np.ndarray,
    num_classes: int,
    batch_size: int,
    output_path: str,
) -> int:
    logits_model = (
        classify_model.get_logits_model()
        if hasattr(classify_model, "get_logits_model")
        else classify_model
    )
    discriminator_net = (
        discri_model.model if hasattr(discri_model, "model") else discri_model
    )

    total_rows = 0
    with open(output_path, "wb") as fp:
        for start in range(0, len(X_open), batch_size):
            end = min(start + batch_size, len(X_open))
            X_batch = X_open[start:end]
            logits_batch = logits_model(X_batch, training=False).numpy()
            probs_batch = tf.nn.softmax(logits_batch).numpy()

            dis_pred = discriminator_net(X_batch, training=False).numpy().reshape(-1)
            uncertain_mask = dis_pred > 0.5
            if np.any(uncertain_mask):
                probs_batch[uncertain_mask] = 1.0 / num_classes

            fp.write(probs_batch.astype(np.float32).tobytes())
            total_rows += probs_batch.shape[0]
            del logits_batch, probs_batch, dis_pred, X_batch

    return total_rows


def predict_to_file_plain(
    classify_model,
    X_open: np.ndarray,
    num_classes: int,
    batch_size: int,
    output_path: str,
) -> int:
    logits_model = (
        classify_model.get_logits_model()
        if hasattr(classify_model, "get_logits_model")
        else classify_model
    )

    total_rows = 0
    with open(output_path, "wb") as fp:
        for start in range(0, len(X_open), batch_size):
            end = min(start + batch_size, len(X_open))
            X_batch = X_open[start:end]
            logits_batch = logits_model(X_batch, training=False).numpy()
            probs_batch = tf.nn.softmax(logits_batch).numpy()
            fp.write(probs_batch.astype(np.float32).tobytes())
            total_rows += probs_batch.shape[0]
            del logits_batch, probs_batch, X_batch

    return total_rows


def robust_filter_labels(
    pred_files: List[str],
    n_samples: int,
    num_classes: int,
    epsilon: float,
) -> Tuple[np.ndarray, Optional[float], Set[int]]:
    """
    Returns:
        pseudo_labels: per-sample predicted class (-1 = uncertain)
        round_max_eigenvalue: highest eigenvalue seen across all samples,
                              or None if spectral norm never exceeded threshold.
        removed_client_indices: 0-based indices (into pred_files) of clients
                                filtered out at least once.
    """
    rf = RobustFilter(epsilon=epsilon, tau=0.1, preset="logits")

    chunk_rows = 50_000
    row_bytes = num_classes * 4
    pseudo_labels = np.empty(n_samples, dtype=np.int32)

    round_max_eigenvalue: Optional[float] = None
    removed_client_indices: Set[int] = set()

    handles = [open(f, "rb") for f in pred_files]
    n_clients = len(handles)

    boundary = 1.0 / num_classes
    offset = 0
    pbar = tqdm(total=n_samples, desc="Robust filter voting", unit="sample")
    while offset < n_samples:
        rows = min(chunk_rows, n_samples - offset)
        client_chunks = []
        for fh in handles:
            raw = fh.read(rows * row_bytes)
            chunk = np.frombuffer(raw, dtype=np.float32).reshape(rows, num_classes)
            client_chunks.append(chunk)

        for i in range(rows):
            sample_preds = [client_chunks[c][i] for c in range(n_clients)]
            robust_mean, max_eig, removed = rf.compute_robust_mean_debug(sample_preds)

            if max_eig is not None:
                if round_max_eigenvalue is None or max_eig > round_max_eigenvalue:
                    round_max_eigenvalue = max_eig
            removed_client_indices.update(removed)

            max_prob = float(np.max(robust_mean))
            if max_prob > boundary:
                pseudo_labels[offset + i] = int(np.argmax(robust_mean))
            else:
                pseudo_labels[offset + i] = -1
            del sample_preds, robust_mean, removed

        pbar.update(rows)
        offset += rows
        del client_chunks
    pbar.close()

    for fh in handles:
        fh.close()

    return pseudo_labels, round_max_eigenvalue, removed_client_indices


def train_discriminator(
    classify_model,
    discri_model,
    open_feature: np.ndarray,
    dis_rounds: int,
    batch_size: int,
    num_classes: int,
    private_X_path: str,
) -> bool:
    logits_model = (
        classify_model.get_logits_model()
        if hasattr(classify_model, "get_logits_model")
        else classify_model
    )
    discriminator_net = (
        discri_model.model if hasattr(discri_model, "model") else discri_model
    )

    prediction_batch_size = min(10000, max(batch_size, 1))
    logits_batches = []
    for start in range(0, len(open_feature), prediction_batch_size):
        end = min(start + prediction_batch_size, len(open_feature))
        batch_logits = logits_model(open_feature[start:end], training=False)
        logits_batches.append(tf.nn.softmax(batch_logits).numpy())

    dis_logits = np.vstack(logits_batches)
    max_probs = np.max(dis_logits, axis=1)
    del logits_batches, dis_logits

    theta = float(np.median(max_probs))
    sure_unknown_mask = max_probs < theta
    sure_unknown_feature = open_feature[sure_unknown_mask]
    del max_probs

    if sure_unknown_feature.size == 0:
        return False

    X_mmap = np.load(private_X_path, mmap_mode="r")
    sure_known_feature = np.array(X_mmap[: len(sure_unknown_feature)], dtype=np.float32)
    del X_mmap

    dis_X = np.vstack([sure_known_feature, sure_unknown_feature])
    dis_y = np.concatenate(
        [
            np.zeros(len(sure_known_feature), dtype=np.float32),
            np.ones(len(sure_unknown_feature), dtype=np.float32),
        ]
    )
    del sure_known_feature, sure_unknown_feature

    indices = np.random.permutation(len(dis_X))
    dis_X = dis_X[indices]
    dis_y = dis_y[indices]

    dataset = tf.data.Dataset.from_tensor_slices((dis_X, dis_y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    for _ in range(dis_rounds):
        discri_model.fit(dataset, epochs=1, verbose=0)

    del dis_X, dis_y, dataset
    return True


def _create_merged_dataset(
    private_X_path: str,
    private_y_path: str,
    public_X_path: str,
    pseudo_labels_path: str,
    input_dim: int,
    num_classes: int,
    batch_size: int,
    chunk_size: int = 50_000,
) -> tf.data.Dataset:
    def generator():
        priv_X = np.load(private_X_path, mmap_mode="r")
        priv_y = np.load(private_y_path, mmap_mode="r")
        for s in range(0, priv_X.shape[0], chunk_size):
            e = min(s + chunk_size, priv_X.shape[0])
            X_c = np.array(priv_X[s:e], dtype=np.float32)
            y_c = tf.keras.utils.to_categorical(
                np.array(priv_y[s:e], dtype=np.int32), num_classes
            ).astype(np.float32)
            yield X_c, y_c
            del X_c, y_c
        del priv_X, priv_y

        pub_X = np.load(public_X_path, mmap_mode="r")
        pseudo_y = np.load(pseudo_labels_path, mmap_mode="r")
        for s in range(0, pub_X.shape[0], chunk_size):
            e = min(s + chunk_size, pub_X.shape[0])
            labels_chunk = np.array(pseudo_y[s:e], dtype=np.int32)
            valid_mask = labels_chunk >= 0
            if not np.any(valid_mask):
                continue
            X_c = np.array(pub_X[s:e], dtype=np.float32)[valid_mask]
            y_c = tf.keras.utils.to_categorical(
                labels_chunk[valid_mask], num_classes
            ).astype(np.float32)
            yield X_c, y_c
            del X_c, y_c, labels_chunk, valid_mask
        del pub_X, pseudo_y

    output_signature = (
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.unbatch().shuffle(20_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


class Cronus(DistillationStrategy):
    name = "Cronus"

    def extra_log_tokens(self) -> Dict[str, float]:
        tokens = {
            "dis_rounds": self.config.dis_rounds,
            "dist_rounds": self.config.dist_rounds,
        }
        if getattr(self.config, "remove_dis", False):
            tokens["no_dis"] = 1
        return tokens

    def setup(self, context: PipelineContext) -> None:
        config = context.config
        _ensure_cache_dir()

        public_dataset, total_public = load_public_dataset_from_clients(
            context.paths,
            batch_size=config.batch_size,
            num_classes=context.num_classes,
            shuffle=False,
            return_labels=False,
        )
        public_features = numpy_from_dataset(public_dataset)
        del public_dataset

        base_pub_path = _base_public_path()
        np.save(base_pub_path, public_features)
        n_public = public_features.shape[0]
        del public_features

        context.shared_state.update(
            {
                "public_sample_count": n_public,
            }
        )

        use_dis = not getattr(config, "remove_dis", False)
        context.shared_state["use_dis"] = use_dis
        label = "clients and discriminators" if use_dis else "clients (no discriminator)"
        print(f"{COLORS.OKGREEN}Initializing Cronus {label}{COLORS.ENDC}")

        if use_dis:
            disc_pool = ModelPool(
                pool_size=min(5, len(context.paths)),
                factory_fn=lambda: create_discriminator(context.input_dim),
                weights_dir=DISC_WEIGHTS_DIR,
            )
            context.shared_state["disc_pool"] = disc_pool

        pool = context.model_pool
        for client_id, paths in enumerate(context.paths):
            state = context.add_client_state(client_id, None, paths)

            y_mmap = np.load(paths["train_y"], mmap_mode="r")
            class_counts = np.bincount(y_mmap.astype(np.int32), minlength=context.num_classes)
            state.data["class_counts"] = class_counts
            del y_mmap

        global_model = create_model(
            context.input_dim,
            context.num_classes,
            config.batch_size,
            model_type=config.model_type,
        )
        context.shared_state["global_model"] = global_model
        print(f"{COLORS.OKGREEN}Global evaluation model created{COLORS.ENDC}")

    def run_round(
        self, context: PipelineContext, round_number: int
    ) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        n_public = context.shared_state["public_sample_count"]

        base_mmap = np.load(_base_public_path(), mmap_mode="r")
        permutation = np.random.permutation(n_public)
        open_feature = np.array(base_mmap[permutation], dtype=np.float32)
        del base_mmap

        pub_X_path = _public_features_path(round_number)
        np.save(pub_X_path, open_feature)

        # ------ Poison config (parsed once per round) ------
        attack_type, poison_value, _ = parse_poison_config(
            getattr(config, "poison", None)
        )

        # ------ Stage I: local training + collect predictions ------
        print(f"\n{COLORS.HEADER}Round {round_number} Stage I — local training & prediction{COLORS.ENDC}")
        pred_files: List[str] = []
        pred_client_ids: List[int] = []  # parallel list: pred_files[i] belongs to pred_client_ids[i]

        use_dis = context.shared_state["use_dis"]

        pool = context.model_pool
        disc_pool = context.shared_state.get("disc_pool")

        for state in context.client_states:
            cid = state.client_id
            is_poisoned = cid in context.poisoned_clients
            print(f"\n{COLORS.BOLD}Client {cid} Stage I training{COLORS.ENDC}")

            if is_poisoned and attack_type == "label_flip":
                print(f"  \u26a0\ufe0f  POISONED CLIENT — Labels flipped")

            model = pool.checkout(cid)
            priv_ds = self._private_dataset(state, context)
            for _ in range(config.train_rounds):
                model.fit(priv_ds, epochs=1, verbose=0)

            if use_dis:
                class_counts = state.data["class_counts"]
                if np.sum(class_counts > 0) <= 1:
                    print("  Skipping discriminator (insufficient classes)")
                    pool.checkin(cid, model)
                    del priv_ds
                    aggressive_memory_cleanup()
                    continue

                disc = disc_pool.checkout(cid)
                success = train_discriminator(
                    model,
                    disc,
                    open_feature,
                    config.dis_rounds,
                    config.batch_size,
                    context.num_classes,
                    state.paths["train_X"],
                )

                del priv_ds
                aggressive_memory_cleanup()

                if not success:
                    print("  Discriminator training skipped (no uncertain samples)")
                    disc_pool.checkin(cid, disc)
                    pool.checkin(cid, model)
                    continue

                if is_poisoned and attack_type == "gradient_scale":
                    apply_gaussian_noise_scale(model, poison_value)
                    print(f"  \u26a0\ufe0f  POISONED CLIENT — Gaussian noise + scale x{poison_value:.1f}")

                pred_path = _prediction_path(cid, round_number)
                predict_to_file(
                    model,
                    disc,
                    open_feature,
                    context.num_classes,
                    config.batch_size,
                    pred_path,
                )
                disc_pool.checkin(cid, disc)
            else:
                del priv_ds
                aggressive_memory_cleanup()

                if is_poisoned and attack_type == "gradient_scale":
                    apply_gaussian_noise_scale(model, poison_value)
                    print(f"  \u26a0\ufe0f  POISONED CLIENT — Gaussian noise + scale x{poison_value:.1f}")

                pred_path = _prediction_path(cid, round_number)
                predict_to_file_plain(
                    model,
                    open_feature,
                    context.num_classes,
                    config.batch_size,
                    pred_path,
                )

            pool.checkin(cid, model)
            pred_files.append(pred_path)
            pred_client_ids.append(cid)
            aggressive_memory_cleanup()

        del open_feature
        aggressive_memory_cleanup()

        if not pred_files:
            print(
                f"{COLORS.WARNING}No client provided confident predictions; "
                f"skipping public training{COLORS.ENDC}"
            )
            return self._eval_fallback(context, round_start, round_number)

        # ------ Robust filtering instead of majority vote ------
        print(f"\n{COLORS.HEADER}Round {round_number} — Robust filter pseudo-labeling{COLORS.ENDC}")

        epsilon = getattr(config, "robust_epsilon", 0.2)
        pseudo_labels, round_max_eigenvalue, removed_client_indices = robust_filter_labels(
            pred_files, n_public, context.num_classes, epsilon
        )

        # --- Debug: aggregation diagnostics ---
        removed_client_ids = sorted({pred_client_ids[i] for i in removed_client_indices})
        eigenvalue_str = (
            f"{round_max_eigenvalue:.6f}" if round_max_eigenvalue is not None else "N/A"
        )
        removed_str = str(removed_client_ids) if removed_client_ids else "none"
        print(f"  [RobustFilter] epsilon={epsilon}")
        print(f"  [RobustFilter] Highest eigenvalue this round: {eigenvalue_str}")
        print(f"  [RobustFilter] Clients removed at least once: {removed_str}")
        context.logger.info(
            "Round %s | RobustFilter | epsilon=%.4f | max_eigenvalue=%s | removed_clients=%s",
            round_number, epsilon, eigenvalue_str, removed_str,
        )

        valid_count = int(np.sum(pseudo_labels >= 0))
        print(
            f"  {valid_count}/{n_public} samples received a confident pseudo-label"
        )

        pseudo_y_path = _pseudo_labels_path(round_number)
        np.save(pseudo_y_path, pseudo_labels)
        del pseudo_labels

        for f in pred_files:
            os.remove(f)

        # ------ Stage II: train on merged (private + pseudo-labeled public) ------
        print(f"\n{COLORS.HEADER}Round {round_number} Stage II — merged training{COLORS.ENDC}")

        for state in context.client_states:
            print(f"  Client {state.client_id}: merged private + pseudo-public training")
            model = pool.checkout(state.client_id)
            merged_ds = _create_merged_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                pub_X_path,
                pseudo_y_path,
                context.input_dim,
                context.num_classes,
                config.batch_size,
            )

            model.fit(merged_ds, epochs=config.dist_rounds, verbose=0)
            pool.checkin(state.client_id, model)
            del merged_ds
            aggressive_memory_cleanup()

        print(f"\n{COLORS.HEADER}Training global model on pseudo-labeled public data{COLORS.ENDC}")
        global_model = context.shared_state["global_model"]
        self._train_on_pseudo_public(
            global_model,
            pub_X_path,
            pseudo_y_path,
            context.num_classes,
            config.batch_size,
            config.dist_rounds,
        )

        self._cleanup_round_files(round_number)
        aggressive_memory_cleanup()

        # ------ Evaluation ------
        is_last_round = round_number == config.rounds
        round_metrics: Dict[int, Dict[str, float]] = {}

        if is_last_round:
            print(f"\n{COLORS.HEADER}Evaluation (per-client, last round){COLORS.ENDC}")
            for state in context.client_states:
                cid = state.client_id
                model = pool.checkout(cid)
                m = evaluate_model(model, context.test_dataset, context.test_labels)
                pool.release(model)
                round_metrics[cid] = m
                context.logger.info(
                    "Round %s | Client %s | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Loss: %.4f",
                    round_number, cid, m["Acc"], m["F1"], m["Precision"], m["Recall"], m["Loss"],
                )
                print(
                    f"  Client {cid}: Acc={m['Acc']:.4f}, F1={m['F1']:.4f}, "
                    f"Precision={m['Precision']:.4f}, Recall={m['Recall']:.4f}, Loss={m['Loss']:.4f}"
                )

        print(f"\n{COLORS.HEADER}Evaluation (global model){COLORS.ENDC}")
        global_metrics = evaluate_model(
            global_model, context.test_dataset, context.test_labels
        )
        round_metrics[-1] = global_metrics

        context.logger.info(
            "Round %s | GLOBAL | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Loss: %.4f",
            round_number,
            global_metrics["Acc"],
            global_metrics["F1"],
            global_metrics["Precision"],
            global_metrics["Recall"],
            global_metrics["Loss"],
        )
        print(
            f"{COLORS.OKGREEN}Global Model: Acc={global_metrics['Acc']:.4f}, "
            f"F1={global_metrics['F1']:.4f}, Precision={global_metrics['Precision']:.4f}, "
            f"Recall={global_metrics['Recall']:.4f}, Loss={global_metrics['Loss']:.4f}{COLORS.ENDC}"
        )

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = (
            context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        )
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")
        return round_metrics

    # ---- helpers ----

    @staticmethod
    def _private_dataset(state, context):
        from .common import create_private_dataset

        poison_loader = None
        if state.client_id in context.poisoned_clients and context.poison_loader is not None:
            poison_loader = context.poison_loader

        return create_private_dataset(
            state.paths["train_X"],
            state.paths["train_y"],
            context.input_dim,
            context.num_classes,
            context.config.batch_size,
            poison_loader=poison_loader,
        )

    @staticmethod
    def _train_on_pseudo_public(
        model_wrapper,
        pub_X_path: str,
        pseudo_y_path: str,
        num_classes: int,
        batch_size: int,
        epochs: int,
        chunk_size: int = 50_000,
    ):
        keras_model = (
            model_wrapper.model
            if hasattr(model_wrapper, "model")
            else model_wrapper
        )
        optimizer = keras_model.optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
        keras_model.optimizer = optimizer
        ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        if not hasattr(model_wrapper, '_cronus_train_step'):
            @tf.function
            def train_step(bx, by):
                with tf.GradientTape() as tape:
                    preds = keras_model(bx, training=True)
                    loss = ce_loss_fn(by, preds)
                grads = tape.gradient(loss, keras_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
                return loss

            model_wrapper._cronus_train_step = train_step

        train_step = model_wrapper._cronus_train_step

        def generator():
            pub_X = np.load(pub_X_path, mmap_mode="r")
            pseudo_y = np.load(pseudo_y_path, mmap_mode="r")
            for s in range(0, pub_X.shape[0], chunk_size):
                e = min(s + chunk_size, pub_X.shape[0])
                labels_chunk = np.array(pseudo_y[s:e], dtype=np.int32)
                valid = labels_chunk >= 0
                if not np.any(valid):
                    continue
                X_c = np.array(pub_X[s:e], dtype=np.float32)[valid]
                y_c = tf.keras.utils.to_categorical(labels_chunk[valid], num_classes).astype(np.float32)
                yield X_c, y_c
                del X_c, y_c, labels_chunk, valid
            del pub_X, pseudo_y

        input_dim = np.load(pub_X_path, mmap_mode="r").shape[1]
        sig = (
            tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
        )
        dataset = (
            tf.data.Dataset.from_generator(generator, output_signature=sig)
            .unbatch()
            .shuffle(20_000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = 0
            for bx, by in dataset:
                loss = train_step(bx, by)
                epoch_loss += float(loss)
                batches += 1
            if batches > 0:
                print(f"    Global public epoch {epoch + 1}/{epochs} — Loss {epoch_loss / batches:.4f}")

        del dataset

    def _eval_fallback(
        self, context: PipelineContext, round_start: float, round_number: int
    ) -> Dict[int, Dict[str, float]]:
        pool = context.model_pool
        all_metrics = []
        round_metrics: Dict[int, Dict[str, float]] = {}
        for state in context.client_states:
            model = pool.checkout(state.client_id)
            m = evaluate_model(model, context.test_dataset, context.test_labels)
            pool.release(model)
            all_metrics.append(m)
            round_metrics[state.client_id] = m
        round_metrics[-1] = {
            k: np.mean([m[k] for m in all_metrics])
            for k in ("Acc", "F1", "Precision", "Recall")
        }
        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = (
            context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        )
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")
        return round_metrics

    @staticmethod
    def _cleanup_round_files(round_number: int):
        for name in os.listdir(CRONUS_CACHE_DIR):
            if name.startswith(f"r{round_number}_"):
                os.remove(os.path.join(CRONUS_CACHE_DIR, name))

    def finalize(self, context: PipelineContext) -> None:
        for name in os.listdir(CRONUS_CACHE_DIR):
            os.remove(os.path.join(CRONUS_CACHE_DIR, name))
