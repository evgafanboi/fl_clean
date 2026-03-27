from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext, ModelPool, evaluate_model
from .base import DistillationStrategy
from .common import (
    create_model,
    create_private_dataset,
    load_public_dataset_from_clients,
    numpy_from_dataset,
)
from models.dense_discri import create_discriminator

SSFLIDS_DISC_DIR = os.path.join("temp_weights", "ssflids_disc_weights")

SSFLIDS_CACHE_DIR = os.path.join("temp_weights", "ssflids_cache")


def _ensure_cache_dir():
    os.makedirs(SSFLIDS_CACHE_DIR, exist_ok=True)


def _base_public_path() -> str:
    return os.path.join(SSFLIDS_CACHE_DIR, "public_features.npy")


def _round_public_path(round_number: int) -> str:
    return os.path.join(SSFLIDS_CACHE_DIR, f"r{round_number}_public_X.npy")


def _pred_path(client_id: int, round_number: int) -> str:
    return os.path.join(SSFLIDS_CACHE_DIR, f"r{round_number}_c{client_id}_hard.npy")


def predict_with_discriminator(
    classify_model,
    discri_model,
    X_open: np.ndarray,
    num_classes: int,
    batch_size: int,
    output_path: str,
) -> None:
    logits_model = classify_model.get_logits_model() if hasattr(classify_model, "get_logits_model") else classify_model
    discriminator_net = discri_model.model if hasattr(discri_model, "model") else discri_model

    boundary = 1.0 / num_classes
    hard_labels: List[int] = []

    for start_idx in range(0, len(X_open), batch_size):
        end_idx = min(start_idx + batch_size, len(X_open))
        X_batch = X_open[start_idx:end_idx]
        logits_batch = logits_model(X_batch, training=False).numpy()
        probs_batch = tf.nn.softmax(logits_batch).numpy()
        dis_pred = discriminator_net(X_batch, training=False).numpy().reshape(-1)

        uncertain_mask = dis_pred > 0.5
        if np.any(uncertain_mask):
            probs_batch[uncertain_mask] = 1.0 / num_classes

        for row in probs_batch:
            max_prob = float(np.max(row))
            if max_prob > boundary:
                hard_labels.append(int(np.argmax(row)))
            else:
                hard_labels.append(num_classes)

        del logits_batch, probs_batch, dis_pred, X_batch

    np.save(output_path, np.array(hard_labels, dtype=np.int32))
    del hard_labels


def hard_label_vote(pred_files: List[str], num_classes: int) -> np.ndarray:
    mmaps = [np.load(f, mmap_mode="r") for f in pred_files]
    sample_cnt = mmaps[0].shape[0]
    voted = np.empty(sample_cnt, dtype=np.int32)

    chunk = 50_000
    for s in range(0, sample_cnt, chunk):
        e = min(s + chunk, sample_cnt)
        label_votes = np.zeros((e - s, num_classes), dtype=np.int32)
        for m in mmaps:
            labels_chunk = np.array(m[s:e], dtype=np.int32)
            valid = labels_chunk < num_classes
            rows = np.arange(e - s)
            label_votes[rows[valid], labels_chunk[valid]] += 1
        voted[s:e] = np.argmax(label_votes, axis=1)
        del label_votes

    del mmaps
    return voted


def train_on_pseudo_labeled_public(
    model_wrapper,
    pseudo_labels_path: str,
    X_path: str,
    epochs: int,
    batch_size: int,
    num_classes: int,
    chunk_size: int = 50_000,
) -> None:
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper

    optimizer = keras_model.optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model.optimizer = optimizer
    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    if not hasattr(model_wrapper, '_ssflids_train_step'):
        @tf.function
        def train_step(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                loss = ce_loss_fn(batch_y, predictions)
            gradients = tape.gradient(loss, keras_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))

        model_wrapper._ssflids_train_step = train_step

    train_step = model_wrapper._ssflids_train_step

    def generator():
        X_mmap = np.load(X_path, mmap_mode="r")
        y_mmap = np.load(pseudo_labels_path, mmap_mode="r")
        for s in range(0, X_mmap.shape[0], chunk_size):
            e = min(s + chunk_size, X_mmap.shape[0])
            X_c = np.array(X_mmap[s:e], dtype=np.float32)
            y_c = tf.keras.utils.to_categorical(
                np.array(y_mmap[s:e], dtype=np.int32), num_classes
            ).astype(np.float32)
            yield X_c, y_c
            del X_c, y_c
        del X_mmap, y_mmap

    input_dim = np.load(X_path, mmap_mode="r").shape[1]
    sig = (
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
    )
    dataset = (
        tf.data.Dataset.from_generator(generator, output_signature=sig)
        .unbatch()
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch_X, batch_y in dataset:
            loss = train_step(batch_X, batch_y)
            epoch_loss += float(loss)
            batches += 1
        if batches > 0:
            print(f"    Public CE epoch {epoch + 1}/{epochs} - Loss {epoch_loss / batches:.4f}")
    
    del dataset


def train_discriminator(
    classify_model,
    discri_model,
    open_feature: np.ndarray,
    dis_rounds: int,
    batch_size: int,
    num_classes: int,
    private_X_path: str,
) -> bool:
    logits_model = classify_model.get_logits_model() if hasattr(classify_model, "get_logits_model") else classify_model

    logits_batches: List[np.ndarray] = []
    prediction_batch_size = min(10000, max(batch_size, 1))
    discriminator_net = discri_model.model if hasattr(discri_model, "model") else discri_model

    for start_idx in range(0, len(open_feature), prediction_batch_size):
        end_idx = min(start_idx + prediction_batch_size, len(open_feature))
        batch_logits = logits_model(open_feature[start_idx:end_idx], training=False)
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
        [np.zeros(len(sure_known_feature), dtype=np.float32), np.ones(len(sure_unknown_feature), dtype=np.float32)]
    )

    indices = np.random.permutation(len(dis_X))
    dis_X = dis_X[indices]
    dis_y = dis_y[indices]

    dataset = tf.data.Dataset.from_tensor_slices((dis_X, dis_y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for _ in range(dis_rounds):
        discri_model.fit(dataset, epochs=1, verbose=0)

    return True


class SSFLIDS(DistillationStrategy):
    name = "SSFL-IDS"

    def extra_log_tokens(self) -> Dict[str, float]:
        return {"dis_rounds": self.config.dis_rounds, "dist_rounds": self.config.dist_rounds}

    def setup(self, context: PipelineContext) -> None:
        config = context.config
        if config.model_type.lower() != "dense":
            raise ValueError("SSFL-IDS currently supports only the 'dense' model_type")

        is_sequence = False  # Current discriminator expects flat features

        _ensure_cache_dir()

        public_dataset, total_public = load_public_dataset_from_clients(
            context.paths,
            batch_size=config.batch_size,
            num_classes=context.num_classes,
            shuffle=False,
            return_labels=False,
            is_sequence=is_sequence,
        )
        public_features = numpy_from_dataset(public_dataset)
        del public_dataset

        np.save(_base_public_path(), public_features)
        n_public = public_features.shape[0]
        del public_features

        context.shared_state.update(
            {
                "is_sequence": is_sequence,
                "public_sample_count": n_public,
            }
        )

        print(f"{COLORS.OKGREEN}Initializing clients and discriminators{COLORS.ENDC}")

        disc_pool = ModelPool(
            pool_size=min(5, len(context.paths)),
            factory_fn=lambda: create_discriminator(context.input_dim),
            weights_dir=SSFLIDS_DISC_DIR,
        )
        context.shared_state["disc_pool"] = disc_pool

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

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        n_public = context.shared_state["public_sample_count"]

        base_mmap = np.load(_base_public_path(), mmap_mode="r")
        permutation = np.random.permutation(n_public)
        open_feature = np.array(base_mmap[permutation], dtype=np.float32)
        del base_mmap

        pub_X_path = _round_public_path(round_number)
        np.save(pub_X_path, open_feature)

        print(f"\n{COLORS.HEADER}Round {round_number} Stage I{COLORS.ENDC}")
        pred_files: List[str] = []
        pool = context.model_pool
        disc_pool = context.shared_state["disc_pool"]

        cleanup_interval = min(getattr(config, 'cleanup_interval', 10), len(context.client_states))
        for client_idx, state in enumerate(context.client_states):
            cid = state.client_id
            print(f"\n{COLORS.BOLD}Client {cid} Stage I training{COLORS.ENDC}")
            private_dataset = create_private_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                context.input_dim,
                context.num_classes,
                config.batch_size,
                is_sequence=context.shared_state["is_sequence"],
            )

            model = pool.checkout(cid)
            for _ in range(config.train_rounds):
                model.fit(private_dataset, epochs=1, verbose=0)

            class_counts = state.data["class_counts"]
            if np.sum(class_counts > 0) <= 1:
                print("  Skipping discriminator (insufficient classes)")
                pool.checkin(cid, model)
                del private_dataset
                if (client_idx + 1) % cleanup_interval == 0:
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

            if not success:
                print("  Discriminator training skipped (no uncertain samples)")
                disc_pool.checkin(cid, disc)
                pool.checkin(cid, model)
                del private_dataset
                if (client_idx + 1) % cleanup_interval == 0:
                    aggressive_memory_cleanup()
                continue

            pred_path = _pred_path(cid, round_number)
            predict_with_discriminator(
                model,
                disc,
                open_feature,
                context.num_classes,
                config.batch_size,
                pred_path,
            )
            disc_pool.checkin(cid, disc)
            pool.checkin(cid, model)
            pred_files.append(pred_path)

            del private_dataset
            if (client_idx + 1) % cleanup_interval == 0:
                aggressive_memory_cleanup()

        del open_feature
        aggressive_memory_cleanup()

        if not pred_files:
            print(f"{COLORS.WARNING}No client provided confident predictions; skipping public training{COLORS.ENDC}")
            all_client_metrics = []
            round_metrics: Dict[int, Dict[str, float]] = {}
            for state in context.client_states:
                model = pool.checkout(state.client_id)
                metrics = evaluate_model(model, context.test_dataset, context.test_labels)
                pool.release(model)
                all_client_metrics.append(metrics)
                round_metrics[state.client_id] = metrics
            avg_metrics = {
                "Acc": np.mean([m["Acc"] for m in all_client_metrics]),
                "F1": np.mean([m["F1"] for m in all_client_metrics]),
                "Precision": np.mean([m["Precision"] for m in all_client_metrics]),
                "Recall": np.mean([m["Recall"] for m in all_client_metrics]),
            }
            round_metrics[-1] = avg_metrics
            round_time = time.time() - round_start
            context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
            context.logger.info("Round %s completed in %.2fs", round_number, round_time)
            print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")
            return round_metrics

        print(f"\n{COLORS.HEADER}Round {round_number} Stage II{COLORS.ENDC}")
        global_labels_np = hard_label_vote(pred_files, context.num_classes)

        pseudo_y_path = os.path.join(SSFLIDS_CACHE_DIR, f"r{round_number}_pseudo_y.npy")
        np.save(pseudo_y_path, global_labels_np)
        del global_labels_np

        for f in pred_files:
            os.remove(f)
        aggressive_memory_cleanup()

        for state in context.client_states:
            print(f"Client {state.client_id}: training on pseudo-labeled public data")
            model = pool.checkout(state.client_id)
            train_on_pseudo_labeled_public(
                model,
                pseudo_y_path,
                pub_X_path,
                config.dist_rounds,
                config.batch_size,
                context.num_classes,
            )
            pool.checkin(state.client_id, model)
            aggressive_memory_cleanup()

        print(f"\n{COLORS.HEADER}Training global evaluation model on pseudo-labeled public data{COLORS.ENDC}")
        global_model = context.shared_state["global_model"]
        train_on_pseudo_labeled_public(
            global_model,
            pseudo_y_path,
            pub_X_path,
            config.dist_rounds,
            config.batch_size,
            context.num_classes,
        )

        for name in os.listdir(SSFLIDS_CACHE_DIR):
            if name.startswith(f"r{round_number}_"):
                os.remove(os.path.join(SSFLIDS_CACHE_DIR, name))
        aggressive_memory_cleanup()

        print(f"\n{COLORS.HEADER}Evaluation (global model){COLORS.ENDC}")

        global_metrics = evaluate_model(global_model, context.test_dataset, context.test_labels)
        round_metrics: Dict[int, Dict[str, float]] = {-1: global_metrics}

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
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        return round_metrics

    def finalize(self, context: PipelineContext) -> None:
        if os.path.isdir(SSFLIDS_CACHE_DIR):
            for name in os.listdir(SSFLIDS_CACHE_DIR):
                os.remove(os.path.join(SSFLIDS_CACHE_DIR, name))
