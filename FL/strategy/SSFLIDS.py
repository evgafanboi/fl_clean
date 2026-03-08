from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext, evaluate_model
from .base import DistillationStrategy
from .common import (
    create_model,
    create_private_dataset,
    load_public_dataset_from_clients,
    numpy_from_dataset,
)
from models.dense_discri import create_discriminator


def predict_with_discriminator(
    classify_model,
    discri_model,
    X_open: np.ndarray,
    num_classes: int,
    batch_size: int,
) -> np.ndarray:
    logits_model = classify_model.get_logits_model() if hasattr(classify_model, "get_logits_model") else classify_model

    all_logits: List[np.ndarray] = []
    discriminator_net = discri_model.model if hasattr(discri_model, "model") else discri_model

    for start_idx in range(0, len(X_open), batch_size):
        end_idx = min(start_idx + batch_size, len(X_open))
        X_batch = X_open[start_idx:end_idx]
        logits_batch = logits_model(X_batch, training=False).numpy()
        probs_batch = tf.nn.softmax(logits_batch).numpy()
        dis_pred = discriminator_net(X_batch, training=False).numpy().reshape(-1)

        uncertain_mask = dis_pred > 0.5
        if np.any(uncertain_mask):
            probs_batch[uncertain_mask] = 1.0 / num_classes

        all_logits.append(probs_batch)

    return np.vstack(all_logits)


def hard_label_from_soft(soft_label: np.ndarray, num_classes: int) -> List[int]:
    boundary = 1.0 / num_classes
    hard_labels: List[int] = []
    for row in soft_label:
        max_prob = float(np.max(row))
        if max_prob > boundary:
            hard_labels.append(int(np.argmax(row)))
        else:
            hard_labels.append(num_classes)
    return hard_labels


def hard_label_vote(all_client_hard_labels: List[List[int]], num_classes: int) -> List[int]:
    if not all_client_hard_labels:
        raise ValueError("No client predictions available for voting")

    client_cnt = len(all_client_hard_labels)
    sample_cnt = len(all_client_hard_labels[0])

    voted: List[int] = []
    for sample_idx in range(sample_cnt):
        label_votes = np.zeros(num_classes, dtype=np.int32)
        for client_idx in range(client_cnt):
            pred_label = all_client_hard_labels[client_idx][sample_idx]
            if pred_label != num_classes:
                label_votes[pred_label] += 1
        voted.append(int(np.argmax(label_votes)))
    return voted


def train_on_pseudo_labeled_public(
    model_wrapper,
    pseudo_labels: np.ndarray,
    X_data: np.ndarray,
    epochs: int,
    batch_size: int,
    num_classes: int,
) -> None:
    """Train model on pseudo-labeled public data using CE loss on hard labels."""
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper

    optimizer = keras_model.optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model.optimizer = optimizer
    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    one_hot_labels = tf.keras.utils.to_categorical(pseudo_labels, num_classes=num_classes).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((X_data, one_hot_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch_X, batch_y in dataset:
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                loss = ce_loss_fn(batch_y, predictions)
            gradients = tape.gradient(loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
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
    del logits_batches

    theta = float(np.median(max_probs))

    sure_unknown_mask = max_probs < theta
    sure_unknown_feature = open_feature[sure_unknown_mask]

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

        public_dataset, total_public = load_public_dataset_from_clients(
            context.paths,
            batch_size=config.batch_size,
            num_classes=context.num_classes,
            shuffle=False,
            return_labels=False,
            is_sequence=is_sequence,
        )
        public_features = numpy_from_dataset(public_dataset)

        context.shared_state.update(
            {
                "public_features": public_features,
                "is_sequence": is_sequence,
                "public_sample_count": total_public,
            }
        )

        print(f"{COLORS.OKGREEN}Initializing clients and discriminators{COLORS.ENDC}")
        for client_id, paths in enumerate(context.paths):
            classify_model = create_model(
                context.input_dim,
                context.num_classes,
                config.batch_size,
                model_type=config.model_type,
            )
            state = context.add_client_state(client_id, classify_model, paths)

            discri_model = create_discriminator(context.input_dim)
            state.data["discriminator"] = discri_model

            y_mmap = np.load(paths["train_y"], mmap_mode="r")
            class_counts = np.bincount(y_mmap.astype(np.int32), minlength=context.num_classes)
            state.data["class_counts"] = class_counts
            del y_mmap

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        public_features = context.shared_state["public_features"]

        permutation = np.random.permutation(public_features.shape[0])
        open_feature = public_features[permutation]

        print(f"\n{COLORS.HEADER}Round {round_number} Stage I{COLORS.ENDC}")
        all_client_hard_labels: List[List[int]] = []

        for state in context.client_states:
            print(f"\n{COLORS.BOLD}Client {state.client_id} Stage I training{COLORS.ENDC}")
            private_dataset = create_private_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                context.input_dim,
                context.num_classes,
                config.batch_size,
                is_sequence=context.shared_state["is_sequence"],
            )

            for _ in range(config.train_rounds):
                state.model.fit(private_dataset, epochs=1, verbose=0)

            class_counts = state.data["class_counts"]
            if np.sum(class_counts > 0) <= 1:
                print("  Skipping discriminator (insufficient classes)")
                del private_dataset
                aggressive_memory_cleanup()
                continue

            success = train_discriminator(
                state.model,
                state.data["discriminator"],
                open_feature,
                config.dis_rounds,
                config.batch_size,
                context.num_classes,
                state.paths["train_X"],
            )

            if not success:
                print("  Discriminator training skipped (no uncertain samples)")
                del private_dataset
                aggressive_memory_cleanup()
                continue

            local_logits = predict_with_discriminator(
                state.model,
                state.data["discriminator"],
                open_feature,
                context.num_classes,
                config.batch_size,
            )
            hard_labels = hard_label_from_soft(local_logits, context.num_classes)
            all_client_hard_labels.append(hard_labels)

            del private_dataset, local_logits, hard_labels
            aggressive_memory_cleanup()

        if not all_client_hard_labels:
            print(f"{COLORS.WARNING}No client provided confident predictions; skipping public training{COLORS.ENDC}")
            all_client_metrics = []
            round_metrics: Dict[int, Dict[str, float]] = {}
            for state in context.client_states:
                metrics = evaluate_model(state.model, context.test_dataset, context.test_labels)
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
        global_labels = hard_label_vote(all_client_hard_labels, context.num_classes)
        global_labels_np = np.array(global_labels, dtype=np.int32)
        
        del all_client_hard_labels
        aggressive_memory_cleanup()

        for state in context.client_states:
            print(f"Client {state.client_id}: training on pseudo-labeled public data")
            train_on_pseudo_labeled_public(
                state.model,
                global_labels_np,
                open_feature,
                config.dist_rounds,
                config.batch_size,
                context.num_classes,
            )
            aggressive_memory_cleanup()

        del global_labels_np, open_feature
        aggressive_memory_cleanup()

        print(f"\n{COLORS.HEADER}Evaluation{COLORS.ENDC}")

        all_client_metrics = []
        round_metrics: Dict[int, Dict[str, float]] = {}

        for state in context.client_states:
            metrics = evaluate_model(state.model, context.test_dataset, context.test_labels)
            all_client_metrics.append(metrics)
            round_metrics[state.client_id] = metrics
            context.logger.info(
                "Round %s | Client %s | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
                round_number,
                state.client_id,
                metrics["Acc"],
                metrics["F1"],
                metrics["Precision"],
                metrics["Recall"],
            )
            print(
                f"{COLORS.OKGREEN}Client {state.client_id}: Acc={metrics['Acc']:.4f}, F1={metrics['F1']:.4f}, "
                f"Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}{COLORS.ENDC}"
            )

        avg_metrics = {
            "Acc": np.mean([m["Acc"] for m in all_client_metrics]),
            "F1": np.mean([m["F1"] for m in all_client_metrics]),
            "Precision": np.mean([m["Precision"] for m in all_client_metrics]),
            "Recall": np.mean([m["Recall"] for m in all_client_metrics]),
        }
        round_metrics[-1] = avg_metrics

        print(
            f"{COLORS.OKGREEN}Round {round_number} - Avg Acc={avg_metrics['Acc']:.4f}, "
            f"F1={avg_metrics['F1']:.4f}, Precision={avg_metrics['Precision']:.4f}, "
            f"Recall={avg_metrics['Recall']:.4f}{COLORS.ENDC}"
        )
        context.logger.info(
            "Round %s | Avg | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
            round_number,
            avg_metrics["Acc"],
            avg_metrics["F1"],
            avg_metrics["Precision"],
            avg_metrics["Recall"],
        )

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        return round_metrics
