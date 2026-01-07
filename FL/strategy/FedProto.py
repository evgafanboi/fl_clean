from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..pipeline import PipelineContext, evaluate_model
from .base import DistillationStrategy
from .common import create_model, create_private_dataset


def extract_class_prototypes(
    model_wrapper,
    X_path: str,
    y_path: str,
    num_classes: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper

    feature_extractor = tf.keras.Model(
        inputs=keras_model.input,
        outputs=keras_model.layers[-2].output,
    )

    X_mmap = np.load(X_path, mmap_mode="r")
    y_mmap = np.load(y_path, mmap_mode="r")
    total_samples = X_mmap.shape[0]

    feature_dim = int(feature_extractor.output_shape[-1])
    class_features_sum = {c: np.zeros(feature_dim, dtype=np.float32) for c in range(num_classes)}
    class_counts = {c: 0 for c in range(num_classes)}

    chunk_size = 10000
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
        y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.int32)

        features = feature_extractor(X_chunk, training=False).numpy()

        for idx, label in enumerate(y_chunk):
            class_features_sum[label] += features[idx]
            class_counts[label] += 1

        del X_chunk, y_chunk, features

    prototypes: Dict[int, np.ndarray] = {}
    supports: Dict[int, int] = {}
    for class_id in range(num_classes):
        if class_counts[class_id] > 0:
            prototypes[class_id] = class_features_sum[class_id] / class_counts[class_id]
            supports[class_id] = class_counts[class_id]

    return prototypes, supports


def aggregate_prototypes(
    all_client_prototypes: Dict[int, Dict[int, Dict[str, np.ndarray | int]]],
    num_classes: int,
) -> Dict[int, np.ndarray]:
    global_prototypes: Dict[int, np.ndarray] = {}

    for class_id in range(num_classes):
        weighted_sums = []
        supports = []
        for protos in all_client_prototypes.values():
            if class_id in protos:
                entry = protos[class_id]
                weighted_sums.append(entry["prototype"] * entry["support"])
                supports.append(entry["support"])

        if weighted_sums and sum(supports) > 0:
            total_support = float(sum(supports))
            stacked = np.stack(weighted_sums, axis=0)
            weights = np.array(supports, dtype=np.float32) / total_support
            global_prototypes[class_id] = np.average(stacked, axis=0, weights=weights)

    return global_prototypes


def local_training_with_prototypes(
    model_wrapper,
    private_dataset: tf.data.Dataset,
    global_prototypes: Dict[int, np.ndarray],
    num_classes: int,
    epochs: int,
    gamma: float,
) -> None:
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper

    feature_extractor = tf.keras.Model(
        inputs=keras_model.input,
        outputs=keras_model.layers[-2].output,
    )

    feature_dim = int(feature_extractor.output_shape[-1])

    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    if hasattr(keras_model, "optimizer") and keras_model.optimizer is not None:
        optimizer = keras_model.optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        keras_model.optimizer = optimizer

    global_proto_array = np.zeros((num_classes, feature_dim), dtype=np.float32)
    has_proto = np.zeros(num_classes, dtype=bool)

    for class_id, proto in global_prototypes.items():
        global_proto_array[class_id] = proto
        has_proto[class_id] = True

    global_proto_tensor = tf.constant(global_proto_array, dtype=tf.float32)
    has_proto_tensor = tf.constant(has_proto, dtype=tf.bool)

    @tf.function
    def train_step(batch_x, batch_y):
        with tf.GradientTape() as tape:
            features = feature_extractor(batch_x, training=True)
            predictions = keras_model(batch_x, training=True)

            ce_loss = ce_loss_fn(batch_y, predictions)

            labels = tf.argmax(batch_y, axis=1)
            mask = tf.gather(has_proto_tensor, labels)

            proto_loss = 0.0
            if tf.reduce_any(mask):
                valid_features = tf.boolean_mask(features, mask)
                valid_labels = tf.boolean_mask(labels, mask)
                proto_targets = tf.gather(global_proto_tensor, valid_labels)
                distances = tf.sqrt(tf.reduce_sum(tf.square(valid_features - proto_targets), axis=1) + 1e-8)
                proto_loss = gamma * tf.reduce_mean(distances)

            loss = ce_loss + proto_loss

        gradients = tape.gradient(loss, keras_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))

    for _ in range(epochs):
        for batch_x, batch_y in private_dataset:
            train_step(batch_x, batch_y)


class FedProto(DistillationStrategy):
    name = "FedProto"

    def extra_log_tokens(self) -> Dict[str, float]:
        return {"gamma": self.config.gamma}

    def setup(self, context: PipelineContext) -> None:
        print(f"{COLORS.OKGREEN}Using FULL test set{COLORS.ENDC}")
        for client_id, paths in enumerate(context.paths):
            model = create_model(context.input_dim, context.num_classes, context.config.batch_size)
            context.add_client_state(client_id, model, paths)

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        config = context.config
        round_start = time.time()

        print(f"\n{COLORS.OKCYAN}[STEP 1/4] Local training{COLORS.ENDC}")

        for state in context.client_states:
            dataset = create_private_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                context.input_dim,
                context.num_classes,
                config.batch_size,
            )
            history = state.model.fit(dataset, epochs=config.epochs, verbose=1)
            if "loss" in history.history:
                final_loss = history.history["loss"][-1]
                print(f"Client {state.client_id}: loss {final_loss:.4f}")
            del dataset
            aggressive_memory_cleanup()

        print(f"\n{COLORS.OKCYAN}[STEP 2/4] Computing prototypes{COLORS.ENDC}")
        all_client_prototypes: Dict[int, Dict[int, Dict[str, np.ndarray | int]]] = {}

        for state in context.client_states:
            prototypes, supports = extract_class_prototypes(
                state.model,
                state.paths["train_X"],
                state.paths["train_y"],
                context.num_classes,
            )

            proto_dict: Dict[int, Dict[str, np.ndarray | int]] = {}
            for class_id, proto in prototypes.items():
                proto_dict[class_id] = {"prototype": proto, "support": supports[class_id]}
            all_client_prototypes[state.client_id] = proto_dict

        print(f"\n{COLORS.OKCYAN}[STEP 3/4] Aggregating prototypes{COLORS.ENDC}")
        global_prototypes = aggregate_prototypes(all_client_prototypes, context.num_classes)
        context.shared_state["global_prototypes"] = global_prototypes

        print(f"\n{COLORS.OKCYAN}[STEP 4/4] Local training with global prototypes{COLORS.ENDC}")
        for state in context.client_states:
            dataset = create_private_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                context.input_dim,
                context.num_classes,
                config.batch_size,
            )
            if global_prototypes:
                local_training_with_prototypes(
                    state.model,
                    dataset,
                    global_prototypes,
                    context.num_classes,
                    config.epochs,
                    config.gamma,
                )
            else:
                state.model.fit(dataset, epochs=config.epochs, verbose=1)
            del dataset
            aggressive_memory_cleanup()

        all_client_metrics = []
        round_metrics: Dict[int, Dict[str, float]] = {}
        
        for state in context.client_states:
            metrics = evaluate_model(state.model, context.test_dataset, context.test_labels)
            all_client_metrics.append(metrics)
            round_metrics[state.client_id] = metrics
            if config.personalized_eval:
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