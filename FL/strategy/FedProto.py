from __future__ import annotations

import gc
import time
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..context import PipelineContext, evaluate_model
from .base import DistillationStrategy
from .common import create_model, create_private_dataset


def extract_class_prototypes(
    model_wrapper,
    X_path: str,
    y_path: str,
    num_classes: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    feature_model = model_wrapper.get_feature_model()

    X_mmap = np.load(X_path, mmap_mode="r")
    y_mmap = np.load(y_path, mmap_mode="r")
    total_samples = X_mmap.shape[0]

    feature_dim = int(feature_model.output_shape[-1])
    class_features_sum = {c: np.zeros(feature_dim, dtype=np.float32) for c in range(num_classes)}
    class_counts = {c: 0 for c in range(num_classes)}

    chunk_size = 100_000
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
        y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.int32)

        features = feature_model.predict(X_chunk, batch_size=8192, verbose=0)

        for c in range(num_classes):
            mask = y_chunk == c
            if np.any(mask):
                class_features_sum[c] += features[mask].sum(axis=0)
                class_counts[c] += int(mask.sum())

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

    feature_model = model_wrapper.get_feature_model()
    dual_model = tf.keras.Model(
        inputs=keras_model.input,
        outputs=[feature_model.output, keras_model.output],
    )

    feature_dim = int(dual_model.output[0].shape[-1])

    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    l1_loss_fn = tf.keras.losses.MeanAbsoluteError()

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
    gamma_tf = tf.constant(gamma, dtype=tf.float32)

    @tf.function
    def train_step(batch_x, batch_y):
        with tf.GradientTape() as tape:
            features, predictions = dual_model(batch_x, training=True)
            ce_loss = ce_loss_fn(batch_y, predictions)

            labels = tf.argmax(batch_y, axis=1)
            proto_targets = tf.gather(global_proto_tensor, labels)
            mask = tf.gather(has_proto_tensor, labels)
            mask_expanded = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            proto_new = mask_expanded * proto_targets + (1.0 - mask_expanded) * features
            proto_loss = gamma_tf * l1_loss_fn(proto_new, features)

            loss = ce_loss + proto_loss
        gradients = tape.gradient(loss, keras_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))

    for _ in range(epochs):
        for batch_x, batch_y in private_dataset:
            train_step(batch_x, batch_y)

    del dual_model, feature_model
    model_wrapper._feature_model = None


class FedProto(DistillationStrategy):
    name = "FedProto"

    def extra_log_tokens(self) -> Dict[str, float]:
        return {"gamma": self.config.gamma}

    def setup(self, context: PipelineContext) -> None:
        print(f"{COLORS.OKGREEN}Using FULL test set{COLORS.ENDC}")
        for client_id, paths in enumerate(context.paths):
            context.add_client_state(client_id, None, paths)

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        config = context.config
        round_start = time.time()
        pool = context.model_pool

        global_prototypes: Dict[int, np.ndarray] = context.shared_state.get("global_prototypes", {})

        print(f"\n{COLORS.OKCYAN}[STEP 1/2] Local training + prototype extraction{COLORS.ENDC}")

        all_client_prototypes: Dict[int, Dict[int, Dict[str, np.ndarray | int]]] = {}
        all_client_metrics = []
        round_metrics: Dict[int, Dict[str, float]] = {}
        is_last_round = round_number == config.rounds
        do_eval = not config.skip_eval or is_last_round

        for client_idx, state in enumerate(context.client_states):
            print(f"\n{COLORS.BOLD}Client {state.client_id}{COLORS.ENDC}")
            model = pool.checkout(state.client_id)

            dataset = create_private_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                context.input_dim,
                context.num_classes,
                config.batch_size,
            )
            local_training_with_prototypes(
                model, dataset, global_prototypes,
                context.num_classes, config.epochs, config.gamma,
            )
            del dataset

            prototypes, supports = extract_class_prototypes(
                model,
                state.paths["train_X"],
                state.paths["train_y"],
                context.num_classes,
            )
            model._feature_model = None

            proto_dict: Dict[int, Dict[str, np.ndarray | int]] = {}
            for class_id, proto in prototypes.items():
                proto_dict[class_id] = {"prototype": proto, "support": supports[class_id]}
            all_client_prototypes[state.client_id] = proto_dict
            del prototypes, supports

            if do_eval:
                metrics = evaluate_model(model, context.test_dataset, context.test_labels)
                all_client_metrics.append(metrics)
                round_metrics[state.client_id] = metrics
                context.logger.info(
                    "Round %s | Client %s | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Loss: %.4f",
                    round_number, state.client_id,
                    metrics["Acc"], metrics["F1"], metrics["Precision"], metrics["Recall"], metrics["Loss"],
                )
                print(
                    f"{COLORS.OKGREEN}Client {state.client_id}: Acc={metrics['Acc']:.4f}, F1={metrics['F1']:.4f}, "
                    f"Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, Loss={metrics['Loss']:.4f}{COLORS.ENDC}"
                )

            pool.checkin(state.client_id, model)
            gc.collect()

        print(f"\n{COLORS.OKCYAN}[STEP 2/2] Aggregating prototypes & summary{COLORS.ENDC}")

        global_prototypes = aggregate_prototypes(all_client_prototypes, context.num_classes)
        context.shared_state["global_prototypes"] = global_prototypes
        del all_client_prototypes

        if all_client_metrics:
            avg_metrics = {
                k: np.mean([m[k] for m in all_client_metrics])
                for k in ("Acc", "F1", "Precision", "Recall", "Loss")
            }
            round_metrics[-1] = avg_metrics

            print(
                f"{COLORS.OKGREEN}Round {round_number} - Avg Acc={avg_metrics['Acc']:.4f}, "
                f"F1={avg_metrics['F1']:.4f}, Precision={avg_metrics['Precision']:.4f}, "
                f"Recall={avg_metrics['Recall']:.4f}, Loss={avg_metrics['Loss']:.4f}{COLORS.ENDC}"
            )
            context.logger.info(
                "Round %s | Avg | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Loss: %.4f",
                round_number,
                avg_metrics["Acc"], avg_metrics["F1"], avg_metrics["Precision"],
                avg_metrics["Recall"], avg_metrics["Loss"],
            )

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        return round_metrics