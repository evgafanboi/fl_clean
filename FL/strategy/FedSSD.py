from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..pipeline import PipelineContext, evaluate_model
from .base import DistillationStrategy
from .common import create_model, create_private_dataset, load_public_dataset_from_clients


def aggregate_weights(client_weights: Sequence[List[np.ndarray]], sample_sizes: Sequence[int]) -> List[np.ndarray]:
    total_samples = float(sum(sample_sizes))
    aggregated: List[np.ndarray] = []
    for params in zip(*client_weights):
        stacked = np.stack(params, axis=0)
        weighted_sum = np.tensordot(np.array(sample_sizes, dtype=np.float32), stacked, axes=(0, 0)) / total_samples
        aggregated.append(weighted_sum)
    return aggregated


def compute_class_metrics(model_wrapper, aux_dataset: tf.data.Dataset, num_classes: int) -> np.ndarray:
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper

    all_preds = []
    all_labels = []
    for batch_X, batch_y in aux_dataset:
        preds = keras_model(batch_X, training=False)
        pred_classes = tf.argmax(preds, axis=1).numpy()
        true_classes = tf.argmax(batch_y, axis=1).numpy()
        all_preds.extend(pred_classes)
        all_labels.extend(true_classes)

    confusion = tf.math.confusion_matrix(all_labels, all_preds, num_classes=num_classes).numpy()
    M_class = np.zeros(num_classes, dtype=np.float32)

    for k in range(num_classes):
        class_total = confusion[k, :].sum()
        if class_total == 0:
            continue
        A_k_k = confusion[k, k] / class_total if class_total > 0 else 0.0
        confusion_rates = []
        for j in range(num_classes):
            if j == k:
                continue
            row_sum = confusion[j, :].sum()
            if row_sum > 0:
                confusion_rates.append(confusion[j, k] / row_sum)
        max_confusion = max(confusion_rates) if confusion_rates else 0.0
        M_class[k] = A_k_k * (1.0 - max_confusion)

    return M_class


def train_with_ssd_loss(
    model_wrapper,
    private_dataset: tf.data.Dataset,
    global_model,
    M_class: np.ndarray,
    m_max: float,
    num_classes: int,
    epochs: int,
) -> None:
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper
    global_keras = global_model.model if hasattr(global_model, "model") else global_model

    optimizer = keras_model.optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model.optimizer = optimizer
    ce_loss_fn = keras_model.loss if hasattr(keras_model, "loss") else tf.keras.losses.CategoricalCrossentropy()

    logits_layer = keras_model.get_layer("logits")
    logits_model = tf.keras.Model(inputs=keras_model.input, outputs=[logits_layer.output, keras_model.output])

    global_logits_layer = global_keras.get_layer("logits")
    global_logits_model = tf.keras.Model(inputs=global_keras.input, outputs=global_logits_layer.output)

    M_class_expanded = tf.constant(tf.expand_dims(M_class, axis=0), dtype=tf.float32)

    @tf.function
    def train_step(batch_X, batch_y):
        with tf.GradientTape() as tape:
            local_logits, predictions = logits_model(batch_X, training=True)
            ce_loss = ce_loss_fn(batch_y, predictions)

            global_logits = global_logits_model(batch_X, training=False)
            global_probs = tf.nn.softmax(global_logits)

            true_labels = tf.cast(tf.argmax(batch_y, axis=1), tf.int32)
            batch_size = tf.shape(batch_y)[0]

            indices = tf.stack([tf.range(batch_size), true_labels], axis=1)
            p_g_k2 = tf.gather_nd(global_probs, indices)
            M_sample_expanded = tf.expand_dims(1.0 - tf.sqrt(tf.maximum(1.0 - p_g_k2, 0.0)), axis=1)

            M = tf.nn.relu(m_max * M_class_expanded * M_sample_expanded - 0.1)

            logit_diff = M * (global_logits - local_logits)
            ssd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logit_diff), axis=1))

            total_loss = ce_loss + ssd_loss

        gradients = tape.gradient(total_loss, keras_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))

        return ce_loss, ssd_loss, total_loss

    print(f"  Training with SSD loss (m_max={m_max:.2f}) for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_losses = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        num_batches = 0

        for batch_X, batch_y in private_dataset:
            ce_loss, ssd_loss, total_loss = train_step(batch_X, batch_y)
            epoch_losses = epoch_losses + tf.stack([ce_loss, ssd_loss, total_loss])
            num_batches += 1

        if num_batches > 0:
            avg_losses = epoch_losses / num_batches
            print(
                f"    Epoch {epoch + 1}/{epochs} - CE: {avg_losses[0]:.4f}, "
                f"SSD: {avg_losses[1]:.4f}, Total: {avg_losses[2]:.4f}"
            )


class FedSSD(DistillationStrategy):
    name = "FedSSD"

    def setup(self, context: PipelineContext) -> None:
        config = context.config
        is_sequence = config.model_type.lower() == "gru"
        aux_dataset, public_len = load_public_dataset_from_clients(
            context.paths,
            batch_size=config.batch_size,
            num_classes=context.num_classes,
            shuffle=True,
            return_labels=True,
            is_sequence=is_sequence,
        )
        context.shared_state.update(
            {
                "aux_dataset": aux_dataset,
                "is_sequence": is_sequence,
                "sample_sizes": [],
            }
        )

        print(f"{COLORS.OKGREEN}Warmup training for all clients{COLORS.ENDC}")

        client_weights: List[List[np.ndarray]] = []
        sample_sizes: List[int] = []

        for client_id, paths in enumerate(context.paths):
            model = create_model(
                context.input_dim,
                context.num_classes,
                config.batch_size,
                model_type=config.model_type,
            )
            state = context.add_client_state(client_id, model, paths)

            train_dataset = create_private_dataset(
                paths["train_X"],
                paths["train_y"],
                context.input_dim,
                context.num_classes,
                config.batch_size,
                is_sequence=is_sequence,
            )
            state.model.fit(train_dataset, epochs=config.epochs, verbose=1)
            client_weights.append(state.model.get_weights())

            y_mmap = np.load(paths["train_y"], mmap_mode="r")
            sample_sizes.append(int(y_mmap.shape[0]))
            del y_mmap, train_dataset
            aggressive_memory_cleanup()

        aggregated_weights = aggregate_weights(client_weights, sample_sizes)
        global_model = create_model(
            context.input_dim,
            context.num_classes,
            config.batch_size,
            model_type=config.model_type,
        )
        global_model.set_weights(aggregated_weights)

        context.shared_state["global_model"] = global_model
        context.shared_state["sample_sizes"] = sample_sizes

        global_metrics = evaluate_model(global_model, context.test_dataset, context.test_labels)
        print(
            f"{COLORS.OKGREEN}Round 0 - Global Acc={global_metrics['Acc']:.4f}, F1={global_metrics['F1']:.4f}, "
            f"Precision={global_metrics['Precision']:.4f}, Recall={global_metrics['Recall']:.4f}{COLORS.ENDC}"
        )
        context.logger.info(
            "Round 0 | GLOBAL | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
            global_metrics["Acc"],
            global_metrics["F1"],
            global_metrics["Precision"],
            global_metrics["Recall"],
        )

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        global_model = context.shared_state["global_model"]
        aux_dataset = context.shared_state["aux_dataset"]
        sample_sizes: List[int] = context.shared_state["sample_sizes"]

        print(f"\n{COLORS.OKCYAN}[STEP 1/2] Computing class metrics on auxiliary dataset{COLORS.ENDC}")
        M_class = compute_class_metrics(global_model, aux_dataset, context.num_classes)
        print(
            f"  M_class statistics -> min: {M_class.min():.4f}, max: {M_class.max():.4f}, mean: {M_class.mean():.4f}"
        )

        print(f"\n{COLORS.OKCYAN}[STEP 2/2] Client selective soft distillation training{COLORS.ENDC}")
        client_weights: List[List[np.ndarray]] = []

        for idx, state in enumerate(context.client_states):
            print(f"\n{COLORS.BOLD}Client {state.client_id}{COLORS.ENDC}")
            state.model.set_weights(global_model.get_weights())

            train_dataset = create_private_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                context.input_dim,
                context.num_classes,
                config.batch_size,
                is_sequence=context.shared_state["is_sequence"],
            )

            train_with_ssd_loss(
                state.model,
                train_dataset,
                global_model,
                M_class,
                config.m_max,
                context.num_classes,
                config.epochs,
            )

            client_weights.append(state.model.get_weights())
            del train_dataset
            aggressive_memory_cleanup()

        aggregated_weights = aggregate_weights(client_weights, sample_sizes)
        global_model.set_weights(aggregated_weights)

        context.shared_state["global_model"] = global_model
        
        global_metrics = evaluate_model(global_model, context.test_dataset, context.test_labels)
        print(
            f"{COLORS.OKGREEN}Round {round_number} - Global Acc={global_metrics['Acc']:.4f}, "
            f"F1={global_metrics['F1']:.4f}, Precision={global_metrics['Precision']:.4f}, "
            f"Recall={global_metrics['Recall']:.4f}{COLORS.ENDC}"
        )
        context.logger.info(
            "Round %s | GLOBAL | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
            round_number,
            global_metrics["Acc"],
            global_metrics["F1"],
            global_metrics["Precision"],
            global_metrics["Recall"],
        )

        round_metrics: Dict[int, Dict[str, float]] = {-1: global_metrics}
        
        if config.personalized_eval:
            for state in context.client_states:
                metrics = evaluate_model(state.model, context.test_dataset, context.test_labels)
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

        for state in context.client_states:
            state.model.set_weights(global_model.get_weights())

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        return round_metrics
