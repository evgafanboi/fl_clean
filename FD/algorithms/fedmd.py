from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..pipeline import PipelineContext, evaluate_model
from .base import DistillationAlgorithm
from .common import (
    create_model,
    create_private_dataset,
    load_public_dataset_from_clients,
    numpy_from_dataset,
)


def generate_public_logits(model_wrapper, public_features: np.ndarray, batch_size: int) -> np.ndarray:
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else model_wrapper
    dataset = tf.data.Dataset.from_tensor_slices(public_features).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    logits: List[np.ndarray] = []
    for batch in dataset:
        outputs = logits_model(batch, training=False)
        logits.append(outputs.numpy())
    return np.concatenate(logits, axis=0)


def digest_phase(
    model_wrapper,
    consensus_logits: np.ndarray,
    public_features: np.ndarray,
    batch_size: int,
    epochs: int,
) -> None:
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else keras_model

    optimizer = keras_model.optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model.optimizer = optimizer
    mae_loss = tf.keras.losses.MeanAbsoluteError()

    dataset = (
        tf.data.Dataset.from_tensor_slices((public_features, consensus_logits))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch_X, batch_consensus in dataset:
            with tf.GradientTape() as tape:
                student_logits = logits_model(batch_X, training=True)
                loss = mae_loss(batch_consensus, student_logits)

            gradients = tape.gradient(loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))

            epoch_loss += float(loss)
            batches += 1

        if batches > 0:
            print(f"    DIGEST epoch {epoch + 1}/{epochs} - MAE {epoch_loss / batches:.4f}")


def revisit_phase(model_wrapper, private_dataset: tf.data.Dataset, epochs: int) -> None:
    print(f"    REVISIT training for {epochs} epochs")
    history = model_wrapper.fit(private_dataset, epochs=epochs, verbose=0)
    if hasattr(history, "history") and "loss" in history.history:
        print(f"      Final REVISIT loss: {history.history['loss'][-1]:.4f}")


class FedMD(DistillationAlgorithm):
    name = "FedMD"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.digest_epochs = 1
        self.transfer_epochs = config.epochs
        self.revisit_epochs = config.epochs
        self.revisit_batch_fraction = 0.25

    def setup(self, context: PipelineContext) -> None:
        print(f"{COLORS.OKGREEN}Preparing FedMD with public data from client slices{COLORS.ENDC}")
        is_sequence = context.config.model_type.lower() == "gru"

        public_unlabeled_ds, total_public = load_public_dataset_from_clients(
            context.paths,
            batch_size=context.config.batch_size,
            num_classes=context.num_classes,
            shuffle=False,
            return_labels=False,
            is_sequence=is_sequence,
        )
        public_features = numpy_from_dataset(public_unlabeled_ds)

        public_labeled_ds, _ = load_public_dataset_from_clients(
            context.paths,
            batch_size=context.config.batch_size,
            num_classes=context.num_classes,
            shuffle=True,
            return_labels=True,
            is_sequence=is_sequence,
        )
        public_labeled_ds = public_labeled_ds.cache()

        context.shared_state.update(
            {
                "public_features": public_features,
                "public_sample_count": total_public,
                "public_dataset_labeled": public_labeled_ds,
                "is_sequence": is_sequence,
            }
        )

        print(f"{COLORS.OKGREEN}Using FULL test set{COLORS.ENDC}")

        for client_id, paths in enumerate(context.paths):
            model = create_model(
                context.input_dim,
                context.num_classes,
                context.config.batch_size,
                model_type=context.config.model_type,
            )
            state = context.add_client_state(client_id, model, paths)
            print(f"Client {state.client_id}: initial transfer learning")
            state.model.fit(public_labeled_ds, epochs=self.transfer_epochs, verbose=0)

            private_dataset = create_private_dataset(
                paths["train_X"],
                paths["train_y"],
                context.input_dim,
                context.num_classes,
                context.config.batch_size,
                is_sequence=is_sequence,
            )
            state.model.fit(private_dataset, epochs=self.revisit_epochs, verbose=0)
            del private_dataset
            aggressive_memory_cleanup()

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        public_features: np.ndarray = context.shared_state["public_features"]
        public_dataset_labeled: tf.data.Dataset = context.shared_state["public_dataset_labeled"]

        print(f"\n{COLORS.OKCYAN}[STEP 1/3] Generating public logits{COLORS.ENDC}")
        all_logits = []
        for state in context.client_states:
            logits = generate_public_logits(state.model, public_features, config.batch_size)
            print(f"  Client {state.client_id}: logits shape {logits.shape}")
            all_logits.append(logits)

        print(f"\n{COLORS.OKCYAN}[STEP 2/3] Computing consensus logits{COLORS.ENDC}")
        consensus_logits = np.mean(all_logits, axis=0)
        print(f"  Consensus shape: {consensus_logits.shape}")

        print(f"\n{COLORS.OKCYAN}[STEP 3/3] Digest and revisit phases{COLORS.ENDC}")
        revisit_batch_size = max(1, int(config.batch_size * self.revisit_batch_fraction))

        for state in context.client_states:
            print(f"\n{COLORS.BOLD}Client {state.client_id}{COLORS.ENDC}")
            digest_phase(state.model, consensus_logits, public_features, config.batch_size, self.digest_epochs)

            private_dataset = create_private_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                context.input_dim,
                context.num_classes,
                revisit_batch_size,
                is_sequence=context.shared_state["is_sequence"],
            )
            revisit_phase(state.model, private_dataset, self.revisit_epochs)
            del private_dataset
            aggressive_memory_cleanup()

        round_metrics: Dict[int, Dict[str, float]] = {}
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

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        return round_metrics
