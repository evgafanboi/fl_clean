from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np
from ..backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext
from ..logging_utils import log_timestamp
from .base import DistillationStrategy
from .common import create_model, create_private_dataset, load_public_dataset_from_clients


def _compute_class_metrics_pt(model_wrapper, aux_dataset, num_classes):
    import torch
    from ..backend import get_torch_device
    dev = get_torch_device()
    net = model_wrapper.nn
    net.to(dev)
    net.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in aux_dataset:
            X_b = X_b.to(dev)
            preds = net(X_b).argmax(dim=1).cpu().numpy()
            labels = y_b.argmax(dim=1).numpy() if y_b.ndim > 1 else y_b.numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        confusion[t, p] += 1
    M_class = np.zeros(num_classes, dtype=np.float32)
    for k in range(num_classes):
        row_sum = confusion[k, :].sum()
        if row_sum == 0:
            continue
        A_k_k = confusion[k, k] / row_sum
        confusion_rates = []
        for j in range(num_classes):
            if j == k:
                continue
            js = confusion[j, :].sum()
            if js > 0:
                confusion_rates.append(confusion[j, k] / js)
        max_confusion = max(confusion_rates) if confusion_rates else 0.0
        M_class[k] = A_k_k * (1.0 - max_confusion)
    return M_class


def _train_with_ssd_loss_pt(model_wrapper, dataloader, global_model, m_class_np, m_max, num_classes, epochs):
    import torch
    import torch.nn.functional as F
    from ..backend import get_torch_device
    dev = get_torch_device()
    net = model_wrapper.nn
    global_net = global_model.nn
    opt = model_wrapper.optimizer
    net.to(dev); net.train()
    global_net.to(dev); global_net.eval()
    m_class_t = torch.from_numpy(m_class_np.reshape(1, -1)).to(dev)
    print(f"  Training with CE+SSD loss (m_max={m_max:.2f}) for {epochs} epochs...")
    for epoch in range(epochs):
        ce_sum, ssd_sum, total_sum, num_b = 0.0, 0.0, 0.0, 0
        for X_b, y_b in dataloader:
            X_b = X_b.to(dev, non_blocking=True)
            y_b = y_b.to(dev, non_blocking=True)
            y_labels = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
            local_logits = net(X_b, return_logits=True)
            local_probs = F.softmax(local_logits, dim=-1)
            with torch.no_grad():
                global_logits = global_net(X_b, return_logits=True)
                global_probs = F.softmax(global_logits, dim=-1)
            p_g_k2 = global_probs[torch.arange(len(y_labels), device=dev), y_labels]
            M_sample = (1.0 - torch.sqrt(torch.clamp(1.0 - p_g_k2, min=0.0))).unsqueeze(1)
            M = m_max * F.relu(m_class_t * M_sample - 0.1)
            ssd_loss = torch.mean(torch.sum((M * (global_logits - local_logits)) ** 2, dim=1))
            ce_loss = F.cross_entropy(local_logits, y_labels)
            loss = ce_loss + ssd_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            ce_sum += ce_loss.item(); ssd_sum += ssd_loss.item(); total_sum += loss.item(); num_b += 1
        if num_b > 0:
            print(f"    Epoch {epoch + 1}/{epochs} - CE: {ce_sum/num_b:.4f}, SSD: {ssd_sum/num_b:.4f}, Total: {total_sum/num_b:.4f}")


def aggregate_weights(client_weights: Sequence[List[np.ndarray]], sample_sizes: Sequence[int]) -> List[np.ndarray]:
    total_samples = float(sum(sample_sizes))
    aggregated: List[np.ndarray] = []
    for params in zip(*client_weights):
        stacked = np.stack(params, axis=0)
        weighted_sum = np.tensordot(np.array(sample_sizes, dtype=np.float32), stacked, axes=(0, 0)) / total_samples
        aggregated.append(weighted_sum)
    return aggregated


def compute_class_metrics(model_wrapper, aux_dataset: tf.data.Dataset, num_classes: int) -> np.ndarray:
    if not _use_tf():
        return _compute_class_metrics_pt(model_wrapper, aux_dataset, num_classes)
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


def _get_ssd_models(model_wrapper, global_model):
    """Build or retrieve cached logits sub-models for SSD training."""
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper
    global_keras = global_model.model if hasattr(global_model, "model") else global_model

    if not hasattr(model_wrapper, "_ssd_logits_model"):
        logits_layer = keras_model.get_layer("logits")
        model_wrapper._ssd_logits_model = tf.keras.Model(
            inputs=keras_model.input, outputs=[logits_layer.output, keras_model.output]
        )

    if not hasattr(model_wrapper, "_ssd_global_logits_model") or model_wrapper._ssd_global_keras is not global_keras:
        global_logits_layer = global_keras.get_layer("logits")
        model_wrapper._ssd_global_logits_model = tf.keras.Model(
            inputs=global_keras.input, outputs=global_logits_layer.output
        )
        model_wrapper._ssd_global_keras = global_keras

    return model_wrapper._ssd_logits_model, model_wrapper._ssd_global_logits_model


def train_with_ssd_loss(
    model_wrapper,
    private_dataset: tf.data.Dataset,
    global_model,
    m_class_expanded: tf.Tensor,
    m_max: float,
    num_classes: int,
    epochs: int,
) -> None:
    if not _use_tf():
        return _train_with_ssd_loss_pt(model_wrapper, private_dataset, global_model, m_class_expanded, m_max, num_classes, epochs)
    """Train with CE + selective self-distillation (L = L_CE + L_SSD)."""
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper

    optimizer = keras_model.optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model.optimizer = optimizer

    ce_loss_fn = keras_model.loss if hasattr(keras_model, "loss") else tf.keras.losses.CategoricalCrossentropy()

    logits_model, global_logits_model = _get_ssd_models(model_wrapper, global_model)

    m_max_tensor = tf.constant(m_max, dtype=tf.float32)

    if not hasattr(model_wrapper, "_ssd_train_step"):
        @tf.function(reduce_retracing=True)
        def train_step(batch_X, batch_y, m_class, m_max_val):
            with tf.GradientTape() as tape:
                local_logits, predictions = logits_model(batch_X, training=True)

                global_logits = global_logits_model(batch_X, training=False)
                global_probs = tf.nn.softmax(global_logits)

                true_labels = tf.cast(tf.argmax(batch_y, axis=1), tf.int32)
                batch_size = tf.shape(batch_y)[0]

                indices = tf.stack([tf.range(batch_size), true_labels], axis=1)
                p_g_k2 = tf.gather_nd(global_probs, indices)
                M_sample_expanded = tf.expand_dims(1.0 - tf.sqrt(tf.maximum(1.0 - p_g_k2, 0.0)), axis=1)

                M = m_max_val * tf.nn.relu(m_class * M_sample_expanded - 0.1)

                logit_diff = M * (global_logits - local_logits)
                ssd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logit_diff), axis=1))

                ce_loss = ce_loss_fn(batch_y, predictions)
                total_loss = ce_loss + ssd_loss

            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))

            return ce_loss, ssd_loss, total_loss

        model_wrapper._ssd_train_step = train_step

    train_step = model_wrapper._ssd_train_step

    print(f"  Training with CE+SSD loss (m_max={m_max:.2f}) for {epochs} epochs...")

    for epoch in range(epochs):
        ce_sum, ssd_sum, total_sum = 0.0, 0.0, 0.0
        num_batches = 0

        for batch_X, batch_y in private_dataset:
            ce_loss, ssd_loss, total_loss = train_step(batch_X, batch_y, m_class_expanded, m_max_tensor)
            ce_sum += float(ce_loss)
            ssd_sum += float(ssd_loss)
            total_sum += float(total_loss)
            num_batches += 1

        if num_batches > 0:
            print(
                f"    Epoch {epoch + 1}/{epochs} - CE: {ce_sum / num_batches:.4f}, "
                f"SSD: {ssd_sum / num_batches:.4f}, Total: {total_sum / num_batches:.4f}"
            )


class FedSSD(DistillationStrategy):
    name = "FedSSD"
    has_global_model = True

    def extra_log_tokens(self) -> Dict[str, float]:
        return {"m_max": self.config.m_max}

    def setup(self, context: PipelineContext) -> None:
        config = context.config
        aux_dataset, public_len = load_public_dataset_from_clients(
            context.paths,
            batch_size=config.batch_size,
            num_classes=context.num_classes,
            shuffle=True,
            return_labels=True,
        )
        context.shared_state.update(
            {
                "aux_dataset": aux_dataset,
                "sample_sizes": [],
            }
        )

        print(f"{COLORS.OKGREEN}Initializing global model with random weights{COLORS.ENDC}")

        global_model = create_model(
            context.input_dim,
            context.num_classes,
            config.batch_size,
            model_type=config.model_type,
        )

        print(f"{COLORS.OKCYAN}Computing initial credibility matrix on auxiliary dataset{COLORS.ENDC}")
        M_class = compute_class_metrics(global_model, aux_dataset, context.num_classes)
        print(
            f"  M_class statistics -> mean: {M_class.mean():.4f}, max: {M_class.max():.4f}"
        )

        sample_sizes: List[int] = []
        for client_id, paths in enumerate(context.paths):
            context.add_client_state(client_id, None, paths)

            y_mmap = np.load(paths["train_y"], mmap_mode="r")
            sample_sizes.append(int(y_mmap.shape[0]))
            del y_mmap

        context.shared_state["global_model"] = global_model
        context.shared_state["sample_sizes"] = sample_sizes

        print(f"{COLORS.OKGREEN}Setup complete - broadcasting initial model to clients{COLORS.ENDC}")

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        log_timestamp(context.logger, f"--- Round {round_number} started ---")
        
        config = context.config
        global_model = context.shared_state["global_model"]
        aux_dataset = context.shared_state["aux_dataset"]
        sample_sizes: List[int] = context.shared_state["sample_sizes"]

        print(f"\n{COLORS.OKCYAN}[STEP 1/2] Computing class metrics on auxiliary dataset{COLORS.ENDC}")
        M_class = compute_class_metrics(global_model, aux_dataset, context.num_classes)
        print(
            f"  M_class statistics -> mean: {M_class.mean():.4f}, max: {M_class.max():.4f}"
        )

        m_class_expanded = np.expand_dims(M_class, axis=0) if not _use_tf() else tf.constant(np.expand_dims(M_class, axis=0), dtype=tf.float32)
        global_weights = global_model.get_weights()

        print(f"\n{COLORS.OKCYAN}[STEP 2/2] Client selective soft distillation training{COLORS.ENDC}")
        client_weights: List[List[np.ndarray]] = []
        pool = context.model_pool
        wrapper = pool._available[-1]

        for idx, state in enumerate(context.client_states):
            print(f"\n{COLORS.BOLD}Client {state.client_id}{COLORS.ENDC}")
            wrapper.set_weights(global_weights)

            train_dataset = create_private_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                context.input_dim,
                context.num_classes,
                config.batch_size,
            )

            train_with_ssd_loss(
                wrapper,
                train_dataset,
                global_model,
                m_class_expanded,
                config.m_max,
                context.num_classes,
                config.epochs,
            )

            client_weights.append(wrapper.get_weights())
            del train_dataset
            aggressive_memory_cleanup()

        aggregated_weights = aggregate_weights(client_weights, sample_sizes)
        global_model.set_weights(aggregated_weights)

        context.shared_state["global_model"] = global_model
        
        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        
        context.logger.info(
            f"Round {round_number} completed"
        )
        log_timestamp(context.logger, f"--- Round {round_number} completed in {round_time:.2f}s ---")
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        return {}
