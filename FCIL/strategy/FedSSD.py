"""Selective Self-Distillation (SSD) module for FCIL.

Adds L_SSD to any CIL local objective. The distillation weight is dynamically
computed per-sample from two metrics:
  M_class  - server-side, derived from confusion matrix on a public dataset
  M_sample - client-side, per-sample, from global model confidence

Final weight: M = m_max * relu(M_class * M_sample - 0.1)
L_SSD = mean( sum( (M * (global_logits - local_logits))^2 ) )
"""

import numpy as np
import tensorflow as tf

PRED_BATCH = 4096


def compute_class_metrics(model_wrapper, X, y, num_classes, label_map=None):
    """Compute M_class from confusion matrix on labeled data.
    Mirrors FL/strategy/FedSSD.py compute_class_metrics exactly."""
    keras_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper

    all_preds, all_labels = [], []
    y_raw = y if len(y.shape) == 1 else np.argmax(y, axis=1)

    for i in range(0, X.shape[0], PRED_BATCH):
        chunk = np.array(X[i:i + PRED_BATCH], dtype=np.float32)
        preds = keras_model(chunk, training=False)
        all_preds.extend(tf.argmax(preds, axis=1).numpy())
        if label_map:
            all_labels.extend(np.vectorize(label_map.get)(y_raw[i:i + PRED_BATCH]))
        else:
            all_labels.extend(y_raw[i:i + PRED_BATCH])
        del chunk, preds

    confusion = tf.math.confusion_matrix(all_labels, all_preds, num_classes=num_classes).numpy()
    M_class = np.zeros(num_classes, dtype=np.float32)

    for k in range(num_classes):
        class_total = confusion[k, :].sum()
        if class_total == 0:
            continue
        A_k_k = confusion[k, k] / class_total
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


def build_ssd_components(model, global_model, M_class, m_max):
    """Build frozen global logits model and TF constants for SSD loss.
    Returns (global_logits_model, M_class_tf, m_max_tf)."""
    global_keras = global_model.model if hasattr(global_model, 'model') else global_model
    frozen = tf.keras.models.clone_model(global_keras)
    frozen.set_weights(global_keras.get_weights())
    frozen.trainable = False
    global_logits_model = tf.keras.Model(
        frozen.input, frozen.get_layer('logits').output)

    M_class_tf = tf.constant(tf.expand_dims(M_class, axis=0), dtype=tf.float32)
    m_max_tf = tf.constant(m_max, dtype=tf.float32)
    return global_logits_model, M_class_tf, m_max_tf
