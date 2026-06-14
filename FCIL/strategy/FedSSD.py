"""Selective Self-Distillation (SSD) module for FCIL.

Adds L_SSD to any CIL local objective. The distillation weight is dynamically
computed per-sample from two metrics:
  M_class  - server-side, derived from confusion matrix on a public dataset
  M_sample - client-side, per-sample, from global model confidence

Final weight: M = m_max * relu(M_class * M_sample - 0.1)
L_SSD = mean( sum( (M * (global_logits - local_logits))^2 ) )
"""

import numpy as np
from ..backend import use_tf as _use_tf

PRED_BATCH = 4096


def compute_class_metrics(model_wrapper, X, y, num_classes, label_map=None):
    y_raw = y if len(y.shape) == 1 else np.argmax(y, axis=1)
    all_preds, all_labels = [], []

    if not _use_tf():
        for i in range(0, X.shape[0], PRED_BATCH):
            chunk = np.array(X[i:i + PRED_BATCH], dtype=np.float32)
            preds = model_wrapper.predict(chunk, batch_size=PRED_BATCH)
            all_preds.extend(np.argmax(preds, axis=1))
            if label_map:
                all_labels.extend(np.vectorize(label_map.get)(y_raw[i:i + PRED_BATCH]))
            else:
                all_labels.extend(y_raw[i:i + PRED_BATCH])
            del chunk, preds
    else:
        keras_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
        for i in range(0, X.shape[0], PRED_BATCH):
            chunk = np.array(X[i:i + PRED_BATCH], dtype=np.float32)
            preds = keras_model(chunk, training=False)
            all_preds.extend(tf.argmax(preds, axis=1).numpy())
            if label_map:
                all_labels.extend(np.vectorize(label_map.get)(y_raw[i:i + PRED_BATCH]))
            else:
                all_labels.extend(y_raw[i:i + PRED_BATCH])
            del chunk, preds

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(all_preds, all_labels):
        confusion[int(t), int(p)] += 1

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
    if not _use_tf():
        import copy
        frozen = copy.deepcopy(global_model)
        for p in frozen.nn.parameters():
            p.requires_grad_(False)
        frozen.nn.eval()
        return frozen.get_logits_model(), np.expand_dims(M_class, 0), float(m_max)

    global_keras = global_model.model if hasattr(global_model, 'model') else global_model
    import tensorflow as tf
    frozen = tf.keras.models.clone_model(global_keras)
    frozen.set_weights(global_keras.get_weights())
    frozen.trainable = False
    global_logits_model = tf.keras.Model(frozen.input, frozen.get_layer('logits').output)
    M_class_tf = tf.constant(tf.expand_dims(M_class, axis=0), dtype=tf.float32)
    m_max_tf = tf.constant(m_max, dtype=tf.float32)
    return global_logits_model, M_class_tf, m_max_tf
