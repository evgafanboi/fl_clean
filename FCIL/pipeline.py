"""FCIL Pipeline - Main execution logic for continual learning"""

import os
import json
import pickle
import dataclasses
import hashlib
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import gc

from .config import FCILConfig
from .colors import COLORS
from .backend import use_tf as _use_tf


def load_task_order(task_order_file: str) -> dict:
    """Load task order from JSON file"""
    with open(task_order_file, 'r') as f:
        task_order_str = json.load(f)
        return {int(k): v for k, v in task_order_str.items()}


def cil_partition_dir(partition_root: str, partition_type: str, n_clients: int) -> Path:
    suffix = "" if partition_type.endswith("_cil") else "_cil"
    return Path(partition_root) / f"{n_clients}_client" / f"{partition_type}{suffix}"


def get_task_files(partition_root: str, partition_type: str, n_clients: int,
                    client_id: int, task_id: int, strategy: str):
    base_dir = cil_partition_dir(partition_root, partition_type, n_clients)
    if strategy == "Centralized":
        base_dir = base_dir / "centralized"
        return {
            'X': str(base_dir / f'task_{task_id}_X_train.npy'),
            'y': str(base_dir / f'task_{task_id}_y_train.npy'),
        }
    return {
        'X': str(base_dir / f'client_{client_id}_task_{task_id}_X_train.npy'),
        'y': str(base_dir / f'client_{client_id}_task_{task_id}_y_train.npy'),
    }


def _get_opt_loss(model):
    if _use_tf():
        return model.model.optimizer, model.model.loss
    return model.optimizer, model.criterion


def _task_active_n(n_clients: int, num_tasks: int, task_id: int) -> int:
    if num_tasks != 7:
        return n_clients
    return min(n_clients, max(1, int(n_clients / 10) * (4 + task_id)))


def _eval_rounds(total_rounds: int) -> list:
    if total_rounds <= 0:
        return []
    mid = (total_rounds + 1) // 2 - 1
    return sorted({0, mid, total_rounds - 1})


def _save_client_weights(weight_dir, round_num, client_ids, client_weights):
    round_dir = weight_dir / f'round_{round_num + 1}'
    round_dir.mkdir(parents=True, exist_ok=True)
    for client_id, weights in zip(client_ids, client_weights):
        with open(round_dir / f'client_{client_id}_weight.bin', 'wb') as f:
            pickle.dump(weights, f)


def _support_weighted_metrics(task_metrics, supports, loss):
    if not task_metrics:
        keys = ['acc', 'prec_macro', 'rec_macro', 'f1_macro',
                'prec_micro', 'rec_micro', 'f1_micro',
                'prec_weighted', 'rec_weighted', 'f1_weighted']
        out = {'loss': loss}
        for k in keys:
            out[k] = 0.0
        out['cm'] = np.zeros((34, 34), dtype=int)
        return out
    pooled_cm = np.sum([m['cm'] for m in task_metrics], axis=0)
    return _metrics_from_cm(pooled_cm, loss)


def create_dataset(X_path: str, y_path: str, batch_size: int, num_classes: int, label_map: dict | None = None):
    X = np.load(X_path)
    y = np.load(y_path)

    if len(X) == 0:
        return None, 0

    if label_map:
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        y = np.vectorize(label_map.get, otypes=[np.int64])(labels)
    elif len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    y = y.astype(int)

    if len(X) == 0:
        return None, 0

    if _use_tf():
        import tensorflow as tf
        y_oh = tf.keras.utils.to_categorical(y, num_classes)
        dataset = tf.data.Dataset.from_tensor_slices((X.astype('float32'), y_oh))
        dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset, X.shape[0]

    import torch
    from torch.utils.data import TensorDataset, DataLoader
    y_oh = np.zeros((len(y), num_classes), dtype=np.float32)
    y_oh[np.arange(len(y)), y] = 1.0
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y_oh)),
        batch_size=batch_size, shuffle=True, drop_last=False)
    return loader, X.shape[0]


def create_model(num_classes: int, input_dim: int, batch_size: int, model: str = "dense"):
    if _use_tf():
        if model == "gru":
            from models.gru import GRUModel
            return GRUModel(num_classes=num_classes, input_dim=input_dim, batch_size=batch_size)
        if model == "dcblstm":
            from models.dcblstm import DCBLSTMModel
            return DCBLSTMModel(num_classes=num_classes, input_dim=input_dim, batch_size=batch_size)
        from models.dense import DenseModel
        return DenseModel(num_classes=num_classes, input_dim=input_dim, batch_size=batch_size)
    if model == "gru":
        from models.pt_gru import PTGRUModel
        return PTGRUModel(input_dim, num_classes, batch_size)
    if model == "dcblstm":
        from models.pt_dcblstm import PTDCBLSTMModel
        return PTDCBLSTMModel(input_dim, num_classes, batch_size)
    from models.pt_dense import PTDenseModel
    return PTDenseModel(input_dim, num_classes, batch_size)

def _active_class_count(cil_method, fallback: int) -> int:
    return len(cil_method.class_order) if hasattr(cil_method, "class_order") else fallback

def _prepare_active_model(cil_method, model, fallback_classes: int):
    current_classes = _active_class_count(cil_method, fallback_classes)
    if hasattr(model, "expand_classes"):
        model.expand_classes(current_classes)
    if hasattr(cil_method, "prepare_model"):
        cil_method.prepare_model(model)
    return current_classes


def train_client(model, dataset, cil_method, epochs: int):
    """Train client model with CIL method"""
    keras_model = model.model if hasattr(model, 'model') else model
    optimizer = keras_model.optimizer
    loss_fn = keras_model.loss
    
    custom_train_step = cil_method.get_train_step(model, optimizer, loss_fn)
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch in dataset:
            if custom_train_step:
                if len(batch) == 3:
                    batch_X, batch_y, batch_old = batch
                    ce_loss, ewc_loss, total = custom_train_step(batch_X, batch_y, batch_old)
                else:
                    batch_X, batch_y = batch
                    ce_loss, ewc_loss, total = custom_train_step(batch_X, batch_y)
                total_loss += total.numpy()
            else:
                batch_X, batch_y = batch
                with tf.GradientTape() as tape:
                    predictions = keras_model(batch_X, training=True)
                    loss = loss_fn(batch_y, predictions)
                
                gradients = tape.gradient(loss, keras_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
                total_loss += loss.numpy()
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_client_fast(model, dataset, train_step, epochs: int):
    if not _use_tf():
        if train_step is None:
            hist = model.fit(dataset, epochs)
            return hist.history['loss']
        import torch
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.nn.to(dev)
        epoch_losses = []
        for _ in range(epochs):
            model.nn.train()
            total_loss, num_batches = 0.0, 0
            for batch in dataset:
                _, _, total = train_step(batch[0], batch[1]) if len(batch) == 2 else train_step(batch[0], batch[1], batch[2])
                total_loss += float(total)
                num_batches += 1
            epoch_losses.append(total_loss / num_batches if num_batches else 0.0)
        return epoch_losses

    import tensorflow as tf
    keras_model = model.model if hasattr(model, 'model') else model
    epoch_losses = []
    for epoch in range(epochs):
        total_loss, num_batches = 0.0, 0
        for batch in dataset:
            if train_step:
                if len(batch) == 3:
                    _, _, total = train_step(batch[0], batch[1], batch[2])
                else:
                    _, _, total = train_step(batch[0], batch[1])
                total_loss += total.numpy()
            else:
                with tf.GradientTape() as tape:
                    predictions = keras_model(batch[0], training=True)
                    loss = keras_model.loss(batch[1], predictions)
                gradients = tape.gradient(loss, keras_model.trainable_variables)
                keras_model.optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
                total_loss += loss.numpy()
            num_batches += 1
        epoch_losses.append(total_loss / num_batches if num_batches else 0.0)
    return epoch_losses


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, batch_size: int, num_classes: int):
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  confusion_matrix, accuracy_score)
    y_true = (y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)).astype(int)
    labels = list(range(num_classes))

    if not _use_tf():
        import torch
        import torch.nn.functional as F
        y_proba = model.predict(X_test, batch_size)
        y_pred = np.argmax(y_proba, axis=1)
        logits = model.get_logits_model().predict(X_test, batch_size)
        loss = F.cross_entropy(
            torch.from_numpy(logits),
            torch.from_numpy(y_true.astype(np.int64))
        ).item()
    else:
        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        y_proba = keras_model.predict(X_test.astype('float32'), batch_size=batch_size, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        y_oh = tf.keras.utils.to_categorical(y_true, num_classes)
        loss = float(tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)(y_oh, y_proba).numpy())

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(float)
    total_support = float(support.sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        prec_per = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec_per = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1_per = np.where(prec_per + rec_per > 0, 2 * prec_per * rec_per / (prec_per + rec_per), 0.0)

    return {
        'loss': loss,
        'acc': float(accuracy_score(y_true, y_pred)),
        'prec_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'rec_macro':  float(recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'f1_macro':   float(f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'prec_micro': float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
        'rec_micro':  float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
        'f1_micro':   float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'prec_weighted': float(np.sum(prec_per * support) / total_support) if total_support > 0 else 0.0,
        'rec_weighted':  float(np.sum(rec_per * support) / total_support) if total_support > 0 else 0.0,
        'f1_weighted':   float(np.sum(f1_per * support) / total_support) if total_support > 0 else 0.0,
        'cm': cm,
    }


def evaluate_active_model(cil_method, model, X_test: np.ndarray, y_test: np.ndarray,
                          batch_size: int, num_classes: int):
    if hasattr(cil_method, 'evaluate_full'):
        return cil_method.evaluate_full(model, X_test, y_test, num_classes, batch_size=batch_size)
    eval_y = y_test
    if hasattr(cil_method, 'label_map') and cil_method.label_map:
        eval_y = cil_method.map_labels_np(eval_y)
    return evaluate_model(model, X_test, eval_y, batch_size, num_classes)


def _metrics_from_cm(cm: np.ndarray, loss: float) -> dict:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    total = int(cm.sum())
    support = cm.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    tp_s = int(tp.sum())
    fp_s = int(fp.sum())
    fn_s = int(fn.sum())
    prec_mic = tp_s / (tp_s + fp_s) if tp_s + fp_s > 0 else 0.0
    rec_mic = tp_s / (tp_s + fn_s) if tp_s + fn_s > 0 else 0.0
    f1_mic = 2 * prec_mic * rec_mic / (prec_mic + rec_mic) if prec_mic + rec_mic > 0 else 0.0
    total_support = float(support.sum())
    return {
        'loss': loss,
        'acc': float(tp.sum()) / total if total > 0 else 0.0,
        'prec_macro': float(prec.mean()),
        'rec_macro': float(rec.mean()),
        'f1_macro': float(f1.mean()),
        'prec_micro': prec_mic,
        'rec_micro': rec_mic,
        'f1_micro': f1_mic,
        'prec_weighted': float(np.sum(prec * support) / total_support) if total_support > 0 else 0.0,
        'rec_weighted': float(np.sum(rec * support) / total_support) if total_support > 0 else 0.0,
        'f1_weighted': float(np.sum(f1 * support) / total_support) if total_support > 0 else 0.0,
        'acc_per_class': np.where(support > 0, tp / support, 0.0).tolist(),
        'prec_per_class': prec.tolist(),
        'rec_per_class': rec.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.astype(int).tolist(),
        'cm': cm,
    }

def _per_task_breakdown(cm: np.ndarray, task_order: dict, label_map: dict | None, current_task_id: int) -> list[dict]:
    out = []
    for task_idx in range(current_task_id + 1):
        idxs = sorted({label_map[c] for c in task_order[task_idx] if c in label_map}) if label_map else sorted(task_order[task_idx])
        tp = np.diag(cm)[idxs].astype(float)
        support = cm[idxs, :].sum(axis=1)
        pred = cm[:, idxs].sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            prec = np.where(pred > 0, tp / pred, 0.0)
            rec = np.where(support > 0, tp / support, 0.0)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        total = int(support.sum())
        out.append({'task': task_idx, 'acc': float(tp.sum()) / total if total > 0 else 0.0,
                    'f1_macro': float(f1.mean()) if len(f1) else 0.0, 'support': total})
    return out

def _log_class_task_breakdown(log, results: dict, task_order: dict, label_map: dict | None,
                              current_task_id: int, class_order: list | None, prefix: str):
    support = results['support_per_class']
    inv_map = {v: k for k, v in label_map.items()} if label_map else {}
    class_parts = []
    for i, n in enumerate(support):
        if n <= 0:
            continue
        cls = inv_map.get(i, class_order[i] if class_order and i < len(class_order) else i)
        class_parts.append(f"C{cls}:acc={results['acc_per_class'][i]:.4f},prec={results['prec_per_class'][i]:.4f},rec={results['rec_per_class'][i]:.4f},f1={results['f1_per_class'][i]:.4f},n={n}")
    task_parts = [f"T{x['task']}:acc={x['acc']:.4f},f1={x['f1_macro']:.4f},n={x['support']}"
                  for x in _per_task_breakdown(results['cm'], task_order, label_map, current_task_id)]
    log(f"{prefix} | per-class | " + " | ".join(class_parts),
        f"{prefix} | per-class | " + " | ".join(class_parts))
    log(f"{prefix} | per-task | " + " | ".join(task_parts),
        f"{prefix} | per-task | " + " | ".join(task_parts))


def _support_weighted_metrics(task_metrics: list[dict], supports: list[int], loss: float) -> dict:
    if not task_metrics:
        keys = ['acc', 'prec_macro', 'rec_macro', 'f1_macro', 'prec_micro', 'rec_micro', 'f1_micro', 'prec_weighted', 'rec_weighted', 'f1_weighted']
        return {'loss': loss, **{k: 0.0 for k in keys}, 'cm': np.zeros((34, 34), dtype=int)}
    pooled_cm = np.sum([m['cm'] for m in task_metrics], axis=0)
    return _metrics_from_cm(pooled_cm, loss)


def _plot_confusion_matrix(cm: np.ndarray, plot_path: str, tick_labels: list | None = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = cm.shape[0]
    if tick_labels is None or len(tick_labels) != n:
        tick_labels = list(range(n))
    fig_size = max(20, int(n * 0.7))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_ylabel('True label', fontsize=14)
    ax.set_title(f'Confusion Matrix ({n}x{n})', fontsize=16)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(axis='both', labelsize=10)
    font_size = max(6, 12 - n // 8)
    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(n):
        for j in range(n):
            pct = 100.0 * cm[i, j] / row_sums[i, 0] if row_sums[i, 0] > 0 else 0.0
            ax.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha='center', va='center', fontsize=font_size)
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _fcil_fingerprint(config) -> str:
    d = dataclasses.asdict(config)
    for k in ('log_file', 'checkpoint', 'fresh_run'):
        d.pop(k, None)
    return hashlib.sha256(str(sorted(d.items())).encode()).hexdigest()[:16]


def _fcil_ckpt_dir(log_stem: str) -> str:
    return os.path.join("checkpoint", log_stem)


def _fcil_atomic_write(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _fcil_write_info(ckpt_dir: str, task_id: int, round_num: int, round_complete: bool,
                     last_client: int, config) -> None:
    lines = [
        f"task_id: {task_id}",
        f"round_num: {round_num}",
        f"round_complete: {str(round_complete).lower()}",
        f"last_client: {last_client}",
        f"config_hash: {_fcil_fingerprint(config)}",
        f"backend: {'tensorflow' if _use_tf() else 'pytorch'}",
    ]
    with open(os.path.join(ckpt_dir, "info.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _fcil_save_ckpt(ckpt_dir, task_id, round_num, global_model, cil_method, old_classes, config) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    _fcil_atomic_write(os.path.join(ckpt_dir, "global_weights.bin"), global_model.get_weights())
    _fcil_atomic_write(os.path.join(ckpt_dir, "cil_state.bin"), cil_method.get_ckpt_state())
    _fcil_atomic_write(os.path.join(ckpt_dir, "pipeline_state.bin"), {'old_classes': old_classes})
    for stale in ("round_meta.bin",):
        p = os.path.join(ckpt_dir, stale)
        if os.path.exists(p):
            os.remove(p)
    _fcil_write_info(ckpt_dir, task_id, round_num, True, -1, config)
    print(f"{COLORS.OKCYAN}Checkpoint saved (T{task_id} R{round_num + 1}) -> {ckpt_dir}{COLORS.ENDC}")


def _fcil_save_mid_ckpt(ckpt_dir, task_id, round_num, last_client,
                         client_weights, sample_sizes, global_model,
                         cil_method, old_classes, config) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    _fcil_atomic_write(os.path.join(ckpt_dir, "global_weights.bin"), global_model.get_weights())
    _fcil_atomic_write(os.path.join(ckpt_dir, "cil_state.bin"), cil_method.get_ckpt_state())
    _fcil_atomic_write(os.path.join(ckpt_dir, "pipeline_state.bin"), {'old_classes': old_classes})
    _fcil_atomic_write(os.path.join(ckpt_dir, "round_meta.bin"),
                       {'client_weights': list(client_weights), 'sample_sizes': list(sample_sizes)})
    _fcil_write_info(ckpt_dir, task_id, round_num, False, last_client, config)


def _fcil_load_ckpt(ckpt_dir: str, config) -> dict | None:
    info_path = os.path.join(ckpt_dir, "info.txt")
    if not os.path.exists(info_path):
        return None
    info = {}
    with open(info_path) as f:
        for line in f:
            k, _, v = line.strip().partition(": ")
            if k:
                info[k] = v
    if info.get("config_hash", "") != _fcil_fingerprint(config):
        print(f"{COLORS.WARNING}FCIL checkpoint config mismatch — ignoring.{COLORS.ENDC}")
        return None
    if info.get("backend", "") != ("tensorflow" if _use_tf() else "pytorch"):
        print(f"{COLORS.WARNING}FCIL checkpoint backend mismatch — ignoring.{COLORS.ENDC}")
        return None
    with open(os.path.join(ckpt_dir, "global_weights.bin"), "rb") as f:
        weights = pickle.load(f)
    cil_state = None
    cil_path = os.path.join(ckpt_dir, "cil_state.bin")
    if os.path.exists(cil_path):
        with open(cil_path, "rb") as f:
            cil_state = pickle.load(f)
    pipeline_state = {}
    ps_path = os.path.join(ckpt_dir, "pipeline_state.bin")
    if os.path.exists(ps_path):
        with open(ps_path, "rb") as f:
            pipeline_state = pickle.load(f)
    round_meta = None
    meta_path = os.path.join(ckpt_dir, "round_meta.bin")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            round_meta = pickle.load(f)
    return {
        "task_id": int(info.get("task_id", 0)),
        "round_num": int(info.get("round_num", 0)),
        "round_complete": info.get("round_complete", "true") == "true",
        "last_client": int(info.get("last_client", -1)),
        "weights": weights,
        "cil_state": cil_state,
        "old_classes": pipeline_state.get("old_classes", set()),
        "round_meta": round_meta,
    }


def _nog_ckpt_save(ckpt_dir, task_id, round_num, client_weights, cil_method, config, task_complete=False) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    _fcil_atomic_write(os.path.join(ckpt_dir, "client_models.bin"), dict(client_weights))
    _fcil_atomic_write(os.path.join(ckpt_dir, "cil_state.bin"), cil_method.get_ckpt_state())
    lines = [
        f"task_id: {task_id}",
        f"round_num: {round_num}",
        f"task_complete: {str(task_complete).lower()}",
        f"config_hash: {_fcil_fingerprint(config)}",
        f"backend: {'tensorflow' if _use_tf() else 'pytorch'}",
    ]
    with open(os.path.join(ckpt_dir, "info.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{COLORS.OKCYAN}Checkpoint saved (T{task_id} R{round_num + 1}) -> {ckpt_dir}{COLORS.ENDC}")


def _nog_ckpt_load(ckpt_dir: str, config) -> dict | None:
    info_path = os.path.join(ckpt_dir, "info.txt")
    if not os.path.exists(info_path):
        return None
    info = {}
    with open(info_path) as f:
        for line in f:
            k, _, v = line.strip().partition(": ")
            if k:
                info[k] = v
    if info.get("config_hash", "") != _fcil_fingerprint(config):
        print(f"{COLORS.WARNING}FCIL checkpoint config mismatch — ignoring.{COLORS.ENDC}")
        return None
    if info.get("backend", "") != ("tensorflow" if _use_tf() else "pytorch"):
        print(f"{COLORS.WARNING}FCIL checkpoint backend mismatch — ignoring.{COLORS.ENDC}")
        return None
    with open(os.path.join(ckpt_dir, "client_models.bin"), "rb") as f:
        weights = pickle.load(f)
    with open(os.path.join(ckpt_dir, "cil_state.bin"), "rb") as f:
        cil_state = pickle.load(f)
    return {
        "task_id": int(info.get("task_id", 0)),
        "round_num": int(info.get("round_num", 0)),
        "task_complete": info.get("task_complete", "false") == "true",
        "weights": weights,
        "cil_state": cil_state,
    }


def run_fcil_pipeline(config: FCILConfig):
    """Main FCIL pipeline execution"""
    
    # Ours variants are strictly logits-based: no global-model aggregation path.
    if config.cil_method in ("ours", "ours2"):
        return run_no_global_pipeline(config)
    
    # Load task order
    task_order = load_task_order(config.task_order_file)
    num_tasks = len(task_order)
    all_classes = sorted(set(c for classes in task_order.values() for c in classes))
    num_classes = len(all_classes)
    
    print(f"\n{COLORS.OKCYAN}Task Order: {num_tasks} tasks, {num_classes} total classes{COLORS.ENDC}")
    for task_id, classes in task_order.items():
        print(f"  Task {task_id}: {len(classes)} classes - {classes}")
    
    def load_test_data():
        import pickle
        with open('data/CIC23_test.pkl', 'rb') as fh:
            test_data = pickle.load(fh)
        if isinstance(test_data, tuple):
            X_full, y_full = test_data
        else:
            X_full = test_data.iloc[:, :-1].values
            y_full = test_data.iloc[:, -1].values
        return X_full, y_full
    
    X_test, y_test = load_test_data()
    input_dim = X_test.shape[1]
    print(f"{COLORS.OKCYAN}Test set: {X_test.shape[0]:,} samples, {input_dim} features{COLORS.ENDC}")
    del X_test, y_test
    gc.collect()
    
    # Import strategy and CIL method
    if config.strategy in ("FedAvg", "FedSSD"):
        from .strategy.FedAvg import FedAvg
        aggregator = FedAvg()
    elif config.strategy == "Centralized":
        from .strategy.Centralized import Centralized
        aggregator = Centralized()
    elif config.strategy == "FedCoMed":
        from .strategy.FedCoMed import FedCoMed
        aggregator = FedCoMed()
    
    if config.cil_method == "finetune":
        from .cil.finetune import Finetune
        cil_method = Finetune(num_classes=num_classes)
    elif config.cil_method == "ewc":
        from .cil.ewc import EWC
        cil_method = EWC(num_classes=num_classes, ewc_lambda=config.ewc_lambda)
    elif config.cil_method == "mas":
        from .cil.mas import MAS
        cil_method = MAS(num_classes=num_classes, mas_lambda=config.mas_lambda)
    elif config.cil_method == "lwf":
        from .cil.lwf import LwF
        cil_method = LwF(num_classes=num_classes, alpha=config.lwf_alpha, temperature=config.lwf_temperature)
    elif config.cil_method == "icarl":
        from .cil.icarl import ICaRL
        cil_method = ICaRL(num_classes=num_classes, memory=config.icarl_memory, use_bce=config.icarl_bce)
    elif config.cil_method == "bic":
        from .cil.bic import BiC
        cil_method = BiC(num_classes=num_classes, memory=config.icarl_memory,
                         alpha=config.lwf_alpha, temperature=config.lwf_temperature,
                         val_split=config.bic_val_split)
    elif config.cil_method == "foster":
        from .cil.foster import FOSTER
        cil_method = FOSTER(num_classes=num_classes, memory=config.icarl_memory,
                            beta1=config.foster_beta1, beta2=config.foster_beta2,
                            lambda_okd=config.foster_lambda_okd,
                            temperature=config.lwf_temperature,
                            compression_epochs=config.foster_compression_epochs)
    elif config.cil_method == "glfc":
        from .cil.glfc import GLFC
        cil_method = GLFC(num_classes=num_classes, memory=config.icarl_memory,
                          encoder_epochs=config.glfc_encoder_epochs,
                          model_selection=config.glfc_model_selection,
                          grad_enc=config.glfc_grad_enc,
                          model_name=config.model)
    elif config.cil_method == "cbkd":
        from .cil.cbkd import CBKD
        cil_method = CBKD(num_classes=num_classes, lam=config.cbkd_lambda,
                          alpha=config.cbkd_alpha, beta=config.cbkd_beta,
                          proto_size=config.cbkd_proto_size)
    elif config.cil_method == "pass":
        from .cil.proto_pass import PASS
        cil_method = PASS(num_classes=num_classes, lam=config.cbkd_lambda,
                          gamma=config.pass_gamma,
                          proto_size=config.cbkd_proto_size)
    elif config.cil_method == "feat":
        from .cil.feat import FEAT
        cil_method = FEAT(num_classes=num_classes, memory=config.icarl_memory,
                          lam=config.feat_lambda, rho=config.feat_rho,
                          temp=config.feat_temp)
    elif config.cil_method == "exp":
        from .cil.exp import EXP
        cil_method = EXP(num_classes=num_classes, memory=config.icarl_memory,
                         lam=config.feat_lambda, rho=config.feat_rho,
                         temp=config.feat_temp)
    
    # Initialize global model
    initial_classes = len(task_order[0]) if config.cil_method in ["lwf", "icarl", "bic", "foster", "glfc", "cbkd", "pass", "feat", "exp"] else num_classes
    global_model = create_model(initial_classes, input_dim, config.batch_size, config.model)
    
    # Logging
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(config.log_file, 'w')
        print(f"{COLORS.OKCYAN}Logging to {config.log_file}{COLORS.ENDC}")
    else:
        log_file = None
    
    def log(msg, file_msg=None):
        print(msg)
        if log_file:
            log_file.write((file_msg if file_msg is not None else msg) + '\n')
            log_file.flush()

    log_stem = Path(config.log_file).stem if config.log_file else "unnamed"
    label_map = None
    old_classes = set()  # all classes seen in previous tasks

    # --- Checkpoint setup ---
    ckpt_dir = _fcil_ckpt_dir(log_stem) if log_stem != "unnamed" else None
    _ckpt_enabled = bool(config.checkpoint) and ckpt_dir is not None
    weight_record_dir = Path("temp_weights") / f"{log_stem}_weight_record"
    if config.fresh_run and ckpt_dir and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        log(f"{COLORS.OKCYAN}fresh_run: checkpoint cleared{COLORS.ENDC}", "fresh_run: checkpoint cleared")
    if config.fresh_run and weight_record_dir.exists():
        shutil.rmtree(weight_record_dir, ignore_errors=True)
        log(f"{COLORS.OKCYAN}fresh_run: weight records cleared{COLORS.ENDC}", "fresh_run: weight records cleared")

    _resume_task, _resume_round, _resume_client = 0, 0, 0
    _resume_cw: list = []
    _resume_ss: list = []
    _has_ckpt = False
    if _ckpt_enabled and ckpt_dir and not config.fresh_run:
        _ckpt = _fcil_load_ckpt(ckpt_dir, config)
        if _ckpt is not None:
            _has_ckpt = True
            if _ckpt["cil_state"] is not None:
                cil_method.set_ckpt_state(_ckpt["cil_state"])
            old_classes = _ckpt["old_classes"]
            label_map = cil_method.label_map if hasattr(cil_method, "label_map") else None
            if _ckpt["round_complete"]:
                _resume_task = _ckpt["task_id"]
                _resume_round = _ckpt["round_num"] + 1
                if _resume_round >= config.rounds_per_task:
                    _resume_task += 1
                    _resume_round = 0
            else:
                _resume_task = _ckpt["task_id"]
                _resume_round = _ckpt["round_num"]
                _resume_client = _ckpt["last_client"] + 1
                if _ckpt["round_meta"]:
                    _resume_cw = _ckpt["round_meta"]["client_weights"]
                    _resume_ss = _ckpt["round_meta"]["sample_sizes"]
            _prepare_active_model(cil_method, global_model, num_classes)
            global_model.set_weights(_ckpt["weights"])
            _prepare_active_model(cil_method, global_model, num_classes)
            log(f"{COLORS.OKGREEN}Resuming from T{_resume_task} R{_resume_round + 1} C{_resume_client}{COLORS.ENDC}",
                f"Resuming from T{_resume_task} R{_resume_round + 1} C{_resume_client}")

    # Main loop: iterate through tasks
    for task_id in range(num_tasks):
        # Skip tasks already completed before the checkpoint
        if _has_ckpt and task_id < _resume_task:
            old_classes.update(task_order[task_id])
            continue

        task_classes = task_order[task_id]
        _is_mid_task_resume = (
            _has_ckpt and task_id == _resume_task
            and (_resume_round > 0 or _resume_client > 0)
        )

        log(
            f"\n{COLORS.HEADER}TASK {task_id}: {len(task_classes)} classes - {task_classes}{COLORS.ENDC}",
            f"TASK {task_id}: {len(task_classes)} classes - {task_classes}"
        )

        if not _is_mid_task_resume:
            cil_method.before_task(task_id, task_classes)
        if hasattr(cil_method, "set_old_model") and task_id > 0 and not _is_mid_task_resume:
            cil_method.set_old_model(global_model)
        current_classes = _prepare_active_model(cil_method, global_model, num_classes)
        
        # FOSTER: zero old-class logits so boosted residual starts at 0
        if config.cil_method == "foster" and task_id > 0:
            cil_method.reinit_logits(global_model)
        
        # Create one reusable client model per task
        n_clients = 1 if config.strategy == "Centralized" else config.n_clients
        active_n = _task_active_n(n_clients, num_tasks, task_id)
        label_map = cil_method.label_map if hasattr(cil_method, "label_map") else None
        client_model = create_model(current_classes, input_dim, config.batch_size, config.model)
        _prepare_active_model(cil_method, client_model, num_classes)
        _opt, _loss_fn = _get_opt_loss(client_model)
        cached_train_step = cil_method.get_train_step(client_model, _opt, _loss_fn)
        
        # SSD: merge public data to temp file once per task
        ssd_pub_X_path, ssd_pub_y_path = None, None
        if config.strategy == "FedSSD" and hasattr(cil_method, "get_ssd_train_step"):
            ssd_dir = Path("temp_weights/ssd_public")
            ssd_dir.mkdir(parents=True, exist_ok=True)
            ssd_pub_X_path = str(ssd_dir / f"task_{task_id}_X_public.npy")
            ssd_pub_y_path = str(ssd_dir / f"task_{task_id}_y_public.npy")
            base_dir = cil_partition_dir(config.partition_root, config.partition_type, config.n_clients)
            pub_X_parts, pub_y_parts = [], []
            for t in range(task_id + 1):
                for c in range(_task_active_n(n_clients, num_tasks, t)):
                    px = base_dir / f"client_{c}_task_{t}_X_public.npy"
                    py = base_dir / f"client_{c}_task_{t}_y_public.npy"
                    if px.exists():
                        pub_X_parts.append(np.load(str(px)))
                        pub_y_parts.append(np.load(str(py)))
            np.save(ssd_pub_X_path, np.concatenate(pub_X_parts))
            np.save(ssd_pub_y_path, np.concatenate(pub_y_parts))
            del pub_X_parts, pub_y_parts
            gc.collect()

        # Run federated rounds for this task
        eval_round_ids = set(_eval_rounds(config.rounds_per_task))
        task_weight_dir = weight_record_dir / f"task_{task_id}"
        for round_num in range(config.rounds_per_task):
            # Skip rounds already completed before the checkpoint
            if _has_ckpt and task_id == _resume_task and round_num < _resume_round:
                continue

            # Determine where this round starts (mid-round resume or fresh)
            _round_start_client = 0
            _initial_cw: list = []
            _initial_ss: list = []
            if _has_ckpt and task_id == _resume_task and round_num == _resume_round:
                _round_start_client = _resume_client
                _initial_cw = list(_resume_cw)
                _initial_ss = list(_resume_ss)
                _resume_client = 0
                _resume_cw = []
                _resume_ss = []
            # SSD: compute M_class each round from merged public dataset
            ssd_train_step = None
            if ssd_pub_X_path is not None:
                from .strategy.FedSSD import compute_class_metrics, build_ssd_components
                X_pub = np.load(ssd_pub_X_path, mmap_mode='r')
                y_pub = np.load(ssd_pub_y_path, mmap_mode='r')
                M_class = compute_class_metrics(
                    global_model, X_pub, y_pub,
                    current_classes, label_map)
                log(
                    f"{COLORS.OKCYAN}SSD M_class: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}{COLORS.ENDC}",
                    f"SSD M_class: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}")
                global_logits_model, M_class_tf, m_max_tf = build_ssd_components(
                    client_model, global_model, M_class, config.m_max)
                local_logits_model = client_model.get_logits_model()
                _ssd_opt, _ssd_loss = _get_opt_loss(client_model)
                ssd_train_step = cil_method.get_ssd_train_step(
                    client_model, _ssd_opt, _ssd_loss,
                    global_logits_model, local_logits_model,
                    M_class_tf, m_max_tf)
                del X_pub, y_pub
                gc.collect()

            active_train_step = ssd_train_step if ssd_train_step is not None else cached_train_step
            
            client_weights = list(_initial_cw)
            sample_sizes = list(_initial_ss)
            client_ids_trained = list(range(len(_initial_cw)))
            _round_losses: list = []
            _round_extra: dict = {}   # term -> list of float
            
            for client_id in range(_round_start_client, active_n):
                if hasattr(cil_method, "set_client"):
                    cil_method.set_client(client_id)

                # Load task data
                paths = get_task_files(
                    config.partition_root, config.partition_type,
                    config.n_clients, client_id, task_id, config.strategy
                )
                
                if hasattr(cil_method, "build_dataset"):
                    dataset, n_samples = cil_method.build_dataset(
                        paths['X'], paths['y'], global_model, config.batch_size
                    )
                else:
                    dataset, n_samples = create_dataset(
                        paths['X'], paths['y'], config.batch_size, current_classes, label_map
                    )

                if n_samples == 0:
                    continue
                
                # Reuse client model, just reset weights
                client_model.set_weights(global_model.get_weights())
                current_classes = _prepare_active_model(cil_method, global_model, num_classes)
                _prepare_active_model(cil_method, client_model, num_classes)
                active_train_step = ssd_train_step if ssd_train_step is not None else cached_train_step
                if config.cil_method in ("ewc", "mas", "cbkd", "pass", "exp", "feat"):
                    _c_opt, _c_loss = _get_opt_loss(client_model)
                    if ssd_train_step is not None:
                        local_logits_model = client_model.get_logits_model()
                        active_train_step = cil_method.get_ssd_train_step(
                            client_model, _c_opt, _c_loss,
                            global_logits_model, local_logits_model,
                            M_class_tf, m_max_tf)
                    else:
                        active_train_step = cil_method.get_train_step(client_model, _c_opt, _c_loss)
                
                # Train using active train step (SSD or cached CIL)
                epoch_losses = train_client_fast(client_model, dataset, active_train_step, config.epochs_per_round)
                loss = epoch_losses[-1] if epoch_losses else 0.0
                _round_losses.append(loss)

                client_weights.append(client_model.get_weights())
                sample_sizes.append(n_samples)
                client_ids_trained.append(client_id)

                # EWC/MAS: compute per-client importance at last round of each task
                if (config.cil_method in ("ewc", "mas")
                        and round_num == config.rounds_per_task - 1
                        and task_id < num_tasks - 1):
                    imp_dataset, _ = create_dataset(
                        paths['X'], paths['y'], config.batch_size, current_classes, label_map)
                    cil_method.after_task(client_model, imp_dataset)
                    del imp_dataset

                # CBKD/PASS: compute per-client prototypes + variance at last round of each task
                if (config.cil_method in ("cbkd", "pass")
                        and round_num == config.rounds_per_task - 1
                        and task_id < num_tasks - 1):
                    cil_method.after_task(client_model, paths['X'], paths['y'])

                # GLFC: collect proto gradients after client trains
                if hasattr(cil_method, "collect_proto_gradients"):
                    cil_method.collect_proto_gradients(
                        client_model, client_id, paths['X'], paths['y'])
                
                del dataset
                gc.collect()

                # Accumulate per-term losses for round summary
                if config.cil_method == "cbkd" and task_id > 0:
                    _round_extra.setdefault('L_CE',    []).append(cil_method._last_ce)
                    _round_extra.setdefault('L_CFD',   []).append(cil_method._last_cfd)
                    _round_extra.setdefault('L_proto', []).append(cil_method._last_proto)
                if config.cil_method == "pass" and task_id > 0:
                    _round_extra.setdefault('L_CE',    []).append(cil_method._last_ce)
                    _round_extra.setdefault('L_KD',    []).append(cil_method._last_kd)
                    _round_extra.setdefault('L_proto', []).append(cil_method._last_proto)
                if config.cil_method == "feat" and task_id > 0:
                    _round_extra.setdefault('L_CE',  []).append(cil_method._last_ce)
                    _round_extra.setdefault('L_GSA', []).append(cil_method._last_gsa)
                if config.cil_method == "exp" and task_id > 0:
                    _round_extra.setdefault('L_CE', []).append(cil_method._last_ce)
                    _round_extra.setdefault('L_GSA', []).append(cil_method._last_gsa)
                    _round_extra.setdefault('L_replay', []).append(cil_method._last_replay)
                    _round_extra.setdefault('L_synth', []).append(cil_method._last_synth)
                    _round_extra.setdefault('L_ex', []).append(cil_method._last_exemplar)

                # Mid-round checkpoint
                if (_ckpt_enabled and ckpt_dir and config.checkpoint > 0
                        and ((client_id + 1) % config.checkpoint == 0 or client_id == active_n - 1)):
                    _fcil_save_mid_ckpt(ckpt_dir, task_id, round_num, client_id,
                                        client_weights, sample_sizes, global_model,
                                        cil_method, old_classes, config)
            
            # Log round avg loss summary
            if _round_losses:
                _avg_loss = sum(_round_losses) / len(_round_losses)
                _loss_str = f"Avg loss={_avg_loss:.4f} ({len(_round_losses)} clients)"
                if _round_extra:
                    _extra_str = " | ".join(
                        f"{k}={sum(v)/len(v):.4f}" for k, v in _round_extra.items()
                    )
                    _loss_str += f" | {_extra_str}"
                log(f"{COLORS.WARNING}R{round_num + 1} T{task_id}: {_loss_str}{COLORS.ENDC}",
                    f"LOSS | T{task_id} | R{round_num + 1} | {_loss_str}")

            if not client_weights:
                log(f"{COLORS.WARNING}Round {round_num + 1}: all clients skipped, keeping global model{COLORS.ENDC}",
                    f"Round {round_num + 1}: all clients skipped")
            else:
                _prepare_active_model(cil_method, global_model, num_classes)
                global_weights = aggregator.aggregate(client_weights, sample_sizes)
                global_model.set_weights(global_weights)
                current_classes = _prepare_active_model(cil_method, global_model, num_classes)
                del global_weights

            if hasattr(cil_method, "proxy_update"):
                proxy_perf = cil_method.proxy_update(global_model)
                if proxy_perf is not None:
                    log(
                        f"{COLORS.OKCYAN}GLFC Proxy: accuracy={proxy_perf:.4f}{COLORS.ENDC}",
                        f"GLFC Proxy: accuracy={proxy_perf:.4f}")

            if hasattr(cil_method, "update_exemplars"):
                reduce_old = round_num == 0
                for client_id in range(active_n):
                    if hasattr(cil_method, "set_client"):
                        cil_method.set_client(client_id)
                    paths = get_task_files(
                        config.partition_root, config.partition_type,
                        config.n_clients, client_id, task_id, config.strategy
                    )
                    cil_method.update_exemplars(global_model, paths['X'], paths['y'], task_classes, reduce_old=reduce_old)
                cil_method.rebuild_global_exemplars(global_model)
                if config.cil_method == "feat" and task_id > 0:
                    log(
                        f"{COLORS.OKCYAN}FEAT prior: eH={cil_method.global_head_energy:.4f}, eT={cil_method.global_tail_energy:.4f}{COLORS.ENDC}",
                        f"FEAT | T{task_id} | R{round_num + 1} | eH={cil_method.global_head_energy:.4f} | eT={cil_method.global_tail_energy:.4f}")
                if config.cil_method == "exp" and task_id > 0:
                    log(
                        f"{COLORS.OKCYAN}EXP prior: eH={cil_method.global_head_energy:.4f}, eT={cil_method.global_tail_energy:.4f}, shift={float(np.mean(cil_method.shift_alpha[:cil_method.old_num_classes])) if cil_method.old_num_classes > 0 else 0.0:.4f}{COLORS.ENDC}",
                        f"EXP | T{task_id} | R{round_num + 1} | eH={cil_method.global_head_energy:.4f} | eT={cil_method.global_tail_energy:.4f} | shift={float(np.mean(cil_method.shift_alpha[:cil_method.old_num_classes])) if cil_method.old_num_classes > 0 else 0.0:.4f}")
            
            if round_num in eval_round_ids and client_weights:
                _save_client_weights(task_weight_dir, round_num, client_ids_trained, client_weights)

            # Clean up client data
            del client_weights, client_ids_trained
            gc.collect()

            # BiC: per-client bias correction → aggregate coefficients
            if config.cil_method == "bic" and task_id > 0:
                corrector_weights = []
                for cid in range(active_n):
                    a, b = cil_method.run_client_bias_correction(global_model, cid)
                    corrector_weights.append([np.array([a, b])])
                agg_ab = aggregator.aggregate(corrector_weights, sample_sizes)
                cil_method.bias_alpha = float(agg_ab[0][0])
                cil_method.bias_beta = float(agg_ab[0][1])
                cil_method.apply_bias_correction(global_model)
                log(
                    f"{COLORS.OKCYAN}BiC correction: alpha={cil_method.bias_alpha:.4f}, beta={cil_method.bias_beta:.4f}{COLORS.ENDC}",
                    f"BiC correction: alpha={cil_method.bias_alpha:.4f}, beta={cil_method.bias_beta:.4f}")
                del corrector_weights

            del sample_sizes
            gc.collect()

            # Per-task then merged evaluation (only at scheduled rounds)
            if round_num in eval_round_ids:
                X_test, y_test = load_test_data()
                _yt_labels = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)
                seen_classes = sorted({c for _t in range(task_id + 1) for c in task_order[_t]})
                eval_mask = np.isin(_yt_labels, seen_classes)
                eval_results = evaluate_active_model(
                    cil_method, global_model, X_test[eval_mask], y_test[eval_mask],
                    config.batch_size, current_classes)
                _ms = (f"Acc={eval_results['acc']:.4f} | Loss={eval_results['loss']:.4f} | "
                       f"F1_mac={eval_results['f1_macro']:.4f} | F1_mic={eval_results['f1_micro']:.4f} | "
                       f"F1_w={eval_results['f1_weighted']:.4f} | "
                       f"Prec_mac={eval_results['prec_macro']:.4f} | Rec_mac={eval_results['rec_macro']:.4f} | "
                       f"Prec_w={eval_results['prec_weighted']:.4f} | Rec_w={eval_results['rec_weighted']:.4f}")
                log(f"{COLORS.OKGREEN}Round {round_num + 1} | Tasks 0-{task_id} | {_ms}{COLORS.ENDC}",
                    f"EVAL | T{task_id} | R{round_num + 1} | {_ms}")
                if log_stem != "unnamed":
                    _plot_confusion_matrix(
                        eval_results['cm'],
                        str(Path("results_cil/plots") / log_stem / f"T{task_id}_R{round_num + 1}.png"),
                        list(cil_method.class_order) if hasattr(cil_method, "class_order") else None)
                del _yt_labels, eval_mask, X_test, y_test
                gc.collect()

            # BiC: undo correction so next round trains on uncorrected model
            if config.cil_method == "bic" and task_id > 0:
                cil_method.undo_bias_correction(global_model)

            # Round-complete checkpoint
            if _ckpt_enabled and ckpt_dir:
                _fcil_save_ckpt(ckpt_dir, task_id, round_num, global_model, cil_method, old_classes, config)

        # FOSTER Stage 2: feature compression after last round of each task
        if config.cil_method == "foster" and task_id > 0:
            log(f"{COLORS.OKCYAN}FOSTER Stage 2: Feature Compression{COLORS.ENDC}",
                "FOSTER Stage 2: Feature Compression")
            task_paths = [
                get_task_files(config.partition_root, config.partition_type,
                               config.n_clients, cid, task_id, config.strategy)
                for cid in range(active_n)
            ]
            cil_method.run_compression(global_model, config, task_paths)
            log(f"{COLORS.OKCYAN}FOSTER compression complete{COLORS.ENDC}",
                "FOSTER compression complete")

            # Post-compression eval so log reflects actual student quality
            X_test_pc, y_test_pc = load_test_data()
            _yt_pc_labels = y_test_pc if len(y_test_pc.shape) == 1 else np.argmax(y_test_pc, axis=1)
            seen_classes = sorted({c for _t in range(task_id + 1) for c in task_order[_t]})
            eval_mask = np.isin(_yt_pc_labels, seen_classes)
            eval_results = evaluate_active_model(
                cil_method, global_model, X_test_pc[eval_mask], y_test_pc[eval_mask],
                config.batch_size, current_classes)
            _ms = (f"Acc={eval_results['acc']:.4f} | Loss={eval_results['loss']:.4f} | "
                   f"F1_mac={eval_results['f1_macro']:.4f} | F1_mic={eval_results['f1_micro']:.4f} | "
                   f"F1_w={eval_results['f1_weighted']:.4f} | "
                   f"Prec_mac={eval_results['prec_macro']:.4f} | Rec_mac={eval_results['rec_macro']:.4f} | "
                   f"Prec_w={eval_results['prec_weighted']:.4f} | Rec_w={eval_results['rec_weighted']:.4f}")
            log(f"{COLORS.OKCYAN}Post-compress | Tasks 0-{task_id} | {_ms}{COLORS.ENDC}",
                f"Post-compress | Tasks 0-{task_id} | {_ms}")
            del eval_mask, _yt_pc_labels
            del X_test_pc, y_test_pc
            gc.collect()

        old_classes.update(task_classes)

        # Clean up reusable client model for this task
        del client_model, cached_train_step
        gc.collect()
        
        # EWC/MAS importance now computed per-client inside the client loop above
    
    log(f"{COLORS.OKGREEN}Pipeline completed!{COLORS.ENDC}", "Pipeline completed!")
    
    # Final evaluation: pooled over all classes
    X_test, y_test = load_test_data()
    final_classes = len(cil_method.class_order) if hasattr(cil_method, "class_order") else num_classes
    _yt_labels = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)
    seen_classes = sorted({c for _t in range(num_tasks) for c in task_order[_t]})
    eval_mask = np.isin(_yt_labels, seen_classes)
    final_results = evaluate_active_model(
        cil_method, global_model, X_test[eval_mask], y_test[eval_mask],
        config.batch_size, final_classes)
    del X_test, y_test, _yt_labels, eval_mask
    gc.collect()
    _ms = (f"Acc={final_results['acc']:.4f} | Loss={final_results['loss']:.4f} | "
           f"F1_mac={final_results['f1_macro']:.4f} | F1_mic={final_results['f1_micro']:.4f} | "
           f"F1_w={final_results['f1_weighted']:.4f} | "
           f"Prec_mac={final_results['prec_macro']:.4f} | Rec_mac={final_results['rec_macro']:.4f} | "
           f"Prec_w={final_results['prec_weighted']:.4f} | Rec_w={final_results['rec_weighted']:.4f}")
    log(f"{COLORS.OKCYAN}FINAL | all {num_classes} classes | {_ms}{COLORS.ENDC}",
        f"FINAL | all {num_classes} classes | {_ms}")
    if log_stem != "unnamed":
        _plot_confusion_matrix(
            final_results['cm'],
            str(Path("results_cil/plots") / log_stem / "final.png"),
            list(cil_method.class_order) if hasattr(cil_method, "class_order") else None)
    
    if log_file:
        log_file.close()

    if hasattr(cil_method, "cleanup_temp"):
        cil_method.cleanup_temp()

    # Clean up checkpoint dir on successful completion
    if _ckpt_enabled and ckpt_dir and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)

def run_no_global_pipeline(config: FCILConfig):
    """Pipeline for PASS method - no global model, per-client prototype distillation"""
    task_order = load_task_order(config.task_order_file)
    num_tasks = len(task_order)
    num_classes = sum(len(task_classes) for task_classes in task_order.values())

    def load_test_data():
        import pickle
        with open('data/CIC23_test.pkl', 'rb') as fh:
            test_data = pickle.load(fh)
        if isinstance(test_data, tuple):
            X_full, y_full = test_data
        else:
            X_full = test_data.iloc[:, :-1].values
            y_full = test_data.iloc[:, -1].values
        return X_full, y_full
    
    X_test_sample, y_test_sample = load_test_data()
    input_dim = X_test_sample.shape[1]
    del X_test_sample, y_test_sample
    
    if config.cil_method == "ours2":
        from .cil.ours2 import Ours2
        cil_method = Ours2(num_classes=num_classes, memory=config.icarl_memory,
                           kd_gamma=config.ours_kd_gamma,
                           robust_threshold=config.robust_threshold,
                           robust_workers=config.robust_workers)
    else:
        from .cil.ours import Ours
        cil_method = Ours(num_classes=num_classes, lam=config.ours_proto_lambda,
                          gamma=config.ours_kd_gamma, proto_size=config.cbkd_proto_size,
                          ekd_epochs=config.ours_ekd_epochs,
                          ekd_lambda=config.ours_ekd_lambda,
                          proto_rel_lambda=config.ours_proto_rel_lambda,
                          encoder_lr_factor=config.ours_encoder_lr_factor,
                          drift_temp=config.ours_drift_temp,
                          robust_threshold=config.robust_threshold,
                          robust_workers=config.robust_workers,
                          no_filter=config.no_filter)
    
    # Logging setup
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(config.log_file, 'w')
        print(f"{COLORS.OKCYAN}Logging to {config.log_file}{COLORS.ENDC}")
    else:
        log_file = None
    
    def log(msg, file_msg=None):
        print(msg)
        if log_file:
            log_file.write((file_msg if file_msg is not None else msg) + '\n')
            log_file.flush()
    
    log_stem = Path(config.log_file).stem if config.log_file else "ours_unnamed"

    ckpt_dir = _fcil_ckpt_dir(log_stem) if log_stem != "ours_unnamed" else None
    _ckpt_enabled = bool(config.checkpoint) and ckpt_dir is not None
    weight_record_dir = Path("temp_weights") / f"{log_stem}_weight_record"
    if config.fresh_run and ckpt_dir and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        log(f"{COLORS.OKCYAN}fresh_run: checkpoint cleared{COLORS.ENDC}", "fresh_run: checkpoint cleared")
    if config.fresh_run and weight_record_dir.exists():
        shutil.rmtree(weight_record_dir, ignore_errors=True)
        log(f"{COLORS.OKCYAN}fresh_run: weight records cleared{COLORS.ENDC}", "fresh_run: weight records cleared")

    _resume_task = 0
    _resume_round = 0
    _has_ckpt = False
    _resume_task_end = False
    _resume_mid_task = False
    _ckpt = _nog_ckpt_load(ckpt_dir, config) if _ckpt_enabled and ckpt_dir and not config.fresh_run else None
    if _ckpt is not None:
        _has_ckpt = True
        cil_method.set_ckpt_state(_ckpt["cil_state"])
        if _ckpt["task_complete"]:
            _resume_task = _ckpt["task_id"] + 1
            _resume_round = 0
            log(f"{COLORS.OKGREEN}Resuming from T{_resume_task}{COLORS.ENDC}",
                f"Resuming from T{_resume_task}")
        else:
            _resume_task = _ckpt["task_id"]
            _resume_round = _ckpt["round_num"] + 1
            _resume_mid_task = True
        if _resume_mid_task and _resume_round >= config.rounds_per_task:
            _resume_round = config.rounds_per_task
            _resume_task_end = True
            _resume_mid_task = False
            log(f"{COLORS.OKGREEN}Resuming from T{_resume_task} task-end freeze{COLORS.ENDC}",
                f"Resuming from T{_resume_task} task-end freeze")
        elif _resume_mid_task:
            log(f"{COLORS.OKGREEN}Resuming from T{_resume_task} R{_resume_round + 1}{COLORS.ENDC}",
                f"Resuming from T{_resume_task} R{_resume_round + 1}")

    # Per-client model pool - no global aggregation
    n_clients = config.n_clients
    client_models = {}
    
    # Initialize client models
    for task_id in range(num_tasks):
        if _has_ckpt and task_id < _resume_task:
            continue
        task_classes = task_order[task_id]
        if not (_has_ckpt and task_id == _resume_task and _resume_mid_task):
            if task_id > 0 and hasattr(cil_method, "set_old_model") and 0 in client_models:
                cil_method.set_old_model(client_models[0])
            cil_method.before_task(task_id, task_classes)
        
        current_classes = _active_class_count(cil_method, num_classes)
        active_n = _task_active_n(n_clients, num_tasks, task_id)
        saved_weights = {}
        if _has_ckpt and task_id == _resume_task:
            if _resume_mid_task:
                saved_weights = _ckpt["weights"]
            elif _ckpt["task_complete"] and _resume_task == _ckpt["task_id"] + 1:
                saved_weights = _ckpt["weights"]
        if hasattr(cil_method, "build_public_candidates"):
            public_n = cil_method.build_public_candidates(config, task_id, active_n, num_tasks)
            log(f"{COLORS.OKCYAN}Ours2 task-{task_id} public candidates: {public_n} samples (frozen at task end){COLORS.ENDC}",
                f"Ours2 | T{task_id} | PublicCandidates samples={public_n}")
        
        # Create/expand client models for this task
        for client_id in range(active_n):
            if client_id not in client_models:
                client_models[client_id] = create_model(current_classes, input_dim, config.batch_size, config.model)
            current_classes = _prepare_active_model(cil_method, client_models[client_id], num_classes)
            if client_id in saved_weights:
                client_models[client_id].set_weights(saved_weights[client_id])
                current_classes = _prepare_active_model(cil_method, client_models[client_id], num_classes)
        
        log(f"\n{COLORS.HEADER}TASK {task_id}: {len(task_classes)} classes - {task_classes}{COLORS.ENDC}",
            f"TASK {task_id}: {len(task_classes)} classes - {task_classes}")
        
        local_epochs = max(config.epochs_per_round - 2, 0)

        # Federated rounds for this task
        eval_round_ids = set(_eval_rounds(config.rounds_per_task))
        task_weight_dir = weight_record_dir / f"task_{task_id}"
        
        for round_num in range(config.rounds_per_task):
            if _has_ckpt and task_id == _resume_task and round_num < _resume_round:
                continue
            log(f"\n{COLORS.WARNING}Round {round_num + 1}/{config.rounds_per_task} Task {task_id}{COLORS.ENDC}",
                f"Round {round_num + 1}/{config.rounds_per_task} Task {task_id}")
            
            client_weights = []
            trained_client_ids = []
            round_losses = []
            
            # Train each client locally with PASS
            for client_id in range(active_n):
                if hasattr(cil_method, "set_client"):
                    cil_method.set_client(client_id)
                
                paths = get_task_files(
                    config.partition_root, config.partition_type,
                    config.n_clients, client_id, task_id, config.strategy
                )
                
                # Build dataset for current task data (Ours2 mixes private + public memory)
                if hasattr(cil_method, "build_dataset"):
                    dataset, n_samples = cil_method.build_dataset(paths['X'], paths['y'], config.batch_size)
                else:
                    dataset, n_samples = create_dataset(
                        paths['X'], paths['y'], config.batch_size, current_classes, cil_method.label_map
                    )
                
                if n_samples == 0:
                    continue
                
                if task_id > 0 and hasattr(cil_method, "capture_anchor_prototypes"):
                    cil_method.capture_anchor_prototypes(client_models[client_id], paths['X'], paths['y'])

                # Stage 1: CE on task 0; drift-calibrated PASS on later tasks
                _opt, _loss_fn = _get_opt_loss(client_models[client_id])
                train_step = cil_method.get_finetune_step(client_models[client_id], _opt, _loss_fn) if hasattr(cil_method, "get_finetune_step") else cil_method.get_train_step(client_models[client_id], _opt, _loss_fn)
                stage_epochs = config.epochs_per_round if task_id == 0 else local_epochs
                epoch_losses = train_client_fast(client_models[client_id], dataset, train_step, stage_epochs)
                loss = epoch_losses[-1] if epoch_losses else 0.0
                round_losses.append(loss)

                if hasattr(cil_method, "update_client_drift"):
                    cil_method.update_client_drift(client_models[client_id], paths['X'], paths['y'])

                if config.cil_method == "ours2" and hasattr(cil_method, "update_eva_threshold"):
                    cil_method.update_eva_threshold(client_id, client_models[client_id], paths['X'], config.batch_size)
                
                # Store client weights for consensus building
                trained_client_ids.append(client_id)
                client_weights.append(client_models[client_id].get_weights())
                
                del dataset
                gc.collect()
            
            # Log round summary
            if round_losses:
                avg_loss = sum(round_losses) / len(round_losses)
                log(f"{COLORS.WARNING}R{round_num + 1} T{task_id}: Avg loss={avg_loss:.4f} ({len(round_losses)} clients){COLORS.ENDC}",
                    f"LOSS | T{task_id} | R{round_num + 1} | Avg loss={avg_loss:.4f} ({len(round_losses)} clients)")
            
            # Stage 2: EKD distillation on shared consensus (single scratch model)
            if client_weights:
                scratch_model = create_model(current_classes, input_dim, config.batch_size, config.model)
                _prepare_active_model(cil_method, scratch_model, num_classes)
                active_models = {cid: client_models[cid] for cid in trained_client_ids}
                if config.cil_method == "ours2":
                    avg_ekd = 0.0
                    cil_method._last_survivor_clients = list(range(len(trained_client_ids)))
                    cil_method._last_survivors = len(trained_client_ids)
                elif hasattr(cil_method, "update_consensus"):
                    cil_method.update_consensus(scratch_model, client_weights, config)
                    avg_ekd = 0.0
                else:
                    avg_ekd = cil_method.distill_round(scratch_model, active_models, client_weights, config, task_id, active_n, num_tasks)
                del scratch_model
                gc.collect()
                survivor_client_ids = [trained_client_ids[i] for i in cil_method._last_survivor_clients]
                if survivor_client_ids and hasattr(cil_method, "aggregate_prototypes"):
                    client_prototypes = [cil_method.prototypes.get(cid, {}) for cid in survivor_client_ids]
                    client_counts = [cil_method.client_proto_counts.get(cid, {}) for cid in survivor_client_ids]
                    if any(client_prototypes):
                        cil_method.aggregate_prototypes(client_prototypes, client_counts)
                        log(f"{COLORS.OKCYAN}Proto aggregation: {len(cil_method.global_prototypes)} classes from {len(survivor_client_ids)}/{len(trained_client_ids)} Blom survivors | var={cil_method.global_variance:.4f}{COLORS.ENDC}",
                            f"Ours | T{task_id} | R{round_num + 1} | ProtoAgg classes={len(cil_method.global_prototypes)} survivor_clients={len(survivor_client_ids)}/{len(trained_client_ids)} var={cil_method.global_variance:.4f}")
                client_weights = [client_models[cid].get_weights() for cid in trained_client_ids]

                if config.cil_method == "ours2":
                    ours2_total = cil_method._last_ce + cil_method._last_kd
                    log(f"{COLORS.OKCYAN}Ours2 losses: CE={cil_method._last_ce:.4f} | KD={cil_method._last_kd:.4f} | total={ours2_total:.4f} | round_clients={cil_method._last_survivors}{COLORS.ENDC}",
                        f"Ours2 | T{task_id} | R{round_num + 1} | CE={cil_method._last_ce:.4f} | KD={cil_method._last_kd:.4f} | total={ours2_total:.4f} | RoundClients={cil_method._last_survivors}")
                else:
                    pass_local = cil_method._last_ce + cil_method._last_kd + cil_method._last_proto + cil_method._last_rel
                    pass_total = pass_local + avg_ekd
                    log(f"{COLORS.OKCYAN}Ours losses: CE={cil_method._last_ce:.4f} | old_KD={cil_method._last_kd:.4f} | protoAug={cil_method._last_proto:.4f} | protoRel={cil_method._last_rel:.4f} | EKD={avg_ekd:.4f} | total={pass_total:.4f} | survivors≈{cil_method._last_survivors}{COLORS.ENDC}",
                        f"Ours | T{task_id} | R{round_num + 1} | CE={cil_method._last_ce:.4f} | old_KD={cil_method._last_kd:.4f} | protoAug={cil_method._last_proto:.4f} | protoRel={cil_method._last_rel:.4f} | EKD={avg_ekd:.4f} | total={pass_total:.4f} | Survivors={cil_method._last_survivors}")
            
            # Evaluation at scheduled rounds
            if round_num in eval_round_ids and client_weights:
                X_test, y_test = load_test_data()
                _yt_labels = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)
                seen_classes = sorted({c for _t in range(task_id + 1) for c in task_order[_t]})
                eval_mask = np.isin(_yt_labels, seen_classes)
                X_eval = X_test[eval_mask]
                y_eval = y_test[eval_mask]
                client_results = [evaluate_active_model(
                    cil_method, client_models[cid], X_eval, y_eval,
                    config.batch_size, current_classes)
                    for cid in trained_client_ids if cid in client_models]
                if client_results:
                    # Average across client models instead of summing to avoid inflating support counts
                    pooled_cm = np.mean([r['cm'] for r in client_results], axis=0)
                    pooled_loss = float(np.mean([r['loss'] for r in client_results]))
                    overall_result = _metrics_from_cm(pooled_cm, pooled_loss)
                    overall_result['cm'] = pooled_cm
                    _ms = (f"Acc={overall_result['acc']:.4f} | Loss={overall_result['loss']:.4f} | "
                           f"F1_mac={overall_result['f1_macro']:.4f} | F1_mic={overall_result['f1_micro']:.4f} | "
                           f"F1_w={overall_result['f1_weighted']:.4f} | "
                           f"Prec_mac={overall_result['prec_macro']:.4f} | Rec_mac={overall_result['rec_macro']:.4f} | "
                           f"Prec_w={overall_result['prec_weighted']:.4f} | Rec_w={overall_result['rec_weighted']:.4f}")
                    log(f"{COLORS.OKGREEN}Round {round_num + 1} | Tasks 0-{task_id} | {_ms}{COLORS.ENDC}",
                        f"EVAL | T{task_id} | R{round_num + 1} | {_ms}")
                    _log_class_task_breakdown(log, overall_result, task_order, cil_method.label_map,
                                              task_id, getattr(cil_method, 'class_order', None),
                                              f"EVAL | T{task_id} | R{round_num + 1}")
                del eval_mask, X_eval, y_eval
                del X_test, y_test, _yt_labels
                gc.collect()
            
            if client_weights and round_num in eval_round_ids:
                _save_client_weights(task_weight_dir, round_num, trained_client_ids, client_weights)
            if _ckpt_enabled and ckpt_dir and trained_client_ids:
                round_weights = {cid: client_models[cid].get_weights() for cid in trained_client_ids}
                _nog_ckpt_save(ckpt_dir, task_id, round_num, round_weights, cil_method, config)
            del client_weights
            gc.collect()

        if config.cil_method == "ours2" and hasattr(cil_method, "freeze_task_memory"):
            seed_client_ids = [cid for cid in range(active_n) if cid in client_models]
            seed_weights = [client_models[cid].get_weights() for cid in seed_client_ids]
            if seed_weights:
                scratch_model = create_model(current_classes, input_dim, config.batch_size, config.model)
                _prepare_active_model(cil_method, scratch_model, num_classes)
                cil_method.freeze_task_memory(scratch_model, seed_weights, config, task_id, seed_client_ids)
                del scratch_model
                gc.collect()
                if _ckpt_enabled and ckpt_dir:
                    end_weights = {cid: client_models[cid].get_weights() for cid in seed_client_ids}
                    _nog_ckpt_save(ckpt_dir, task_id, config.rounds_per_task - 1, end_weights, cil_method, config, task_complete=True)
        if _resume_task_end and task_id == _resume_task:
            _resume_task_end = False
    
    # Final evaluation
    log(f"\n{COLORS.OKGREEN}No-Global Pipeline completed!{COLORS.ENDC}", "No-Global Pipeline completed!")
    
    X_test, y_test = load_test_data()
    _yt_labels = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)
    final_classes = len(cil_method.class_order)
    seen_classes = sorted({c for _t in range(num_tasks) for c in task_order[_t]})
    eval_mask = np.isin(_yt_labels, seen_classes)
    X_eval = X_test[eval_mask]
    y_eval = y_test[eval_mask]
    client_results = [evaluate_active_model(
        cil_method, cm, X_eval, y_eval,
        config.batch_size, final_classes)
        for cm in client_models.values()]
    if client_results:
        pooled_cm = np.mean([r['cm'] for r in client_results], axis=0)
        pooled_loss = float(np.mean([r['loss'] for r in client_results]))
        final_result = _metrics_from_cm(pooled_cm, pooled_loss)
        final_result['cm'] = pooled_cm
        _ms = (f"Acc={final_result['acc']:.4f} | Loss={final_result['loss']:.4f} | "
               f"F1_mac={final_result['f1_macro']:.4f} | F1_mic={final_result['f1_micro']:.4f} | "
               f"F1_w={final_result['f1_weighted']:.4f} | "
               f"Prec_mac={final_result['prec_macro']:.4f} | Rec_mac={final_result['rec_macro']:.4f} | "
               f"Prec_w={final_result['prec_weighted']:.4f} | Rec_w={final_result['rec_weighted']:.4f}")
        log(f"{COLORS.OKCYAN}FINAL | all {num_classes} classes | {_ms}{COLORS.ENDC}",
            f"FINAL | all {num_classes} classes | {_ms}")
        _log_class_task_breakdown(log, final_result, task_order, cil_method.label_map,
                                  num_tasks - 1, getattr(cil_method, 'class_order', None),
                                  "FINAL")
    del X_test, y_test, _yt_labels, eval_mask, X_eval, y_eval
    gc.collect()
    
    if log_file:
        log_file.close()
    
    if hasattr(cil_method, "cleanup_temp"):
        cil_method.cleanup_temp()
