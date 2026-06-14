"""Bias Correction (BiC) for continual learning
Stage 1: LwF-style distillation + exemplar replay (same as iCaRL exemplar management)
Stage 2: Learn linear bias correction layer (alpha, beta) on balanced validation set
"""

import gc
import numpy as np
from pathlib import Path
from ..backend import use_tf as _use_tf
from .base import CILMethod

PRED_BATCH = 4096
def _batched_predict(model, X, batch_size=PRED_BATCH):
    if not _use_tf() and hasattr(model, 'predict'):
        return model.predict(X, batch_size=batch_size)
    parts = []
    for i in range(0, X.shape[0], batch_size):
        parts.append(model(X[i:i + batch_size], training=False).numpy())
    return np.concatenate(parts, axis=0)


class BiC(CILMethod):
    def __init__(self, num_classes: int, memory: int = 2000,
                 alpha: float = 0.5, temperature: float = 2.0,
                 val_split: float = 0.1, **kwargs):
        super().__init__(num_classes)
        self.name = "BiC"
        self.memory = memory
        self.lwf_alpha = alpha
        self.temperature = temperature
        self.val_ratio = val_split
        self.class_order = []
        self.label_map = {}
        self.client_id = None
        self.client_seen = {}
        self.exemplars = {}
        self.val_exemplars = {}
        self.old_num_classes = 0
        self.old_model = None
        self.old_logits_model = None
        self.bias_alpha = None       # scalar
        self.bias_beta = None        # scalar
        self.exemplar_dir = Path("temp_weights/bic_exemplars")
        self.val_dir = Path("temp_weights/bic_val")
        self.client_val_X = {}

    def set_client(self, client_id: int):
        self.client_id = client_id
        if client_id not in self.client_seen:
            self.client_seen[client_id] = set()
        if client_id not in self.exemplars:
            self.exemplars[client_id] = {}
        if client_id not in self.val_exemplars:
            self.val_exemplars[client_id] = {}
        if client_id not in self.client_val_X:
            self.client_val_X[client_id] = {}

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)
        for c in task_classes:
            if c not in self.label_map:
                self.label_map[c] = len(self.class_order)
                self.class_order.append(c)
        self.old_num_classes = len(self.class_order) - len(task_classes)

    def set_old_model(self, model):
        if not _use_tf():
            import copy
            old = copy.deepcopy(model.nn)
            old.eval()
            for p in old.parameters():
                p.requires_grad_(False)
            self.old_model = old
            self.old_logits_model = old
            self.old_num_classes = model.num_classes
            return
        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        self.old_model = tf.keras.models.clone_model(keras_model)
        self.old_model.set_weights(keras_model.get_weights())
        self.old_model.trainable = False
        logits_layer = self.old_model.get_layer('logits')
        self.old_logits_model = tf.keras.Model(self.old_model.input, logits_layer.output)

    def map_labels_np(self, y):
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        return np.vectorize(self.label_map.get, otypes=[int])(labels)

    # exemplar management (iCaRL herding)

    def _get_m_per_class(self, client_id):
        classes = sorted(self.client_seen.get(client_id, []))
        t = len(classes)
        base = self.memory // t if t > 0 else 0
        remainder = self.memory - base * t
        sizes = {c: base for c in classes}
        for c in classes[:remainder]:
            sizes[c] += 1
        return sizes

    def _reduce_exemplars(self, client_id, m_per_class):
        for c, path in list(self.exemplars.get(client_id, {}).items()):
            m = m_per_class.get(c, 0)
            val_count = int(m * self.val_ratio)
            train_count = m - val_count
            X_ex = np.load(path, mmap_mode='r')
            X_train = np.array(X_ex[:train_count])
            X_val = np.array(X_ex[train_count:train_count + val_count])
            np.save(path, X_train)
            val_path = self.val_dir / f"client_{client_id}_class_{c}.npy"
            self.val_dir.mkdir(parents=True, exist_ok=True)
            np.save(val_path, X_val)
            self.val_exemplars[client_id][c] = val_path

    def _construct_exemplars(self, feature_model, X_c, m):
        if m <= 0 or X_c.shape[0] == 0:
            return np.empty((0,) + X_c.shape[1:], dtype=X_c.dtype)
        if feature_model is None:
            return np.array(X_c[:m])
        feats = _batched_predict(feature_model, X_c)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        class_mean = np.mean(feats, axis=0)
        n = feats.shape[0]
        available = np.ones(n, dtype=bool)
        selected_idx = []
        sum_sel = np.zeros_like(class_mean)
        for k in range(min(m, n)):
            target = class_mean * (k + 1) - sum_sel
            dists = np.linalg.norm(feats[available] - target, axis=1)
            avail_indices = np.where(available)[0]
            best = avail_indices[int(np.argmin(dists))]
            selected_idx.append(best)
            sum_sel += feats[best]
            available[best] = False
        del feats
        return np.array(X_c[selected_idx])

    def update_exemplars(self, model, X_path, y_path, task_classes, reduce_old=False):
        feature_model = model.get_feature_model() if hasattr(model, "get_feature_model") else None
        X_new = np.load(X_path, mmap_mode='r')
        y_new = np.load(y_path, mmap_mode='r')
        labels = np.array(y_new if len(y_new.shape) == 1 else np.argmax(y_new, axis=1))
        present = [c for c in task_classes if np.any(labels == c)]
        for c in present:
            self.client_seen[self.client_id].add(c)
        m_per_class = self._get_m_per_class(self.client_id)
        if reduce_old:
            self._reduce_exemplars(self.client_id, m_per_class)
        for c in present:
            mask = labels == c
            X_c = np.array(X_new[mask], dtype=np.float32)
            m = m_per_class.get(c, 0)
            exemplars = self._construct_exemplars(feature_model, X_c, m)
            if exemplars.shape[0] == 0:
                del X_c, exemplars
                continue
            val_count = int(exemplars.shape[0] * self.val_ratio)
            train_count = exemplars.shape[0] - val_count
            self.exemplar_dir.mkdir(parents=True, exist_ok=True)
            self.val_dir.mkdir(parents=True, exist_ok=True)
            train_path = self.exemplar_dir / f"client_{self.client_id}_class_{c}.npy"
            val_path = self.val_dir / f"client_{self.client_id}_class_{c}.npy"
            np.save(train_path, exemplars[:train_count])
            np.save(val_path, exemplars[train_count:])
            self.exemplars[self.client_id][c] = train_path
            self.val_exemplars[self.client_id][c] = val_path
            del X_c, exemplars
        del X_new, y_new, labels, feature_model
        gc.collect()

    def rebuild_global_exemplars(self, model):
        pass  # BiC does not need global means

    # ── Stage 1 dataset: current task data + train-split exemplars ──

    def build_dataset(self, X_path, y_path, model, batch_size):
        if not _use_tf():
            return self._build_dataset_pt(X_path, y_path, model, batch_size)
        import tensorflow as tf
        old_tgt_file = Path("temp_weights/bic_old_targets.npy")
        if old_tgt_file.exists():
            old_tgt_file.unlink(missing_ok=True)
        X_new = np.load(X_path, mmap_mode='r')
        y_new = np.load(y_path, mmap_mode='r')
        num_classes = len(self.class_order)
        old_nc = self.old_num_classes
        input_dim = X_new.shape[1]
        client_exemplars = list(self.exemplars.get(self.client_id, {}).items())
        label_map = dict(self.label_map)

        # Split current task data into train / val
        labels_full = np.array(y_new if len(y_new.shape) == 1 else np.argmax(y_new, axis=1))
        task_classes = [c for c in self.class_order[old_nc:]]
        train_mask = np.ones(X_new.shape[0], dtype=bool)
        new_val = {}
        for c in task_classes:
            c_idx = np.where(labels_full == c)[0]
            if len(c_idx) == 0:
                continue
            val_n = int(len(c_idx) * self.val_ratio)
            if val_n == 0:
                continue
            val_idx = c_idx[-val_n:]
            train_mask[val_idx] = False
            new_val[c] = np.array(X_new[val_idx], dtype=np.float32)
        self.client_val_X[self.client_id] = new_val

        train_indices = np.where(train_mask)[0]

        # Pre-compute old soft targets (LwF distillation)
        old_targets_path = None
        if old_nc > 0 and self.old_logits_model is not None:
            T = self.temperature
            all_old = []
            for i in range(0, len(train_indices), PRED_BATCH):
                idx = train_indices[i:i + PRED_BATCH]
                chunk = np.array(X_new[idx], dtype=np.float32)
                logits = self.old_logits_model(chunk, training=False).numpy()
                all_old.append((logits[:, :old_nc] / T).astype(np.float32))
                del chunk, logits
            for c, path in client_exemplars:
                X_ex = np.load(path, mmap_mode='r')
                for i in range(0, X_ex.shape[0], PRED_BATCH):
                    chunk = np.array(X_ex[i:i + PRED_BATCH], dtype=np.float32)
                    logits = self.old_logits_model(chunk, training=False).numpy()
                    all_old.append((logits[:, :old_nc] / T).astype(np.float32))
                    del chunk, logits
                del X_ex
            old_targets = np.concatenate(all_old, axis=0)
            del all_old
            gc.collect()
            old_targets_path = Path("temp_weights/bic_old_targets.npy")
            old_targets_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(old_targets_path), old_targets)
            del old_targets
            gc.collect()

        total = len(train_indices)
        for c, path in client_exemplars:
            X_ex = np.load(path, mmap_mode='r')
            total += X_ex.shape[0]
            del X_ex
        if total == 0:
            return None, 0

        x_path_str = X_path
        y_path_str = y_path
        train_idx_list = train_indices.tolist()
        ex_info = [(c, str(path)) for c, path in client_exemplars]
        old_path_str = str(old_targets_path) if old_targets_path else None

        def gen():
            X_n = np.load(x_path_str, mmap_mode='r')
            y_n = np.load(y_path_str, mmap_mode='r')
            old_tgt = np.load(old_path_str, mmap_mode='r') if old_path_str else None
            row = 0
            for i in range(0, len(train_idx_list), batch_size):
                idx = train_idx_list[i:i + batch_size]
                X_b = np.array(X_n[idx], dtype=np.float32)
                lab = np.array(y_n[idx])
                lab = lab if len(lab.shape) == 1 else np.argmax(lab, axis=1)
                lab = np.vectorize(label_map.get, otypes=[int])(lab)
                y_b = tf.keras.utils.to_categorical(lab, num_classes).astype(np.float32)
                if old_tgt is not None:
                    o_b = np.array(old_tgt[row:row + X_b.shape[0]], dtype=np.float32)
                    row += X_b.shape[0]
                    yield X_b, y_b, o_b
                else:
                    yield X_b, y_b
            del X_n, y_n
            for c, p in ex_info:
                X_ex = np.load(p, mmap_mode='r')
                mapped_c = label_map[c]
                for i in range(0, X_ex.shape[0], batch_size):
                    X_b = np.array(X_ex[i:i + batch_size], dtype=np.float32)
                    lab = np.full((X_b.shape[0],), mapped_c)
                    y_b = tf.keras.utils.to_categorical(lab, num_classes).astype(np.float32)
                    if old_tgt is not None:
                        o_b = np.array(old_tgt[row:row + X_b.shape[0]], dtype=np.float32)
                        row += X_b.shape[0]
                        yield X_b, y_b, o_b
                    else:
                        yield X_b, y_b
                del X_ex
            del old_tgt

        if old_nc > 0:
            output_signature = (
                tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
                tf.TensorSpec(shape=(None, old_nc), dtype=tf.float32),
            )
        else:
            output_signature = (
                tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
            )
        dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        dataset = dataset.prefetch(1)
        return dataset, total

    def _build_dataset_pt(self, X_path, y_path, model, batch_size):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        X_new = np.load(X_path)
        y_new = np.load(y_path)
        num_classes = len(self.class_order)
        old_nc = self.old_num_classes
        label_map = dict(self.label_map)
        # split val same as TF path
        labels_full = np.array(y_new if len(y_new.shape) == 1 else np.argmax(y_new, axis=1))
        old_nc_classes = list(self.class_order[:old_nc])
        task_classes = list(self.class_order[old_nc:])
        train_mask = np.ones(X_new.shape[0], dtype=bool)
        new_val = {}
        for c in task_classes:
            c_idx = np.where(labels_full == c)[0]
            val_n = int(len(c_idx) * self.val_ratio)
            if val_n == 0:
                continue
            val_idx = c_idx[-val_n:]
            train_mask[val_idx] = False
            new_val[c] = np.array(X_new[val_idx], dtype=np.float32)
        self.client_val_X[self.client_id] = new_val
        train_indices = np.where(train_mask)[0]

        client_exemplars = list(self.exemplars.get(self.client_id, {}).items())
        X_parts = [np.array(X_new[train_indices], dtype=np.float32)]
        y_parts = []
        lab = labels_full[train_indices]
        mapped = np.vectorize(label_map.get, otypes=[int])(lab)
        y_oh = np.zeros((len(mapped), num_classes), dtype=np.float32)
        y_oh[np.arange(len(mapped)), mapped.astype(int)] = 1.0
        y_parts.append(y_oh)
        for c, path in client_exemplars:
            X_ex = np.load(path).astype(np.float32)
            X_parts.append(X_ex)
            y_ex = np.zeros((len(X_ex), num_classes), dtype=np.float32)
            y_ex[:, label_map[c]] = 1.0
            y_parts.append(y_ex)
        X_all = np.concatenate(X_parts)
        y_all = np.concatenate(y_parts)
        total = X_all.shape[0]
        del X_parts, y_parts

        if total == 0:
            return None, 0

        if old_nc > 0 and self.old_logits_model is not None:
            import torch
            T = self.temperature
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.old_logits_model.to(dev)
            logits_np = []
            for i in range(0, len(X_all), PRED_BATCH):
                chunk = torch.from_numpy(X_all[i:i + PRED_BATCH]).to(dev)
                with torch.no_grad():
                    logits_np.append(self.old_logits_model(chunk, return_logits=True).cpu().numpy())
            old_targets = np.concatenate(logits_np, axis=0)[:, :old_nc].astype(np.float32) / T
            dataset = TensorDataset(
                torch.from_numpy(X_all),
                torch.from_numpy(y_all),
                torch.from_numpy(old_targets))
        else:
            dataset = TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all))
        del X_all, y_all
        gc.collect()
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), total

    # Stage 1 train step: LwF distillation + CE

    def get_train_step(self, model, optimizer, loss_fn):
        if not _use_tf():
            if self.old_logits_model is None or self.old_num_classes == 0:
                return None
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            self.old_logits_model.to(dev)
            T = self.temperature
            lam = self.lwf_alpha
            old_nc = self.old_num_classes

            def step_pt(X_b, y_b, old_b=None):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                logits = model.nn(X_b, return_logits=True)
                ce_loss = F.cross_entropy(logits, y_cls, label_smoothing=0.05)
                if old_b is not None and old_nc > 0:
                    old_probs = F.softmax(old_b.to(dev), dim=1)
                    new_probs = F.log_softmax(logits[:, :old_nc] / T, dim=1)
                    distill = F.kl_div(new_probs, old_probs, reduction='batchmean') * (T * T)
                else:
                    distill = torch.tensor(0.0, device=dev)
                total = lam * distill + (1.0 - lam) * ce_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                return ce_loss.item(), distill.item(), total.item()

            return step_pt
        keras_model = model.model if hasattr(model, 'model') else model
        new_logits_model = model.get_logits_model() if hasattr(model, 'get_logits_model') else None
        T = self.temperature
        lam = self.lwf_alpha
        old_nc = self.old_num_classes

        if self.old_logits_model is None or old_nc == 0:
            return None

        @tf.function(reduce_retracing=True)
        def train_step_bic(batch_X, batch_y, batch_old_logits):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                new_logits = new_logits_model(batch_X, training=True)
                old_probs = tf.stop_gradient(tf.nn.softmax(batch_old_logits))
                new_probs = tf.nn.softmax(new_logits[:, :old_nc] / T)
                distill = tf.keras.losses.KLDivergence()(old_probs, new_probs) * (T * T)
                total_loss = lam * distill + (1.0 - lam) * ce_loss
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, distill, total_loss

        return train_step_bic

    def get_ssd_train_step(self, model, optimizer, loss_fn,
                           global_logits_model, local_logits_model,
                           M_class_tf, m_max_tf):
        if self.old_logits_model is None or self.old_num_classes == 0:
            from .finetune import Finetune
            return Finetune(num_classes=self.num_classes).get_ssd_train_step(
                model, optimizer, loss_fn, global_logits_model, local_logits_model, M_class_tf, m_max_tf)

        if not _use_tf():
            import torch
            import torch.nn.functional as F
            from .finetune import _pt_ssd_loss
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            T = self.temperature
            lam = self.lwf_alpha
            old_nc = self.old_num_classes
            m_max = float(m_max_tf)
            old_model = self.old_logits_model

            def step_pt(X_b, y_b, old_logits_b=None):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                local_logits = model.nn(X_b, return_logits=True)
                ce_loss = F.cross_entropy(local_logits, y_cls, label_smoothing=0.05)
                if old_logits_b is not None:
                    old_probs = F.softmax(old_logits_b.to(dev), dim=1)
                    new_probs = F.log_softmax(local_logits[:, :old_nc] / T, dim=1)
                    distill = F.kl_div(new_probs, old_probs, reduction='batchmean') * (T * T)
                else:
                    distill = torch.tensor(0.0, device=dev)
                with torch.no_grad():
                    global_log = torch.from_numpy(
                        global_logits_model.predict(X_b.cpu().numpy(), batch_size=len(X_b))).to(dev)
                ssd_loss = _pt_ssd_loss(local_logits, global_log, y_b, M_class_tf, m_max)
                total = lam * distill + (1.0 - lam) * ce_loss + ssd_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                return ce_loss.item(), ssd_loss.item(), total.item()

            return step_pt

        keras_model = model.model if hasattr(model, 'model') else model
        T = self.temperature
        lam = self.lwf_alpha
        old_nc = self.old_num_classes

        @tf.function(reduce_retracing=True)
        def train_step_bic_ssd(batch_X, batch_y, batch_old_logits):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)

                new_logits = local_logits_model(batch_X, training=True)
                old_probs = tf.stop_gradient(tf.nn.softmax(batch_old_logits))
                new_probs = tf.nn.softmax(new_logits[:, :old_nc] / T)
                distill = tf.keras.losses.KLDivergence()(old_probs, new_probs) * (T * T)

                global_lg = global_logits_model(batch_X, training=False)
                global_probs = tf.nn.softmax(global_lg)
                true_labels = tf.cast(tf.argmax(batch_y, axis=1), tf.int32)
                batch_size = tf.shape(batch_y)[0]
                indices = tf.stack([tf.range(batch_size), true_labels], axis=1)
                p_g_k2 = tf.gather_nd(global_probs, indices)
                M_sample = tf.expand_dims(
                    1.0 - tf.sqrt(tf.maximum(1.0 - p_g_k2, 0.0)), axis=1)
                M = m_max_tf * tf.nn.relu(M_class_tf * M_sample - 0.1)
                logit_diff = M * (global_lg - new_logits)
                ssd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logit_diff), axis=1))

                total_loss = lam * distill + (1.0 - lam) * ce_loss + ssd_loss
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, ssd_loss, total_loss

        return train_step_bic_ssd

    # Stage 2: per-client bias correction

    def run_client_bias_correction(self, model, client_id, n_epochs=50, lr=0.01):
        old_nc = self.old_num_classes
        num_classes = len(self.class_order)
        if self.val_ratio == 0.0 or old_nc == 0 or old_nc >= num_classes:
            return 1.0, 0.0
        logits_model = model.get_logits_model() if hasattr(model, 'get_logits_model') else None

        per_class = {}
        for c, path in self.val_exemplars.get(client_id, {}).items():
            mapped = self.label_map[c]
            X_v = np.load(path, mmap_mode='r')
            if X_v.shape[0] > 0:
                per_class.setdefault(mapped, []).append(np.array(X_v, dtype=np.float32))
            del X_v
        for c, X_v in self.client_val_X.get(client_id, {}).items():
            if X_v.shape[0] > 0:
                mapped = self.label_map[c]
                per_class.setdefault(mapped, []).append(X_v)

        if not per_class:
            return 1.0, 0.0

        for k in per_class:
            per_class[k] = np.concatenate(per_class[k], axis=0)
        min_n = min(v.shape[0] for v in per_class.values())
        if min_n == 0:
            return 1.0, 0.0

        X_parts, y_parts = [], []
        for mapped_c, data in per_class.items():
            X_parts.append(data[:min_n])
            y_parts.append(np.full(min_n, mapped_c, dtype=np.int32))
        X_val = np.concatenate(X_parts, axis=0)
        y_val = np.concatenate(y_parts, axis=0)
        del per_class, X_parts, y_parts

        if not _use_tf():
            import torch
            import torch.nn.functional as F
            logits_all = logits_model.predict(X_val, batch_size=PRED_BATCH)
            del X_val
            logits_t = torch.from_numpy(logits_all)
            y_t = torch.from_numpy(y_val.astype(np.int64))
            alpha_p = torch.tensor(1.0, requires_grad=True)
            beta_p = torch.tensor(0.0, requires_grad=True)
            opt_pt = torch.optim.Adam([alpha_p, beta_p], lr=lr)
            for _ in range(n_epochs):
                opt_pt.zero_grad()
                corrected = torch.cat([logits_t[:, :old_nc], logits_t[:, old_nc:] * alpha_p + beta_p], dim=1)
                loss = F.cross_entropy(corrected, y_t)
                loss.backward()
                opt_pt.step()
            result = (float(alpha_p.detach()), float(beta_p.detach()))
            del logits_all, logits_t, y_t, alpha_p, beta_p, opt_pt
            gc.collect()
            return result

        import tensorflow as tf
        y_onehot = tf.keras.utils.to_categorical(y_val, num_classes).astype(np.float32)
        logits_all = _batched_predict(logits_model, X_val)
        del X_val

        alpha_var = tf.Variable(1.0, dtype=tf.float32)
        beta_var = tf.Variable(0.0, dtype=tf.float32)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        logits_tf = tf.constant(logits_all, dtype=tf.float32)
        y_tf = tf.constant(y_onehot, dtype=tf.float32)
        del logits_all, y_onehot

        for _ in range(n_epochs):
            with tf.GradientTape() as tape:
                corrected = tf.concat([
                    logits_tf[:, :old_nc],
                    logits_tf[:, old_nc:] * alpha_var + beta_var
                ], axis=1)
                probs = tf.nn.softmax(corrected)
                loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(y_tf, probs)
                )
            grads = tape.gradient(loss, [alpha_var, beta_var])
            opt.apply_gradients(zip(grads, [alpha_var, beta_var]))

        result = (float(alpha_var.numpy()), float(beta_var.numpy()))
        del logits_tf, y_tf, alpha_var, beta_var, opt
        gc.collect()
        return result

    # Apply / undo bias correction on the model's logits layer

    def apply_bias_correction(self, model):
        if self.bias_alpha is None or self.old_num_classes == 0:
            return
        old_nc = self.old_num_classes
        if not _use_tf():
            import torch
            sd = model.nn.state_dict()
            self._saved_logits_weights = {k: v.clone() for k, v in sd.items() if 'logits' in k}
            for k in list(sd.keys()):
                if 'logits.weight' in k:
                    sd[k][:, old_nc:] *= self.bias_alpha
                elif 'logits.bias' in k:
                    sd[k][old_nc:] = sd[k][old_nc:] * self.bias_alpha + self.bias_beta
            model.nn.load_state_dict(sd)
            return
        logits_layer = model.model.get_layer('logits')
        self._saved_logits_weights = logits_layer.get_weights()
        kernel, bias = [w.copy() for w in self._saved_logits_weights]
        kernel[:, old_nc:] *= self.bias_alpha
        bias[old_nc:] = bias[old_nc:] * self.bias_alpha + self.bias_beta
        logits_layer.set_weights([kernel, bias])
        model._logits_model = None
        model._feature_model = None

    def undo_bias_correction(self, model):
        if not _use_tf():
            if not hasattr(self, '_saved_logits_weights') or self._saved_logits_weights is None:
                return
            sd = model.nn.state_dict()
            sd.update(self._saved_logits_weights)
            model.nn.load_state_dict(sd)
            self._saved_logits_weights = None
            return
        logits_layer = model.model.get_layer('logits')
        logits_layer.set_weights(self._saved_logits_weights)
        model._logits_model = None
        model._feature_model = None
        self._saved_logits_weights = None

    # Evaluation: softmax CE + logits argmax

    def evaluate(self, model, X, y, num_classes, logger=None):
        logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else None
        y_labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        y_mapped = self.map_labels_np(y_labels)
        loss_sum = 0.0
        n = X.shape[0]
        all_preds = []
        for i in range(0, n, PRED_BATCH):
            x_chunk = np.array(X[i:i + PRED_BATCH], dtype=np.float32)
            y_idx = y_mapped[i:i + PRED_BATCH].astype(int)
            y_chunk = np.zeros((len(y_idx), num_classes), dtype=np.float32)
            y_chunk[np.arange(len(y_idx)), y_idx] = 1.0
            if _use_tf():
                log_chunk = logits_model(x_chunk, training=False).numpy()
            else:
                log_chunk = logits_model.predict(x_chunk, batch_size=len(x_chunk))
            all_preds.append(np.argmax(log_chunk, axis=1))
            e = np.exp(log_chunk - log_chunk.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            p = np.clip(p, 1e-7, 1.0)
            loss_sum += float(-np.sum(y_chunk * np.log(p)))
            del x_chunk, y_chunk, log_chunk, p
        preds = np.concatenate(all_preds)
        return {"acc": float(np.mean(preds == y_mapped)), "loss": loss_sum / n}

    # Cleanup

    def cleanup_temp(self):
        for d in (self.exemplar_dir, self.val_dir):
            if d.exists():
                for f in d.glob("*.npy"):
                    f.unlink(missing_ok=True)
        old_tgt = Path("temp_weights/bic_old_targets.npy")
        if old_tgt.exists():
            old_tgt.unlink(missing_ok=True)

    def extra_metrics(self) -> dict:
        return {
            "bic_memory": self.memory,
            "bic_alpha": self.lwf_alpha,
            "bic_temperature": self.temperature,
            "bias_alpha": self.bias_alpha,
            "bias_beta": self.bias_beta,
        }
