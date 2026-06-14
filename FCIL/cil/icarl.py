"""Incremental Classifier and Representation Learning (iCaRL)"""

import gc
import math
import numpy as np
from pathlib import Path
from ..backend import use_tf as _use_tf
from .base import CILMethod
tf = __import__('tensorflow') if _use_tf() else None

PRED_BATCH = 4096

def _batched_predict(model, X, batch_size=PRED_BATCH):
    if not _use_tf() and hasattr(model, 'predict'):
        return model.predict(X, batch_size=batch_size)
    parts = []
    for i in range(0, X.shape[0], batch_size):
        parts.append(model(X[i:i+batch_size], training=False).numpy())
    return np.concatenate(parts, axis=0)


class ICaRL(CILMethod):
    def __init__(self, num_classes: int, memory: int = 2000, use_bce: bool = False, **kwargs):
        super().__init__(num_classes)
        self.name = "ICaRL"
        self.memory = memory
        self.memory_percent = float(memory)
        self.use_bce = use_bce
        self.class_order = []
        self.label_map = {}
        self.client_id = None
        self.client_seen = {}
        self.exemplars = {}
        self.client_means = {}
        self.client_counts = {}
        self.client_memory_budget = {}
        self.client_budget_tasks = set()
        self.old_num_classes = 0
        self.old_targets = None
        self.global_means = None
        self.global_mean_labels = None
        self.exemplar_dir = Path("temp_weights/icarl_exemplars")

    def set_client(self, client_id: int):
        self.client_id = client_id
        if client_id not in self.client_seen:
            self.client_seen[client_id] = set()
        if client_id not in self.exemplars:
            self.exemplars[client_id] = {}
        if client_id not in self.client_means:
            self.client_means[client_id] = {}
        if client_id not in self.client_counts:
            self.client_counts[client_id] = {}
        if client_id not in self.client_memory_budget:
            self.client_memory_budget[client_id] = 0

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)
        for c in task_classes:
            if c not in self.label_map:
                self.label_map[c] = len(self.class_order)
                self.class_order.append(c)
        self.old_num_classes = len(self.class_order) - len(task_classes)

    def map_labels_np(self, y):
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        mapped = np.vectorize(self.label_map.get, otypes=[int])(labels)
        return mapped

    def _get_m_per_class(self, client_id):
        classes = sorted(self.client_seen.get(client_id, []))
        t = len(classes)
        budget = int(self.client_memory_budget.get(client_id, 0))
        base = budget // t if t > 0 else 0
        remainder = budget - base * t
        sizes = {c: base for c in classes}
        for c in classes[:remainder]:
            sizes[c] += 1
        return sizes

    def _reduce_exemplars(self, client_id, m_per_class):
        for c, path in list(self.exemplars.get(client_id, {}).items()):
            m = m_per_class.get(c, 0)
            X_ex = np.load(path, mmap_mode='r')
            X_trim = np.array(X_ex[:m])
            np.save(path, X_trim)

    def _construct_exemplars(self, feature_model, X_c, m):
        if m <= 0 or X_c.shape[0] == 0:
            return np.empty((0,) + X_c.shape[1:], dtype=X_c.dtype), None, 0
        if feature_model is None:
            return np.array(X_c[:m]), None, 0
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
        result = np.array(X_c[selected_idx])
        exemplar_mean = sum_sel / max(1, len(selected_idx))
        del feats
        return result, exemplar_mean, len(selected_idx)

    def update_exemplars(self, model, X_path, y_path, task_classes, reduce_old=False):
        feature_model = model.get_feature_model() if hasattr(model, "get_feature_model") else None
        if feature_model is None:
            return
        X_new = np.load(X_path, mmap_mode='r')
        y_new = np.load(y_path, mmap_mode='r')
        key = (self.client_id, self.current_task)
        if key not in self.client_budget_tasks:
            self.client_memory_budget[self.client_id] = self.client_memory_budget.get(self.client_id, 0) + int(math.ceil(X_new.shape[0] * self.memory_percent / 100.0))
            self.client_budget_tasks.add(key)
        labels = np.array(y_new if len(y_new.shape) == 1 else np.argmax(y_new, axis=1))
        present = []
        for c in task_classes:
            if np.any(labels == c):
                present.append(c)
        for c in present:
            self.client_seen[self.client_id].add(c)
        m_per_class = self._get_m_per_class(self.client_id)
        if reduce_old:
            self._reduce_exemplars(self.client_id, m_per_class)
        for c in present:
            mask = labels == c
            X_c = np.array(X_new[mask], dtype=np.float32)
            m = m_per_class.get(c, 0)
            exemplars, mean, count = self._construct_exemplars(feature_model, X_c, m)
            self.exemplar_dir.mkdir(parents=True, exist_ok=True)
            path = self.exemplar_dir / f"client_{self.client_id}_class_{c}.npy"
            np.save(path, exemplars)
            self.exemplars[self.client_id][c] = path
            if self.use_bce and count > 0:
                mean = mean / (np.linalg.norm(mean) + 1e-12)
                self.client_means[self.client_id][c] = mean.astype(np.float32)
                self.client_counts[self.client_id][c] = count
            del X_c, exemplars
        del X_new, y_new, labels, feature_model
        gc.collect()

    def rebuild_global_exemplars(self, model):
        if not self.use_bce:
            self.global_means = None
            self.global_mean_labels = None
            return
        means = []
        labels = []
        for c in self.class_order:
            sum_feat = None
            count = 0
            for client_id in self.client_means:
                if c in self.client_means[client_id]:
                    mean = self.client_means[client_id][c]
                    n = self.client_counts[client_id][c]
                    sum_feat = mean * n if sum_feat is None else sum_feat + mean * n
                    count += n
            if count > 0:
                mean = sum_feat / count
                mean = mean / (np.linalg.norm(mean) + 1e-12)
                means.append(mean)
                labels.append(self.label_map[c])
        gc.collect()
        self.global_means = np.stack(means, axis=0) if means else None
        self.global_mean_labels = labels

    def build_dataset(self, X_path, y_path, model, batch_size):
        if not _use_tf():
            return self._build_dataset_pt(X_path, y_path, model, batch_size)
        old_tgt_file = Path("temp_weights/icarl_old_targets.npy")
        if old_tgt_file.exists():
            old_tgt_file.unlink(missing_ok=True)
        X_new = np.load(X_path, mmap_mode='r')
        y_new = np.load(y_path, mmap_mode='r')
        num_classes = len(self.class_order)
        old_nc = self.old_num_classes
        input_dim = X_new.shape[1]
        client_exemplars = list(self.exemplars.get(self.client_id, {}).items())
        label_map = dict(self.label_map)
        total_check = X_new.shape[0]
        for _, _ex_path in client_exemplars:
            _X_ex = np.load(_ex_path, mmap_mode='r')
            total_check += _X_ex.shape[0]
            del _X_ex
        if total_check == 0:
            return None, 0

        # Pre-compute old logits on disk to avoid calling model inside generator
        old_targets_path = None
        if old_nc > 0:
            logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else None
            all_old = []
            # new data logits
            for i in range(0, X_new.shape[0], PRED_BATCH):
                chunk = np.array(X_new[i:i+PRED_BATCH], dtype=np.float32)
                logits = logits_model(chunk, training=False).numpy()
                all_old.append(tf.nn.sigmoid(logits[:, :old_nc]).numpy().astype(np.float32))
                del chunk, logits
            # exemplar logits
            for c, path in client_exemplars:
                X_ex = np.load(path, mmap_mode='r')
                for i in range(0, X_ex.shape[0], PRED_BATCH):
                    chunk = np.array(X_ex[i:i+PRED_BATCH], dtype=np.float32)
                    logits = logits_model(chunk, training=False).numpy()
                    all_old.append(tf.nn.sigmoid(logits[:, :old_nc]).numpy().astype(np.float32))
                    del chunk, logits
                del X_ex
            old_targets = np.concatenate(all_old, axis=0)
            del all_old, logits_model
            gc.collect()
            old_targets_path = Path("temp_weights/icarl_old_targets.npy")
            old_targets_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(old_targets_path), old_targets)
            del old_targets
            gc.collect()

        total = X_new.shape[0]
        for c, path in client_exemplars:
            X_ex = np.load(path, mmap_mode='r')
            total += X_ex.shape[0]
            del X_ex

        x_path_str = X_path
        y_path_str = y_path
        ex_info = [(c, str(path)) for c, path in client_exemplars]
        old_path_str = str(old_targets_path) if old_targets_path else None

        def gen():
            X_n = np.load(x_path_str, mmap_mode='r')
            y_n = np.load(y_path_str, mmap_mode='r')
            old_tgt = np.load(old_path_str, mmap_mode='r') if old_path_str else None
            row = 0
            # yield new data
            for i in range(0, X_n.shape[0], batch_size):
                X_b = np.array(X_n[i:i+batch_size], dtype=np.float32)
                lab = np.array(y_n[i:i+batch_size])
                lab = lab if len(lab.shape) == 1 else np.argmax(lab, axis=1)
                lab = np.vectorize(label_map.get, otypes=[int])(lab)
                y_b = tf.keras.utils.to_categorical(lab, num_classes).astype(np.float32)
                if old_tgt is not None:
                    o_b = np.array(old_tgt[row:row+X_b.shape[0]], dtype=np.float32)
                    row += X_b.shape[0]
                    yield X_b, y_b, o_b
                else:
                    yield X_b, y_b
            del X_n, y_n
            for c, p in ex_info:
                X_ex = np.load(p, mmap_mode='r')
                mapped_c = label_map[c]
                for i in range(0, X_ex.shape[0], batch_size):
                    X_b = np.array(X_ex[i:i+batch_size], dtype=np.float32)
                    lab = np.full((X_b.shape[0],), mapped_c)
                    y_b = tf.keras.utils.to_categorical(lab, num_classes).astype(np.float32)
                    if old_tgt is not None:
                        o_b = np.array(old_tgt[row:row+X_b.shape[0]], dtype=np.float32)
                        row += X_b.shape[0]
                        yield X_b, y_b, o_b
                    else:
                        yield X_b, y_b
                del X_ex
            del old_tgt

        if old_nc > 0:
            output_signature=(
                tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
                tf.TensorSpec(shape=(None, old_nc), dtype=tf.float32)
            )
        else:
            output_signature=(
                tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
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
        labels = y_new if len(y_new.shape) == 1 else np.argmax(y_new, axis=1)
        mapped = np.vectorize(label_map.get, otypes=[int])(labels)
        y_oh = np.zeros((len(mapped), num_classes), dtype=np.float32)
        y_oh[np.arange(len(mapped)), mapped] = 1.0
        X_parts = [X_new.astype(np.float32)]
        y_parts = [y_oh]
        client_exemplars = list(self.exemplars.get(self.client_id, {}).items())
        for c, path in client_exemplars:
            X_ex = np.load(path)
            y_ex = np.zeros((len(X_ex), num_classes), dtype=np.float32)
            y_ex[:, label_map[c]] = 1.0
            X_parts.append(X_ex.astype(np.float32))
            y_parts.append(y_ex)
        X_all = np.concatenate(X_parts)
        y_all = np.concatenate(y_parts)
        total = X_all.shape[0]
        del X_parts, y_parts
        if total == 0:
            return None, 0
        if old_nc > 0:
            logits_model = model.get_logits_model()
            logits_np = logits_model.predict(X_all, batch_size=PRED_BATCH)
            old_tgt = 1.0 / (1.0 + np.exp(-logits_np[:, :old_nc])).astype(np.float32)
            dataset = TensorDataset(
                torch.from_numpy(X_all),
                torch.from_numpy(y_all),
                torch.from_numpy(old_tgt))
        else:
            dataset = TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all))
        del X_all, y_all
        gc.collect()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader, total

    def cleanup_temp(self):
        if self.exemplar_dir.exists():
            for f in self.exemplar_dir.glob("*.npy"):
                f.unlink(missing_ok=True)
        old_tgt = Path("temp_weights/icarl_old_targets.npy")
        if old_tgt.exists():
            old_tgt.unlink(missing_ok=True)

    def get_train_step(self, model, optimizer, loss_fn):
        if not _use_tf():
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            use_bce = self.use_bce
            old_nc = self.old_num_classes

            def step_pt(X_b, y_b, old_b=None):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                logits = model.nn(X_b, return_logits=True)
                if use_bce:
                    probs = torch.sigmoid(logits)
                    ce_loss = F.binary_cross_entropy(probs, y_b)
                else:
                    ce_loss = F.cross_entropy(logits, y_cls, label_smoothing=0.05)
                distill = torch.tensor(0.0, device=dev)
                if old_b is not None and old_nc > 0:
                    probs_old = torch.sigmoid(logits[:, :old_nc])
                    distill = F.binary_cross_entropy(probs_old, old_b.to(dev))
                total = ce_loss + distill
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                return ce_loss.item(), distill.item(), total.item()

            if self.old_num_classes > 0:
                return step_pt
            return lambda X_b, y_b: step_pt(X_b, y_b, None)

        keras_model = model.model if hasattr(model, 'model') else model
        logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else None
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        @tf.function
        def train_step_with_icarl(batch_X, batch_y, batch_old=None):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                if self.use_bce:
                    logits = logits_model(batch_X, training=True)
                    probs = tf.nn.sigmoid(logits)
                    ce_loss = bce(batch_y, probs)
                else:
                    ce_loss = loss_fn(batch_y, predictions)
                distill = 0.0
                if batch_old is not None and self.old_num_classes > 0:
                    logits = logits_model(batch_X, training=True)
                    probs_old = tf.nn.sigmoid(logits[:, :self.old_num_classes])
                    distill = bce(batch_old, probs_old)
                total_loss = ce_loss + distill
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, distill, total_loss

        if self.old_num_classes > 0:
            return train_step_with_icarl
        else:
            @tf.function
            def train_step_no_old(batch_X, batch_y):
                return train_step_with_icarl(batch_X, batch_y, None)
            return train_step_no_old

    def evaluate(self, model, X, y, num_classes, logger=None):
        logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else None
        y_labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        y_mapped = self.map_labels_np(y_labels)
        if self.use_bce:
            preds = self.classify(model, X)
        loss_sum = 0.0
        n = X.shape[0]
        all_preds_logits = []
        for i in range(0, n, PRED_BATCH):
            x_chunk = np.array(X[i:i+PRED_BATCH], dtype=np.float32)
            y_idx = y_mapped[i:i+PRED_BATCH].astype(int)
            y_chunk = np.zeros((len(y_idx), num_classes), dtype=np.float32)
            y_chunk[np.arange(len(y_idx)), y_idx] = 1.0
            if _use_tf():
                log_chunk = logits_model(x_chunk, training=False).numpy()
            else:
                log_chunk = logits_model.predict(x_chunk, batch_size=len(x_chunk))
            if self.use_bce:
                p = 1.0 / (1.0 + np.exp(-log_chunk))
                p = np.clip(p, 1e-7, 1.0 - 1e-7)
                loss_sum += float(-np.sum(y_chunk * np.log(p) + (1 - y_chunk) * np.log(1 - p))) / y_chunk.shape[1]
            else:
                all_preds_logits.append(np.argmax(log_chunk, axis=1))
                e = np.exp(log_chunk - log_chunk.max(axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
                p = np.clip(p, 1e-7, 1.0)
                loss_sum += float(-np.sum(y_chunk * np.log(p)))
            del x_chunk, y_chunk, log_chunk, p
        loss = loss_sum / n
        if not self.use_bce:
            preds = np.concatenate(all_preds_logits)
        acc = float(np.mean(preds == y_mapped))
        return {"acc": acc, "loss": loss}

    def classify(self, model, X):
        feature_model = model.get_feature_model() if hasattr(model, "get_feature_model") else None
        if self.global_means is None or len(self.global_mean_labels) == 0:
            logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else None
            logits = _batched_predict(logits_model, X)
            return np.argmax(logits, axis=1)
        all_preds = []
        for i in range(0, X.shape[0], PRED_BATCH):
            chunk = np.array(X[i:i+PRED_BATCH], dtype=np.float32)
            feats = feature_model(chunk, training=False).numpy()
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
            dists = np.linalg.norm(feats[:, None, :] - self.global_means[None, :, :], axis=2)
            idx = np.argmin(dists, axis=1)
            all_preds.append(np.array([self.global_mean_labels[j] for j in idx]))
            del chunk, feats, dists
        return np.concatenate(all_preds)

    def extra_metrics(self) -> dict:
        return {
            "icarl_memory": self.memory,
            "icarl_bce": int(self.use_bce)
        }

    def get_ssd_train_step(self, model, optimizer, loss_fn,
                           global_logits_model, local_logits_model,
                           M_class_tf, m_max_tf):
        keras_model = model.model if hasattr(model, 'model') else model
        logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else None
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        old_nc = self.old_num_classes
        use_bce = self.use_bce

        @tf.function(reduce_retracing=True)
        def train_step_icarl_ssd(batch_X, batch_y, batch_old=None):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                if use_bce:
                    logits = logits_model(batch_X, training=True)
                    probs = tf.nn.sigmoid(logits)
                    ce_loss = bce(batch_y, probs)
                else:
                    ce_loss = loss_fn(batch_y, predictions)
                distill = 0.0
                if batch_old is not None and old_nc > 0:
                    logits = logits_model(batch_X, training=True)
                    probs_old = tf.nn.sigmoid(logits[:, :old_nc])
                    distill = bce(batch_old, probs_old)

                local_lg = local_logits_model(batch_X, training=True)
                global_lg = global_logits_model(batch_X, training=False)
                global_probs = tf.nn.softmax(global_lg)
                true_labels = tf.cast(tf.argmax(batch_y, axis=1), tf.int32)
                batch_size = tf.shape(batch_y)[0]
                indices = tf.stack([tf.range(batch_size), true_labels], axis=1)
                p_g_k2 = tf.gather_nd(global_probs, indices)
                M_sample = tf.expand_dims(
                    1.0 - tf.sqrt(tf.maximum(1.0 - p_g_k2, 0.0)), axis=1)
                M = m_max_tf * tf.nn.relu(M_class_tf * M_sample - 0.1)
                logit_diff = M * (global_lg - local_lg)
                ssd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logit_diff), axis=1))

                total_loss = ce_loss + distill + ssd_loss
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, ssd_loss, total_loss

        if old_nc > 0:
            return train_step_icarl_ssd
        else:
            @tf.function(reduce_retracing=True)
            def train_step_no_old_ssd(batch_X, batch_y):
                return train_step_icarl_ssd(batch_X, batch_y, None)
            return train_step_no_old_ssd