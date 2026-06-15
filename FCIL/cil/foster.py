"""FOSTER: Feature Boosting and Compression for Class-Incremental Learning
Stage 1 (Feature Boosting): 3-loss objective - Logits Alignment + Feature Enhancement + KD
Stage 2 (Feature Compression): Distill boosted ensemble into single model via BKD
Uses iCaRL herding for exemplar management.
"""

import gc
import copy
import numpy as np
from pathlib import Path
from ..backend import use_tf as _use_tf
from .base import CILMethod
tf = __import__('tensorflow') if _use_tf() else None

def _logits_feature_model(keras_model):
    logits_layer = keras_model.get_layer('logits')
    return tf.keras.Model(keras_model.input, logits_layer.input)

def _new_student_like(model, config, total_nc):
    if not _use_tf():
        if config.model == "gru":
            from models.pt_gru import PTGRUModel
            return PTGRUModel(model.input_dim, total_nc, config.batch_size)
        from models.pt_dense import PTDenseModel
        return PTDenseModel(model.input_dim, total_nc, config.batch_size)
    if config.model == "gru":
        from models.gru import GRUModel
        return GRUModel(input_dim=model.input_dim, num_classes=total_nc, batch_size=config.batch_size)
    from models.dense import DenseModel
    return DenseModel(input_dim=model.input_dim, num_classes=total_nc, batch_size=config.batch_size)

def _copy_backbone_and_old_head(student_model, old_model, old_nc):
    old_layers = {layer.name: layer for layer in old_model.layers}
    for layer in student_model.layers:
        if layer.name == 'logits':
            old_kernel, old_bias = old_layers['logits'].get_weights()
            new_kernel, new_bias = layer.get_weights()
            new_kernel[:, :old_nc] = old_kernel
            new_bias[:old_nc] = old_bias
            layer.set_weights([new_kernel, new_bias])
        elif layer.name in old_layers and layer.get_weights():
            layer.set_weights(old_layers[layer.name].get_weights())

def _copy_pt_backbone_and_old_head(student_nn, old_nn, old_nc):
    old_sd = old_nn.state_dict()
    new_sd = student_nn.state_dict()
    for k in new_sd:
        if k not in old_sd:
            continue
        if k == 'logits.weight':
            new_sd[k][:old_nc].copy_(old_sd[k])
        elif k == 'logits.bias':
            new_sd[k][:old_nc].copy_(old_sd[k])
        elif new_sd[k].shape == old_sd[k].shape:
            new_sd[k].copy_(old_sd[k])
    student_nn.load_state_dict(new_sd)

def _pt_feature_hook(model):
    box = {}
    hook = model.nn.logits.register_forward_pre_hook(lambda module, inp: box.__setitem__('features', inp[0]))
    return hook, box

PRED_BATCH = 4096


def _batched_predict(model, X, batch_size=PRED_BATCH):
    if not _use_tf() and hasattr(model, 'predict'):
        return model.predict(X, batch_size=batch_size)
    parts = []
    for i in range(0, X.shape[0], batch_size):
        parts.append(model(X[i:i + batch_size], training=False).numpy())
    return np.concatenate(parts, axis=0)


def _kd_loss(pred, soft, T):
    """Standard KD loss: -sum(softmax(soft/T) * log_softmax(pred/T)) / batch"""
    pred_log = tf.nn.log_softmax(pred / T, axis=1)
    soft_prob = tf.nn.softmax(soft / T, axis=1)
    return -tf.reduce_sum(soft_prob * pred_log) / tf.cast(tf.shape(pred)[0], tf.float32)


def _bkd_loss(pred, soft, T, per_cls_weights):
    """Balanced KD loss: per-class weighted soft targets, then KD"""
    pred_log = tf.nn.log_softmax(pred / T, axis=1)
    soft_prob = tf.nn.softmax(soft / T, axis=1)
    soft_weighted = soft_prob * per_cls_weights
    soft_weighted = soft_weighted / tf.reduce_sum(soft_weighted, axis=1, keepdims=True)
    return -tf.reduce_sum(soft_weighted * pred_log) / tf.cast(tf.shape(pred)[0], tf.float32)


class FOSTER(CILMethod):
    def __init__(self, num_classes: int, memory: int = 2000,
                 beta1: float = 0.97, beta2: float = 0.97,
                 lambda_okd: float = 1.0, temperature: float = 2.0,
                 compression_epochs: int = 50, **kwargs):
        super().__init__(num_classes)
        self.name = "FOSTER"
        self.memory = memory
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_okd = lambda_okd
        self.temperature = temperature
        self.compression_epochs = compression_epochs
        self.class_order = []
        self.label_map = {}
        self.client_id = None
        self.client_seen = {}
        self.exemplars = {}
        self.old_num_classes = 0
        self.old_model = None
        self.old_logits_model = None
        self.old_feature_model = None
        self._in_boosting = False
        self._class_sample_counts = {}  # mapped_label -> total count across clients this task
        self.per_cls_weights_stage1 = None
        self.per_cls_weights_stage2 = None
        self.old_backbone_weights = None
        self.exemplar_dir = Path("temp_weights/foster_exemplars")

    def set_client(self, client_id: int):
        self.client_id = client_id
        if client_id not in self.client_seen:
            self.client_seen[client_id] = set()
        if client_id not in self.exemplars:
            self.exemplars[client_id] = {}
        if client_id == 0:
            self._class_sample_counts = {}

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)
        for c in task_classes:
            if c not in self.label_map:
                self.label_map[c] = len(self.class_order)
                self.class_order.append(c)
        self.old_num_classes = len(self.class_order) - len(task_classes)
        self._class_sample_counts = {}
        self.per_cls_weights_stage1 = None
        self.per_cls_weights_stage2 = None

    def set_old_model(self, model):
        if not _use_tf():
            self.old_model = copy.deepcopy(model.nn)
            self.old_model.eval()
            for p in self.old_model.parameters():
                p.requires_grad_(False)
            self.old_logits_model = self.old_model
            self.old_feature_model = None
            self.old_backbone_weights = model.get_weights()
            self._in_boosting = True
            return
        keras_model = model.model if hasattr(model, 'model') else model
        self.old_model = tf.keras.models.clone_model(keras_model)
        self.old_model.set_weights(keras_model.get_weights())
        self.old_model.trainable = False
        for layer in self.old_model.layers:
            layer.trainable = False
        logits_layer = self.old_model.get_layer('logits')
        self.old_logits_model = tf.keras.Model(self.old_model.input, logits_layer.output)
        self.old_feature_model = _logits_feature_model(self.old_model)
        self.old_backbone_weights = keras_model.get_weights()
        self._in_boosting = True

    def reinit_logits(self, model):
        """Zero old-class logit weights so boosted residual starts at zero.
        Backbone is preserved (carries compressed knowledge from prior task).
        New-class logit weights stay random (from expand_classes)."""
        if not _use_tf():
            with __import__('torch').no_grad():
                old_nc = self.old_num_classes
                model.nn.logits.weight[:old_nc].zero_()
                model.nn.logits.bias[:old_nc].zero_()
            model._logits_model = None
            model._feature_model = None
            return
        logits_layer = (model.model if hasattr(model, 'model') else model).get_layer('logits')
        kernel, bias = logits_layer.get_weights()
        old_nc = self.old_num_classes
        kernel[:, :old_nc] = 0.0
        bias[:old_nc] = 0.0
        logits_layer.set_weights([kernel, bias])
        model._logits_model = None
        model._feature_model = None

    def _compute_per_cls_weights(self, beta, sample_counts):
        """Normalized effective number weighting (Eq. 13)"""
        effective_num = 1.0 - np.power(beta, sample_counts)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(sample_counts)
        return per_cls_weights

    def compute_stage1_weights(self):
        old_nc = self.old_num_classes
        samples_old = max(1, int(self.memory) // old_nc) if old_nc > 0 else 0
        cls_num_list = [samples_old] * old_nc + [
            max(1, self._class_sample_counts.get(self.label_map[c], 1))
            for c in self.class_order[old_nc:]
        ]
        weights = self._compute_per_cls_weights(self.beta1, cls_num_list).astype(np.float32)
        if not _use_tf():
            self.per_cls_weights_stage1 = weights
            return
        if self.per_cls_weights_stage1 is None:
            self.per_cls_weights_stage1 = tf.Variable(weights, trainable=False)
        else:
            self.per_cls_weights_stage1.assign(weights)

    def compute_stage2_weights(self):
        old_nc = self.old_num_classes
        samples_old = max(1, int(self.memory) // old_nc) if old_nc > 0 else 0
        cls_num_list = [samples_old] * old_nc + [
            max(1, self._class_sample_counts.get(self.label_map[c], 1))
            for c in self.class_order[old_nc:]
        ]
        weights = self._compute_per_cls_weights(self.beta2, cls_num_list).astype(np.float32)
        if not _use_tf():
            self.per_cls_weights_stage2 = weights
            return
        if self.per_cls_weights_stage2 is None:
            self.per_cls_weights_stage2 = tf.Variable(weights, trainable=False)
        else:
            self.per_cls_weights_stage2.assign(weights)

    def map_labels_np(self, y):
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        return np.vectorize(self.label_map.get, otypes=[int])(labels)

    # ── Exemplar management (iCaRL herding) ──

    def _get_m_per_class(self, client_id):
        classes = sorted(self.client_seen.get(client_id, []))
        t = len(classes)
        budget = int(self.memory)
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
            np.save(path, np.array(X_ex[:m]))

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
            self.exemplar_dir.mkdir(parents=True, exist_ok=True)
            path = self.exemplar_dir / f"client_{self.client_id}_class_{c}.npy"
            np.save(path, exemplars)
            self.exemplars[self.client_id][c] = path
            del X_c, exemplars
        del X_new, y_new, labels, feature_model
        gc.collect()

    def rebuild_global_exemplars(self, model):
        pass

    # ── Dataset: current task data + exemplars ──

    def build_dataset(self, X_path, y_path, model, batch_size):
        if not _use_tf():
            return self._build_dataset_pt(X_path, y_path, batch_size)
        X_new = np.load(X_path, mmap_mode='r')
        y_new = np.load(y_path, mmap_mode='r')
        num_classes = len(self.class_order)
        old_nc = self.old_num_classes
        input_dim = X_new.shape[1]
        client_exemplars = list(self.exemplars.get(self.client_id, {}).items())
        label_map = dict(self.label_map)

        if old_nc > 0:
            labels_all = np.array(y_new if len(y_new.shape) == 1 else np.argmax(y_new, axis=1))
            for c in self.class_order[old_nc:]:
                cnt = int(np.sum(labels_all == c))
                mapped = self.label_map[c]
                self._class_sample_counts[mapped] = self._class_sample_counts.get(mapped, 0) + cnt
            self.compute_stage1_weights()
            del labels_all

        total = X_new.shape[0]
        for c, path in client_exemplars:
            X_ex = np.load(path, mmap_mode='r')
            total += X_ex.shape[0]
            del X_ex
        if total == 0:
            return None, 0

        x_path_str = X_path
        y_path_str = y_path
        ex_info = [(c, str(path)) for c, path in client_exemplars]

        def gen():
            X_n = np.load(x_path_str, mmap_mode='r')
            y_n = np.load(y_path_str, mmap_mode='r')
            for i in range(0, X_n.shape[0], batch_size):
                X_b = np.array(X_n[i:i + batch_size], dtype=np.float32)
                lab = np.array(y_n[i:i + batch_size])
                lab = lab if len(lab.shape) == 1 else np.argmax(lab, axis=1)
                lab = np.vectorize(label_map.get, otypes=[int])(lab)
                y_b = tf.keras.utils.to_categorical(lab, num_classes).astype(np.float32)
                yield X_b, y_b
            del X_n, y_n
            for c, p in ex_info:
                X_ex = np.load(p, mmap_mode='r')
                mapped_c = label_map[c]
                for i in range(0, X_ex.shape[0], batch_size):
                    X_b = np.array(X_ex[i:i + batch_size], dtype=np.float32)
                    lab = np.full((X_b.shape[0],), mapped_c)
                    y_b = tf.keras.utils.to_categorical(lab, num_classes).astype(np.float32)
                    yield X_b, y_b
                del X_ex

        output_signature = (
            tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
        )
        dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        dataset = dataset.prefetch(1)
        return dataset, total

    def _build_dataset_pt(self, X_path, y_path, batch_size):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X_new = np.load(X_path)
        y_new = np.load(y_path)
        num_classes = len(self.class_order)
        old_nc = self.old_num_classes
        label_map = dict(self.label_map)
        if old_nc > 0:
            labels_all = np.array(y_new if len(y_new.shape) == 1 else np.argmax(y_new, axis=1))
            for c in self.class_order[old_nc:]:
                cnt = int(np.sum(labels_all == c))
                mapped = self.label_map[c]
                self._class_sample_counts[mapped] = self._class_sample_counts.get(mapped, 0) + cnt
            self.compute_stage1_weights()
            del labels_all
        labels = y_new if len(y_new.shape) == 1 else np.argmax(y_new, axis=1)
        mapped = np.vectorize(label_map.get, otypes=[int])(labels)
        y_oh = np.zeros((len(mapped), num_classes), dtype=np.float32)
        y_oh[np.arange(len(mapped)), mapped] = 1.0
        X_parts = [X_new.astype(np.float32)]
        y_parts = [y_oh]
        for c, path in self.exemplars.get(self.client_id, {}).items():
            X_ex = np.load(path)
            if X_ex.shape[0] == 0:
                continue
            y_ex = np.zeros((len(X_ex), num_classes), dtype=np.float32)
            y_ex[:, label_map[c]] = 1.0
            X_parts.append(X_ex.astype(np.float32))
            y_parts.append(y_ex)
        X_all = np.concatenate(X_parts, axis=0)
        y_all = np.concatenate(y_parts, axis=0)
        total = int(X_all.shape[0])
        del X_parts, y_parts
        if total == 0:
            return None, 0
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all)),
            batch_size=batch_size, shuffle=True, drop_last=False)
        del X_all, y_all
        gc.collect()
        return loader, total

    # ── Stage 1: Feature Boosting train step ──

    def get_train_step(self, model, optimizer, loss_fn):
        old_nc = self.old_num_classes
        if old_nc == 0 or self.old_logits_model is None:
            return None

        if not _use_tf():
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            self.old_model.to(dev)
            self.old_model.eval()
            feat_box = {}
            feat_dim = model.nn.logits.in_features
            total_nc = len(self.class_order)
            aux = nn.Linear(feat_dim, total_nc).to(dev)
            optimizer.add_param_group({'params': aux.parameters(), 'weight_decay': 0.0})
            self.compute_stage1_weights()
            w = torch.from_numpy(np.array(self.per_cls_weights_stage1, dtype=np.float32)).to(dev)
            T = self.temperature
            lambda_okd = self.lambda_okd

            def step_pt(X_b, y_b):
                X_b = X_b.to(dev, non_blocking=True)
                y_b = y_b.to(dev, non_blocking=True)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                aux.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    old_logits = self.old_model(X_b, return_logits=True)
                hook = model.nn.logits.register_forward_pre_hook(lambda module, inp: feat_box.__setitem__('features', inp[0]))
                new_logits = model.nn(X_b, return_logits=True)
                hook.remove()
                feats = feat_box['features']
                boosted_old = old_logits + new_logits[:, :old_nc]
                boosted_logits = torch.cat([boosted_old, new_logits[:, old_nc:]], dim=1)
                loss_clf = F.cross_entropy(boosted_logits / w, y_cls, label_smoothing=0.05)
                loss_fe = F.cross_entropy(aux(feats), y_cls, label_smoothing=0.05)
                soft_prob = F.softmax(old_logits / T, dim=1)
                pred_log = F.log_softmax(boosted_logits[:, :old_nc] / T, dim=1)
                loss_kd = lambda_okd * (-(soft_prob * pred_log).sum(dim=1).mean())
                total = loss_clf + loss_fe + loss_kd
                total.backward()
                torch.nn.utils.clip_grad_norm_(list(model.nn.parameters()) + list(aux.parameters()), 0.5)
                optimizer.step()
                return loss_clf.item(), loss_kd.item(), total.item()

            return step_pt

        keras_model = model.model if hasattr(model, 'model') else model
        new_logits_model = model.get_logits_model()
        new_feature_model = model.get_feature_model()
        old_logits_fn = self.old_logits_model
        total_nc = len(self.class_order)
        T = self.temperature
        lambda_okd = self.lambda_okd

        self.compute_stage1_weights()
        per_cls_weights = self.per_cls_weights_stage1

        feat_dim = new_feature_model.output_shape[-1]
        aux_kernel = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(feat_dim, total_nc)), name='foster_aux_k')
        aux_bias = tf.Variable(tf.zeros(total_nc), name='foster_aux_b')

        @tf.function(reduce_retracing=True)
        def train_step_boosting(batch_X, batch_y):
            with tf.GradientTape() as tape:
                # Dual forward: frozen old backbone + trainable new backbone
                old_logits = tf.stop_gradient(old_logits_fn(batch_X, training=False))
                new_logits = new_logits_model(batch_X, training=True)
                new_features = new_feature_model(batch_X, training=True)

                boosted_old = old_logits + new_logits[:, :old_nc]
                boosted_logits = tf.concat([boosted_old, new_logits[:, old_nc:]], axis=1)

                loss_clf = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=batch_y,
                        logits=boosted_logits / per_cls_weights))

                fe_logits = tf.matmul(new_features, aux_kernel) + aux_bias
                loss_fe = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=batch_y, logits=fe_logits))

                loss_kd = lambda_okd * _kd_loss(boosted_logits[:, :old_nc], old_logits, T)

                total_loss = loss_clf + loss_fe + loss_kd

            all_vars = keras_model.trainable_variables + [aux_kernel, aux_bias]
            gradients = tape.gradient(total_loss, all_vars)
            optimizer.apply_gradients(zip(gradients, all_vars))
            return loss_clf, loss_kd, total_loss

        return train_step_boosting

    # ── Stage 2: Feature Compression ──

    def run_compression(self, global_model, config, task_data_paths=None):
        if not _use_tf():
            return self._run_compression_pt(global_model, config, task_data_paths)
        """Compress the boosted ensemble (old_model + global_model) into a single model.
        Called after all FL rounds for a task (task_id > 0)."""
        old_nc = self.old_num_classes
        total_nc = len(self.class_order)
        T = self.temperature

        self.compute_stage2_weights()
        per_cls_weights = self.per_cls_weights_stage2

        keras_boosted = global_model.model if hasattr(global_model, 'model') else global_model
        boosted_logits_model = global_model.get_logits_model()

        from models.dense import DenseModel
        student = _new_student_like(global_model, config, total_nc)
        if self.old_model is not None:
            _copy_backbone_and_old_head(student.model, self.old_model, old_nc)

        student_keras = student.model
        student_logits_model = student.get_logits_model()
        student_optimizer = student_keras.optimizer

        # Gather data: current-task data from all clients + exemplars (matches original FOSTER)
        all_X, all_y = [], []
        if task_data_paths:
            for paths in task_data_paths:
                X_d = np.load(paths['X'], mmap_mode='r')
                y_d = np.load(paths['y'], mmap_mode='r')
                labs = np.array(y_d if len(y_d.shape) == 1 else np.argmax(y_d, axis=1))
                all_X.append(np.array(X_d, dtype=np.float32))
                all_y.append(np.vectorize(self.label_map.get, otypes=[int])(labs).astype(np.int32))
                del X_d, y_d, labs
        for client_id in self.exemplars:
            for c, path in self.exemplars[client_id].items():
                X_ex = np.load(path, mmap_mode='r')
                if X_ex.shape[0] == 0:
                    del X_ex
                    continue
                all_X.append(np.array(X_ex, dtype=np.float32))
                mapped_c = self.label_map[c]
                all_y.append(np.full(X_ex.shape[0], mapped_c, dtype=np.int32))
                del X_ex
        if not all_X:
            global_model.set_weights(student.get_weights())
            global_model._logits_model = None
            global_model._feature_model = None
            return
        X_comp = np.concatenate(all_X, axis=0)
        y_comp = np.concatenate(all_y, axis=0)
        del all_X, all_y
        gc.collect()

        # Pre-compute teacher boosted logits
        teacher_logits_parts = []
        for i in range(0, X_comp.shape[0], PRED_BATCH):
            chunk = X_comp[i:i + PRED_BATCH]
            new_logits = boosted_logits_model(chunk, training=False).numpy()
            old_logits = self.old_logits_model(chunk, training=False).numpy()
            boosted_old = old_logits + new_logits[:, :old_nc]
            full = np.concatenate([boosted_old, new_logits[:, old_nc:]], axis=1)
            teacher_logits_parts.append(full.astype(np.float32))
            del chunk, new_logits, old_logits
        teacher_logits_all = np.concatenate(teacher_logits_parts, axis=0)
        del teacher_logits_parts
        gc.collect()

        per_cls_weights_tf = tf.constant(per_cls_weights, dtype=tf.float32)

        @tf.function(reduce_retracing=True)
        def compression_step(batch_X, teacher_logits):
            with tf.GradientTape() as tape:
                student_logits = student_logits_model(batch_X, training=True)
                loss = _bkd_loss(student_logits, teacher_logits, T, per_cls_weights_tf)
            gradients = tape.gradient(loss, student_keras.trainable_variables)
            student_optimizer.apply_gradients(zip(gradients, student_keras.trainable_variables))
            return loss

        batch_size = config.batch_size
        n = X_comp.shape[0]
        for epoch in range(self.compression_epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            steps = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                loss = compression_step(
                    tf.constant(X_comp[idx], dtype=tf.float32),
                    tf.constant(teacher_logits_all[idx], dtype=tf.float32))
                epoch_loss += loss.numpy()
                steps += 1

        del X_comp, y_comp, teacher_logits_all
        gc.collect()

        global_model.set_weights(student.get_weights())
        global_model._logits_model = None
        global_model._feature_model = None
        self._in_boosting = False
        del student
        gc.collect()

    def _run_compression_pt(self, global_model, config, task_data_paths=None):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        old_nc = self.old_num_classes
        total_nc = len(self.class_order)
        T = self.temperature
        self.compute_stage2_weights()
        student = _new_student_like(global_model, config, total_nc)
        if self.old_model is not None:
            _copy_pt_backbone_and_old_head(student.nn, self.old_model, old_nc)
        all_X = []
        if task_data_paths:
            for paths in task_data_paths:
                X_d = np.load(paths['X'], mmap_mode='r')
                all_X.append(np.array(X_d, dtype=np.float32))
                del X_d
        for client_id in self.exemplars:
            for _, path in self.exemplars[client_id].items():
                X_ex = np.load(path, mmap_mode='r')
                if X_ex.shape[0] > 0:
                    all_X.append(np.array(X_ex, dtype=np.float32))
                del X_ex
        if not all_X:
            global_model.set_weights(student.get_weights())
            global_model._logits_model = None
            global_model._feature_model = None
            return
        X_comp = np.concatenate(all_X, axis=0)
        del all_X
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        global_model.nn.to(dev).eval()
        self.old_model.to(dev).eval()
        teacher_parts = []
        with torch.no_grad():
            for i in range(0, X_comp.shape[0], PRED_BATCH):
                x = torch.from_numpy(X_comp[i:i + PRED_BATCH]).to(dev)
                new_logits = global_model.nn(x, return_logits=True)
                old_logits = self.old_model(x, return_logits=True)
                teacher_parts.append(torch.cat([old_logits + new_logits[:, :old_nc], new_logits[:, old_nc:]], dim=1).cpu())
        teacher = torch.cat(teacher_parts, dim=0)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_comp), teacher),
            batch_size=config.batch_size, shuffle=True, drop_last=False)
        student.nn.to(dev)
        opt = student.optimizer
        w = torch.from_numpy(np.array(self.per_cls_weights_stage2, dtype=np.float32)).to(dev)
        for _ in range(self.compression_epochs):
            student.nn.train()
            for X_b, t_b in loader:
                X_b = X_b.to(dev, non_blocking=True)
                t_b = t_b.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = student.nn(X_b, return_logits=True)
                pred_log = F.log_softmax(logits / T, dim=1)
                soft_prob = F.softmax(t_b / T, dim=1)
                soft_weighted = soft_prob * w
                soft_weighted = soft_weighted / soft_weighted.sum(dim=1, keepdim=True)
                loss = -(soft_weighted * pred_log).sum(dim=1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.nn.parameters(), 0.5)
                opt.step()
        global_model.set_weights(student.get_weights())
        global_model._logits_model = None
        global_model._feature_model = None
        self._in_boosting = False
        del X_comp, teacher, loader, student
        gc.collect()

    # ── Evaluation ──

    def evaluate(self, model, X, y, num_classes, logger=None):
        logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else None
        old_nc = self.old_num_classes
        use_boosted = self._in_boosting and old_nc > 0 and self.old_logits_model is not None
        y_labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        y_mapped = self.map_labels_np(y_labels)
        if not _use_tf():
            import torch
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev).eval()
            if self.old_model is not None:
                self.old_model.to(dev).eval()
            loss_sum = 0.0
            n = X.shape[0]
            all_preds = []
            with torch.no_grad():
                for i in range(0, n, PRED_BATCH):
                    x_chunk = torch.from_numpy(np.array(X[i:i + PRED_BATCH], dtype=np.float32)).to(dev)
                    y_idx = y_mapped[i:i + PRED_BATCH].astype(int)
                    new_logits = model.nn(x_chunk, return_logits=True)
                    if use_boosted:
                        old_logits = self.old_model(x_chunk, return_logits=True)
                        log_t = torch.cat([old_logits + new_logits[:, :old_nc], new_logits[:, old_nc:]], dim=1)
                    else:
                        log_t = new_logits
                    log_np = log_t.cpu().numpy()
                    all_preds.append(np.argmax(log_np, axis=1))
                    y_chunk = np.zeros((len(y_idx), num_classes), dtype=np.float32)
                    y_chunk[np.arange(len(y_idx)), y_idx] = 1.0
                    e = np.exp(log_np - log_np.max(axis=1, keepdims=True))
                    p = np.clip(e / e.sum(axis=1, keepdims=True), 1e-7, 1.0)
                    loss_sum += float(-np.sum(y_chunk * np.log(p)))
            preds = np.concatenate(all_preds)
            return {"acc": float(np.mean(preds == y_mapped)), "loss": loss_sum / n}
        loss_sum = 0.0
        n = X.shape[0]
        all_preds = []
        for i in range(0, n, PRED_BATCH):
            x_chunk = np.array(X[i:i + PRED_BATCH], dtype=np.float32)
            y_chunk = tf.keras.utils.to_categorical(y_mapped[i:i + PRED_BATCH], num_classes)
            new_logits = logits_model(x_chunk, training=False).numpy()
            if use_boosted:
                old_logits = self.old_logits_model(x_chunk, training=False).numpy()
                log_chunk = np.concatenate([
                    old_logits + new_logits[:, :old_nc],
                    new_logits[:, old_nc:]
                ], axis=1)
            else:
                log_chunk = new_logits
            all_preds.append(np.argmax(log_chunk, axis=1))
            e = np.exp(log_chunk - log_chunk.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            p = np.clip(p, 1e-7, 1.0)
            loss_sum += float(-np.sum(y_chunk * np.log(p)))
            del x_chunk, y_chunk, log_chunk, p
        loss = loss_sum / n
        preds = np.concatenate(all_preds)
        acc = float(np.mean(preds == y_mapped))
        return {"acc": acc, "loss": loss}

    # ── Cleanup ──

    def cleanup_temp(self):
        if self.exemplar_dir.exists():
            for f in self.exemplar_dir.glob("*.npy"):
                f.unlink(missing_ok=True)

    def extra_metrics(self) -> dict:
        return {
            "foster_memory": self.memory,
            "foster_beta1": self.beta1,
            "foster_beta2": self.beta2,
            "foster_lambda_okd": self.lambda_okd,
            "foster_T": self.temperature,
        }
