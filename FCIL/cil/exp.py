import numpy as np
from ..backend import use_tf as _use_tf
from .feat import FEAT, _build_etf, _projector
from .icarl import _batched_predict, PRED_BATCH

tf = __import__('tensorflow') if _use_tf() else None


class EXP(FEAT):
    def __init__(self, num_classes: int, memory: int = 2000,
                 lam: float = 1.0, rho: float = 0.9, temp: float = 0.5, **kwargs):
        super().__init__(num_classes=num_classes, memory=memory, lam=lam, rho=rho, temp=temp, **kwargs)
        self.name = 'EXP'
        self.prev_etf_weight = None
        self.shift_alpha = np.zeros((0,), dtype=np.float32)
        self._last_replay = 0.0
        self._last_synth = 0.0
        self._last_exemplar = 0.0

    def before_task(self, task_id: int, task_classes: list):
        if self.etf_weight is not None:
            self.prev_etf_weight = self.etf_weight.copy()
        super().before_task(task_id, task_classes)

    def prepare_model(self, model):
        num_classes = len(self.class_order)
        if num_classes == 0:
            return
        if _use_tf():
            logits_layer = model.model.get_layer('logits') if hasattr(model, 'model') else model.get_layer('logits')
            feat_dim = int(logits_layer.get_weights()[0].shape[0])
            logits_layer.trainable = False
        else:
            feat_dim = int(model.nn.logits.in_features)
            model.nn.logits.weight.requires_grad_(False)
            model.nn.logits.bias.requires_grad_(False)
        self.etf_weight = _build_etf(feat_dim, num_classes)
        self._update_shift_alpha()
        cols = self.etf_weight.T
        head = cols[:, self.head_classes] if self.head_classes else np.zeros((cols.shape[0], 0), dtype=np.float32)
        tail = cols[:, self.tail_classes] if self.tail_classes else np.zeros((cols.shape[0], 0), dtype=np.float32)
        self.project_head = _projector(head)
        self.project_tail = _projector(tail)

    def _update_shift_alpha(self):
        self.shift_alpha = np.zeros((len(self.class_order),), dtype=np.float32)
        if self.prev_etf_weight is None or self.old_num_classes == 0:
            return
        n = min(self.old_num_classes, self.prev_etf_weight.shape[0], self.etf_weight.shape[0])
        old = self.prev_etf_weight[:n]
        new = self.etf_weight[:n]
        old = old / (np.linalg.norm(old, axis=1, keepdims=True) + 1e-12)
        new = new / (np.linalg.norm(new, axis=1, keepdims=True) + 1e-12)
        cos = np.sum(old * new, axis=1)
        cos = np.minimum(1.0, np.maximum(-1.0, cos))
        self.shift_alpha[:n] = (np.arccos(cos) / np.pi).astype(np.float32)

    def _tf_synth_arrays(self):
        if self.old_num_classes == 0:
            return None, None, None
        proto_per_class = self.memory // self.old_num_classes
        if proto_per_class <= 0:
            return None, None, None
        labels = np.repeat(np.arange(self.old_num_classes, dtype=np.int64), proto_per_class)
        vecs = self._angular_cap_synth(labels)
        alpha = self.shift_alpha[labels].astype(np.float32)
        return vecs, labels, alpha

    def _angular_cap_synth(self, labels):
        etf = self.etf_weight.astype(np.float32)
        etf = etf / (np.linalg.norm(etf, axis=1, keepdims=True) + 1e-12)
        rng = np.random.default_rng(self.current_task + len(labels) + self.old_num_classes)
        out = []
        cos_all = np.clip(etf @ etf.T, -1.0, 1.0)
        np.fill_diagonal(cos_all, -1.0)
        nearest = np.arccos(np.max(cos_all[:self.old_num_classes], axis=1)) if self.old_num_classes > 1 else np.full((self.old_num_classes,), np.pi / 2, dtype=np.float32)
        for c in labels:
            p = etf[int(c)]
            u = rng.normal(size=p.shape).astype(np.float32)
            u = u - np.dot(u, p) * p
            u = u / (np.linalg.norm(u) + 1e-12)
            sigma = min(np.pi / 8, 0.45 * float(nearest[int(c)])) * (0.25 + 0.75 * float(self.shift_alpha[int(c)]))
            theta = min(abs(float(rng.normal(0.0, sigma))), 0.45 * float(nearest[int(c)]))
            out.append(np.cos(theta) * p + np.sin(theta) * u)
        return np.asarray(out, dtype=np.float32)

    def get_train_step(self, model, optimizer, loss_fn):
        self.prepare_model(model)
        if not _use_tf():
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            etf = torch.from_numpy(self.etf_weight).float().to(dev)
            alpha_np = self.shift_alpha.astype(np.float32)
            alpha_t = torch.from_numpy(alpha_np).float().to(dev)
            if self._pt_hook is not None:
                self._pt_hook.remove()
            feat_box = {}

            def _capture(module, inp):
                feat_box['value'] = inp[0]

            self._pt_hook = model.nn.logits.register_forward_pre_hook(_capture)
            proto_per_class = self.memory // self.old_num_classes if self.old_num_classes > 0 else 0
            synth_labels = []
            for cls_idx in range(self.old_num_classes):
                synth_labels.extend([cls_idx] * proto_per_class)
            synth_lab = torch.tensor(synth_labels, dtype=torch.long, device=dev) if synth_labels else None
            synth_vec = torch.from_numpy(self._angular_cap_synth(np.array(synth_labels, dtype=np.int64))).float().to(dev) if synth_labels else None

            def _synth_loss(feat_n, y_cls):
                if synth_lab is None:
                    return torch.tensor(0.0, device=dev)
                etf_n = F.normalize(etf, dim=1)
                vec_n = F.normalize(synth_vec, dim=1)
                replay_w = 0.5 + 0.5 * alpha_t[synth_lab]
                pos_logits = vec_n @ etf_n.T
                pos_loss = (F.cross_entropy(pos_logits, synth_lab, reduction='none') * replay_w).mean()
                new_mask = y_cls >= self.old_num_classes
                if not new_mask.any():
                    return pos_loss
                f = feat_n[new_mask]
                y = y_cls[new_mask]
                true_sim = (f * etf_n[y]).sum(dim=1, keepdim=True)
                old_sim = f @ vec_n.T
                weight = alpha_t[synth_lab].view(1, -1)
                margin = 1.0 / max(self.old_num_classes, 1)
                excl_loss = (F.softplus((old_sim - true_sim + margin) / self.temperature) * weight).mean()
                return pos_loss + excl_loss

            def step_pt(X_b, y_b, old_b=None):
                X_b = X_b.to(dev)
                y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad(set_to_none=True)
                model.nn(X_b, return_logits=True)
                feats = feat_box['value']
                feat_n = F.normalize(feats, dim=1)
                logits = feat_n @ F.normalize(etf, dim=1).T
                ce_loss = F.cross_entropy(logits, y_cls, label_smoothing=0.05)
                sim_f = feat_n @ feat_n.T
                proto = F.normalize(etf[y_cls], dim=1)
                sim_p = proto @ proto.T
                prob_f = torch.softmax(sim_f / self.temperature, dim=1)
                prob_p = torch.softmax(sim_p / self.temperature, dim=1)
                kl_rows = (prob_f * (torch.log(prob_f + 1e-12) - torch.log(prob_p + 1e-12))).sum(dim=1)
                cls_losses = [kl_rows[y_cls == cls].mean() for cls in torch.unique(y_cls)]
                gsa_loss = torch.stack(cls_losses).mean()
                exemplar_loss = torch.tensor(0.0, device=dev)
                if self.old_num_classes > 0:
                    old_mask = y_cls < self.old_num_classes
                    if old_mask.any():
                        ex = F.cross_entropy(logits[old_mask], y_cls[old_mask], reduction='none')
                        exemplar_loss = (ex * (1.0 - alpha_t[y_cls[old_mask]])).mean()
                synth_loss = _synth_loss(feat_n, y_cls)
                replay_loss = synth_loss + exemplar_loss
                total = ce_loss + self.lam * gsa_loss + replay_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                self._last_ce = float(ce_loss.item())
                self._last_gsa = float(gsa_loss.item())
                self._last_synth = float(synth_loss.item())
                self._last_exemplar = float(exemplar_loss.item())
                self._last_replay = float(replay_loss.item())
                return ce_loss.item(), gsa_loss.item(), total.item()

            return step_pt

        keras_model = model.model if hasattr(model, 'model') else model
        logits_layer = keras_model.get_layer('logits')
        feature_model = tf.keras.Model(keras_model.input, logits_layer.input)
        ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05)
        etf = tf.constant(self.etf_weight, dtype=tf.float32)
        alpha = tf.constant(self.shift_alpha, dtype=tf.float32)
        synth_vecs, synth_labels, synth_alpha = self._tf_synth_arrays()
        if synth_vecs is not None:
            synth_vecs = tf.constant(synth_vecs, dtype=tf.float32)
            synth_labels = tf.constant(synth_labels, dtype=tf.int32)
            synth_alpha = tf.constant(synth_alpha, dtype=tf.float32)

        def step_tf(batch_X, batch_y, batch_old=None):
            with tf.GradientTape() as tape:
                feats = feature_model(batch_X, training=True)
                feat_n = tf.math.l2_normalize(feats, axis=1)
                etf_n = tf.math.l2_normalize(etf, axis=1)
                logits = tf.matmul(feat_n, etf_n, transpose_b=True)
                ce_loss = ce_loss_fn(batch_y, logits)
                y_cls = tf.argmax(batch_y, axis=1, output_type=tf.int32)
                sim_f = tf.matmul(feat_n, feat_n, transpose_b=True)
                proto = tf.math.l2_normalize(tf.gather(etf, y_cls), axis=1)
                sim_p = tf.matmul(proto, proto, transpose_b=True)
                prob_f = tf.nn.softmax(sim_f / self.temperature, axis=1)
                prob_p = tf.nn.softmax(sim_p / self.temperature, axis=1)
                kl_rows = tf.reduce_sum(prob_f * (tf.math.log(prob_f + 1e-12) - tf.math.log(prob_p + 1e-12)), axis=1)
                _, class_idx = tf.unique(y_cls)
                num_cls = tf.reduce_max(class_idx) + 1
                gsa_loss = tf.reduce_mean(tf.math.unsorted_segment_mean(kl_rows, class_idx, num_cls))
                exemplar_loss = tf.constant(0.0, dtype=tf.float32)
                synth_loss = tf.constant(0.0, dtype=tf.float32)
                if self.old_num_classes > 0:
                    old_mask = y_cls < self.old_num_classes
                    if bool(tf.reduce_any(old_mask).numpy()):
                        ex_logits = tf.boolean_mask(logits, old_mask)
                        ex_y = tf.boolean_mask(y_cls, old_mask)
                        ex_loss = tf.keras.losses.sparse_categorical_crossentropy(ex_y, ex_logits, from_logits=True)
                        ex_alpha = tf.gather(alpha, ex_y)
                        exemplar_loss = tf.reduce_mean(ex_loss * (1.0 - ex_alpha))
                    if synth_vecs is not None:
                        synth_n = tf.math.l2_normalize(synth_vecs, axis=1)
                        pos_logits = tf.matmul(synth_n, etf_n, transpose_b=True)
                        replay_w = 0.5 + 0.5 * synth_alpha
                        pos_rows = tf.keras.losses.sparse_categorical_crossentropy(synth_labels, pos_logits, from_logits=True)
                        synth_loss = tf.reduce_mean(pos_rows * replay_w)
                        new_mask = y_cls >= self.old_num_classes
                        if bool(tf.reduce_any(new_mask).numpy()):
                            f_new = tf.boolean_mask(feat_n, new_mask)
                            y_new = tf.boolean_mask(y_cls, new_mask)
                            true_sim = tf.reduce_sum(f_new * tf.gather(etf_n, y_new), axis=1, keepdims=True)
                            old_sim = tf.matmul(f_new, synth_n, transpose_b=True)
                            margin = tf.constant(1.0 / max(self.old_num_classes, 1), dtype=tf.float32)
                            rows = tf.nn.softplus((old_sim - true_sim + margin) / self.temperature)
                            synth_loss = synth_loss + tf.reduce_mean(rows * tf.reshape(synth_alpha, (1, -1)))
                replay_loss = synth_loss + exemplar_loss
                total = ce_loss + self.lam * gsa_loss + replay_loss
            grads = tape.gradient(total, keras_model.trainable_variables)
            optimizer.apply_gradients((g, v) for g, v in zip(grads, keras_model.trainable_variables) if g is not None)
            self._last_ce = float(ce_loss.numpy())
            self._last_gsa = float(gsa_loss.numpy())
            self._last_synth = float(synth_loss.numpy())
            self._last_exemplar = float(exemplar_loss.numpy())
            self._last_replay = float(replay_loss.numpy())
            return ce_loss, gsa_loss, total

        return step_tf

    def _predict_logits(self, model, X, batch_size=PRED_BATCH):
        feats = _batched_predict(model.get_feature_model(), X, batch_size=batch_size)
        feats = self._correct_features(feats)
        etf = self.etf_weight.astype(np.float32)
        etf = etf / (np.linalg.norm(etf, axis=1, keepdims=True) + 1e-12)
        return feats @ etf.T

    def extra_metrics(self) -> dict:
        return {
            'exp_lambda': self.lam,
            'exp_rho': self.rho,
            'exp_temp': self.temperature,
            'exp_shift': float(np.mean(self.shift_alpha[:self.old_num_classes])) if self.old_num_classes > 0 else 0.0,
            'exp_replay': self._last_replay,
        }
