"""Prototype Augmentation and Self-Supervision (PASS) for Exemplar-Free CIL.
Same prototype learning as CBKD, but uses L2 feature distillation instead of CFD.
L_total = L_CE + lambda * L_protoAug + gamma * L_kd
L_kd = ||F_new(x) - F_old(x)||^2  (feature extractor output)"""

import numpy as np
from ..backend import use_tf as _use_tf
from .base import CILMethod


class PASS(CILMethod):
    def __init__(self, num_classes: int, lam: float = 10.0,
                 gamma: float = 10.0, proto_size: int = 50, **kwargs):
        super().__init__(num_classes)
        self.name = "PASS"
        self.lam = lam
        self.gamma = gamma
        self.proto_size = proto_size
        self.class_order = []
        self.label_map = {}
        self.old_model = None
        self.old_feature_model = None
        self.old_num_classes = 0
        self.active_client = 0
        self.prototypes = {}
        self.variance = {}
        self._last_ce = 0.0
        self._last_kd = 0.0
        self._last_proto = 0.0
        self._compiled_step = None
        self._compiled_step_no_proto = None
        self._compiled_ssd_step = None
        self._compiled_ssd_step_no_proto = None

    def set_client(self, client_id: int):
        self.active_client = client_id
        self.prototypes.setdefault(client_id, {})
        self.variance.setdefault(client_id, 0.0)

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)
        for c in task_classes:
            if c not in self.label_map:
                self.label_map[c] = len(self.class_order)
                self.class_order.append(c)

    def set_old_model(self, model):
        if not _use_tf():
            import copy
            old = copy.deepcopy(model.nn)
            old.eval()
            for p in old.parameters():
                p.requires_grad_(False)
            self.old_model = old
            self.old_feature_model = old
            self.old_num_classes = model.num_classes
            self._compiled_step = None
            self._compiled_step_no_proto = None
            self._compiled_ssd_step = None
            self._compiled_ssd_step_no_proto = None
            return
        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        self.old_model = tf.keras.models.clone_model(keras_model)
        self.old_model.set_weights(keras_model.get_weights())
        self.old_model.trainable = False
        self.old_num_classes = int(self.old_model.get_layer('logits').units)
        feature_layer = self.old_model.get_layer('ln_3')
        self.old_feature_model = tf.keras.Model(self.old_model.input, feature_layer.output)
        self._compiled_step = None
        self._compiled_step_no_proto = None
        self._compiled_ssd_step = None
        self._compiled_ssd_step_no_proto = None

    def map_labels_np(self, y):
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        return np.vectorize(self.label_map.get, otypes=[int])(labels)

    def after_task(self, model, X_path: str, y_path: str):
        cid = self.active_client
        feature_model = model.get_feature_model() if hasattr(model, 'get_feature_model') else None
        if feature_model is None:
            return

        X = np.load(X_path, mmap_mode='r')
        y = np.load(y_path, mmap_mode='r')
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)

        unique_classes = np.unique(labels)
        D = feature_model.output_shape[-1]
        K_new = len(unique_classes)
        trace_sum = 0.0
        total_new_samples = 0

        batch_size = 4096
        for c in unique_classes:
            mask = labels == c
            X_c = np.array(X[mask], dtype=np.float32)
            n = X_c.shape[0]
            if _use_tf():
                feat_chunks = [
                    feature_model(X_c[i:i + batch_size], training=False).numpy()
                    for i in range(0, n, batch_size)
                ]
            else:
                feat_chunks = [
                    feature_model.predict(X_c[i:i + batch_size], batch_size=batch_size)
                    for i in range(0, n, batch_size)
                ]
            feats = np.concatenate(feat_chunks, axis=0)
            del feat_chunks, X_c
            mu = feats.mean(axis=0)
            self.prototypes[cid][int(c)] = mu
            cov = np.cov(feats, rowvar=False)
            trace_sum += np.trace(cov) if cov.ndim >= 2 else float(cov)
            total_new_samples += feats.shape[0]

        old_r2 = self.variance.get(cid, 0.0)
        old_count = sum(1 for cls in self.prototypes[cid] if cls not in set(np.unique(labels).tolist()))
        K_old_approx = max(old_count * 100, 0)
        new_r2 = trace_sum / (K_new * D)
        del X, y, labels
        if self.current_task == 0:
            self.variance[cid] = new_r2
        else:
            total = total_new_samples + K_old_approx
            self.variance[cid] = (K_old_approx * old_r2 + total_new_samples * new_r2) / max(total, 1)

    def _build_compiled_steps(self, keras_model, new_feature_model, optimizer, loss_fn):
        import tensorflow as tf
        old_feature_model = self.old_feature_model
        gamma = self.gamma
        lam = self.lam
        logits_layer = keras_model.get_layer('logits')
        predictions_layer = keras_model.get_layer('predictions')

        @tf.function
        def step_with_proto(batch_X, batch_y, proto_vecs, proto_labels_oh, r_std):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                old_feats = tf.stop_gradient(old_feature_model(batch_X, training=False))
                new_feats = new_feature_model(batch_X, training=True)
                kd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(new_feats - old_feats), axis=-1))
                total_loss = ce_loss + gamma * kd_loss
            grads = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
            noise = tf.random.normal(tf.shape(proto_vecs)) * r_std
            augmented = proto_vecs + noise
            classifier_vars = logits_layer.trainable_variables
            with tf.GradientTape() as tape2:
                proto_logits = logits_layer(augmented)
                proto_preds = predictions_layer(proto_logits)
                l_proto = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(proto_labels_oh, proto_preds))
                l_proto_scaled = lam * l_proto
            cls_grads = tape2.gradient(l_proto_scaled, classifier_vars)
            optimizer.apply_gradients(
                [(g, v) for g, v in zip(cls_grads, classifier_vars) if g is not None])
            return ce_loss, kd_loss, l_proto

        @tf.function
        def step_no_proto(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                old_feats = tf.stop_gradient(old_feature_model(batch_X, training=False))
                new_feats = new_feature_model(batch_X, training=True)
                kd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(new_feats - old_feats), axis=-1))
                total_loss = ce_loss + gamma * kd_loss
            grads = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
            return ce_loss, kd_loss, tf.constant(0.0)

        self._compiled_step = step_with_proto
        self._compiled_step_no_proto = step_no_proto

    def _build_compiled_ssd_steps(self, keras_model, new_feature_model, optimizer, loss_fn,
                                  global_logits_model, local_logits_model):
        import tensorflow as tf
        old_feature_model = self.old_feature_model
        gamma = self.gamma
        lam = self.lam
        logits_layer = keras_model.get_layer('logits')
        predictions_layer = keras_model.get_layer('predictions')

        @tf.function
        def ssd_with_proto(batch_X, batch_y, M_class, m_max, proto_vecs, proto_labels_oh, r_std):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                old_feats = tf.stop_gradient(old_feature_model(batch_X, training=False))
                new_feats = new_feature_model(batch_X, training=True)
                kd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(new_feats - old_feats), axis=-1))
                global_lg = global_logits_model(batch_X, training=False)
                local_lg = local_logits_model(batch_X, training=True)
                global_probs = tf.nn.softmax(global_lg)
                true_labels = tf.cast(tf.argmax(batch_y, axis=1), tf.int32)
                batch_size = tf.shape(batch_y)[0]
                indices = tf.stack([tf.range(batch_size), true_labels], axis=1)
                p_g_k2 = tf.gather_nd(global_probs, indices)
                M_sample = tf.expand_dims(
                    1.0 - tf.sqrt(tf.maximum(1.0 - p_g_k2, 0.0)), axis=1)
                M = m_max * tf.nn.relu(M_class * M_sample - 0.1)
                logit_diff = M * (global_lg - local_lg)
                ssd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logit_diff), axis=1))
                total_loss = ce_loss + gamma * kd_loss + ssd_loss
            grads = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
            noise = tf.random.normal(tf.shape(proto_vecs)) * r_std
            augmented = proto_vecs + noise
            classifier_vars = logits_layer.trainable_variables
            with tf.GradientTape() as tape2:
                proto_logits = logits_layer(augmented)
                proto_preds = predictions_layer(proto_logits)
                l_proto = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(proto_labels_oh, proto_preds))
                l_proto_scaled = lam * l_proto
            cls_grads = tape2.gradient(l_proto_scaled, classifier_vars)
            optimizer.apply_gradients(
                [(g, v) for g, v in zip(cls_grads, classifier_vars) if g is not None])
            return ce_loss, ssd_loss, total_loss

        @tf.function
        def ssd_no_proto(batch_X, batch_y, M_class, m_max):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                old_feats = tf.stop_gradient(old_feature_model(batch_X, training=False))
                new_feats = new_feature_model(batch_X, training=True)
                kd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(new_feats - old_feats), axis=-1))
                global_lg = global_logits_model(batch_X, training=False)
                local_lg = local_logits_model(batch_X, training=True)
                global_probs = tf.nn.softmax(global_lg)
                true_labels = tf.cast(tf.argmax(batch_y, axis=1), tf.int32)
                batch_size = tf.shape(batch_y)[0]
                indices = tf.stack([tf.range(batch_size), true_labels], axis=1)
                p_g_k2 = tf.gather_nd(global_probs, indices)
                M_sample = tf.expand_dims(
                    1.0 - tf.sqrt(tf.maximum(1.0 - p_g_k2, 0.0)), axis=1)
                M = m_max * tf.nn.relu(M_class * M_sample - 0.1)
                logit_diff = M * (global_lg - local_lg)
                ssd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logit_diff), axis=1))
                total_loss = ce_loss + gamma * kd_loss + ssd_loss
            grads = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
            return ce_loss, ssd_loss, total_loss

        self._compiled_ssd_step = ssd_with_proto
        self._compiled_ssd_step_no_proto = ssd_no_proto

    def get_train_step(self, model, optimizer, loss_fn):
        if self.old_feature_model is None:
            return None

        if not _use_tf():
            import torch
            import torch.nn.functional as F
            from models.pt_common import _PTFeatureWrapper
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            self.old_feature_model.to(dev)
            lam = self.lam
            gamma = self.gamma
            proto_size = self.proto_size
            old_model = self.old_feature_model
            cid = self.active_client
            protos = self.prototypes.get(cid, {})
            r2 = self.variance.get(cid, 1.0)
            proto_classes = sorted(protos.keys())
            if proto_classes:
                base_vecs = torch.from_numpy(
                    np.repeat(np.stack([protos[c] for c in proto_classes]), proto_size, axis=0)
                ).float().to(dev)
                base_labels = torch.tensor(
                    np.repeat([self.label_map[c] for c in proto_classes], proto_size),
                    dtype=torch.long, device=dev)
                r_std = float(np.sqrt(max(r2, 1e-8)))

            def step_pt(X_b, y_b):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                logits = model.nn(X_b, return_logits=True)
                n_out = logits.shape[1]
                valid = (y_cls >= 0) & (y_cls < n_out)
                ce_loss = F.cross_entropy(logits[valid], y_cls[valid], label_smoothing=0.05) if valid.any() else torch.tensor(0.0, device=dev)
                feat_wrapper = model.get_feature_model()
                new_feats = feat_wrapper._features if feat_wrapper._features is not None else \
                    torch.zeros(len(X_b), feat_wrapper._feat_dim, device=dev)
                model.nn(X_b, return_logits=True)
                new_feats = feat_wrapper._features
                with torch.no_grad():
                    old_wrapper = type('OldFeatureWrapperHost', (), {'nn': old_model, 'batch_size': model.batch_size})()
                    old_feat_wrapper = _PTFeatureWrapper(old_wrapper)
                    old_chunks = []
                    for i in range(0, len(X_b), 4096):
                        old_model(X_b[i:i+4096], return_logits=True)
                        old_chunks.append(old_feat_wrapper._features.detach())
                    old_feats = torch.cat(old_chunks)
                    old_feat_wrapper._hook.remove()
                    old_feat_wrapper._hook = None
                kd_loss = ((new_feats - old_feats) ** 2).sum(dim=-1).mean()
                total = ce_loss + gamma * kd_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                l_proto = torch.tensor(0.0)
                if proto_classes:
                    optimizer.zero_grad()
                    noise = torch.randn_like(base_vecs) * r_std
                    augmented = base_vecs + noise
                    l_proto = F.cross_entropy(model.nn.logits(augmented), base_labels)
                    (lam * l_proto).backward()
                    optimizer.step()
                self._last_ce = float(ce_loss)
                self._last_kd = float(kd_loss) * gamma
                self._last_proto = float(l_proto) * lam
                return ce_loss.item(), kd_loss.item(), total.item()

            return step_pt

        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        new_feature_model = model.get_feature_model() if hasattr(model, 'get_feature_model') else None
        lam = self.lam

        if self._compiled_step is None:
            self._build_compiled_steps(keras_model, new_feature_model, optimizer, loss_fn)

        cid = self.active_client
        protos = self.prototypes.get(cid, {})
        r2 = self.variance.get(cid, 1.0)
        proto_classes = sorted(protos.keys())

        gamma_val = self.gamma

        if proto_classes:
            base_vecs = np.stack([protos[c] for c in proto_classes]).astype(np.float32)
            base_labels = [self.label_map[c] for c in proto_classes]
            proto_vecs_tf = tf.constant(np.repeat(base_vecs, self.proto_size, axis=0))
            proto_labels_oh_tf = tf.one_hot(
                np.repeat(base_labels, self.proto_size), len(self.class_order))
            r_std_tf = tf.constant(float(np.sqrt(max(r2, 1e-8))))
            step_fn = self._compiled_step

            def wrapper(batch_X, batch_y):
                ce, kd, proto = step_fn(batch_X, batch_y,
                                        proto_vecs_tf, proto_labels_oh_tf, r_std_tf)
                self._last_ce = float(ce.numpy())
                self._last_kd = float(kd.numpy()) * gamma_val
                self._last_proto = float(proto.numpy()) * lam
                return ce, kd, ce + gamma_val * kd + proto * lam
        else:
            step_fn = self._compiled_step_no_proto

            def wrapper(batch_X, batch_y):
                ce, kd, _ = step_fn(batch_X, batch_y)
                self._last_ce = float(ce.numpy())
                self._last_kd = float(kd.numpy()) * gamma_val
                self._last_proto = 0.0
                return ce, kd, ce + gamma_val * kd

        return wrapper

    def get_ssd_train_step(self, model, optimizer, loss_fn,
                           global_logits_model, local_logits_model,
                           M_class_tf, m_max_tf):
        if self.old_feature_model is None:
            from .finetune import Finetune
            return Finetune(num_classes=self.num_classes).get_ssd_train_step(
                model, optimizer, loss_fn, global_logits_model, local_logits_model, M_class_tf, m_max_tf)

        if not _use_tf():
            import torch
            import torch.nn.functional as F
            from .finetune import _pt_ssd_loss
            from models.pt_common import _PTFeatureWrapper
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            self.old_feature_model.to(dev)
            lam = self.lam
            gamma = self.gamma
            proto_size = self.proto_size
            old_model = self.old_feature_model
            m_max = float(m_max_tf)
            cid = self.active_client
            protos = self.prototypes.get(cid, {})
            r2 = self.variance.get(cid, 1.0)
            proto_classes = sorted(protos.keys())
            if proto_classes:
                base_vecs = torch.from_numpy(
                    np.repeat(np.stack([protos[c] for c in proto_classes]), proto_size, axis=0)
                ).float().to(dev)
                base_labels = torch.tensor(
                    np.repeat([self.label_map[c] for c in proto_classes], proto_size),
                    dtype=torch.long, device=dev)
                r_std = float(np.sqrt(max(r2, 1e-8)))

            def step_pt(X_b, y_b):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                logits = model.nn(X_b, return_logits=True)
                n_out = logits.shape[1]
                valid = (y_cls >= 0) & (y_cls < n_out)
                ce_loss = F.cross_entropy(logits[valid], y_cls[valid], label_smoothing=0.05) if valid.any() else torch.tensor(0.0, device=dev)
                feat_wrapper = model.get_feature_model()
                model.nn(X_b, return_logits=True)
                new_feats = feat_wrapper._features
                with torch.no_grad():
                    old_wrapper = type('OldFeatureWrapperHost', (), {'nn': old_model, 'batch_size': model.batch_size})()
                    old_feat_wrapper = _PTFeatureWrapper(old_wrapper)
                    old_model(X_b, return_logits=True)
                    old_feats = old_feat_wrapper._features.detach()
                    old_feat_wrapper._hook.remove()
                    old_feat_wrapper._hook = None
                    global_log = torch.from_numpy(
                        global_logits_model.predict(X_b.cpu().numpy(), batch_size=len(X_b))).to(dev)
                kd_loss = ((new_feats - old_feats) ** 2).sum(dim=-1).mean()
                ssd_loss = _pt_ssd_loss(logits, global_log, y_b, M_class_tf, m_max)
                total = ce_loss + gamma * kd_loss + ssd_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                if proto_classes:
                    optimizer.zero_grad()
                    noise = torch.randn_like(base_vecs) * r_std
                    augmented = base_vecs + noise
                    l_proto = F.cross_entropy(model.nn.logits(augmented), base_labels)
                    (lam * l_proto).backward()
                    optimizer.step()
                self._last_ce = float(ce_loss)
                self._last_kd = float(kd_loss) * gamma
                self._last_proto = 0.0
                return ce_loss.item(), ssd_loss.item(), total.item()

            return step_pt

        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        new_feature_model = model.get_feature_model() if hasattr(model, 'get_feature_model') else None
        lam = self.lam

        if self._compiled_ssd_step is None:
            self._build_compiled_ssd_steps(keras_model, new_feature_model, optimizer, loss_fn,
                                           global_logits_model, local_logits_model)

        cid = self.active_client
        protos = self.prototypes.get(cid, {})
        r2 = self.variance.get(cid, 1.0)
        proto_classes = sorted(protos.keys())

        if proto_classes:
            base_vecs = np.stack([protos[c] for c in proto_classes]).astype(np.float32)
            base_labels = [self.label_map[c] for c in proto_classes]
            proto_vecs_tf = tf.constant(np.repeat(base_vecs, self.proto_size, axis=0))
            proto_labels_oh_tf = tf.one_hot(
                np.repeat(base_labels, self.proto_size), len(self.class_order))
            r_std_tf = tf.constant(float(np.sqrt(max(r2, 1e-8))))
            step_fn = self._compiled_ssd_step

            def wrapper(batch_X, batch_y):
                return step_fn(batch_X, batch_y, M_class_tf, m_max_tf,
                               proto_vecs_tf, proto_labels_oh_tf, r_std_tf)
        else:
            step_fn = self._compiled_ssd_step_no_proto

            def wrapper(batch_X, batch_y):
                return step_fn(batch_X, batch_y, M_class_tf, m_max_tf)

        return wrapper

    def extra_metrics(self) -> dict:
        return {"pass_lambda": self.lam, "pass_gamma": self.gamma,
                "proto_size": self.proto_size}
