"""Correlation-Based Knowledge Distillation in Exemplar-Free CIL (CBKDEFCIL)"""

import numpy as np
import tensorflow as tf
from .base import CILMethod


class CBKD(CILMethod):
    def __init__(self, num_classes: int, lam: float = 10.0,
                 alpha: float = 10.0, beta: float = 10.0,
                 proto_size: int = 50, **kwargs):
        super().__init__(num_classes)
        self.name = "CBKD"
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.proto_size = proto_size
        self.class_order = []
        self.label_map = {}
        self.old_model = None
        self.old_logits_model = None
        self.old_feature_model = None
        self.old_num_classes = 0
        self.active_client = 0
        self.prototypes = {}   # client_id -> {class_label -> np.array}
        self.variance = {}     # client_id -> float (r^2)
        self._last_ce = 0.0
        self._last_cfd = 0.0
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
        keras_model = model.model if hasattr(model, 'model') else model
        self.old_model = tf.keras.models.clone_model(keras_model)
        self.old_model.set_weights(keras_model.get_weights())
        self.old_model.trainable = False
        logits_layer = self.old_model.get_layer('logits')
        self.old_logits_model = tf.keras.Model(self.old_model.input, logits_layer.output)
        self.old_num_classes = int(logits_layer.units)
        feature_layer = self.old_model.get_layer('ln_3')
        self.old_feature_model = tf.keras.Model(self.old_model.input, feature_layer.output)
        self._compiled_step = None
        self._compiled_step_no_proto = None
        self._compiled_ssd_step = None
        self._compiled_ssd_step_no_proto = None

    def map_labels_np(self, y):
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        return np.vectorize(self.label_map.get)(labels)

    def after_task(self, model, X_path: str, y_path: str):
        """Compute prototypes and update variance after each task per client."""
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
            X_c = X[mask]
            n = X_c.shape[0]
            feat_chunks = [
                feature_model(X_c[i:i + batch_size], training=False).numpy()
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

    def _pearson_distance(self, a, b):
        """Pearson distance between tensors along last axis."""
        a_mean = tf.reduce_mean(a, axis=-1, keepdims=True)
        b_mean = tf.reduce_mean(b, axis=-1, keepdims=True)
        a_centered = a - a_mean
        b_centered = b - b_mean
        num = tf.reduce_sum(a_centered * b_centered, axis=-1)
        denom = tf.sqrt(tf.reduce_sum(tf.square(a_centered), axis=-1) *
                        tf.reduce_sum(tf.square(b_centered), axis=-1) + 1e-8)
        return 1.0 - num / denom

    def _build_compiled_steps(self, keras_model, new_logits_model, optimizer, loss_fn):
        old_logits_model = self.old_logits_model
        old_nc = self.old_num_classes
        alpha_cfd = self.alpha
        beta_cfd = self.beta
        lam = self.lam
        logits_layer = keras_model.get_layer('logits')
        predictions_layer = keras_model.get_layer('predictions')
        pearson = self._pearson_distance

        @tf.function
        def step_with_proto(batch_X, batch_y, proto_vecs, proto_labels_oh, r_std):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                old_logits = tf.stop_gradient(old_logits_model(batch_X, training=False))
                new_logits = new_logits_model(batch_X, training=True)[:, :old_nc]
                l_inter = tf.reduce_mean(pearson(old_logits, new_logits))
                l_intra = tf.reduce_mean(pearson(tf.transpose(old_logits), tf.transpose(new_logits)))
                l_cfd = alpha_cfd * l_inter + beta_cfd * l_intra
                total_loss = ce_loss + l_cfd
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
            return ce_loss, l_cfd, l_proto

        @tf.function
        def step_no_proto(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                old_logits = tf.stop_gradient(old_logits_model(batch_X, training=False))
                new_logits = new_logits_model(batch_X, training=True)[:, :old_nc]
                l_inter = tf.reduce_mean(pearson(old_logits, new_logits))
                l_intra = tf.reduce_mean(pearson(tf.transpose(old_logits), tf.transpose(new_logits)))
                l_cfd = alpha_cfd * l_inter + beta_cfd * l_intra
                total_loss = ce_loss + l_cfd
            grads = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
            return ce_loss, l_cfd, tf.constant(0.0)

        self._compiled_step = step_with_proto
        self._compiled_step_no_proto = step_no_proto

    def _build_compiled_ssd_steps(self, keras_model, new_logits_model, optimizer, loss_fn,
                                  global_logits_model, local_logits_model):
        old_logits_model = self.old_logits_model
        old_nc = self.old_num_classes
        alpha_cfd = self.alpha
        beta_cfd = self.beta
        lam = self.lam
        logits_layer = keras_model.get_layer('logits')
        predictions_layer = keras_model.get_layer('predictions')
        pearson = self._pearson_distance

        @tf.function
        def ssd_with_proto(batch_X, batch_y, M_class, m_max, proto_vecs, proto_labels_oh, r_std):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                old_logits = tf.stop_gradient(old_logits_model(batch_X, training=False))
                new_logits = new_logits_model(batch_X, training=True)[:, :old_nc]
                l_inter = tf.reduce_mean(pearson(old_logits, new_logits))
                l_intra = tf.reduce_mean(pearson(tf.transpose(old_logits), tf.transpose(new_logits)))
                l_cfd = alpha_cfd * l_inter + beta_cfd * l_intra
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
                total_loss = ce_loss + l_cfd + ssd_loss
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
                old_logits = tf.stop_gradient(old_logits_model(batch_X, training=False))
                new_logits = new_logits_model(batch_X, training=True)[:, :old_nc]
                l_inter = tf.reduce_mean(pearson(old_logits, new_logits))
                l_intra = tf.reduce_mean(pearson(tf.transpose(old_logits), tf.transpose(new_logits)))
                l_cfd = alpha_cfd * l_inter + beta_cfd * l_intra
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
                total_loss = ce_loss + l_cfd + ssd_loss
            grads = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
            return ce_loss, ssd_loss, total_loss

        self._compiled_ssd_step = ssd_with_proto
        self._compiled_ssd_step_no_proto = ssd_no_proto

    def get_train_step(self, model, optimizer, loss_fn):
        if self.old_logits_model is None:
            return None

        keras_model = model.model if hasattr(model, 'model') else model
        new_logits_model = model.get_logits_model() if hasattr(model, 'get_logits_model') else None
        lam = self.lam

        if self._compiled_step is None:
            self._build_compiled_steps(keras_model, new_logits_model, optimizer, loss_fn)

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
            step_fn = self._compiled_step

            def wrapper(batch_X, batch_y):
                ce, cfd, proto = step_fn(batch_X, batch_y,
                                         proto_vecs_tf, proto_labels_oh_tf, r_std_tf)
                self._last_ce = float(ce.numpy())
                self._last_cfd = float(cfd.numpy())
                self._last_proto = float(proto.numpy()) * lam
                return ce, cfd, ce + cfd + proto * lam
        else:
            step_fn = self._compiled_step_no_proto

            def wrapper(batch_X, batch_y):
                ce, cfd, _ = step_fn(batch_X, batch_y)
                self._last_ce = float(ce.numpy())
                self._last_cfd = float(cfd.numpy())
                self._last_proto = 0.0
                return ce, cfd, ce + cfd

        return wrapper

    def get_ssd_train_step(self, model, optimizer, loss_fn,
                           global_logits_model, local_logits_model,
                           M_class_tf, m_max_tf):
        if self.old_logits_model is None:
            from .finetune import Finetune
            ft = Finetune(num_classes=self.num_classes)
            return ft.get_ssd_train_step(model, optimizer, loss_fn,
                                         global_logits_model, local_logits_model,
                                         M_class_tf, m_max_tf)

        keras_model = model.model if hasattr(model, 'model') else model
        new_logits_model = model.get_logits_model() if hasattr(model, 'get_logits_model') else None
        lam = self.lam

        if self._compiled_ssd_step is None:
            self._build_compiled_ssd_steps(keras_model, new_logits_model, optimizer, loss_fn,
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
        return {"cbkd_lambda": self.lam, "cbkd_alpha": self.alpha,
                "cbkd_beta": self.beta, "proto_size": self.proto_size}
