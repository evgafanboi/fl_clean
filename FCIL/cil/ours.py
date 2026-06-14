import gc
import numpy as np
from FL.strategy.robust_filter import RobustFilterV3
from ..backend import use_tf as _use_tf
from .proto_pass import PASS

LOGIT_CLIP = 30.0

class Ours(PASS):
    def __init__(self, num_classes: int, lam: float = 10.0, gamma: float = 10.0, proto_size: int = 50,
                 ekd_epochs: int = 1, ekd_lambda: float = 1.0,
                 robust_threshold: float = 0.3, robust_workers: int = 8, **kwargs):
        super().__init__(num_classes=num_classes, lam=lam, gamma=gamma, proto_size=proto_size, **kwargs)
        self.name = 'Ours'
        self.ekd_epochs = ekd_epochs
        self.ekd_lambda = ekd_lambda
        self.robust_filter = RobustFilterV3(robust_threshold=robust_threshold, reference="Blom", workers=robust_workers)
        self._last_ekd = 0.0
        self._last_survivors = 0
        self._last_survivor_clients = []
        self.global_prototypes = {}
        self.global_variance = 1.0

    def _public_X_generator(self, config, task_id: int, active_n: int, num_tasks: int):
        from ..pipeline import cil_partition_dir, _task_active_n
        base = cil_partition_dir(config.partition_root, config.partition_type, config.n_clients)
        known = set(self.class_order)
        
        for t in range(task_id + 1):
            for cid in range(_task_active_n(config.n_clients, num_tasks, t)):
                x_path = base / f"client_{cid}_task_{t}_X_public.npy"
                y_path = base / f"client_{cid}_task_{t}_y_public.npy"
                if x_path.exists() and y_path.exists():
                    X = np.load(str(x_path), mmap_mode='r')
                    y = np.load(str(y_path), mmap_mode='r')
                    mask = np.isin(y, list(known))
                    if mask.any():
                        yield np.asarray(X[mask], dtype=np.float32)

    def _raw_logits(self, model, X, batch_size):
        predictor = model.get_logits_model() if hasattr(model, 'get_logits_model') else model
        return predictor.predict(X, batch_size=batch_size)

    def _blom_mean(self, logits, discard_mask=None):
        if discard_mask is None:
            discard_mask, _ = self.robust_filter.count_discards_mask(logits)
        keep = ~discard_mask
        support = keep.sum(axis=1)
        bad = support == 0
        if bad.any():
            keep[bad] = True
            support[bad] = keep.shape[1]
        w = keep.astype(np.float32)
        return (np.einsum('nk,nkd->nd', w, logits) / support[:, None]).astype(np.float32)

    def aggregate_prototypes(self, client_prototypes):
        merged = {}
        for proto_dict in client_prototypes:
            for cls, vec in proto_dict.items():
                merged.setdefault(cls, []).append(vec)
        out = {}
        for cls, vecs in merged.items():
            stack = np.stack(vecs, axis=0).astype(np.float32)
            out[cls] = stack.mean(axis=0)
        
        # Preserve older classes not present in current survivors to prevent accidental dropping
        for cls, vec in self.global_prototypes.items():
            if cls not in out:
                out[cls] = vec
                
        self.global_prototypes = out
        
        if merged:
            self.global_variance = float(np.mean([np.var(np.stack(v, axis=0)) for v in merged.values()]))
        elif not self.global_prototypes:
            self.global_variance = 1.0
            
        return out

    def after_aggregation(self, global_model, client_model, client_weights, config, task_id, round_num, active_n, num_tasks, client_prototypes=None):
        if not client_weights:
            return None
        if client_prototypes:
            self.aggregate_prototypes(client_prototypes)
        return self._streaming_distillation(global_model, client_model, client_weights, config, task_id, active_n, num_tasks)

    def _ekd(self, model, X, teacher_logits, batch_size):
        if not _use_tf():
            return self._ekd_pt(model, X, teacher_logits, batch_size)
        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        logits_model = model.get_logits_model()
        n = len(X)
        final = 0.0
        for _ in range(self.ekd_epochs):
            order = np.random.permutation(n)
            total, batches = 0.0, 0
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                with tf.GradientTape() as tape:
                    student = tf.clip_by_value(logits_model(X[idx], training=True), -LOGIT_CLIP, LOGIT_CLIP)
                    teacher = tf.clip_by_value(teacher_logits[idx], -LOGIT_CLIP, LOGIT_CLIP)
                    alpha_t = tf.exp(teacher) + 1.0
                    alpha_s = tf.exp(student) + 1.0
                    a0_t = tf.reduce_sum(alpha_t, axis=-1, keepdims=True)
                    a0_s = tf.reduce_sum(alpha_s, axis=-1, keepdims=True)
                    l1 = tf.reduce_sum((alpha_t / a0_t) * (tf.math.log(alpha_t / a0_t + 1e-8) - tf.math.log(alpha_s / a0_s + 1e-8)), axis=-1)
                    l2 = (tf.math.lgamma(a0_t[:, 0]) - tf.math.lgamma(a0_s[:, 0])
                          - tf.reduce_sum(tf.math.lgamma(alpha_t) - tf.math.lgamma(alpha_s), axis=-1)
                          + tf.reduce_sum((alpha_t - alpha_s) * (tf.math.digamma(alpha_t) - tf.math.digamma(a0_t)), axis=-1))
                    loss = tf.reduce_mean(l1 + self.ekd_lambda * l2)
                keras_model.optimizer.apply_gradients((g, v) for g, v in zip(tape.gradient(loss, keras_model.trainable_variables), keras_model.trainable_variables) if g is not None)
                total += float(loss.numpy()); batches += 1
            final = total / max(batches, 1)
        return final

    def _ekd_pt(self, model, X, teacher_logits, batch_size):
        import torch
        import torch.nn.functional as F
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.nn.to(dev).train()
        X_t = torch.from_numpy(X.astype(np.float32)).to(dev)
        z_t = torch.from_numpy(teacher_logits.astype(np.float32)).to(dev)
        n = len(X)
        final = 0.0
        for _ in range(self.ekd_epochs):
            order = np.random.permutation(n)
            total, batches = 0.0, 0
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                student = torch.clamp(model.nn(X_t[idx], return_logits=True), -LOGIT_CLIP, LOGIT_CLIP)
                teacher = torch.clamp(z_t[idx], -LOGIT_CLIP, LOGIT_CLIP)
                alpha_t = torch.exp(teacher) + 1.0
                alpha_s = torch.exp(student) + 1.0
                a0_t = alpha_t.sum(-1, keepdim=True)
                a0_s = alpha_s.sum(-1, keepdim=True)
                l1 = ((alpha_t / a0_t) * (torch.log(alpha_t / a0_t + 1e-8) - torch.log(alpha_s / a0_s + 1e-8))).sum(-1)
                l2 = (torch.lgamma(a0_t[:, 0]) - torch.lgamma(a0_s[:, 0])
                      - (torch.lgamma(alpha_t) - torch.lgamma(alpha_s)).sum(-1)
                      + ((alpha_t - alpha_s) * (torch.digamma(alpha_t) - torch.digamma(a0_t))).sum(-1))
                loss = (l1 + self.ekd_lambda * l2).mean()
                model.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                model.optimizer.step()
                total += float(loss.item()); batches += 1
            final = total / max(batches, 1)
        return final

    def extra_metrics(self):
        out = super().extra_metrics()
        out.update({'ours_ekd': self._last_ekd, 'ours_blom_survivors': self._last_survivors})
        return out

    def get_finetune_step(self, model, optimizer, loss_fn):
        if not _use_tf():
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            def step_pt(X_b, y_b):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                logits = model.nn(X_b, return_logits=True)
                ce_loss = F.cross_entropy(logits, y_cls, label_smoothing=0.05)
                ce_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                return ce_loss.item(), 0.0, ce_loss.item()
            return step_pt
        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        @tf.function(reduce_retracing=True)
        def step_tf(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
            grads = tape.gradient(ce_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
            return ce_loss, tf.constant(0.0), ce_loss
        return step_tf

    def get_global_proto_step(self, model, optimizer, loss_fn):
        if not _use_tf():
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            proto_classes = sorted(self.global_prototypes.keys())
            if not proto_classes:
                return None
            base_vecs = torch.from_numpy(np.stack([self.global_prototypes[c] for c in proto_classes])).float().to(dev)
            base_labels = torch.tensor([self.label_map[c] for c in proto_classes], dtype=torch.long, device=dev)
            r_std = float(np.sqrt(max(self.global_variance, 1e-8)))
            def step_pt(X_b, y_b):
                optimizer.zero_grad()
                noise = torch.randn_like(base_vecs) * r_std
                augmented = base_vecs + noise
                l_proto = F.cross_entropy(model.nn.logits(augmented), base_labels)
                (self.lam * l_proto).backward()
                optimizer.step()
                return 0.0, 0.0, l_proto.item()
            return step_pt
        import tensorflow as tf
        proto_classes = sorted(self.global_prototypes.keys())
        if not proto_classes:
            return None
        base_vecs = np.stack([self.global_prototypes[c] for c in proto_classes]).astype(np.float32)
        base_labels = [self.label_map[c] for c in proto_classes]
        proto_vecs_tf = tf.constant(np.repeat(base_vecs, self.proto_size, axis=0))
        proto_labels_oh_tf = tf.one_hot(np.repeat(base_labels, self.proto_size), len(self.class_order))
        r_std_tf = tf.constant(float(np.sqrt(max(self.global_variance, 1e-8))))
        keras_model = model.model if hasattr(model, 'model') else model
        logits_layer = keras_model.get_layer('logits')
        predictions_layer = keras_model.get_layer('predictions')
        @tf.function(reduce_retracing=True)
        def step_tf(batch_X, batch_y):
            classifier_vars = logits_layer.trainable_variables
            with tf.GradientTape() as tape:
                noise = tf.random.normal(tf.shape(proto_vecs_tf)) * r_std_tf
                augmented = proto_vecs_tf + noise
                proto_logits = logits_layer(augmented)
                proto_preds = predictions_layer(proto_logits)
                l_proto = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(proto_labels_oh_tf, proto_preds))
                l_proto_scaled = self.lam * l_proto
            cls_grads = tape.gradient(l_proto_scaled, classifier_vars)
            optimizer.apply_gradients([(g, v) for g, v in zip(cls_grads, classifier_vars) if g is not None])
            return tf.constant(0.0), tf.constant(0.0), l_proto_scaled
        return step_tf

    def _streaming_distillation(self, global_model, client_model, client_weights, config, task_id, active_n, num_tasks):
        total_ekd = 0.0
        chunks = 0
        old_global_w = global_model.get_weights()
        for X_chunk in self._public_X_generator(config, task_id, active_n, num_tasks):
            preds = []
            for weights in client_weights:
                global_model.set_weights(weights)
                pred = self._raw_logits(global_model, X_chunk, config.batch_size)
                preds.append(np.clip(pred, -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32))
            global_model.set_weights(old_global_w)
            consensus = self._blom_mean(np.stack(preds, axis=1))
            chunk_ekd = self._ekd(global_model, X_chunk, consensus, config.batch_size)
            total_ekd += chunk_ekd
            chunks += 1
            del X_chunk, preds, consensus
            gc.collect()
        self._last_ekd = total_ekd / max(chunks, 1)
        return self._last_ekd

    def distill_round(self, scratch_model, client_models, client_weights, config, task_id, active_n, num_tasks):
        total_ekd = 0.0
        chunks = 0
        total_counts = np.zeros(len(client_weights), dtype=np.int64)
        total_rows = 0
        for X_chunk in self._public_X_generator(config, task_id, active_n, num_tasks):
            preds = []
            for weights in client_weights:
                scratch_model.set_weights(weights)
                pred = self._raw_logits(scratch_model, X_chunk, config.batch_size)
                preds.append(np.clip(pred, -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32))
            logits = np.stack(preds, axis=1)
            discard_mask, _ = self.robust_filter.count_discards_mask(logits)
            total_counts += discard_mask.sum(axis=0)
            total_rows += len(X_chunk)
            consensus = self._blom_mean(logits, discard_mask)
            del preds, logits
            for student in client_models.values():
                total_ekd += self._ekd(student, X_chunk, consensus, config.batch_size)
            chunks += 1
            del X_chunk, consensus
            gc.collect()
        discard_frac = total_counts.astype(np.float64) / max(total_rows, 1)
        survivor = discard_frac <= self.robust_filter.robust_threshold
        self._last_survivor_clients = [i for i in range(len(client_weights)) if survivor[i]]
        self._last_survivors = int(survivor.sum())
        n_failed = int((~survivor).sum())
        failed_ratios = [(client_id, round(float(discard_frac[client_id]), 4)) for client_id in range(len(client_weights)) if not survivor[client_id]][:10]
        print(f"  V3 filter: threshold={self.robust_filter.robust_threshold:.2f} | {n_failed}/{len(client_weights)} clients failed | top discard ratios: {failed_ratios}")
        self._last_ekd = total_ekd / max(chunks * max(len(client_models), 1), 1)
        return self._last_ekd
