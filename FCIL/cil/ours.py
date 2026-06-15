import gc
import numpy as np
from FL.strategy.robust_filter import RobustFilterV3
from ..backend import use_tf as _use_tf
from .proto_pass import PASS

LOGIT_CLIP = 30.0

class Ours(PASS):
    def __init__(self, num_classes: int, lam: float = 10.0, gamma: float = 10.0, proto_size: int = 50,
                 ekd_epochs: int = 1, ekd_lambda: float = 1.0,
                 proto_rel_lambda: float = 1.0, encoder_lr_factor: float = 0.1, drift_temp: float = 0.5,
                 robust_threshold: float = 0.3, robust_workers: int = 8, no_filter: bool = False, **kwargs):
        super().__init__(num_classes=num_classes, lam=lam, gamma=gamma, proto_size=proto_size, **kwargs)
        self.name = 'Ours'
        self.ekd_epochs = ekd_epochs
        self.ekd_lambda = ekd_lambda
        self.proto_rel_lambda = proto_rel_lambda
        self.encoder_lr_factor = encoder_lr_factor
        self.drift_temp = drift_temp
        self.robust_filter = RobustFilterV3(robust_threshold=robust_threshold, reference="Blom", workers=robust_workers)
        self.no_filter = no_filter
        self._last_ekd = 0.0
        self._last_survivors = 0
        self._last_survivor_clients = []
        self.global_prototypes = {}
        self.global_variance = 1.0
        self.global_counts = {}
        self.client_proto_counts = {}
        self.client_anchor_prototypes = {}
        self.client_new_prototypes = {}
        self.client_drifts = {}
        self._last_rel = 0.0
        self.current_task_id = 0

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)
        self.current_task_id = task_id
        self.current_task_classes = set(task_classes)

    def _scale_encoder_grads_pt(self, model):
        if self.encoder_lr_factor >= 1.0:
            return
        for name, p in model.nn.named_parameters():
            if p.grad is not None and not name.startswith('logits.'):
                p.grad.mul_(self.encoder_lr_factor)

    def _scale_encoder_grads_tf(self, grads, variables):
        out = []
        for g, v in zip(grads, variables):
            if g is not None and 'logits' not in v.name:
                g = g * self.encoder_lr_factor
            out.append(g)
        return out

    def _feature_prototypes(self, model, X_path, y_path, batch_size=4096):
        feature_model = model.get_feature_model() if hasattr(model, 'get_feature_model') else None
        X = np.load(X_path, mmap_mode='r')
        y = np.load(y_path, mmap_mode='r')
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        protos, counts = {}, {}
        for c in np.unique(labels):
            X_c = np.asarray(X[labels == c], dtype=np.float32)
            if _use_tf():
                chunks = [feature_model(X_c[i:i + batch_size], training=False).numpy() for i in range(0, len(X_c), batch_size)]
            else:
                chunks = [feature_model.predict(X_c[i:i + batch_size], batch_size=batch_size) for i in range(0, len(X_c), batch_size)]
            feats = np.concatenate(chunks, axis=0)
            protos[int(c)] = feats.mean(axis=0).astype(np.float32)
            counts[int(c)] = int(len(feats))
        return protos, counts

    def capture_anchor_prototypes(self, model, X_path, y_path):
        cid = self.active_client
        protos, counts = self._feature_prototypes(model, X_path, y_path)
        self.client_anchor_prototypes[cid] = protos
        self.client_proto_counts.setdefault(cid, {}).update(counts)

    def update_client_drift(self, model, X_path, y_path):
        cid = self.active_client
        new_protos, counts = self._feature_prototypes(model, X_path, y_path)
        anchors = self.client_anchor_prototypes.get(cid, {})
        old_protos = self.prototypes.get(cid, {})
        new_classes = [c for c in new_protos if c in anchors]
        new_drifts = {c: new_protos[c] - anchors[c] for c in new_classes}
        old_drifts = {}
        old_classes = [c for c in old_protos if c not in new_protos]
        if new_drifts and old_classes:
            drift_classes = sorted(new_drifts)
            drift_stack = np.stack([new_drifts[c] for c in drift_classes]).astype(np.float32)
            anchor_stack = np.stack([anchors[c] for c in drift_classes]).astype(np.float32)
            temp = max(float(self.drift_temp), 1e-6)
            for c in old_classes:
                sims = anchor_stack @ old_protos[c] / (np.linalg.norm(anchor_stack, axis=1) * np.linalg.norm(old_protos[c]) + 1e-8)
                weights = np.exp((sims - sims.max()) / temp)
                weights = weights / weights.sum()
                old_drifts[c] = (weights[:, None] * drift_stack).sum(axis=0).astype(np.float32)
        compensated = {c: (old_protos[c] + old_drifts.get(c, 0.0)).astype(np.float32) for c in old_classes}
        compensated.update({c: v.astype(np.float32) for c, v in new_protos.items()})
        self.prototypes[cid] = compensated
        self.client_new_prototypes[cid] = new_protos
        self.client_drifts[cid] = {**old_drifts, **new_drifts}
        self.client_proto_counts.setdefault(cid, {}).update(counts)
        return compensated

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
        if self.no_filter:
            return logits.mean(axis=1).astype(np.float32)
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

    def aggregate_prototypes(self, client_prototypes, client_counts=None):
        merged, count_sums = {}, {}
        client_counts = client_counts or [{} for _ in client_prototypes]
        for proto_dict, count_dict in zip(client_prototypes, client_counts):
            for cls, vec in proto_dict.items():
                weight = float(count_dict.get(cls, 1))
                merged.setdefault(cls, []).append((vec, weight))
                count_sums[cls] = count_sums.get(cls, 0.0) + weight
        out = {}
        variances = []
        for cls, rows in merged.items():
            stack = np.stack([v for v, _ in rows], axis=0).astype(np.float32)
            weights = np.asarray([w for _, w in rows], dtype=np.float32)
            out[cls] = ((stack * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-8)).astype(np.float32)
            variances.append(np.average(((stack - out[cls]) ** 2).mean(axis=1), weights=weights))
        for cls, vec in self.global_prototypes.items():
            if cls not in out:
                out[cls] = vec
                count_sums[cls] = self.global_counts.get(cls, 1.0)
        self.global_prototypes = out
        self.global_counts = {int(c): float(n) for c, n in count_sums.items()}
        if variances:
            self.global_variance = float(np.mean(variances))
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
                grads = self._scale_encoder_grads_tf(tape.gradient(loss, keras_model.trainable_variables), keras_model.trainable_variables)
                keras_model.optimizer.apply_gradients((g, v) for g, v in zip(grads, keras_model.trainable_variables) if g is not None)
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
                self._scale_encoder_grads_pt(model)
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                model.optimizer.step()
                total += float(loss.item()); batches += 1
            final = total / max(batches, 1)
        return final

    def extra_metrics(self):
        out = super().extra_metrics()
        out.update({'ours_ekd': self._last_ekd, 'ours_blom_survivors': self._last_survivors, 'ours_proto_rel': self._last_rel,
                    'ours_r_std': getattr(self, '_last_r_std', 0.0), 'ours_d_min': getattr(self, '_last_d_min', 0.0)})
        return out

    def get_finetune_step(self, model, optimizer, loss_fn):
        if not _use_tf():
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            if self.old_feature_model is not None:
                self.old_feature_model.to(dev)
            cid = self.active_client
            # protoAug uses ONLY old-class prototypes for stability;
            # plasticity comes from real CE on the current-task batch and from
            # drift-compensated prototypes already updated via update_client_drift.
            proto_classes = sorted(self.global_prototypes.keys()) if self.current_task_id > 0 else []
            old_proto_classes = [c for c in proto_classes if c not in self.current_task_classes]
            base_vecs = base_labels = old_base = old_rel = None
            r_std = 0.0
            if old_proto_classes:
                base_arr = np.stack([self.global_prototypes[c] for c in old_proto_classes]).astype(np.float32)
                base_vecs = torch.from_numpy(base_arr).to(dev)
                base_labels = torch.tensor([self.label_map[c] for c in old_proto_classes], dtype=torch.long, device=dev)
                old_base = base_vecs
                old_rel = F.normalize(old_base, dim=1) @ F.normalize(old_base, dim=1).t()
                # Architecture-aware adaptive noise bound:
                # 1) per-dim std under chi(D) is sqrt(D) * r_std => want sqrt(D) * r_std <= 0.5 * d_min
                #    so synthetic samples stay inside the prototype's Voronoi cell.
                # 2) cap from aggregated empirical variance to avoid blowing up
                #    when prototypes are degenerate / very close.
                D = int(base_arr.shape[1])
                if base_arr.shape[0] >= 2:
                    diff = base_arr[:, None, :] - base_arr[None, :, :]
                    d2 = (diff * diff).sum(axis=-1)
                    np.fill_diagonal(d2, np.inf)
                    d_min = float(np.sqrt(d2.min()))
                else:
                    d_min = float(np.linalg.norm(base_arr[0]) + 1e-6)
                geom_cap = 0.5 * d_min / np.sqrt(max(D, 1))
                emp_std = float(np.sqrt(max(self.global_variance, 1e-8)))
                r_std = float(min(emp_std, geom_cap))
                self._last_r_std = r_std
                self._last_d_min = d_min
            def step_pt(X_b, y_b):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                logits = model.nn(X_b, return_logits=True)
                ce_loss = F.cross_entropy(logits, y_cls, label_smoothing=0.05)
                kd_loss = torch.tensor(0.0, device=dev)
                if self.current_task_id > 0 and self.old_feature_model is not None:
                    feat_wrapper = model.get_feature_model()
                    model.nn(X_b, return_logits=True)
                    new_feats = feat_wrapper._features
                    with torch.no_grad():
                        _buf = []
                        _h = self.old_feature_model.logits.register_forward_pre_hook(lambda m, inp: _buf.append(inp[0].detach()))
                        self.old_feature_model(X_b, return_logits=True)
                        _h.remove()
                        old_feats = _buf[0]
                    kd_loss = F.mse_loss(new_feats, old_feats)
                proto_loss = torch.tensor(0.0, device=dev)
                rel_loss = torch.tensor(0.0, device=dev)
                if base_vecs is not None:
                    noise = torch.randn_like(base_vecs) * r_std
                    proto_loss = F.cross_entropy(model.nn.logits(base_vecs + noise), base_labels)
                if old_proto_classes and len(old_proto_classes) > 1:
                    class_w = model.nn.logits.weight[[self.label_map[c] for c in old_proto_classes], :]
                    current_rel = F.normalize(class_w, dim=1) @ F.normalize(class_w, dim=1).t()
                    rel_loss = ((current_rel - old_rel) ** 2).mean()
                total = ce_loss + self.gamma * kd_loss + self.lam * proto_loss + self.proto_rel_lambda * rel_loss
                total.backward()
                self._scale_encoder_grads_pt(model)
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                self._last_ce = float(ce_loss.item())
                self._last_kd = float(kd_loss.item()) * self.gamma
                self._last_proto = float(proto_loss.item()) * self.lam
                self._last_rel = float(rel_loss.item()) * self.proto_rel_lambda
                return ce_loss.item(), kd_loss.item(), total.item()
            return step_pt
        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        @tf.function(reduce_retracing=True)
        def step_tf(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
            grads = self._scale_encoder_grads_tf(tape.gradient(ce_loss, keras_model.trainable_variables), keras_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
            return ce_loss, tf.constant(0.0), ce_loss
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
            discard_mask = None
            if not self.no_filter:
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
        if self.no_filter:
            self._last_survivor_clients = list(range(len(client_weights)))
            self._last_survivors = len(client_weights)
            print(f"  Mean logits: robust filter disabled | 0/{len(client_weights)} clients failed")
        else:
            discard_frac = total_counts.astype(np.float64) / max(total_rows, 1)
            survivor = discard_frac <= self.robust_filter.robust_threshold
            self._last_survivor_clients = [i for i in range(len(client_weights)) if survivor[i]]
            self._last_survivors = int(survivor.sum())
            n_failed = int((~survivor).sum())
            failed_ratios = [(client_id, round(float(discard_frac[client_id]), 4)) for client_id in range(len(client_weights)) if not survivor[client_id]][:10]
            print(f"  V3 filter: threshold={self.robust_filter.robust_threshold:.2f} | {n_failed}/{len(client_weights)} clients failed | top discard ratios: {failed_ratios}")
        self._last_ekd = total_ekd / max(chunks * max(len(client_models), 1), 1)
        return self._last_ekd
