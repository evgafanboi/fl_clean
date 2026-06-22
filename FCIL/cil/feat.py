import numpy as np
from ..backend import use_tf as _use_tf
from .icarl import ICaRL, _batched_predict, PRED_BATCH

tf = __import__('tensorflow') if _use_tf() else None
GSA_BATCH = 1024


def _build_etf(feat_dim: int, num_classes: int) -> np.ndarray:
    if num_classes == 1:
        out = np.zeros((1, feat_dim), dtype=np.float32)
        out[0, 0] = 1.0
        return out
    basis = np.eye(feat_dim, num_classes, dtype=np.float32)
    simplex = np.eye(num_classes, dtype=np.float32) - np.ones((num_classes, num_classes), dtype=np.float32) / num_classes
    cols = basis @ (simplex * np.sqrt(num_classes / (num_classes - 1)))
    cols /= np.linalg.norm(cols, axis=0, keepdims=True) + 1e-12
    return cols.T.astype(np.float32)


def _projector(cols: np.ndarray) -> np.ndarray:
    if cols.size == 0:
        dim = cols.shape[0] if cols.ndim == 2 else 0
        return np.zeros((dim, dim), dtype=np.float32)
    gram = cols.T @ cols
    return (cols @ np.linalg.pinv(gram) @ cols.T).astype(np.float32)


class FEAT(ICaRL):
    def __init__(self, num_classes: int, memory: int = 2000,
                 lam: float = 1.0, rho: float = 0.9, temp: float = 0.5, **kwargs):
        super().__init__(num_classes=num_classes, memory=memory, use_bce=False, **kwargs)
        self.name = 'FEAT'
        self.lam = lam
        self.rho = rho
        self.temperature = temp
        self.head_classes = []
        self.tail_classes = []
        self.etf_weight = None
        self.project_head = None
        self.project_tail = None
        self.global_head_energy = 0.0
        self.global_tail_energy = 0.0
        self.client_head_energy = {}
        self.client_tail_energy = {}
        self.client_tail_count = {}
        self._updated_clients = set()
        self._last_ce = 0.0
        self._last_gsa = 0.0
        self._pt_hook = None

    def set_client(self, client_id: int):
        super().set_client(client_id)
        self.client_head_energy.setdefault(client_id, 0.0)
        self.client_tail_energy.setdefault(client_id, 0.0)
        self.client_tail_count.setdefault(client_id, 0)

    def get_ckpt_state(self) -> dict:
        state = super().get_ckpt_state()
        state.pop('_pt_hook', None)
        return state

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)
        self.tail_classes = list(range(self.old_num_classes))
        self.head_classes = list(range(self.old_num_classes, len(self.class_order)))

    def prepare_model(self, model):
        num_classes = len(self.class_order)
        if num_classes == 0:
            return
        if _use_tf():
            logits_layer = model.model.get_layer('logits') if hasattr(model, 'model') else model.get_layer('logits')
            feat_dim = int(logits_layer.get_weights()[0].shape[0])
            num_classes = int(logits_layer.get_weights()[0].shape[1])
            self.etf_weight = _build_etf(feat_dim, num_classes)
            kernel = self.etf_weight.T.astype(np.float32)
            bias = np.zeros((num_classes,), dtype=np.float32)
            logits_layer.set_weights([kernel, bias])
            logits_layer.trainable = False
        else:
            import torch
            logits_layer = model.nn.logits
            feat_dim = int(logits_layer.in_features)
            num_classes = int(logits_layer.out_features)
            self.etf_weight = _build_etf(feat_dim, num_classes)
            with torch.no_grad():
                logits_layer.weight.copy_(torch.from_numpy(self.etf_weight).to(logits_layer.weight.device, dtype=logits_layer.weight.dtype))
                logits_layer.bias.zero_()
            logits_layer.weight.requires_grad_(False)
            logits_layer.bias.requires_grad_(False)
        cols = self.etf_weight.T
        head_classes = [c for c in self.head_classes if c < num_classes]
        tail_classes = [c for c in self.tail_classes if c < num_classes]
        head = cols[:, head_classes] if head_classes else np.zeros((cols.shape[0], 0), dtype=np.float32)
        tail = cols[:, tail_classes] if tail_classes else np.zeros((cols.shape[0], 0), dtype=np.float32)
        self.project_head = _projector(head)
        self.project_tail = _projector(tail)

    def build_dataset(self, X_path, y_path, model, batch_size):
        if not _use_tf():
            import torch
            X_new = np.load(X_path, mmap_mode='r')
            y_new = np.load(y_path, mmap_mode='r')
            num_classes = len(self.class_order)
            label_map = dict(self.label_map)
            ex_info = []
            total = int(X_new.shape[0])
            for cls, path in self.exemplars.get(self.client_id, {}).items():
                X_ex = np.load(path, mmap_mode='r')
                n_ex = int(X_ex.shape[0])
                if n_ex > 0:
                    ex_info.append((cls, str(path), n_ex))
                    total += n_ex
                del X_ex
            if total == 0:
                return None, 0
            x_path_str = str(X_path)
            y_path_str = str(y_path)

            class _Stream:
                def __iter__(self_inner):
                    rng = np.random.default_rng()
                    chunks = [('new', x_path_str, y_path_str, i, min(i + batch_size, X_new.shape[0]), None)
                              for i in range(0, X_new.shape[0], batch_size)]
                    for cls, path, n_ex in ex_info:
                        chunks.extend(('ex', path, None, i, min(i + batch_size, n_ex), cls)
                                      for i in range(0, n_ex, batch_size))
                    rng.shuffle(chunks)
                    for kind, path, y_path, start, end, cls in chunks:
                        X_src = np.load(path, mmap_mode='r')
                        X_b = np.array(X_src[start:end], dtype=np.float32)
                        if kind == 'new':
                            y_src = np.load(y_path, mmap_mode='r')
                            lab = np.array(y_src[start:end])
                            lab = lab if len(lab.shape) == 1 else np.argmax(lab, axis=1)
                            mapped = np.vectorize(label_map.get, otypes=[int])(lab)
                            del y_src
                        else:
                            mapped = np.full((X_b.shape[0],), label_map[cls], dtype=int)
                        order = rng.permutation(X_b.shape[0])
                        X_b = X_b[order]
                        mapped = mapped[order]
                        y_b = np.zeros((len(mapped), num_classes), dtype=np.float32)
                        y_b[np.arange(len(mapped)), mapped] = 1.0
                        yield torch.from_numpy(X_b), torch.from_numpy(y_b)
                        del X_src

            return _Stream(), total

        X_new = np.load(X_path, mmap_mode='r')
        y_new = np.load(y_path, mmap_mode='r')
        num_classes = len(self.class_order)
        label_map = dict(self.label_map)
        ex_info = [(cls, str(path)) for cls, path in self.exemplars.get(self.client_id, {}).items()]
        total = int(X_new.shape[0])
        for _, path in ex_info:
            total += int(np.load(path, mmap_mode='r').shape[0])
        if total == 0:
            return None, 0
        input_dim = int(X_new.shape[1])
        x_path_str = X_path
        y_path_str = y_path

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
            for cls, path in ex_info:
                X_ex = np.load(path, mmap_mode='r')
                mapped_cls = label_map[cls]
                for i in range(0, X_ex.shape[0], batch_size):
                    X_b = np.array(X_ex[i:i + batch_size], dtype=np.float32)
                    lab = np.full((X_b.shape[0],), mapped_cls)
                    y_b = tf.keras.utils.to_categorical(lab, num_classes).astype(np.float32)
                    yield X_b, y_b

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
            ))
        dataset = dataset.prefetch(1)
        return dataset, total

    def _update_client_energies(self, model):
        if self.old_num_classes == 0 or not self.tail_classes:
            self.client_tail_count[self.client_id] = 0
            return
        tail_raw = self.class_order[:self.old_num_classes]
        parts = []
        for cls in tail_raw:
            path = self.exemplars.get(self.client_id, {}).get(cls)
            if path is not None:
                arr = np.load(path)
                if arr.shape[0] > 0:
                    parts.append(arr.astype(np.float32))
        if not parts:
            self.client_tail_count[self.client_id] = 0
            return
        X_tail = np.concatenate(parts, axis=0)
        feats = _batched_predict(model.get_feature_model(), X_tail, batch_size=PRED_BATCH)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        proj_h = feats @ self.project_head
        proj_t = feats @ self.project_tail
        rank_h = max(len(self.head_classes) - 1, 1)
        rank_t = max(len(self.tail_classes) - 1, 1)
        energy_h = float(np.mean(np.sum(proj_h * proj_h, axis=1) / rank_h))
        energy_t = float(np.mean(np.sum(proj_t * proj_t, axis=1) / rank_t))
        prev_h = self.client_head_energy.get(self.client_id, energy_h)
        prev_t = self.client_tail_energy.get(self.client_id, energy_t)
        self.client_head_energy[self.client_id] = (1.0 - self.rho) * prev_h + self.rho * energy_h
        self.client_tail_energy[self.client_id] = (1.0 - self.rho) * prev_t + self.rho * energy_t
        self.client_tail_count[self.client_id] = int(X_tail.shape[0])
        self._updated_clients.add(self.client_id)

    def update_exemplars(self, model, X_path, y_path, task_classes, reduce_old=False):
        super().update_exemplars(model, X_path, y_path, task_classes, reduce_old=reduce_old)
        self._update_client_energies(model)

    def rebuild_global_exemplars(self, model):
        total = 0
        sum_h = 0.0
        sum_t = 0.0
        for client_id in self._updated_clients:
            count = self.client_tail_count.get(client_id, 0)
            if count > 0:
                sum_h += self.client_head_energy[client_id] * count
                sum_t += self.client_tail_energy[client_id] * count
                total += count
        self.global_head_energy = sum_h / max(total, 1)
        self.global_tail_energy = sum_t / max(total, 1)
        self._updated_clients.clear()

    def get_train_step(self, model, optimizer, loss_fn):
        self.prepare_model(model)
        if not _use_tf():
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            etf = torch.from_numpy(self.etf_weight).to(dev)
            if self._pt_hook is not None:
                self._pt_hook.remove()
            feat_box = {}

            def _capture(module, inp):
                feat_box['value'] = inp[0]

            self._pt_hook = model.nn.logits.register_forward_pre_hook(_capture)

            def step_pt(X_b, y_b, old_b=None):
                X_b = X_b.to(dev)
                y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model.nn(X_b, return_logits=True)
                feats = feat_box['value']
                ce_loss = F.cross_entropy(logits, y_cls, label_smoothing=0.05)
                gsa_loss = torch.tensor(0.0, device=dev)
                if self.old_num_classes > 0:
                    if feats.shape[0] > GSA_BATCH:
                        idx = torch.randperm(feats.shape[0], device=dev)[:GSA_BATCH]
                        feats_gsa = feats[idx]
                        y_gsa = y_cls[idx]
                    else:
                        feats_gsa = feats
                        y_gsa = y_cls
                    feat_n = F.normalize(feats_gsa, dim=1)
                    sim_f = feat_n @ feat_n.T
                    proto = F.normalize(etf[y_gsa], dim=1)
                    sim_p = proto @ proto.T
                    prob_f = torch.softmax(sim_f / self.temperature, dim=1)
                    prob_p = torch.softmax(sim_p / self.temperature, dim=1)
                    kl_rows = (prob_f * (torch.log(prob_f + 1e-12) - torch.log(prob_p + 1e-12))).sum(dim=1)
                    cls_losses = [kl_rows[y_gsa == cls].mean() for cls in torch.unique(y_gsa)]
                    gsa_loss = torch.stack(cls_losses).mean()
                total = ce_loss + self.lam * gsa_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                self._last_ce = float(ce_loss.item())
                self._last_gsa = float(gsa_loss.item())
                feat_box.clear()
                return ce_loss.item(), gsa_loss.item(), total.item()

            return step_pt

        keras_model = model.model if hasattr(model, 'model') else model
        logits_layer = keras_model.get_layer('logits')
        feature_model = tf.keras.Model(keras_model.input, logits_layer.input)
        ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05)
        etf = tf.constant(self.etf_weight, dtype=tf.float32)

        def step_tf(batch_X, batch_y, batch_old=None):
            with tf.GradientTape() as tape:
                feats = feature_model(batch_X, training=True)
                logits = logits_layer(feats)
                ce_loss = ce_loss_fn(batch_y, logits)
                gsa_loss = tf.constant(0.0, dtype=tf.float32)
                if self.old_num_classes > 0:
                    y_cls = tf.argmax(batch_y, axis=1, output_type=tf.int32)
                    feat_n = tf.math.l2_normalize(feats, axis=1)
                    sim_f = tf.matmul(feat_n, feat_n, transpose_b=True)
                    proto = tf.math.l2_normalize(tf.gather(etf, y_cls), axis=1)
                    sim_p = tf.matmul(proto, proto, transpose_b=True)
                    prob_f = tf.nn.softmax(sim_f / self.temperature, axis=1)
                    prob_p = tf.nn.softmax(sim_p / self.temperature, axis=1)
                    kl_rows = tf.reduce_sum(prob_f * (tf.math.log(prob_f + 1e-12) - tf.math.log(prob_p + 1e-12)), axis=1)
                    _, class_idx = tf.unique(y_cls)
                    num_cls = tf.reduce_max(class_idx) + 1
                    gsa_loss = tf.reduce_mean(tf.math.unsorted_segment_mean(kl_rows, class_idx, num_cls))
                total = ce_loss + self.lam * gsa_loss
            grads = tape.gradient(total, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
            self._last_ce = float(ce_loss.numpy())
            self._last_gsa = float(gsa_loss.numpy())
            return ce_loss, gsa_loss, total

        return step_tf

    def _correct_features(self, feats: np.ndarray) -> np.ndarray:
        feats = feats.astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        if self.old_num_classes == 0 or not self.tail_classes:
            return feats
        proj_h = feats @ self.project_head
        proj_t = feats @ self.project_tail
        rank_h = max(len(self.head_classes) - 1, 1)
        rank_t = max(len(self.tail_classes) - 1, 1)
        energy_h = np.sum(proj_h * proj_h, axis=1, keepdims=True) / rank_h
        energy_t = np.sum(proj_t * proj_t, axis=1, keepdims=True) / rank_t
        gate = np.maximum((energy_h - self.global_head_energy) / (energy_h + energy_t + 1e-12), 0.0)
        feats = feats - gate * proj_h + gate * proj_t
        return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

    def _predict_logits(self, model, X, batch_size=PRED_BATCH):
        feats = _batched_predict(model.get_feature_model(), X, batch_size=batch_size)
        feats = self._correct_features(feats)
        return feats @ self.etf_weight.T

    def evaluate_full(self, model, X, y, num_classes, batch_size=PRED_BATCH):
        from sklearn.metrics import confusion_matrix
        y_true = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        y_true = self.map_labels_np(y_true).astype(int)
        feature_model = model.get_feature_model()
        etf_trunc = self.etf_weight[:num_classes].T
        all_preds = []
        loss_sum = 0.0
        n = X.shape[0]
        for i in range(0, n, batch_size):
            chunk = np.array(X[i:i+batch_size], dtype=np.float32)
            y_chunk = y_true[i:i+batch_size]
            feats = feature_model(chunk, training=False).numpy() if _use_tf() else feature_model.predict(chunk, batch_size=len(chunk))
            feats = self._correct_features(feats)
            logits = feats @ etf_trunc
            logits = logits.astype(np.float32)
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            all_preds.append(np.argmax(probs, axis=1))
            loss_sum += float(-np.sum(np.log(np.clip(probs[np.arange(len(y_chunk)), y_chunk], 1e-12, 1.0))))
            del chunk, feats, logits, probs
        y_pred = np.concatenate(all_preds)
        loss = loss_sum / n
        labels = list(range(num_classes))
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
        acc = float(np.sum(tp) / total_support) if total_support > 0 else 0.0
        prec_macro = float(np.mean(prec_per))
        rec_macro = float(np.mean(rec_per))
        f1_macro = float(np.mean(f1_per))
        prec_micro = float(np.sum(tp) / (np.sum(tp) + np.sum(fp))) if (np.sum(tp) + np.sum(fp)) > 0 else 0.0
        rec_micro = float(np.sum(tp) / (np.sum(tp) + np.sum(fn))) if (np.sum(tp) + np.sum(fn)) > 0 else 0.0
        f1_micro = float(2 * prec_micro * rec_micro / (prec_micro + rec_micro)) if (prec_micro + rec_micro) > 0 else 0.0
        return {
            'loss': loss,
            'acc': acc,
            'prec_macro': prec_macro,
            'rec_macro': rec_macro,
            'f1_macro': f1_macro,
            'prec_micro': prec_micro,
            'rec_micro': rec_micro,
            'f1_micro': f1_micro,
            'prec_weighted': float(np.sum(prec_per * support) / total_support) if total_support > 0 else 0.0,
            'rec_weighted': float(np.sum(rec_per * support) / total_support) if total_support > 0 else 0.0,
            'f1_weighted': float(np.sum(f1_per * support) / total_support) if total_support > 0 else 0.0,
            'cm': cm,
        }

    def extra_metrics(self) -> dict:
        return {
            'feat_lambda': self.lam,
            'feat_rho': self.rho,
            'feat_temp': self.temperature,
            'feat_eh': self.global_head_energy,
            'feat_et': self.global_tail_energy,
        }