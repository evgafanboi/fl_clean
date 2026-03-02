"""Global-Local Forgetting Compensation (GLFC) - CVPR 2022
Monolithic FCIL method: includes its own FedAvg aggregation + proxy server."""

import gc
import copy
import numpy as np
import tensorflow as tf
from pathlib import Path
from .base import CILMethod

PRED_BATCH = 4096


def _batched_predict(model, X, batch_size=PRED_BATCH):
    parts = []
    for i in range(0, X.shape[0], batch_size):
        parts.append(model(X[i:i+batch_size], training=False).numpy())
    return np.concatenate(parts, axis=0)


class GradientEncoder:
    """Small MLP encoder for gradient sharing (replaces LeNet for tabular).
    input_dim -> 64 (sigmoid) -> 64 (sigmoid) -> num_classes.
    Uniform init [-0.5, 0.5] matching original."""

    def __init__(self, input_dim, num_classes):
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(64, activation='sigmoid', name='enc_1')(inputs)
        x = tf.keras.layers.Dense(64, activation='sigmoid', name='enc_2')(x)
        outputs = tf.keras.layers.Dense(num_classes, name='enc_out')(x)
        self.model = tf.keras.Model(inputs, outputs)
        for layer in self.model.layers:
            w = layer.get_weights()
            if w:
                layer.set_weights([np.random.uniform(-0.5, 0.5, s.shape).astype(np.float32) for s in w])


class ProxyServer:
    """Relay: reconstructs proto samples from gradient sets via DLG,
    evaluates global model on reconstructed data, tracks best model."""

    def __init__(self, input_dim, num_classes, encoder):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.encoder = encoder
        self.reconstruction_iters = 100
        self.num_augmentations = 10
        self.proto_data = np.empty((0, input_dim), dtype=np.float32)
        self.proto_labels = np.empty(0, dtype=np.int64)
        self.best_model_weights = None
        self.best_perf = 0.0

    def gradient2label(self, grad_set):
        fc_weight_grad = grad_set[-2]
        return int(np.argmin(np.sum(fc_weight_grad, axis=0)))

    def reconstruct(self, pool_grad):
        labels = np.array([self.gradient2label(g) for g in pool_grad])
        unique_labels = np.unique(labels)
        new_data, new_labels = [], []

        for label_i in unique_labels:
            grad_indices = np.where(labels == label_i)[0]
            augmentations = []
            for j in range(len(grad_indices)):
                grad_truth = [tf.constant(g, dtype=tf.float32) for g in pool_grad[grad_indices[j]]]
                dummy = tf.Variable(tf.random.normal([1, self.input_dim]))
                label_tensor = tf.constant([label_i], dtype=tf.int64)
                criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                recon_enc = tf.keras.models.clone_model(self.encoder.model)
                recon_enc.set_weights(self.encoder.model.get_weights())
                opt = tf.keras.optimizers.Adam(learning_rate=0.1)

                for it in range(self.reconstruction_iters):
                    with tf.GradientTape() as outer:
                        outer.watch(dummy)
                        with tf.GradientTape() as inner:
                            pred = recon_enc(dummy, training=True)
                            loss = criterion(label_tensor, pred)
                        dummy_grads = inner.gradient(loss, recon_enc.trainable_variables)
                        grad_diff = tf.add_n([tf.reduce_sum(tf.square(gx - gy))
                                              for gx, gy in zip(dummy_grads, grad_truth)])
                    opt.apply_gradients([(outer.gradient(grad_diff, [dummy])[0], dummy)])
                    if it >= self.reconstruction_iters - self.num_augmentations:
                        augmentations.append(dummy.numpy().squeeze(0).copy())

                del recon_enc, grad_truth, dummy
            new_data.extend(augmentations)
            new_labels.extend([label_i] * len(augmentations))

        if new_data:
            self.proto_data = np.concatenate(
                [self.proto_data, np.array(new_data, dtype=np.float32)], axis=0)
            self.proto_labels = np.concatenate(
                [self.proto_labels, np.array(new_labels, dtype=np.int64)], axis=0)
        gc.collect()

    def evaluate_model(self, model, label_map):
        if self.proto_data.shape[0] == 0:
            return 0.0
        keras_model = model.model if hasattr(model, 'model') else model
        correct, total = 0, 0
        for i in range(0, self.proto_data.shape[0], PRED_BATCH):
            chunk = self.proto_data[i:i+PRED_BATCH]
            labs = self.proto_labels[i:i+PRED_BATCH]
            preds = tf.argmax(keras_model(chunk, training=False), axis=1).numpy()
            mapped = np.vectorize(lambda x: label_map.get(x, x))(labs) if label_map else labs
            correct += np.sum(preds == mapped)
            total += len(labs)
        return correct / total if total > 0 else 0.0

    def update(self, global_model, pool_grad, label_map):
        if pool_grad:
            self.reconstruct(pool_grad)
        perf = self.evaluate_model(global_model, label_map)
        if perf >= self.best_perf:
            self.best_perf = perf
            self.best_model_weights = [w.copy() for w in global_model.get_weights()]
        return perf

    def reset_for_new_task(self):
        self.best_perf = 0.0
        self.best_model_weights = None
        self.proto_data = np.empty((0, self.proto_data.shape[1]), dtype=np.float32)
        self.proto_labels = np.empty(0, dtype=np.int64)


class GLFC(CILMethod):
    """GLFC: iCaRL exemplars + L_GC (gradient compensation) + L_RD (relation distillation).
    When model_selection=True, adds proxy server for best-model selection via
    gradient-leaked proto samples (encoder + DLG reconstruction)."""

    def __init__(self, num_classes: int, memory: int = 2000,
                 encoder_epochs: int = 50, model_selection: bool = True, **kwargs):
        super().__init__(num_classes)
        self.name = "GLFC"
        self.memory = memory
        self.encoder_epochs = encoder_epochs
        self.model_selection = model_selection
        self.class_order = []
        self.label_map = {}
        self.current_task_classes = []
        self.client_id = None
        self.client_seen = {}
        self.exemplars = {}
        self.old_num_classes = 0
        self.old_logits_model = None
        self.input_dim = None
        self.encoder = None
        self.proxy_server = None
        self.proto_grad_pool = []
        self.exemplar_dir = Path("temp_weights/glfc_exemplars")

    def set_client(self, client_id: int):
        self.client_id = client_id
        if client_id not in self.client_seen:
            self.client_seen[client_id] = set()
        if client_id not in self.exemplars:
            self.exemplars[client_id] = {}

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)
        self.current_task_classes = list(task_classes)
        for c in task_classes:
            if c not in self.label_map:
                self.label_map[c] = len(self.class_order)
                self.class_order.append(c)
        self.old_num_classes = len(self.class_order) - len(task_classes)

    def set_old_model(self, model):
        keras_model = model.model if hasattr(model, 'model') else model
        old_model = tf.keras.models.clone_model(keras_model)
        if self.model_selection and self.proxy_server and self.proxy_server.best_model_weights is not None:
            old_model.set_weights(self.proxy_server.best_model_weights)
        else:
            old_model.set_weights(keras_model.get_weights())
        old_model.trainable = False
        logits_layer = old_model.get_layer('logits')
        self.old_logits_model = tf.keras.Model(old_model.input, logits_layer.output)
        self.old_num_classes = int(logits_layer.units)
        if self.proxy_server:
            self.proxy_server.reset_for_new_task()

    def _init_encoder(self, input_dim):
        if not self.model_selection or self.encoder is not None:
            return
        self.input_dim = input_dim
        self.encoder = GradientEncoder(input_dim, self.num_classes)
        self.proxy_server = ProxyServer(input_dim, self.num_classes, self.encoder)

    def map_labels_np(self, y):
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        return np.vectorize(self.label_map.get)(labels)

    # ── exemplar management (iCaRL herding) ──

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
            X_ex = np.load(path, mmap_mode='r')
            np.save(path, np.array(X_ex[:m]))

    def _construct_exemplars(self, feature_model, X_c, m):
        if m <= 0 or X_c.shape[0] == 0:
            return np.empty((0,) + X_c.shape[1:], dtype=X_c.dtype)
        feats = _batched_predict(feature_model, X_c)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        class_mean = np.mean(feats, axis=0)
        available = np.ones(feats.shape[0], dtype=bool)
        selected_idx = []
        sum_sel = np.zeros_like(class_mean)
        for k in range(min(m, feats.shape[0])):
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
        if feature_model is None:
            return
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
            X_c = np.array(X_new[labels == c], dtype=np.float32)
            exemplars = self._construct_exemplars(feature_model, X_c, m_per_class.get(c, 0))
            self.exemplar_dir.mkdir(parents=True, exist_ok=True)
            path = self.exemplar_dir / f"client_{self.client_id}_class_{c}.npy"
            np.save(path, exemplars)
            self.exemplars[self.client_id][c] = path
            del X_c, exemplars
        del X_new, y_new, labels, feature_model
        gc.collect()

    def rebuild_global_exemplars(self, model):
        pass

    # ── proto gradient sharing (client -> relay), conditional on model_selection ──

    def collect_proto_gradients(self, client_model, client_id, X_path, y_path):
        if not self.model_selection or self.encoder is None or self.current_task == 0:
            return

        X = np.load(X_path, mmap_mode='r')
        y = np.load(y_path, mmap_mode='r')
        labels = np.array(y if len(y.shape) == 1 else np.argmax(y, axis=1))
        feature_model = client_model.get_feature_model() if hasattr(client_model, "get_feature_model") else None
        keras_model = client_model.model if hasattr(client_model, 'model') else client_model

        for c in self.current_task_classes:
            if not np.any(labels == c):
                continue
            mapped_c = self.label_map[c]
            X_c = np.array(X[labels == c], dtype=np.float32)

            if feature_model is not None:
                feats = _batched_predict(feature_model, X_c)
                feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
                proto_idx = int(np.argmin(np.linalg.norm(feats - np.mean(feats, axis=0), axis=1)))
                del feats
            else:
                proto_idx = 0
            proto = X_c[proto_idx:proto_idx+1].copy()

            # perturb: Gaussian noise + SGD/BCE optimisation
            proto_var = tf.Variable(proto + np.random.normal(0, 0.01, proto.shape).astype(np.float32))
            target_oh = tf.one_hot([mapped_c], keras_model.output_shape[-1])
            opt_p = tf.keras.optimizers.SGD(learning_rate=0.01)
            for _ in range(self.encoder_epochs):
                with tf.GradientTape() as tape:
                    tape.watch(proto_var)
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=target_oh, logits=keras_model(proto_var, training=False)))
                opt_p.apply_gradients([(tape.gradient(loss, [proto_var])[0], proto_var)])

            # encode perturbed proto -> gradient set
            enc_model = self.encoder.model
            proto_final = tf.constant(proto_var.numpy())
            criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            with tf.GradientTape() as tape:
                enc_loss = criterion(tf.constant([mapped_c], dtype=tf.int64), enc_model(proto_final, training=True))
            enc_grads = tape.gradient(enc_loss, enc_model.trainable_variables)
            self.proto_grad_pool.append([g.numpy() for g in enc_grads])
            del X_c, proto, proto_var, proto_final

        del X, y, labels, feature_model
        gc.collect()

    def proxy_update(self, global_model):
        if not self.model_selection or self.proxy_server is None or self.current_task == 0:
            self.proto_grad_pool = []
            return None
        perf = self.proxy_server.update(global_model, self.proto_grad_pool, self.label_map)
        self.proto_grad_pool = []
        return perf

    # ── dataset: new data + exemplars + old targets ──

    def build_dataset(self, X_path, y_path, model, batch_size):
        X_new = np.load(X_path, mmap_mode='r')
        y_new = np.load(y_path, mmap_mode='r')
        num_classes = len(self.class_order)
        old_nc = self.old_num_classes
        input_dim = X_new.shape[1]
        if self.input_dim is None:
            self.input_dim = input_dim
        if self.model_selection:
            self._init_encoder(input_dim)

        client_exemplars = list(self.exemplars.get(self.client_id, {}).items())
        label_map = dict(self.label_map)

        old_targets_path = None
        if old_nc > 0 and self.old_logits_model is not None:
            all_old = []
            for i in range(0, X_new.shape[0], PRED_BATCH):
                chunk = np.array(X_new[i:i+PRED_BATCH], dtype=np.float32)
                logits = self.old_logits_model(chunk, training=False).numpy()
                all_old.append(tf.nn.sigmoid(logits[:, :old_nc]).numpy().astype(np.float32))
                del chunk, logits
            for c, path in client_exemplars:
                X_ex = np.load(path, mmap_mode='r')
                for i in range(0, X_ex.shape[0], PRED_BATCH):
                    chunk = np.array(X_ex[i:i+PRED_BATCH], dtype=np.float32)
                    logits = self.old_logits_model(chunk, training=False).numpy()
                    all_old.append(tf.nn.sigmoid(logits[:, :old_nc]).numpy().astype(np.float32))
                    del chunk, logits
                del X_ex
            old_targets = np.concatenate(all_old, axis=0)
            del all_old
            gc.collect()
            old_targets_path = Path("temp_weights/glfc_old_targets.npy")
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
            for i in range(0, X_n.shape[0], batch_size):
                X_b = np.array(X_n[i:i+batch_size], dtype=np.float32)
                lab = np.array(y_n[i:i+batch_size])
                lab = lab if len(lab.shape) == 1 else np.argmax(lab, axis=1)
                lab = np.vectorize(label_map.get)(lab)
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

        if old_nc > 0 and self.old_logits_model is not None:
            sig = (
                tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
                tf.TensorSpec(shape=(None, old_nc), dtype=tf.float32),
            )
        else:
            sig = (
                tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
            )
        dataset = tf.data.Dataset.from_generator(gen, output_signature=sig)
        dataset = dataset.prefetch(1)
        return dataset, total

    # ── train step: 0.5 * L_GC + 0.5 * L_RD ──

    def get_train_step(self, model, optimizer, loss_fn):
        keras_model = model.model if hasattr(model, 'model') else model
        logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else None
        old_nc = self.old_num_classes

        @tf.function
        def train_step_glfc(batch_X, batch_y, batch_old=None):
            with tf.GradientTape() as tape:
                logits = logits_model(batch_X, training=True)
                pred = tf.stop_gradient(tf.nn.sigmoid(logits))
                target = batch_y

                # per-sample gradient: |sigmoid(pred_at_true_class) - 1|
                labels = tf.cast(tf.argmax(batch_y, axis=1), tf.int32)
                batch_sz = tf.shape(labels)[0]
                indices = tf.stack([tf.range(batch_sz), labels], axis=1)
                g = tf.expand_dims(tf.abs(tf.gather_nd(pred, indices) - 1.0), 1)

                # gradient compensation: divide per-sample g by group mean
                if old_nc > 0:
                    is_old = tf.expand_dims(tf.cast(labels < old_nc, tf.float32), 1)
                    is_new = 1.0 - is_old
                    n_old = tf.reduce_sum(is_old)
                    n_new = tf.reduce_sum(is_new)
                    g_old = g * is_old
                    g_new = g * is_new
                    mean_old = tf.where(n_old > 0, tf.reduce_sum(g_old) / n_old, 1.0)
                    mean_new = tf.where(n_new > 0, tf.reduce_sum(g_new) / n_new, 1.0)
                    w = g_old / (mean_old + 1e-12) + g_new / (mean_new + 1e-12)
                else:
                    w = tf.ones_like(g)

                per_sample_bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
                loss_gc = tf.reduce_mean(w * per_sample_bce)

                if batch_old is not None and old_nc > 0:
                    distill_target = tf.concat([batch_old, target[:, old_nc:]], axis=1)
                    loss_rd = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=distill_target, logits=logits))
                    total_loss = 0.5 * loss_gc + 0.5 * loss_rd
                else:
                    loss_rd = tf.constant(0.0)
                    total_loss = loss_gc

            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return loss_gc, loss_rd, total_loss

        if old_nc > 0:
            return train_step_glfc

        @tf.function
        def train_step_first_task(batch_X, batch_y):
            return train_step_glfc(batch_X, batch_y, None)
        return train_step_first_task

    # ── evaluation ──

    def evaluate(self, model, X, y, num_classes, logger=None):
        logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else None
        y_mapped = self.map_labels_np(y if len(y.shape) == 1 else np.argmax(y, axis=1))
        loss_sum = 0.0
        n = X.shape[0]
        all_preds = []
        for i in range(0, n, PRED_BATCH):
            x_chunk = np.array(X[i:i+PRED_BATCH], dtype=np.float32)
            y_chunk = tf.keras.utils.to_categorical(y_mapped[i:i+PRED_BATCH], num_classes)
            log_chunk = logits_model(x_chunk, training=False).numpy()
            all_preds.append(np.argmax(log_chunk, axis=1))
            e = np.exp(log_chunk - log_chunk.max(axis=1, keepdims=True))
            p = np.clip(e / e.sum(axis=1, keepdims=True), 1e-7, 1.0)
            loss_sum += float(-np.sum(y_chunk * np.log(p)))
            del x_chunk, y_chunk, log_chunk, p
        return {"acc": float(np.mean(np.concatenate(all_preds) == y_mapped)),
                "loss": loss_sum / n}

    def cleanup_temp(self):
        if self.exemplar_dir.exists():
            for f in self.exemplar_dir.glob("*.npy"):
                f.unlink(missing_ok=True)
        old_tgt = Path("temp_weights/glfc_old_targets.npy")
        if old_tgt.exists():
            old_tgt.unlink(missing_ok=True)

    def extra_metrics(self) -> dict:
        return {"glfc_memory": self.memory,
                "glfc_model_selection": int(self.model_selection)}
