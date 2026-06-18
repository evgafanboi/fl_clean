import numpy as np
from typing import Optional, Any
from ..backend import use_tf as _use_tf
from .base import CILMethod


class EWC(CILMethod):

    def __init__(self, num_classes: int, ewc_lambda: float = 1.0, **kwargs):
        super().__init__(num_classes)
        self.name = "EWC"
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.optimal_weights = {}
        self.active_client = 0

    def set_client(self, client_id: int):
        self.active_client = client_id
        self.fisher.setdefault(client_id, {})
        self.optimal_weights.setdefault(client_id, {})

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)

    def after_task(self, model, task_data: Optional[Any] = None):
        if task_data is None:
            return
        cid = self.active_client

        if not _use_tf():
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            old_weights = [p.detach().cpu().numpy().copy() for p in model.nn.parameters()]
            params = list(model.nn.parameters())
            fisher = [torch.zeros_like(p, device=dev) for p in params]
            total_samples = 0
            was_training = model.nn.training
            model.nn.train()
            for batch_X, batch_y in task_data:
                X_b = batch_X.to(dev)
                y_cls = (batch_y.argmax(dim=1) if batch_y.ndim > 1 else batch_y.long()).to(dev)
                batch_size = len(y_cls)
                model.optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model.nn(X_b, return_logits=True), y_cls)
                loss.backward()
                for i, p in enumerate(params):
                    if p.grad is not None:
                        fisher[i] += p.grad.detach() ** 2 * batch_size
                total_samples += batch_size
            for i in range(len(fisher)):
                fisher[i] /= max(total_samples, 1)
            fisher = [f.detach().cpu().numpy() for f in fisher]
            self.fisher[cid][self.current_task] = fisher
            self.optimal_weights[cid][self.current_task] = old_weights
            model.nn.train(was_training)
            mean_val = sum(float(np.mean(f)) for f in fisher) / len(fisher)
            max_val = max(float(np.max(f)) for f in fisher)
            print(f"  EWC Client {cid} Task {self.current_task}: {total_samples} samples, "
                  f"mean={mean_val:.6e}, max={max_val:.6e}")
            return

        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        old_weights = [tf.identity(v).numpy() for v in keras_model.trainable_variables]
        fisher_diag = [np.zeros_like(v.numpy()) for v in keras_model.trainable_variables]
        num_samples = 0
        for batch_X, batch_y in task_data:
            y_idx = tf.argmax(batch_y, axis=1) if len(batch_y.shape) > 1 else tf.cast(batch_y, tf.int64)
            batch_size = batch_X.shape[0]
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=False)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_idx, predictions))
            grads = tape.gradient(loss, keras_model.trainable_variables)
            for i, g in enumerate(grads):
                if g is not None:
                    fisher_diag[i] += tf.square(g).numpy() * batch_size
            num_samples += batch_size
        for i in range(len(fisher_diag)):
            fisher_diag[i] /= num_samples
        self.fisher[cid][self.current_task] = fisher_diag
        self.optimal_weights[cid][self.current_task] = old_weights
        mean_val = sum(float(np.sum(f)) for f in fisher_diag) / sum(f.size for f in fisher_diag)
        max_val = max(float(np.max(f)) for f in fisher_diag)
        print(f"  EWC Client {cid} Task {self.current_task}: {num_samples} samples, "
              f"mean={mean_val:.6e}, max={max_val:.6e}")

    def get_train_step(self, model, optimizer, loss_fn):
        cid = self.active_client
        client_fisher = self.fisher.get(cid, {})
        client_optimal = self.optimal_weights.get(cid, {})
        if len(client_fisher) == 0:
            return None

        if not _use_tf():
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            ewc_lambda = self.ewc_lambda
            fisher_t = {tid: [torch.from_numpy(f).to(dev) for f in client_fisher[tid]]
                        for tid in client_fisher}
            optimal_t = {tid: [torch.from_numpy(o).to(dev) for o in client_optimal[tid]]
                         for tid in client_optimal}
            params = list(model.nn.parameters())

            def step_pt(X_b, y_b):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                logits = model.nn(X_b, return_logits=True)
                ce_loss = F.cross_entropy(logits, y_cls, label_smoothing=0.05)
                ewc_loss = sum(
                    (fisher_t[tid][i] * (params[i] - optimal_t[tid][i]) ** 2).sum()
                    for tid in fisher_t for i in range(len(params)))
                total = ce_loss + 0.5 * ewc_lambda * ewc_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                return ce_loss.item(), ewc_loss.item() if hasattr(ewc_loss, 'item') else float(ewc_loss), total.item()

            return step_pt

        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        ewc_lambda = self.ewc_lambda

        @tf.function
        def train_step_with_ewc(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                ewc_loss = 0.0
                for task_id in client_fisher:
                    for i, var in enumerate(keras_model.trainable_variables):
                        f = client_fisher[task_id][i]
                        ref = client_optimal[task_id][i]
                        ewc_loss += tf.reduce_sum(
                            tf.cast(f, tf.float32) * tf.square(var - ref))
                total_loss = ce_loss + 0.5 * ewc_lambda * ewc_loss
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, ewc_loss, total_loss

        return train_step_with_ewc

    def extra_metrics(self) -> dict:
        n_tasks = max((len(v) for v in self.fisher.values()), default=0)
        return {"ewc_lambda": self.ewc_lambda, "protected_tasks": n_tasks}

    def get_ssd_train_step(self, model, optimizer, loss_fn,
                           global_logits_model, local_logits_model,
                           M_class_tf, m_max_tf):
        cid = self.active_client
        client_fisher = self.fisher.get(cid, {})
        client_optimal = self.optimal_weights.get(cid, {})
        ewc_lambda = self.ewc_lambda

        if not _use_tf():
            if len(client_fisher) == 0:
                from .finetune import Finetune
                return Finetune(num_classes=self.num_classes).get_ssd_train_step(
                    model, optimizer, loss_fn, global_logits_model, local_logits_model, M_class_tf, m_max_tf)
            import torch
            import torch.nn.functional as F
            from .finetune import _pt_ssd_loss
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            m_max = float(m_max_tf)
            fisher_t = {tid: [torch.from_numpy(f).to(dev) for f in client_fisher[tid]]
                        for tid in client_fisher}
            optimal_t = {tid: [torch.from_numpy(o).to(dev) for o in client_optimal[tid]]
                         for tid in client_optimal}
            params = list(model.nn.parameters())

            def step_pt(X_b, y_b):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                local_logits = model.nn(X_b, return_logits=True)
                ce_loss = F.cross_entropy(local_logits, y_cls, label_smoothing=0.05)
                ewc_loss = sum(
                    (fisher_t[tid][i] * (params[i] - optimal_t[tid][i]) ** 2).sum()
                    for tid in fisher_t for i in range(len(params)))
                with torch.no_grad():
                    global_logits = torch.from_numpy(
                        global_logits_model.predict(X_b.cpu().numpy(), batch_size=len(X_b))).to(dev)
                ssd_loss = _pt_ssd_loss(local_logits, global_logits, y_b, M_class_tf, m_max)
                total = ce_loss + 0.5 * ewc_lambda * ewc_loss + ssd_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                return ce_loss.item(), ssd_loss.item(), total.item()

            return step_pt

        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model

        @tf.function(reduce_retracing=True)
        def train_step_ewc_ssd(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)

                ewc_loss = 0.0
                for task_id in client_fisher:
                    for i, var in enumerate(keras_model.trainable_variables):
                        f = client_fisher[task_id][i]
                        ref = client_optimal[task_id][i]
                        ewc_loss += tf.reduce_sum(
                            tf.cast(f, tf.float32) * tf.square(var - ref))

                local_logits = local_logits_model(batch_X, training=True)
                global_logits = global_logits_model(batch_X, training=False)
                global_probs = tf.nn.softmax(global_logits)
                true_labels = tf.cast(tf.argmax(batch_y, axis=1), tf.int32)
                batch_size = tf.shape(batch_y)[0]
                indices = tf.stack([tf.range(batch_size), true_labels], axis=1)
                p_g_k2 = tf.gather_nd(global_probs, indices)
                M_sample = tf.expand_dims(
                    1.0 - tf.sqrt(tf.maximum(1.0 - p_g_k2, 0.0)), axis=1)
                M = m_max_tf * tf.nn.relu(M_class_tf * M_sample - 0.1)
                logit_diff = M * (global_logits - local_logits)
                ssd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logit_diff), axis=1))

                total_loss = ce_loss + 0.5 * ewc_lambda * ewc_loss + ssd_loss
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, ssd_loss, total_loss

        if len(client_fisher) == 0:
            from .finetune import Finetune
            ft = Finetune(num_classes=self.num_classes)
            return ft.get_ssd_train_step(model, optimizer, loss_fn,
                                         global_logits_model, local_logits_model,
                                         M_class_tf, m_max_tf)
        return train_step_ewc_ssd
