"""Learning without Forgetting (LwF) for continual learning"""

import numpy as np
from ..backend import use_tf as _use_tf
from .base import CILMethod


class LwF(CILMethod):
    def __init__(self, num_classes: int, alpha: float = 0.5, temperature: float = 2.0, **kwargs):
        super().__init__(num_classes)
        self.name = "LwF"
        self.alpha = alpha
        self.temperature = temperature
        self.class_order = []
        self.label_map = {}
        self.old_model = None
        self.old_logits_model = None
        self.old_num_classes = 0

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
            self.old_logits_model = old
            self.old_num_classes = model.num_classes
            return
        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        self.old_model = tf.keras.models.clone_model(keras_model)
        self.old_model.set_weights(keras_model.get_weights())
        self.old_model.trainable = False
        logits_layer = self.old_model.get_layer('logits')
        self.old_logits_model = tf.keras.Model(self.old_model.input, logits_layer.output)
        self.old_num_classes = int(logits_layer.units)

    def map_labels_np(self, y):
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        mapped = np.vectorize(self.label_map.get, otypes=[int])(labels)
        return mapped

    def get_train_step(self, model, optimizer, loss_fn):
        if self.old_logits_model is None:
            return None

        if not _use_tf():
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            self.old_model.to(dev)
            T = self.temperature
            alpha = self.alpha
            old_model = self.old_model
            old_nc = self.old_num_classes

            def step_pt(X_b, y_b):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                logits = model.nn(X_b, return_logits=True)
                ce_loss = F.cross_entropy(logits, y_cls, label_smoothing=0.05)
                with torch.no_grad():
                    old_logits = old_model(X_b, return_logits=True)
                    old_probs = F.softmax(old_logits / T, dim=1)
                new_probs = F.log_softmax(logits[:, :old_nc] / T, dim=1)
                distill = F.kl_div(new_probs, old_probs, reduction='batchmean') * (T * T)
                total = alpha * distill + (1.0 - alpha) * ce_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                return ce_loss.item(), distill.item(), total.item()

            return step_pt

        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        new_logits_model = model.get_logits_model() if hasattr(model, 'get_logits_model') else tf.keras.Model(keras_model.input, keras_model.get_layer('logits').output)
        T = self.temperature
        alpha = self.alpha

        @tf.function
        def train_step_with_lwf(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                old_logits = self.old_logits_model(batch_X, training=False)
                new_logits = new_logits_model(batch_X, training=True)
                old_probs = tf.stop_gradient(tf.nn.softmax(old_logits / T))
                new_probs = tf.nn.softmax(new_logits[:, :self.old_num_classes] / T)
                distill = tf.keras.losses.KLDivergence()(old_probs, new_probs) * (T * T)
                total_loss = alpha * distill + (1.0 - alpha) * ce_loss
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, distill, total_loss

        return train_step_with_lwf

    def extra_metrics(self) -> dict:
        return {
            "lwf_alpha": self.alpha,
            "lwf_temperature": self.temperature
        }

    def get_ssd_train_step(self, model, optimizer, loss_fn,
                           global_logits_model, local_logits_model,
                           M_class_tf, m_max_tf):
        if self.old_logits_model is None:
            from .finetune import Finetune
            return Finetune(num_classes=self.num_classes).get_ssd_train_step(
                model, optimizer, loss_fn, global_logits_model, local_logits_model, M_class_tf, m_max_tf)

        if not _use_tf():
            import torch
            import torch.nn.functional as F
            from .finetune import _pt_ssd_loss
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            self.old_model.to(dev)
            T = self.temperature
            alpha = self.alpha
            old_model = self.old_model
            old_nc = self.old_num_classes
            m_max = float(m_max_tf)

            def step_pt(X_b, y_b):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                local_logits = model.nn(X_b, return_logits=True)
                ce_loss = F.cross_entropy(local_logits, y_cls, label_smoothing=0.05)
                with torch.no_grad():
                    old_logits = old_model(X_b, return_logits=True)
                    old_probs = F.softmax(old_logits / T, dim=1)
                    global_logits_np = global_logits_model.predict(X_b.cpu().numpy(), batch_size=len(X_b))
                    global_logits = torch.from_numpy(global_logits_np).to(dev)
                new_probs = F.log_softmax(local_logits[:, :old_nc] / T, dim=1)
                lwf_loss = F.kl_div(new_probs, old_probs, reduction='batchmean') * (T * T)
                ssd_loss = _pt_ssd_loss(local_logits, global_logits, y_b, M_class_tf, m_max)
                total = (1.0 - alpha) * ce_loss + alpha * lwf_loss + ssd_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                return ce_loss.item(), ssd_loss.item(), total.item()

            return step_pt

        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model
        T = self.temperature
        alpha = self.alpha
        old_logits_model = self.old_logits_model
        old_nc = self.old_num_classes

        @tf.function(reduce_retracing=True)
        def train_step_lwf_ssd(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)

                old_logits = old_logits_model(batch_X, training=False)
                new_logits = local_logits_model(batch_X, training=True)
                old_probs = tf.stop_gradient(tf.nn.softmax(old_logits / T))
                new_probs = tf.nn.softmax(new_logits[:, :old_nc] / T)
                lwf_loss = tf.keras.losses.KLDivergence()(old_probs, new_probs) * (T * T)

                global_lg = global_logits_model(batch_X, training=False)
                global_probs = tf.nn.softmax(global_lg)
                true_labels = tf.cast(tf.argmax(batch_y, axis=1), tf.int32)
                batch_size = tf.shape(batch_y)[0]
                indices = tf.stack([tf.range(batch_size), true_labels], axis=1)
                p_g_k2 = tf.gather_nd(global_probs, indices)
                M_sample = tf.expand_dims(
                    1.0 - tf.sqrt(tf.maximum(1.0 - p_g_k2, 0.0)), axis=1)
                M = m_max_tf * tf.nn.relu(M_class_tf * M_sample - 0.1)
                logit_diff = M * (global_lg - new_logits)
                ssd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logit_diff), axis=1))

                total_loss = (1.0 - alpha) * ce_loss + alpha * lwf_loss + ssd_loss
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, ssd_loss, total_loss

        return train_step_lwf_ssd