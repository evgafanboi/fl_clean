"""Finetune CIL method - simply train on new task without catastrophic forgetting prevention"""

from ..backend import use_tf as _use_tf
from .base import CILMethod


def _pt_ssd_loss(local_logits, global_logits, y_b, M_class_np, m_max):
    import torch
    dev = local_logits.device
    M_class = torch.from_numpy(M_class_np.astype('float32')).to(dev)
    probs = torch.softmax(global_logits, dim=1)
    true_labels = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
    p_g_k2 = probs[torch.arange(len(true_labels), device=dev), true_labels]
    M_sample = (1.0 - torch.sqrt(torch.clamp(1.0 - p_g_k2, min=0.0))).unsqueeze(1)
    M = m_max * torch.relu(M_class * M_sample - 0.1)
    return torch.mean(torch.sum((M * (global_logits - local_logits)) ** 2, dim=1))


class Finetune(CILMethod):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes)
        self.name = "Finetune"

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)

    def after_task(self, model, task_data=None):
        pass

    def get_ssd_train_step(self, model, optimizer, loss_fn,
                           global_logits_model, local_logits_model,
                           M_class_tf, m_max_tf):
        if not _use_tf():
            import torch
            import torch.nn.functional as F
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.nn.to(dev)
            m_max = float(m_max_tf)
            M_class_np = M_class_tf

            def step_pt(X_b, y_b):
                X_b = X_b.to(dev); y_b = y_b.to(dev)
                y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
                model.nn.train()
                optimizer.zero_grad()
                local_logits = model.nn(X_b, return_logits=True)
                ce_loss = F.cross_entropy(local_logits, y_cls, label_smoothing=0.05)
                with torch.no_grad():
                    global_logits = torch.from_numpy(
                        global_logits_model.predict(X_b.cpu().numpy(), batch_size=len(X_b))).to(dev)
                ssd_loss = _pt_ssd_loss(local_logits, global_logits, y_b, M_class_np, m_max)
                total = ce_loss + ssd_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 0.5)
                optimizer.step()
                return ce_loss.item(), ssd_loss.item(), total.item()

            return step_pt

        import tensorflow as tf
        keras_model = model.model if hasattr(model, 'model') else model

        @tf.function(reduce_retracing=True)
        def train_step_ssd(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
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
                total_loss = ce_loss + ssd_loss
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, ssd_loss, total_loss

        return train_step_ssd
