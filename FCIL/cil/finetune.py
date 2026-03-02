"""Finetune CIL method - simply train on new task without catastrophic forgetting prevention"""

import tensorflow as tf
from .base import CILMethod


class Finetune(CILMethod):
    """Naive finetune on new task without any continual learning mechanism"""
    
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
