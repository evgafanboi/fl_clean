"""Memory Aware Synapses (MAS) for continual learning.
Per-client omega computation — importance values never leave the client.
Measures output sensitivity via gradient of L2 norm of learned function (label-free).
"""

import numpy as np
import tensorflow as tf
from typing import Optional
from .base import CILMethod


class MAS(CILMethod):

    def __init__(self, num_classes: int, mas_lambda: float = 1.0, **kwargs):
        super().__init__(num_classes)
        self.name = "MAS"
        self.mas_lambda = mas_lambda
        self.omega = {}           # client_id -> {task_id -> [arrays]}
        self.optimal_weights = {} # client_id -> {task_id -> [arrays]}
        self.active_client = 0

    def set_client(self, client_id: int):
        self.active_client = client_id
        self.omega.setdefault(client_id, {})
        self.optimal_weights.setdefault(client_id, {})

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)

    def after_task(self, model, task_data: Optional[tf.data.Dataset] = None):
        if task_data is None:
            return
        cid = self.active_client
        keras_model = model.model if hasattr(model, 'model') else model
        logits_model = model.get_logits_model()

        old_weights = [tf.identity(v).numpy() for v in keras_model.trainable_variables]
        omega = [np.zeros_like(v.numpy()) for v in keras_model.trainable_variables]

        num_batches = 0
        for batch_X, batch_y in task_data:
            with tf.GradientTape() as tape:
                logits = logits_model(batch_X, training=False)
                l2_norms = tf.norm(logits, ord=2, axis=1)
                target = tf.reduce_mean(l2_norms)
            grads = tape.gradient(target, keras_model.trainable_variables)
            for i, g in enumerate(grads):
                if g is not None:
                    omega[i] += tf.abs(g).numpy()
            num_batches += 1

        for i in range(len(omega)):
            omega[i] /= num_batches

        self.omega[cid][self.current_task] = omega
        self.optimal_weights[cid][self.current_task] = old_weights

        total_params = sum(w.size for w in old_weights)
        mean_val = sum(float(np.mean(o)) for o in omega) / len(omega)
        max_val = max(float(np.max(o)) for o in omega)
        print(f"  MAS Client {cid} Task {self.current_task}: {num_batches} batches, "
              f"mean={mean_val:.6e}, max={max_val:.6e}")

    def get_train_step(self, model, optimizer, loss_fn):
        cid = self.active_client
        client_omega = self.omega.get(cid, {})
        client_optimal = self.optimal_weights.get(cid, {})
        if len(client_omega) == 0:
            return None

        keras_model = model.model if hasattr(model, 'model') else model
        mas_lambda = self.mas_lambda

        @tf.function
        def train_step_with_mas(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)

                mas_loss = 0.0
                for task_id in client_omega:
                    for i, var in enumerate(keras_model.trainable_variables):
                        o = client_omega[task_id][i]
                        ref = client_optimal[task_id][i]
                        mas_loss += tf.reduce_sum(
                            tf.cast(o, tf.float32) * tf.square(var - ref))

                total_loss = ce_loss + mas_lambda * mas_loss
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, mas_loss, total_loss

        return train_step_with_mas

    def extra_metrics(self) -> dict:
        n_tasks = max((len(v) for v in self.omega.values()), default=0)
        return {"mas_lambda": self.mas_lambda, "protected_tasks": n_tasks}

    def get_ssd_train_step(self, model, optimizer, loss_fn,
                           global_logits_model, local_logits_model,
                           M_class_tf, m_max_tf):
        cid = self.active_client
        client_omega = self.omega.get(cid, {})
        client_optimal = self.optimal_weights.get(cid, {})
        keras_model = model.model if hasattr(model, 'model') else model
        mas_lambda = self.mas_lambda

        @tf.function(reduce_retracing=True)
        def train_step_mas_ssd(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)

                mas_loss = 0.0
                for task_id in client_omega:
                    for i, var in enumerate(keras_model.trainable_variables):
                        o = client_omega[task_id][i]
                        ref = client_optimal[task_id][i]
                        mas_loss += tf.reduce_sum(
                            tf.cast(o, tf.float32) * tf.square(var - ref))

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

                total_loss = ce_loss + mas_lambda * mas_loss + ssd_loss
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return ce_loss, ssd_loss, total_loss

        if len(client_omega) == 0:
            from .finetune import Finetune
            ft = Finetune(num_classes=self.num_classes)
            return ft.get_ssd_train_step(model, optimizer, loss_fn,
                                         global_logits_model, local_logits_model,
                                         M_class_tf, m_max_tf)
        return train_step_mas_ssd
