"""Elastic Weight Consolidation (EWC) for continual learning"""

import numpy as np
import tensorflow as tf
from typing import Optional
from .base import CILMethod


class EWC(CILMethod):
    """
    Elastic Weight Consolidation - penalizes changes to important weights.
    
    After each task, computes Fisher Information Matrix to identify important weights,
    then adds penalty term to loss: lambda * sum((theta - theta_old)^2 * F)
    """
    
    def __init__(self, num_classes: int, ewc_lambda: float = 1.0, **kwargs):
        super().__init__(num_classes)
        self.name = "EWC"
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = {}
        self.optimal_weights = {}
    
    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)
    
    def after_task(self, model, task_data: Optional[tf.data.Dataset] = None):
        """Compute Fisher Information Matrix after task completion"""
        if task_data is None:
            return
        
        print(f"  Computing Fisher Information for Task {self.current_task}...")
        
        keras_model = model.model if hasattr(model, 'model') else model
        
        old_weights = []
        fisher_diag = []
        for var in keras_model.trainable_variables:
            old_weights.append(tf.identity(var).numpy())
            fisher_diag.append(np.zeros_like(var.numpy()))
        
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            reduction=tf.keras.losses.Reduction.SUM
        )
        num_samples = 0
        for batch_X, batch_y in task_data:
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=False)
                loss = loss_fn(batch_y, predictions)
            
            gradients = tape.gradient(loss, keras_model.trainable_variables)
            
            for i, (grad, var) in enumerate(zip(gradients, keras_model.trainable_variables)):
                if grad is not None:
                    grad_squared = tf.square(grad)
                    fisher_diag[i] += grad_squared.numpy()
            
            num_samples += batch_X.shape[0]
        
        for i in range(len(fisher_diag)):
            fisher_diag[i] /= num_samples
        
        self.fisher_dict[self.current_task] = fisher_diag
        self.optimal_weights[self.current_task] = old_weights
        
        # Memory usage info
        total_params = sum(w.size for w in old_weights)
        total_sum = 0.0
        total_count = 0
        max_val = 0.0
        for f in fisher_diag:
            total_sum += float(np.sum(f))
            total_count += f.size
            max_val = max(max_val, float(np.max(f)))
        mean_val = total_sum / total_count if total_count > 0 else 0.0
        print(f"  Fisher computed for Task {self.current_task} ({num_samples} samples, {total_params:,} params)")
        print(f"  Fisher stats: mean={mean_val:.6e}, max={max_val:.6e}")
        print(f"  EWC now protecting {len(self.fisher_dict)} tasks ({total_params * len(self.fisher_dict) * 2 * 4 / 1024 / 1024:.1f} MB)")
    
    def get_train_step(self, model, optimizer, loss_fn):
        """Return custom training step with EWC penalty"""
        
        if len(self.fisher_dict) == 0:
            return None
        
        keras_model = model.model if hasattr(model, 'model') else model
        
        @tf.function
        def train_step_with_ewc(batch_X, batch_y):
            with tf.GradientTape() as tape:
                predictions = keras_model(batch_X, training=True)
                ce_loss = loss_fn(batch_y, predictions)
                
                ewc_loss = 0.0
                for task_id in self.fisher_dict:
                    for i, var in enumerate(keras_model.trainable_variables):
                        fisher = self.fisher_dict[task_id][i]
                        optimal = self.optimal_weights[task_id][i]
                        ewc_loss += tf.reduce_sum(
                            tf.cast(fisher, tf.float32) * tf.square(var - optimal)
                        )
                
                total_loss = ce_loss + self.ewc_lambda * ewc_loss
            
            gradients = tape.gradient(total_loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            
            return ce_loss, ewc_loss, total_loss
        
        return train_step_with_ewc
    
    def extra_metrics(self) -> dict:
        return {
            "ewc_lambda": self.ewc_lambda,
            "protected_tasks": len(self.fisher_dict)
        }
