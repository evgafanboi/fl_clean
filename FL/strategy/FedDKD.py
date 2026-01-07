from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..pipeline import PipelineContext, evaluate_model
from .base import DistillationStrategy
from .common import create_model, create_private_dataset


def compute_dkd_gradient(expert_model, global_model, batch_X, batch_y):
    expert_keras = expert_model.model if hasattr(expert_model, 'model') else expert_model
    global_keras = global_model.model if hasattr(global_model, 'model') else global_model
    
    expert_logits = expert_keras(batch_X, training=False)
    
    with tf.GradientTape() as tape:
        global_logits = global_keras(batch_X, training=True)
        loss = tf.keras.losses.categorical_crossentropy(expert_logits, global_logits, from_logits=False)
        loss = tf.reduce_mean(loss)
    
    trainable_vars = global_model.trainable_variables if hasattr(global_model, 'trainable_variables') else global_keras.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    return gradients, loss.numpy()


def aggregate_gradients(client_gradients, client_weights):
    n_clients = len(client_gradients)
    total_samples = sum(client_weights)
    weights = [w / total_samples for w in client_weights]
    
    aggregated = []
    num_layers = len(client_gradients[0])
    
    for layer_idx in range(num_layers):
        layer_grads = [client_gradients[client_idx][layer_idx] * weights[client_idx] 
                      for client_idx in range(n_clients)]
        aggregated.append(tf.reduce_sum(layer_grads, axis=0))
    
    return aggregated


def apply_gradient_update(model, gradients, learning_rate):
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    if hasattr(model, 'model'):
        trainable_vars = model.model.trainable_variables
    else:
        trainable_vars = model.trainable_variables
    
    optimizer.apply_gradients(zip(gradients, trainable_vars))


class FedDKD(DistillationStrategy):
    name = "FedDKD"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.expert_epochs = config.epochs
        self.dkd_steps = config.dkd_steps
        self.dkd_lr = config.dkd_lr

    def extra_log_tokens(self) -> Dict[str, str]:
        return {
            "dkd_steps": f"dkd{self.dkd_steps}",
            "dkd_lr": f"lr{self.dkd_lr}",
        }

    def setup(self, context: PipelineContext) -> None:
        print(f"{COLORS.OKGREEN}Setting up FedDKD{COLORS.ENDC}")
        is_sequence = context.config.model_type.lower() == "gru"
        context.shared_state["is_sequence"] = is_sequence
        
        sample_sizes = []
        
        print(f"\n{COLORS.HEADER}Training Initial Local Experts{COLORS.ENDC}")
        for client_id, paths in enumerate(context.paths):
            print(f"\n{COLORS.BOLD}Training Expert {client_id}{COLORS.ENDC}")
            
            model = create_model(
                context.input_dim,
                context.num_classes,
                context.config.batch_size,
                model_type=context.config.model_type,
            )
            
            train_dataset = create_private_dataset(
                paths["train_X"],
                paths["train_y"],
                context.input_dim,
                context.num_classes,
                context.config.batch_size,
                is_sequence=is_sequence,
            )
            
            X_train_mmap = np.load(paths["train_X"], mmap_mode='r')
            sample_size = X_train_mmap.shape[0]
            sample_sizes.append(sample_size)
            del X_train_mmap
            
            model.fit(train_dataset, epochs=self.expert_epochs, verbose=1)
            
            context.add_client_state(client_id, model, paths)
            
            del train_dataset
            aggressive_memory_cleanup()
        
        context.shared_state["sample_sizes"] = sample_sizes
        
        print(f"\n{COLORS.PURPLE}Initializing global model (weighted average of experts){COLORS.ENDC}")
        
        total_samples = sum(sample_sizes)
        init_weights = [s / total_samples for s in sample_sizes]
        
        global_model = create_model(
            context.input_dim,
            context.num_classes,
            context.config.batch_size,
            model_type=context.config.model_type,
        )
        
        expert_weights = [state.model.get_weights() for state in context.client_states]
        
        global_weights = []
        for layer_idx in range(len(expert_weights[0])):
            layer_avg = sum([expert_weights[client_idx][layer_idx] * init_weights[client_idx] 
                            for client_idx in range(context.n_clients)])
            global_weights.append(layer_avg)
        
        global_model.set_weights(global_weights)
        del global_weights
        
        context.shared_state["global_model"] = global_model
        
        context.logger.info("Global model initialized from expert average")

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        is_sequence = context.shared_state["is_sequence"]
        sample_sizes = context.shared_state["sample_sizes"]
        global_model = context.shared_state["global_model"]
        
        print(f"\n{COLORS.PURPLE}[LOCAL TRAINING] Updating expert models from global{COLORS.ENDC}")
        
        global_weights = global_model.get_weights()
        
        for state in context.client_states:
            print(f"\n{COLORS.BOLD}Training Expert {state.client_id}{COLORS.ENDC}")
            state.model.set_weights(global_weights)
            
            train_dataset = create_private_dataset(
                state.paths["train_X"],
                state.paths["train_y"],
                context.input_dim,
                context.num_classes,
                config.batch_size,
                is_sequence=is_sequence,
            )
            
            state.model.fit(train_dataset, epochs=self.expert_epochs, verbose=1)
            
            del train_dataset
            aggressive_memory_cleanup()
        
        context.logger.info(f"All experts trained for {self.expert_epochs} epochs from global model")
        
        print(f"\n{COLORS.PURPLE}[DKD DISTILLATION] Running {self.dkd_steps} gradient steps{COLORS.ENDC}")
        
        for dkd_step in range(self.dkd_steps):
            client_grads = []
            step_losses = []
            
            for state in context.client_states:
                train_dataset = create_private_dataset(
                    state.paths["train_X"],
                    state.paths["train_y"],
                    context.input_dim,
                    context.num_classes,
                    config.batch_size,
                    is_sequence=is_sequence,
                )
                
                for batch_X, batch_y in train_dataset.take(1):
                    grads, loss = compute_dkd_gradient(
                        state.model,
                        global_model,
                        batch_X,
                        batch_y,
                    )
                    client_grads.append(grads)
                    step_losses.append(loss)
                
                del train_dataset
            
            aggregated_grads = aggregate_gradients(client_grads, sample_sizes)
            
            apply_gradient_update(global_model, aggregated_grads, self.dkd_lr)
            
            del client_grads, aggregated_grads
            aggressive_memory_cleanup()
            
            if (dkd_step + 1) % 10 == 0:
                avg_loss = np.mean(step_losses)
                print(f"  DKD step {dkd_step + 1}/{self.dkd_steps}: Avg distillation loss={avg_loss:.4f}")
        
        context.logger.info(f"Completed {self.dkd_steps} DKD steps")
        
        print(f"\n{COLORS.PURPLE}[GLOBAL EVALUATION]{COLORS.ENDC}")
        
        global_metrics = evaluate_model(global_model, context.test_dataset, context.test_labels)
        
        print(
            f"{COLORS.OKGREEN}[GLOBAL MODEL] Acc={global_metrics['Acc']:.4f}, "
            f"F1={global_metrics['F1']:.4f}, Precision={global_metrics['Precision']:.4f}, "
            f"Recall={global_metrics['Recall']:.4f}{COLORS.ENDC}"
        )
        
        context.logger.info(
            "Round %s | GLOBAL | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
            round_number,
            global_metrics["Acc"],
            global_metrics["F1"],
            global_metrics["Precision"],
            global_metrics["Recall"],
        )
        
        round_metrics: Dict[int, Dict[str, float]] = {-1: global_metrics}
        
        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        
        return round_metrics
