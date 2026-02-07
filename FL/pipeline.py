import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

from .aggregators import StrategyRuntime, build_strategy
from .colors import COLORS
from .context import ClientState, PipelineContext
from .data_utils import (
    create_client_dataset,
    load_test_dataset,
    parse_partition_type,
    restore_full_partition,
    setup_paths,
)
from .evaluation import create_enhanced_excel_report, evaluate_model_with_metrics
from .gpu import configure_gpu_memory
from .logging_utils import log_timestamp, setup_logger
from .memory import aggressive_memory_cleanup
from .model_factory import create_model
from .poison_utils import parse_poison_config, get_or_create_poisoned_clients, PoisonedDataLoader


@dataclass
class FLConfig:
    n_clients: int = 10
    partition_type: str = "iid-10"
    rounds: int = 10
    strategy: str = "FedAvg"
    feddyn_alpha: float = 0.1
    batch_size: int = 8192
    epochs: int = 5
    weights_cache_dir: str = "temp_weights"
    mu: float = 0.01

    model: str = "dense"
    robust_epsilon: float = 0.2
    robust_tau: float = 0.1
    poison: Optional[str] = None
    personalized_eval: bool = False
    root_iterations: int = 1
    lambda1: float = 1.0
    lambda2: float = 1.0
    temperature: float = 1.0
    rank: int = 8
    trust_score: bool = False
    peer_trust: bool = False
    
    k_clusters: int = 3
    warmup_rounds: int = 2
    m_max: float = 1.0
    support: bool = False
    threshold: float = 0.3

    def to_strategy_params(self) -> Dict[str, object]:
        return {
            'mu': self.mu,

            'feddyn_alpha': self.feddyn_alpha,
            'epsilon': self.robust_epsilon,
            'robust_tau': self.robust_tau,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'temperature': self.temperature,
            'rank': self.rank,
            'trust_score': self.trust_score,
            'peer_trust': self.peer_trust,
            'k_clusters': self.k_clusters,
            'warmup_rounds': self.warmup_rounds,
            'm_max': self.m_max,
            'support': self.support,
            'threshold': self.threshold,
        }


class FederatedLearningPipeline:
    def __init__(self, config: FLConfig):
        self.config = config
        self.strategy_runtime: Optional[StrategyRuntime] = None
        self.logger: Optional[logging.Logger] = None
        self.detailed_logger: Optional[logging.Logger] = None
        self.log_filename: Optional[str] = None
        self.results_df = pd.DataFrame(columns=['Round', 'Loss', 'Accuracy', 'F1_Score', 'Precision', 'Recall'])
        self.poisoned_clients: List[int] = []
        self.poison_loader: Optional[PoisonedDataLoader] = None
        self.test_dataset = None
        self.eval_batch_size: Optional[int] = None
        self.class_names: List[str] = []
        self.partition_label: Optional[str] = None
        self.current_round = 0
        self.poison_attack: Optional[str] = None
        self.poison_value: Optional[float] = None
        self.poison_ratio: Optional[float] = None
        # Secure Aggregation state
        self.dh_private_keys: Dict[int, int] = {}
        self.dh_public_keys: Dict[int, int] = {}

    def _load_class_metadata(self, partition_label: str, client_count: int) -> (List[str], int):
        class_names_file = os.path.join(
            "data", "partitions", f"{client_count}_client", partition_label, "label_classes.npy"
        )
        if os.path.exists(class_names_file):
            class_names = np.load(class_names_file, allow_pickle=True)
            class_names = [str(name) for name in class_names]
            num_classes = len(class_names)
        else:
            y_test = np.load(os.path.join("data", "y_test.npy"), mmap_mode='r')
            num_classes = len(np.unique(y_test))
            class_names = [f"Class_{i}" for i in range(num_classes)]
            del y_test
        return class_names, num_classes

    def _determine_input_dim(self) -> int:
        sample_X = np.load(os.path.join("data", "X_test.npy"), mmap_mode='r')
        input_dim = sample_X.shape[1]
        del sample_X
        return input_dim

    def _prepare_paths(self, n_clients: int, partition_label: str, client_count: int):
        paths_list = []
        for client_idx in range(n_clients):
            paths = setup_paths(str(client_idx), partition_label, client_count)
            paths_list.append(paths)
        return paths_list

    def _restore_partitions(self, paths_list: List[Dict[str, str]]):
        print(f"{COLORS.OKCYAN}Restoring full partitions...{COLORS.ENDC}")
        restored_count = sum(restore_full_partition(paths) for paths in paths_list)
        if restored_count > 0:
            print(f"{COLORS.OKGREEN}Restored {restored_count} partitions{COLORS.ENDC}")

    def _get_test_dataset(self, num_classes: int):
        if self.test_dataset is None:
            batch_size = self.eval_batch_size or min(self.config.batch_size, 2048)
            self.test_dataset = load_test_dataset(batch_size, num_classes)
        return self.test_dataset

    def _extract_weight_matrices(self, model):
        """Extract weight matrices from Dense layers for frequency aggregation"""
        # Handle wrapper classes (e.g., DenseModel)
        keras_model = model.model if hasattr(model, 'model') else model
        
        weight_matrices = {}
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                # Get kernel weights (out_dim, in_dim) - need to transpose for (out, in) format
                kernel = layer.get_weights()[0]  # Shape: (in_dim, out_dim)
                weight_matrices[layer.name] = kernel.T  # Transpose to (out_dim, in_dim)
        return weight_matrices
    
    def _set_weight_matrices(self, model, weight_matrices):
        """Set weight matrices back to Dense layers after aggregation"""
        # Handle wrapper classes
        keras_model = model.model if hasattr(model, 'model') else model
        
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Dense) and layer.name in weight_matrices:
                # Transpose back to (in_dim, out_dim) for Keras Dense layer
                aggregated_kernel = weight_matrices[layer.name].T
                current_weights = layer.get_weights()
                # Keep bias if it exists
                if len(current_weights) > 1:
                    layer.set_weights([aggregated_kernel, current_weights[1]])
                else:
                    layer.set_weights([aggregated_kernel])

    @tf.function
    def _compute_trust_score_tf(self, client_gradient_flat, global_gradient_flat):
        """
        TensorFlow-compiled trust score computation.
        Trust score = max(0, cos(client_gradient, server_gradient))
        Order: client gradient as reference, server gradient being tested.
        """
        dot_product = tf.reduce_sum(client_gradient_flat * global_gradient_flat)
        norm_client = tf.norm(client_gradient_flat)
        norm_global = tf.norm(global_gradient_flat)
        
        # Avoid division by zero
        cos_sim = tf.cond(
            tf.logical_and(norm_client > 0, norm_global > 0),
            lambda: dot_product / (norm_client * norm_global),
            lambda: tf.constant(0.0)
        )
        
        # ReLU clip: max(0, cos_sim)
        return tf.maximum(0.0, cos_sim)
    
    def _compute_peer_trust_scores(self, client_weights_list, client_indices):
        """
        Compute peer trust scores between adjacent clients (id +-1).
        Each client computes cosine similarity with its left and right neighbors.
        
        Args:
            client_weights_list: List of weight arrays for participating clients
            client_indices: List of client IDs
        
        Returns:
            Dict mapping client_id to (left_score, right_score)
        """
        peer_scores = {}
        n_clients = len(client_indices)
        
        for i, client_id in enumerate(client_indices):
            left_score = None
            right_score = None
            
            # Left neighbor (client_id - 1)
            left_neighbor_id = client_id - 1
            if left_neighbor_id in client_indices:
                left_idx = client_indices.index(left_neighbor_id)
                left_score = self._compute_cosine_similarity(
                    client_weights_list[i],
                    client_weights_list[left_idx]
                )
            
            # Right neighbor (client_id + 1)
            right_neighbor_id = client_id + 1
            if right_neighbor_id in client_indices:
                right_idx = client_indices.index(right_neighbor_id)
                right_score = self._compute_cosine_similarity(
                    client_weights_list[i],
                    client_weights_list[right_idx]
                )
            
            peer_scores[client_id] = (left_score, right_score)
        
        return peer_scores
    
    def _compute_cosine_similarity(self, weights1, weights2):
        """Compute cosine similarity between two weight arrays"""
        flat1 = np.concatenate([w.flatten() for w in weights1])
        flat2 = np.concatenate([w.flatten() for w in weights2])
        
        dot_product = np.dot(flat1, flat2)
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 > 0 and norm2 > 0:
            return float(dot_product / (norm1 * norm2))
        return 0.0
    
    def _init_secure_aggregation_keys(self, n_clients: int):
        """Initialize DH keypairs for all clients in Secure Aggregation"""
        if self.config.strategy != "SecureAggregation":
            return
        
        print(f"{COLORS.OKBLUE}Initializing Secure Aggregation DH key exchange...{COLORS.ENDC}")
        
        # Generate keypair for each client
        from .strategy.SecureAggregation import SecureAggregation
        sa = SecureAggregation()
        
        for client_id in range(n_clients):
            private_key, public_key = sa.generate_dh_keypair()
            self.dh_private_keys[client_id] = private_key
            self.dh_public_keys[client_id] = public_key
        
        self.logger.info(f"Secure Aggregation: Generated {n_clients} DH keypairs")
        print(f"{COLORS.OKGREEN}DH key exchange complete ({n_clients} clients){COLORS.ENDC}")
    
    def _apply_secure_aggregation_mask(self, client_id: int, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Apply pairwise masks for Secure Aggregation"""
        from .strategy.SecureAggregation import SecureAggregation
        sa = SecureAggregation()
        
        # Get all public keys except server (public keys are client-only)
        public_keys = {cid: pk for cid, pk in self.dh_public_keys.items()}
        private_key = self.dh_private_keys[client_id]
        
        # Apply masks
        masked_weights = sa.apply_pairwise_masks(
            client_id,
            weights,
            public_keys,
            private_key
        )
        
        return masked_weights

    def _compute_trust_score(self, client_weights, global_weights):
        """
        Compute FLTrust-style trust score (ReLU clipped cosine similarity).
        Compares client gradient (client_weights - global_weights) vs server gradient.
        
        Note: Since this is called after training but before aggregation,
        we treat the global model as the reference point (previous round's model).
        """
        # Client gradient: new_weights - old_weights (the update)
        client_gradient = [c_w - g_w for c_w, g_w in zip(client_weights, global_weights)]
        client_gradient_flat = tf.constant(
            np.concatenate([g.flatten() for g in client_gradient]), 
            dtype=tf.float32
        )
        
        # Server gradient: for comparison, we use the global model direction
        # (in FLTrust, this would be the server's gradient on root dataset)
        # Here we approximate it as the global weights themselves as directional reference
        global_gradient_flat = tf.constant(
            np.concatenate([g_w.flatten() for g_w in global_weights]),
            dtype=tf.float32
        )
        
        # Compute trust score with TF compilation
        trust_score = self._compute_trust_score_tf(client_gradient_flat, global_gradient_flat)
        
        return float(trust_score.numpy())

    def _run_independent_learning(self, n_clients: int, input_dim: int, num_classes: int, paths_list: List[Dict[str, str]]):
        """
        Run independent learning (None strategy).
        Each client trains for (rounds * epochs) total epochs without federation.
        """
        total_epochs = self.config.rounds * self.config.epochs
        print(f"\n{COLORS.HEADER}INDEPENDENT LEARNING MODE{COLORS.ENDC}")
        print(f"Training each client independently for {total_epochs} epochs ({self.config.rounds} rounds × {self.config.epochs} epochs)")
        log_timestamp(self.logger, f"Independent learning: {total_epochs} total epochs per client")
        
        for client_idx in range(n_clients):
            tf.keras.backend.clear_session()
            print(f"\n{COLORS.BOLD}Client {client_idx}{COLORS.ENDC}")
            log_timestamp(self.logger, f"Client {client_idx} training started")
            
            client_start = time.time()
            
            model = create_model(
                architecture=self.config.model,
                input_dim=input_dim,
                num_classes=num_classes,
                batch_size=self.config.batch_size,
                strategy_runtime=self.strategy_runtime,
                client_id=client_idx,
            )
            
            train_dataset = create_client_dataset(
                paths_list[client_idx]['train_X'],
                paths_list[client_idx]['train_y'],
                input_dim,
                num_classes,
                self.config.batch_size,
            )
            
            print(f"Training for {total_epochs} epochs independently")
            history = model.fit(
                train_dataset,
                epochs=total_epochs,
                verbose=1,
            )
            
            final_loss = float(history.history.get("loss", [0.0])[-1])
            client_time = time.time() - client_start
            
            self.logger.info(f"Client {client_idx}: FinalLoss={final_loss:.4f} (trained {total_epochs} epochs)")
            print(f"{COLORS.WARNING}Client {client_idx}: FinalLoss={final_loss:.4f}{COLORS.ENDC}")
            log_timestamp(self.logger, f"Client {client_idx} completed in {client_time:.2f}s")
            
            # Personalized eval
            metrics = evaluate_model_with_metrics(
                model,
                self.test_dataset,
                num_classes,
                class_names=self.class_names,
                round_num=None,
                strategy_name=self.config.strategy,
                partition_type=self.partition_label,
                collect_details=False,
            )
            test_loss, accuracy, f1, precision, recall, _, _, _, _ = metrics
            
            self.logger.info(
                f"Client {client_idx} | PERSONALIZED | Acc: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Loss: {test_loss:.4f}"
            )
            print(
                f"{COLORS.OKGREEN}Client {client_idx}: Acc={accuracy:.4f}, F1={f1:.4f}, Loss={test_loss:.4f}{COLORS.ENDC}"
            )
            
            del model, train_dataset, history
            aggressive_memory_cleanup()
        
        log_timestamp(self.logger, "Independent learning completed")
        print(f"{COLORS.OKGREEN}Independent learning completed!{COLORS.ENDC}")

    def _train_single_client(
        self,
        client_id: int,
        input_dim: int,
        num_classes: int,
        latest_weights,
        paths,
    ):
        tf.keras.backend.clear_session()
        print(f"\n{COLORS.BOLD}Client {client_id}{COLORS.ENDC}")

        client_start_time = time.time()
        log_timestamp(self.logger, f"Client {client_id} training started")

        # FedKEESS: After warmup, use SSD wrapper with k teachers (excluding own bin)
        if self.config.strategy == "FedKEESS" and self.current_round > self.config.warmup_rounds:
            if hasattr(self, 'fedkeess_models') and self.fedkeess_models is not None:
                from models.fedkeess_wrapper import FedKEESSModel
                
                # Get client's bin assignment
                client_bin = self.fedkeess_bin_assignments[client_id]
                k = len(self.fedkeess_models)
                
                # Count clients per bin
                bin_sizes = {}
                for cid, bid in enumerate(self.fedkeess_bin_assignments):
                    bin_sizes[bid] = bin_sizes.get(bid, 0) + 1
                
                client_bin_size = bin_sizes[client_bin]
                
                # Teacher selection: exclude own bin ONLY if singleton
                if client_bin_size == 1:
                    # Singleton: can't distill from self
                    teacher_indices = [i for i in range(k) if i != client_bin]
                    print(f"  Client {client_id} in singleton bin {client_bin}, distilling from {len(teacher_indices)} other bins")
                else:
                    # Multi-client: distill from all bins (including own, since aggregation differs)
                    teacher_indices = list(range(k))
                    print(f"  Client {client_id} in bin {client_bin} (size={client_bin_size}), distilling from all {len(teacher_indices)} bins")
                
                teacher_models = [self.fedkeess_models[i] for i in teacher_indices]
                teacher_M_class = [self.fedkeess_M_class[i] for i in teacher_indices]
                
                # Create base model
                base_model = create_model(
                    architecture=self.config.model,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    batch_size=self.config.batch_size,
                    strategy_runtime=None,
                    client_id=None,
                )
                
                # Wrap with FedKEESS SSD (k-1 teachers, excluding own bin)
                model = FedKEESSModel(
                    base_model=base_model,
                    k_teachers=len(teacher_models),
                    num_classes=num_classes,
                    m_max=self.config.m_max,
                )
                
                # Set teachers (all OTHER bins' models + their M_class)
                if len(teacher_models) > 0:
                    model.set_teachers(
                        teacher_weights_list=teacher_models,
                        M_class_list=teacher_M_class,
                    )
                
                # Initialize client's model with its assigned bin's model
                model.set_weights(self.fedkeess_models[client_bin])
            else:
                model = create_model(
                    architecture=self.config.model,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    batch_size=self.config.batch_size,
                    strategy_runtime=self.strategy_runtime,
                    client_id=client_id,
                )
        else:
            model = create_model(
                architecture=self.config.model,
                input_dim=input_dim,
                num_classes=num_classes,
                batch_size=self.config.batch_size,
                strategy_runtime=self.strategy_runtime,
                client_id=client_id,
            )

        if latest_weights is not None:
            model.set_weights(latest_weights)
        
        if self.config.strategy == "FedMLB" and hasattr(model, 'set_global_weights') and latest_weights is not None:
            model.set_global_weights(latest_weights)

        poisoned = client_id in self.poisoned_clients
        poison_loader = self.poison_loader if poisoned and self.poison_attack == "label_flip" else None
        if poisoned and self.poison_attack == "label_flip":
            print(f"  \u26a0\ufe0f  POISONED CLIENT - Labels flipped")
        if poisoned and self.poison_attack == "gradient_scale":
            scale = self.poison_value or 1.0
            print(f"  \u26a0\ufe0f  POISONED CLIENT - Scaling gradients x{scale:.1f}")

        train_dataset = create_client_dataset(
            paths['train_X'],
            paths['train_y'],
            input_dim,
            num_classes,
            self.config.batch_size,
            poison_loader=poison_loader
        )

        X_train_mmap = np.load(paths['train_X'], mmap_mode='r')
        sample_size = X_train_mmap.shape[0]
        del X_train_mmap

        print(f"Training client {client_id} for {self.config.epochs} epochs")
        
        if self.config.strategy in ["FedSSD1", "FedSSDexp", "FedSSD2"] and self.strategy_runtime and hasattr(self.strategy_runtime.client_strategy, 'train'):
            if latest_weights is not None:
                global_model = create_model(
                    architecture=self.config.model,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    batch_size=self.config.batch_size,
                    strategy_runtime=self.strategy_runtime,
                    client_id=-1,
                )
                global_model.set_weights(latest_weights)
            else:
                global_model = None
            
            self.strategy_runtime.client_strategy.train(
                model=model,
                global_model=global_model,
                train_X_path=paths['train_X'],
                train_y_path=paths['train_y'],
                input_dim=input_dim,
                num_classes=num_classes,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
            )
            loss = 0.0
            history = None
        else:
            history = model.fit(
                train_dataset,
                epochs=self.config.epochs,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='loss',
                        patience=3,
                        restore_best_weights=True,
                        verbose=1
                    )
                ],
                verbose=1,
            )
            loss = float(history.history.get("loss", [0.0])[-1])

        client_train_time = time.time() - client_start_time
        log_timestamp(self.logger, f"Client {client_id} completed in {client_train_time:.2f}s")

        self.logger.info(f"Client {client_id}: Loss={loss:.4f}")
        print(f"{COLORS.WARNING}Client {client_id}: Loss={loss:.4f}{COLORS.ENDC}")

        # Compute trust score if flag is set and not None strategy
        if self.config.trust_score and self.config.strategy != "None" and latest_weights is not None:
            trust_score = self._compute_trust_score(model.get_weights(), latest_weights)
            self.logger.info(f"Client {client_id}: TrustScore={trust_score:.4f}")
            print(f"{COLORS.OKCYAN}Client {client_id}: TrustScore={trust_score:.4f}{COLORS.ENDC}")

        if self.config.personalized_eval:
            self._evaluate_personalized_client(model, client_id, num_classes)

        trained_weights = model.get_weights()
        poisoned_weights = self._apply_poison_to_weights(trained_weights, latest_weights, client_id)

        # Apply Secure Aggregation masking if enabled
        if self.config.strategy == "SecureAggregation":
            poisoned_weights = self._apply_secure_aggregation_mask(
                client_id,
                poisoned_weights
            )

        weights_file = os.path.join(
            self.config.weights_cache_dir,
            f"client_{client_id}_round_{self.current_round}_weights.pkl"
        )

        # For FLTrust, save model update (new_weights - old_weights)
        if self.config.strategy == "FLTrust":
            if latest_weights is None:
                update = poisoned_weights
            else:
                update = [new_w - old_w for new_w, old_w in zip(poisoned_weights, latest_weights)]
            with open(weights_file, 'wb') as file_handler:
                pickle.dump(update, file_handler)
        elif self.config.strategy == "FedDyn":
            feddyn_data = model.get_feddyn_update()
            if 'weights' in feddyn_data:
                feddyn_data['weights'] = poisoned_weights
            with open(weights_file, 'wb') as file_handler:
                pickle.dump(feddyn_data, file_handler)
        else:
            with open(weights_file, 'wb') as file_handler:
                pickle.dump(poisoned_weights, file_handler)

        del model, train_dataset, history
        aggressive_memory_cleanup()

        return weights_file, sample_size, loss

    def _load_client_weights(self, weights_files: List[str]):
        weights_list = []
        for weights_file in weights_files:
            with open(weights_file, 'rb') as file_handler:
                saved_data = pickle.load(file_handler)
                if isinstance(saved_data, dict) and 'weights' in saved_data:
                    weights_list.append(saved_data['weights'])
                else:
                    weights_list.append(saved_data)
        return weights_list

    def _aggregate(self, weights_list, sample_sizes, participating_clients, global_update=None, models=None):
        aggregator = self.strategy_runtime.aggregator
        
        # FedoRA requires weight_matrices extraction from Dense layers
        if self.config.strategy == "FedoRA":
            weight_matrices_list = [self._extract_weight_matrices(model) for model in models]
            aggregated_weights, global_weight_matrices = aggregator.aggregate(
                weights_list, 
                sample_sizes, 
                delta_weights_list=weight_matrices_list  # Reuse parameter name for compatibility
            )
            return aggregated_weights, global_weight_matrices
        # FLTrust requires global_update parameter
        elif self.config.strategy == "FLTrust":
            return aggregator.aggregate(weights_list, sample_sizes, global_update=global_update)
        elif self.strategy_runtime.requires_participant_ids:
            return aggregator.aggregate(weights_list, sample_sizes, participating_clients)
        else:
            return aggregator.aggregate(weights_list, sample_sizes)
    
    def _train_server_on_root_dataset(self, current_weights, input_dim, num_classes):
        tf.keras.backend.clear_session()
        
        model = create_model(
            architecture=self.config.model,
            input_dim=input_dim,
            num_classes=num_classes,
            batch_size=self.config.batch_size,
            strategy_runtime=self.strategy_runtime,
            client_id=None,
        )
        
        if current_weights is not None:
            model.set_weights(current_weights)
        
        partition_path = os.path.join("data", "partitions", f"{self.n_clients}_client", self.partition_label)
        
        all_public_X = []
        all_public_y = []
        for client_idx in range(self.n_clients):
            public_X = np.load(os.path.join(partition_path, f"client_{client_idx}_X_public.npy"))
            public_y = np.load(os.path.join(partition_path, f"client_{client_idx}_y_public.npy"))
            all_public_X.append(public_X)
            all_public_y.append(public_y)
        
        combined_X = np.concatenate(all_public_X, axis=0)
        combined_y = np.concatenate(all_public_y, axis=0)
        
        root_dataset = tf.data.Dataset.from_tensor_slices((combined_X, combined_y))
        root_dataset = root_dataset.map(lambda x, y: (x, tf.keras.utils.to_categorical(y, num_classes)))
        root_dataset = root_dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        print(f"  [SERVER] Training on root dataset ({len(combined_X)} samples) for {self.config.root_iterations} iterations")
        
        model.fit(root_dataset, epochs=self.config.root_iterations, verbose=0)
        
        new_weights = model.get_weights()
        if current_weights is None:
            global_update = new_weights
        else:
            global_update = [new_w - old_w for new_w, old_w in zip(new_weights, current_weights)]
        
        del model, root_dataset, combined_X, combined_y
        aggressive_memory_cleanup()
        
        return global_update

    def _apply_poison_to_weights(self, weights, latest_weights, client_id):
        if self.poison_attack == "gradient_scale" and latest_weights is not None and client_id in self.poisoned_clients:
            scale = self.poison_value or 1.0
            scaled = []
            for new_w, old_w in zip(weights, latest_weights):
                scaled.append(old_w + (new_w - old_w) * scale)
            return scaled
        return weights

    def _evaluate_personalized_client(self, model, client_id: int, num_classes: int):
        test_dataset = self._get_test_dataset(num_classes)
        result = evaluate_model_with_metrics(
            model,
            test_dataset,
            num_classes,
            self.class_names,
            self.current_round,
            self.config.strategy,
            self.partition_label,
            collect_details=False,
        )
        test_loss, accuracy, f1_value, precision, recall = result[:5]
        self.logger.info(
            "Round %s | Client %s | PERSONALIZED | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Loss: %.4f",
            self.current_round,
            client_id,
            accuracy,
            f1_value,
            precision,
            recall,
            test_loss,
        )
        print(
            f"{COLORS.OKBLUE}Client {client_id} personalized eval -> Acc={accuracy:.4f}, F1={f1_value:.4f}, Loss={test_loss:.4f}{COLORS.ENDC}"
        )

    def _evaluate_global_model(
        self,
        aggregated_weights,
        input_dim,
        num_classes,
        class_names,
        round_num,
        partition_label
    ):
        tf.keras.backend.clear_session()
        eval_model = create_model(
            architecture=self.config.model,
            input_dim=input_dim,
            num_classes=num_classes,
            batch_size=self.config.batch_size,
            strategy_runtime=self.strategy_runtime,
            client_id=None,
        )
        eval_model.set_weights(aggregated_weights)

        test_dataset = self._get_test_dataset(num_classes)

        if round_num == self.config.rounds:
            print(f"{COLORS.HEADER}FINAL GLOBAL EVALUATION{COLORS.ENDC}")
        else:
            print(f"\n{COLORS.OKCYAN}Evaluating global model{COLORS.ENDC}")

        result = evaluate_model_with_metrics(
            eval_model,
            test_dataset,
            num_classes,
            class_names,
            round_num,
            self.config.strategy,
            partition_label,
        )
        test_loss, accuracy, f1_score_value, precision, recall = result[:5]
        per_class_metrics = result[5] if len(result) > 5 else None
        confusion_mat = result[6] if len(result) > 6 else None
        class_report = result[7] if len(result) > 7 else ""

        print(
            f"{COLORS.OKGREEN}[GLOBAL] Acc={accuracy:.4f}, Loss={test_loss:.4f}, F1={f1_score_value:.4f}{COLORS.ENDC}"
        )

        self.logger.info(
            f"Round {round_num} | GLOBAL | Acc: {accuracy:.4f} | F1: {f1_score_value:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Loss: {test_loss:.4f}"
        )
        if class_report and self.detailed_logger is not None:
            self.detailed_logger.info(f"Round {round_num}\n{class_report}")

        del eval_model, test_dataset
        aggressive_memory_cleanup()

        return test_loss, accuracy, f1_score_value, precision, recall, per_class_metrics, confusion_mat

    def run(self):
        configure_gpu_memory()
        os.makedirs(self.config.weights_cache_dir, exist_ok=True)

        self.strategy_runtime = build_strategy(
            self.config.strategy,
            self.config.to_strategy_params(),
        )

        partition_label, client_count = parse_partition_type(self.config.partition_type)
        n_clients = min(self.config.n_clients, client_count)

        # Prepare poison suffix for logging
        poison_suffix = ""
        if self.config.poison:
            poison_suffix = self.config.poison.replace(" ", "_")

        extra_log_tokens = self.strategy_runtime.extra_log_tokens() if hasattr(self.strategy_runtime, 'extra_log_tokens') else {}

        self.logger, self.log_filename, self.detailed_logger = setup_logger(
            n_clients,
            partition_label,
            strategy_name=self.config.strategy,
            extra_tokens=list(extra_log_tokens.values()),
            poison_suffix=poison_suffix,
        )
        excel_filename = self.log_filename.replace('.log', '.xlsx')

        class_names, num_classes = self._load_class_metadata(partition_label, client_count)
        self.class_names = class_names
        self.eval_batch_size = min(self.config.batch_size, 2048)
        self.test_dataset = load_test_dataset(self.eval_batch_size, num_classes)
        input_dim = self._determine_input_dim()

        paths_list = self._prepare_paths(n_clients, partition_label, client_count)
        
        # Skip partition restoration for strategies that need separate public/validation data
        if self.config.strategy not in ["FLTrust", "FedSSD1", "FedSSD2"]:
            self._restore_partitions(paths_list)
        
        self.n_clients = n_clients
        self.partition_label = partition_label

        # Setup poisoning if configured
        attack_type, poison_value, poison_ratio = parse_poison_config(self.config.poison)
        self.poison_attack = attack_type
        self.poison_value = poison_value
        self.poison_ratio = poison_ratio
        if attack_type:
            self.poisoned_clients = get_or_create_poisoned_clients(
                partition_label, attack_type, poison_value, poison_ratio, n_clients
            )
            if attack_type == "label_flip":
                self.poison_loader = PoisonedDataLoader(attack_type, num_classes)
            log_timestamp(self.logger, f"POISONING ENABLED: {attack_type} attack value={poison_value} on {len(self.poisoned_clients)} clients")
            log_timestamp(self.logger, f"Poisoned clients: {self.poisoned_clients}")
            print(f"\n  POISONING ENABLED: {attack_type} attack (value={poison_value})")
            print(f"    Poisoned clients: {self.poisoned_clients} ({len(self.poisoned_clients)}/{n_clients})")

        # Initialize Secure Aggregation DH keys if needed
        self._init_secure_aggregation_keys(n_clients)
        
        # Initialize FedKEESS state
        self.fedkeess_models = None
        self.fedkeess_M_class = None
        self.fedkeess_bin_assignments = None

        latest_weights = None
        pipeline_start_time = time.time()
        log_timestamp(self.logger, "=== FL PIPELINE STARTED ===")
        log_timestamp(self.logger, f"Strategy: {self.config.strategy}, Clients: {n_clients}, Rounds: {self.config.rounds}")
        round_times: List[float] = []
        
        # Special handling for None strategy (independent learning)
        if self.config.strategy == "None":
            self._run_independent_learning(n_clients, input_dim, num_classes, paths_list)
            return
        
        if self.config.strategy == "FedMLB":
            print(f"\n{COLORS.OKCYAN}Initializing FedMLB server model{COLORS.ENDC}")
            log_timestamp(self.logger, "Initializing server model")
            init_model = create_model(
                architecture=self.config.model,
                input_dim=input_dim,
                num_classes=num_classes,
                batch_size=self.config.batch_size,
                strategy_runtime=self.strategy_runtime,
                client_id=None,
            )
            latest_weights = init_model.get_weights()
            del init_model
            aggressive_memory_cleanup()
            print(f"{COLORS.OKGREEN}Server model initialized with random weights{COLORS.ENDC}")
        
        if self.config.strategy == "FedSSD1":
            print(f"\n{COLORS.OKCYAN}Initializing FedSSD1 server model{COLORS.ENDC}")
            log_timestamp(self.logger, "Initializing server model")
            init_model = create_model(
                architecture=self.config.model,
                input_dim=input_dim,
                num_classes=num_classes,
                batch_size=self.config.batch_size,
                strategy_runtime=self.strategy_runtime,
                client_id=None,
            )
            latest_weights = init_model.get_weights()
            del init_model
            aggressive_memory_cleanup()
            print(f"{COLORS.OKGREEN}Server model initialized with random weights{COLORS.ENDC}")
            
            # Initialize M_class (will be updated before using)
            self.strategy_runtime.aggregator.M_class = np.ones(num_classes, dtype=np.float32)
            print(f"{COLORS.OKCYAN}M_class initialized to ones (will be computed from CMs in each round){COLORS.ENDC}")

        if self.config.strategy == "FedSSDexp":
            print(f"\n{COLORS.OKCYAN}Initializing FedSSDexp server model{COLORS.ENDC}")
            log_timestamp(self.logger, "Initializing server model")
            init_model = create_model(
                architecture=self.config.model,
                input_dim=input_dim,
                num_classes=num_classes,
                batch_size=self.config.batch_size,
                strategy_runtime=self.strategy_runtime,
                client_id=None,
            )
            latest_weights = init_model.get_weights()
            del init_model
            aggressive_memory_cleanup()
            print(f"{COLORS.OKGREEN}Server model initialized with random weights{COLORS.ENDC}")
            
            self.strategy_runtime.aggregator.M_class = np.ones(num_classes, dtype=np.float32)
            print(f"{COLORS.OKCYAN}M_class initialized to ones (will be computed from full training data in each round){COLORS.ENDC}")

        if self.config.strategy == "FedSSD2":
            print(f"\n{COLORS.OKCYAN}Initializing FedSSD2 server model{COLORS.ENDC}")
            log_timestamp(self.logger, "Initializing server model")
            init_model = create_model(
                architecture=self.config.model,
                input_dim=input_dim,
                num_classes=num_classes,
                batch_size=self.config.batch_size,
                strategy_runtime=self.strategy_runtime,
                client_id=None,
            )
            latest_weights = init_model.get_weights()
            del init_model
            aggressive_memory_cleanup()
            print(f"{COLORS.OKGREEN}Server model initialized with random weights{COLORS.ENDC}")
            
            self.strategy_runtime.aggregator.M_class = np.ones(num_classes, dtype=np.float32)
            print(f"{COLORS.OKCYAN}M_class initialized to ones (will be computed from pseudo-CM in each round){COLORS.ENDC}")
            
            # Load public unlabeled dataset
            partition_path = os.path.join("data", "partitions", f"{n_clients}_client", partition_label)
            all_public_X = []
            for client_idx in range(n_clients):
                public_X = np.load(os.path.join(partition_path, f"client_{client_idx}_X_public.npy"))
                all_public_X.append(public_X)
            
            self.fedssd2_public_X = np.concatenate(all_public_X, axis=0).astype(np.float32)
            print(f"{COLORS.OKGREEN}Loaded public unlabeled dataset ({self.fedssd2_public_X.shape[0]:,} samples){COLORS.ENDC}")
            del all_public_X
            aggressive_memory_cleanup()

        for round_num in range(1, self.config.rounds + 1):
            self.current_round = round_num
            round_start_time = time.time()

            self.logger.info(f"Round {round_num}/{self.config.rounds}")
            print(f"\n{COLORS.HEADER}Round {round_num}/{self.config.rounds}{COLORS.ENDC}")
            log_timestamp(self.logger, f"--- Round {round_num} started ---")

            # FedSSD1: Stage 1 - Collect confusion matrices before training
            if self.config.strategy == "FedSSD1" and latest_weights is not None:
                print(f"\n{COLORS.OKCYAN}[FedSSD1 Stage 1] Collecting confusion matrices from clients{COLORS.ENDC}")
                log_timestamp(self.logger, "FedSSD1 Stage 1: Confusion matrix collection")
                
                global_model_for_cm = create_model(
                    architecture=self.config.model,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    batch_size=self.config.batch_size,
                    strategy_runtime=self.strategy_runtime,
                    client_id=-1,
                )
                global_model_for_cm.set_weights(latest_weights)
                
                # Extract Keras model if wrapped
                keras_model = global_model_for_cm.model if hasattr(global_model_for_cm, 'model') else global_model_for_cm
                
                confusion_matrices = []
                support_counts_list = []
                
                for client_idx in range(n_clients):
                    # Load client public slice (validation set representing local distribution)
                    val_dataset = create_client_dataset(
                        paths_list[client_idx]['public_X'],
                        paths_list[client_idx]['public_y'],
                        input_dim,
                        num_classes,
                        self.config.batch_size,
                    )
                    
                    # Compute confusion matrix incrementally
                    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
                    class_counts = np.zeros(num_classes, dtype=np.int32)
                    
                    for batch_X, batch_y in val_dataset:
                        predictions = keras_model(batch_X, training=False)
                        pred_labels = tf.argmax(predictions, axis=1).numpy()
                        true_labels = tf.argmax(batch_y, axis=1).numpy() if batch_y.shape.ndims > 1 else batch_y.numpy()
                        
                        # Update confusion matrix incrementally
                        for true_label, pred_label in zip(true_labels, pred_labels):
                            cm[true_label, pred_label] += 1
                            class_counts[true_label] += 1
                    
                    # Normalize to proportions
                    cm_normalized = np.zeros_like(cm, dtype=np.float32)
                    for k in range(num_classes):
                        row_sum = cm[k, :].sum()
                        if row_sum > 0:
                            cm_normalized[k, :] = cm[k, :].astype(np.float32) / row_sum
                    
                    confusion_matrices.append(cm_normalized)
                    support_counts_list.append(class_counts)
                    
                    print(f"  Client {client_idx}: CM shape={cm.shape}, support={class_counts.sum()}")
                
                del global_model_for_cm, val_dataset
                aggressive_memory_cleanup()
                
                # Aggregate confusion matrices
                print(f"{COLORS.OKCYAN}[FedSSD1 Stage 1] Aggregating confusion matrices{COLORS.ENDC}")
                aggregated_cm = self.strategy_runtime.aggregator.aggregate_confusion_matrices(
                    confusion_matrices=confusion_matrices,
                    support_counts=support_counts_list if self.config.support else None,
                    num_classes=num_classes,
                )
                
                # Compute M_class
                M_class = self.strategy_runtime.aggregator.compute_M_class(
                    confusion_matrix=aggregated_cm,
                    num_classes=num_classes,
                )
                
                print(f"{COLORS.OKGREEN}  M_class computed: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}{COLORS.ENDC}")
                log_timestamp(self.logger, f"FedSSD1 M_class: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}")

            # FedSSDexp: Stage 1 - Compute M_class from full training data
            if self.config.strategy == "FedSSDexp" and latest_weights is not None:
                print(f"\n{COLORS.OKCYAN}[FedSSDexp Stage 1] Computing M_class from client training data{COLORS.ENDC}")
                log_timestamp(self.logger, "FedSSDexp Stage 1: M_class computation")
                
                global_model_for_mclass = create_model(
                    architecture=self.config.model,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    batch_size=self.config.batch_size,
                    strategy_runtime=self.strategy_runtime,
                    client_id=-1,
                )
                global_model_for_mclass.set_weights(latest_weights)
                
                keras_model = global_model_for_mclass.model if hasattr(global_model_for_mclass, 'model') else global_model_for_mclass
                
                M_class_list = []
                support_counts_list = []
                
                for client_idx in range(n_clients):
                    X_train = np.load(paths_list[client_idx]['train_X'], mmap_mode='r')
                    y_train = np.load(paths_list[client_idx]['train_y'], mmap_mode='r')
                    total_samples = X_train.shape[0]
                    
                    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
                    
                    for start_idx in range(0, total_samples, self.config.batch_size):
                        end_idx = min(start_idx + self.config.batch_size, total_samples)
                        batch_X = np.array(X_train[start_idx:end_idx], dtype=np.float32)
                        batch_y = np.array(y_train[start_idx:end_idx], dtype=np.int32)
                        
                        predictions = keras_model(batch_X, training=False)
                        pred_labels = tf.argmax(predictions, axis=1).numpy()
                        true_labels = batch_y if len(batch_y.shape) == 1 else np.argmax(batch_y, axis=1)
                        
                        for true_label, pred_label in zip(true_labels, pred_labels):
                            cm[true_label, pred_label] += 1
                    
                    del X_train, y_train
                    
                    M_class, class_counts = self.strategy_runtime.aggregator.compute_M_class_from_cm(
                        confusion_matrix=cm,
                        num_classes=num_classes,
                    )
                    
                    M_class_list.append(M_class)
                    support_counts_list.append(class_counts)
                    
                    print(f"  Client {client_idx}: M_class min={M_class.min():.4f}, max={M_class.max():.4f}, support={class_counts.sum()}")
                
                del global_model_for_mclass
                aggressive_memory_cleanup()
                
                print(f"{COLORS.OKCYAN}[FedSSDexp Stage 1] Aggregating M_class values{COLORS.ENDC}")
                M_class = self.strategy_runtime.aggregator.aggregate_M_class_values(
                    client_M_class_list=M_class_list,
                    client_class_counts_list=support_counts_list,
                    num_classes=num_classes,
                )
                
                print(f"{COLORS.OKGREEN}  M_class aggregated: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}{COLORS.ENDC}")
                log_timestamp(self.logger, f"FedSSDexp M_class: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}")

            # FedSSD2: Stage 1 - Pseudo-labeling and pseudo-CM computation
            if self.config.strategy == "FedSSD2" and latest_weights is not None:
                print(f"\n{COLORS.OKCYAN}[FedSSD2 Stage 1] Collecting client predictions on public data{COLORS.ENDC}")
                log_timestamp(self.logger, "FedSSD2 Stage 1: Pseudo-labeling")
                
                prediction_files = []
                
                for client_idx in range(n_clients):
                    model = create_model(
                        architecture=self.config.model,
                        input_dim=input_dim,
                        num_classes=num_classes,
                        batch_size=self.config.batch_size,
                        strategy_runtime=self.strategy_runtime,
                        client_id=client_idx,
                    )
                    model.set_weights(latest_weights)
                    
                    keras_model = model.model if hasattr(model, 'model') else model
                    
                    predictions = []
                    for start_idx in range(0, len(self.fedssd2_public_X), self.config.batch_size):
                        end_idx = min(start_idx + self.config.batch_size, len(self.fedssd2_public_X))
                        batch_X = self.fedssd2_public_X[start_idx:end_idx]
                        batch_pred = keras_model(batch_X, training=False).numpy()
                        predictions.append(batch_pred)
                    
                    client_predictions = np.vstack(predictions)
                    
                    # Save predictions to disk
                    pred_file = os.path.join('temp_weights', f'fedssd2_pred_client_{client_idx}.npy')
                    np.save(pred_file, client_predictions)
                    prediction_files.append(pred_file)
                    
                    print(f"  Client {client_idx}: Predicted {client_predictions.shape[0]} public samples (saved to disk)")
                    
                    del model, keras_model, predictions, client_predictions
                    aggressive_memory_cleanup()
                
                # Load predictions and vote for pseudo-labels
                print(f"{COLORS.OKCYAN}[FedSSD2 Stage 1] Voting for pseudo-labels (threshold={self.config.threshold}){COLORS.ENDC}")
                all_client_predictions = [np.load(f) for f in prediction_files]
                pseudo_labels = self.strategy_runtime.aggregator.hard_label_vote(
                    all_client_predictions,
                    num_classes,
                    threshold=self.config.threshold
                )
                del all_client_predictions
                aggressive_memory_cleanup()
                
                # Filter out unlabeled samples (marked as num_classes)
                labeled_mask = pseudo_labels < num_classes
                n_labeled = labeled_mask.sum()
                n_unlabeled = len(pseudo_labels) - n_labeled
                
                print(f"  Pseudo-labels: {n_labeled:,} labeled, {n_unlabeled:,} unlabeled (kept unlabeled)")
                log_timestamp(self.logger, f"FedSSD2 pseudo-labeling: {n_labeled}/{len(pseudo_labels)} samples labeled")
                
                if n_labeled == 0:
                    print(f"{COLORS.WARNING}No samples exceeded voting threshold - skipping Stage 1{COLORS.ENDC}")
                    log_timestamp(self.logger, "FedSSD2 Stage 1 skipped: no confident pseudo-labels")
                else:
                    # Use only labeled samples
                    X_public_labeled = self.fedssd2_public_X[labeled_mask]
                    pseudo_labels_filtered = pseudo_labels[labeled_mask]
                    
                    # Train global model on pseudo-labeled public data
                    print(f"{COLORS.OKCYAN}[FedSSD2 Stage 1] Training global model on {n_labeled:,} pseudo-labeled samples{COLORS.ENDC}")
                    global_model_for_pseudo = create_model(
                        architecture=self.config.model,
                        input_dim=input_dim,
                        num_classes=num_classes,
                        batch_size=self.config.batch_size,
                        strategy_runtime=self.strategy_runtime,
                        client_id=-1,
                    )
                    global_model_for_pseudo.set_weights(latest_weights)
                    
                    pseudo_y_onehot = tf.keras.utils.to_categorical(pseudo_labels_filtered, num_classes).astype(np.float32)
                    pseudo_dataset = tf.data.Dataset.from_tensor_slices((X_public_labeled, pseudo_y_onehot))
                    pseudo_dataset = pseudo_dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
                    
                    global_model_for_pseudo.fit(pseudo_dataset, epochs=1, verbose=0)
                    
                    # Compute pseudo-confusion matrix (using only labeled samples)
                    print(f"{COLORS.OKCYAN}[FedSSD2 Stage 1] Computing pseudo-confusion matrix{COLORS.ENDC}")
                    pseudo_cm = self.strategy_runtime.aggregator.compute_pseudo_confusion_matrix(
                        model=global_model_for_pseudo,
                        X_public=X_public_labeled,
                        pseudo_labels=pseudo_labels_filtered,
                        num_classes=num_classes,
                        batch_size=self.config.batch_size,
                    )
                    
                    # Compute M_class from pseudo-CM
                    M_class = self.strategy_runtime.aggregator.compute_M_class_from_cm(
                        confusion_matrix=pseudo_cm,
                        num_classes=num_classes,
                    )
                    
                    print(f"{COLORS.OKGREEN}  Pseudo-CM computed, M_class: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}{COLORS.ENDC}")
                    log_timestamp(self.logger, f"FedSSD2 M_class: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}")
                    
                    del global_model_for_pseudo, pseudo_dataset, pseudo_y_onehot, X_public_labeled, pseudo_labels_filtered, pseudo_cm, M_class
                
                # Clean up temporary prediction files
                for pred_file in prediction_files:
                    if os.path.exists(pred_file):
                        os.remove(pred_file)
                
                del pseudo_labels, labeled_mask, prediction_files
                aggressive_memory_cleanup()

            weights_files = []
            sample_sizes: List[int] = []
            client_losses: List[float] = []
            participating_clients: List[int] = []

            for client_idx in range(n_clients):
                weights_file, sample_size, loss = self._train_single_client(
                    client_id=client_idx,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    latest_weights=latest_weights,
                    paths=paths_list[client_idx],
                )
                weights_files.append(weights_file)
                sample_sizes.append(sample_size)
                client_losses.append(loss)
                participating_clients.append(client_idx)

            if not participating_clients:
                break

            weights_list = self._load_client_weights(weights_files)
            
            # Compute peer trust scores if enabled
            if self.config.peer_trust:
                peer_scores = self._compute_peer_trust_scores(weights_list, participating_clients)
                self.logger.info(f"\n=== Peer Trust Scores (Round {round_num}) ===")
                for client_id in participating_clients:
                    left_score, right_score = peer_scores[client_id]
                    left_str = f"{left_score:.4f}" if left_score is not None else "N/A"
                    right_str = f"{right_score:.4f}" if right_score is not None else "N/A"
                    self.logger.info(f"Client {client_id}: Left={left_str}, Right={right_str}")
                    print(f"{COLORS.OKCYAN}Client {client_id} PeerTrust: Left={left_str}, Right={right_str}{COLORS.ENDC}")
            
            # For FedoRA, reload models to extract weight matrices from Dense layers
            models_list = None
            if self.config.strategy == "FedoRA":
                models_list = []
                model_arch = self.config.model
                batch_size = self.config.batch_size
                strategy_runtime = self.strategy_runtime
                for weights_file in weights_files:
                    model = create_model(model_arch, input_dim, num_classes, batch_size, strategy_runtime, client_id=None)
                    saved_data = np.load(weights_file, allow_pickle=True)
                    model.set_weights(saved_data)
                    models_list.append(model)
            
            # For FLTrust, compute server update on root dataset
            global_update = None
            if self.config.strategy == "FLTrust":
                print(f"\n{COLORS.OKBLUE}Training server on root dataset{COLORS.ENDC}")
                global_update = self._train_server_on_root_dataset(latest_weights, input_dim, num_classes)
            
            print(f"\n{COLORS.OKBLUE}Aggregating updates ({self.config.strategy}){COLORS.ENDC}")
            
            # FedKEESS: Handle clustering and multi-model aggregation
            if self.config.strategy == "FedKEESS" and round_num > self.config.warmup_rounds:
                aggregated_result = self._aggregate(weights_list, sample_sizes, participating_clients, global_update, models_list)
                
                # Extract k global models and bin assignments
                global_models_list = aggregated_result['global_models']
                bin_assignments = aggregated_result['bin_assignments']
                k = len(global_models_list)
                
                print(f"{COLORS.OKCYAN}FedKEESS: {k} clusters generated{COLORS.ENDC}")
                log_timestamp(self.logger, f"FedKEESS: Clustered into {k} bins")
                
                # Log bin distribution
                bin_counts = {}
                for client_id, bin_id in enumerate(bin_assignments):
                    bin_counts[bin_id] = bin_counts.get(bin_id, 0) + 1
                bin_dist = ", ".join([f"Bin {i}: {bin_counts.get(i, 0)} clients" for i in range(k)])
                print(f"{COLORS.OKCYAN}  Distribution: {bin_dist}{COLORS.ENDC}")
                log_timestamp(self.logger, f"Bin distribution: {bin_dist}")
                
                # Log detailed bin assignments
                bin_members = {i: [] for i in range(k)}
                for client_id, bin_id in enumerate(bin_assignments):
                    bin_members[bin_id].append(client_id)
                for bin_id in range(k):
                    members_str = ", ".join([str(cid) for cid in bin_members[bin_id]])
                    print(f"{COLORS.OKCYAN}  Bin {bin_id}: clients [{members_str}]{COLORS.ENDC}")
                    log_timestamp(self.logger, f"Bin {bin_id} members: [{members_str}]")
                
                # Collect confusion matrices from each client for all k models
                model_arch = self.config.model
                batch_size = self.config.batch_size
                strategy_runtime = self.strategy_runtime
                
                confusion_matrices_per_client = []
                for client_idx in range(n_clients):
                    client_confusion_matrices = []
                    for model_idx, global_model_weights in enumerate(global_models_list):
                        # Create temp model and set weights
                        temp_model = create_model(
                            architecture=model_arch,
                            input_dim=input_dim,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            strategy_runtime=strategy_runtime,
                            client_id=client_idx,
                        )
                        temp_model.set_weights(global_model_weights)
                        
                        # Load client's private dataset
                        test_dataset = create_client_dataset(
                            paths_list[client_idx]['train_X'],
                            paths_list[client_idx]['train_y'],
                            input_dim,
                            num_classes,
                            batch_size,
                        )
                        
                        # Compute confusion matrix
                        y_true = []
                        y_pred = []
                        for batch_X, batch_y in test_dataset:
                            predictions = temp_model.predict(batch_X, verbose=0)
                            y_pred.extend(np.argmax(predictions, axis=1))
                            y_true.extend(np.argmax(batch_y.numpy(), axis=1) if batch_y.numpy().ndim > 1 else batch_y.numpy())
                        
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
                        client_confusion_matrices.append(cm)
                        del temp_model
                    
                    confusion_matrices_per_client.append(client_confusion_matrices)
                    aggressive_memory_cleanup()
                
                # Aggregate confusion matrices per bin
                aggregated_confusion_matrices = self.strategy_runtime.aggregate_confusion_matrices(
                    confusion_matrices_per_client,
                    bin_assignments,
                    k,
                    num_classes,
                )
                
                # Compute M_class per bin from aggregated confusion matrices
                M_class_per_bin = []
                for bin_idx in range(k):
                    cm = aggregated_confusion_matrices[bin_idx]
                    M_class = self.strategy_runtime.compute_ssd_weights(cm, num_classes)
                    M_class_per_bin.append(M_class)
                
                # Store for next round (clients will use these k models and M_class values for SSD)
                self.fedkeess_models = global_models_list
                self.fedkeess_M_class = M_class_per_bin
                self.fedkeess_bin_assignments = bin_assignments
                
                # Evaluate all k global models
                print(f"\n{COLORS.OKBLUE}Evaluating {k} global models{COLORS.ENDC}")
                log_timestamp(self.logger, f"Evaluating {k} FedKEESS global models")
                
                bin_metrics = []
                for bin_idx, bin_weights in enumerate(global_models_list):
                    print(f"{COLORS.OKCYAN}  Evaluating Bin {bin_idx} model...{COLORS.ENDC}")
                    metrics = self._evaluate_global_model(
                        bin_weights,
                        input_dim,
                        num_classes,
                        class_names,
                        round_num,
                        partition_label,
                    )
                    test_loss, accuracy, f1_value, precision, recall, _, _ = metrics
                    bin_metrics.append({
                        'loss': test_loss,
                        'accuracy': accuracy,
                        'f1': f1_value,
                        'precision': precision,
                        'recall': recall,
                    })
                    print(f"{COLORS.OKGREEN}  Bin {bin_idx}: Loss={test_loss:.4f}, Acc={accuracy:.4f}, F1={f1_value:.4f}{COLORS.ENDC}")
                    log_timestamp(self.logger, f"Bin {bin_idx}: Loss={test_loss:.4f}, Acc={accuracy:.4f}, F1={f1_value:.4f}")
                
                # Compute weighted average based on bin sizes
                total_clients = sum(bin_counts.values())
                avg_loss = sum(bin_metrics[i]['loss'] * bin_counts.get(i, 0) for i in range(k)) / total_clients
                avg_accuracy = sum(bin_metrics[i]['accuracy'] * bin_counts.get(i, 0) for i in range(k)) / total_clients
                avg_f1 = sum(bin_metrics[i]['f1'] * bin_counts.get(i, 0) for i in range(k)) / total_clients
                
                print(f"{COLORS.HEADER}Weighted Average: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}, F1={avg_f1:.4f}{COLORS.ENDC}")
                log_timestamp(self.logger, f"FedKEESS Weighted Avg: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}, F1={avg_f1:.4f}")
                
                # Use first bin for Excel reporting (legacy compatibility)
                latest_weights = global_models_list[0]
                # Store bin metrics for potential multi-model Excel export
                self.fedkeess_bin_metrics = bin_metrics
                
            else:
                aggregated_result = self._aggregate(weights_list, sample_sizes, participating_clients, global_update, models_list)
            
            # For FedoRA, handle special return format (weights, weight_matrices)
            if self.config.strategy == "FedoRA":
                latest_weights, global_weight_matrices = aggregated_result
                # Apply global weight_matrices to model (will be broadcast to clients)
                temp_model = create_model(self.config.model, input_dim, num_classes, self.config.batch_size, self.strategy_runtime, client_id=None)
                temp_model.set_weights(latest_weights)
                self._set_weight_matrices(temp_model, global_weight_matrices)
                latest_weights = temp_model.get_weights()
                del temp_model, models_list
                aggressive_memory_cleanup()
            # For FLTrust, apply update to get new weights
            elif self.config.strategy == "FLTrust":
                if latest_weights is None:
                    latest_weights = aggregated_result
                else:
                    # Apply update: w_new = w_old + aggregated_update
                    latest_weights = [old_w + update for old_w, update in zip(latest_weights, aggregated_result)]
            else:
                latest_weights = aggregated_result

            del weights_list
            aggressive_memory_cleanup()

            eval_start_time = time.time()
            metrics = self._evaluate_global_model(
                latest_weights,
                input_dim,
                num_classes,
                class_names,
                round_num,
                partition_label,
            )
            test_loss, accuracy, f1_value, precision, recall, per_class_metrics, confusion_mat = metrics

            eval_time = time.time() - eval_start_time
            log_timestamp(self.logger, f"Global eval completed in {eval_time:.2f}s")

            new_row = pd.DataFrame({
                'Round': [round_num],
                'Loss': [test_loss],
                'Accuracy': [accuracy],
                'F1_Score': [f1_value],
                'Precision': [precision],
                'Recall': [recall],
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

            if round_num == self.config.rounds and per_class_metrics is not None:
                create_enhanced_excel_report(
                    excel_filename,
                    self.results_df,
                    per_class_metrics,
                    class_names,
                    round_num,
                    confusion_mat,
                )
            else:
                self.results_df.to_excel(excel_filename, index=False)

            round_avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
            self.logger.info(
                f"Round {round_num} summary - ClientLossAvg: {round_avg_loss:.4f}, GlobalLoss: {test_loss:.4f}"
            )
            print(
                f"{COLORS.OKGREEN}Round {round_num} completed - Loss: {test_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1_value:.4f}{COLORS.ENDC}"
            )

            round_time = time.time() - round_start_time
            round_times.append(round_time)
            log_timestamp(self.logger, f"--- Round {round_num} completed in {round_time:.2f}s ---")

            for weights_file in weights_files:
                if round_num != self.config.rounds and os.path.exists(weights_file):
                    os.remove(weights_file)

            aggressive_memory_cleanup()
            time.sleep(1)

        pipeline_end_time = time.time()
        total_time = pipeline_end_time - pipeline_start_time
        avg_round_time = sum(round_times) / len(round_times) if round_times else 0

        log_timestamp(self.logger, "All rounds completed")
        self.logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        self.logger.info(f"Average time per round: {avg_round_time:.2f} seconds")
        self.logger.info(f"Total rounds completed: {len(round_times)}")

        print(f"{COLORS.OKGREEN}Pipeline completed!{COLORS.ENDC}")
        print(f"{COLORS.OKGREEN}Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes){COLORS.ENDC}")
        print(f"{COLORS.OKGREEN}Average time per round: {avg_round_time:.2f} seconds{COLORS.ENDC}")
        print(f"{COLORS.OKCYAN}Results saved to {excel_filename}{COLORS.ENDC}")


def run_pipeline(config: FLConfig) -> None:
    pipeline = FederatedLearningPipeline(config)
    pipeline.run()


def run_distillation_pipeline(config, strategy) -> None:
    from .config import FDConfig
    from .context import evaluate_model
    
    partition_label, client_count = parse_partition_type(config.partition_type)
    n_clients = min(config.n_clients, client_count)
    
    extra_log_tokens = strategy.extra_log_tokens()
    
    poison_suffix = ""
    if config.poison:
        poison_suffix = config.poison.replace(" ", "_")
    
    logger, log_filename, detailed_logger = setup_logger(
        algorithm_name=strategy.name,
        n_clients=n_clients,
        partition_label=partition_label,
        extra_tokens=list(extra_log_tokens.values()),
        poison_suffix=poison_suffix,
    )
    excel_filename = log_filename.replace(".log", ".xlsx")
    
    y_test = np.load(os.path.join("data", "y_test.npy"), mmap_mode="r")
    num_classes = len(np.unique(y_test))
    del y_test
    
    sample_X = np.load(os.path.join("data", "X_test.npy"), mmap_mode="r")
    input_dim = int(sample_X.shape[1])
    del sample_X
    
    paths_list = [setup_paths(str(i), partition_label, client_count) for i in range(n_clients)]
    
    # Restore partitions for strategies that DON'T use public dataset
    strategies_without_public = ["FD", "FedProto", "FedDKD"]
    if strategy.name in strategies_without_public:
        from .data_utils import restore_full_partition
        restored_count = sum(restore_full_partition(paths) for paths in paths_list)
        if restored_count > 0:
            print(f"{COLORS.OKGREEN}Restored {restored_count} partitions for {strategy.name} (no public dataset){COLORS.ENDC}")
    
    test_dataset = load_test_dataset(config.batch_size, num_classes)
    test_labels = _extract_labels(test_dataset, num_classes)
    
    poisoned_clients = []
    poison_loader = None
    attack_type, poison_value, poison_ratio = parse_poison_config(config.poison)
    if attack_type:
        poisoned_clients = get_or_create_poisoned_clients(
            partition_label, attack_type, poison_value, poison_ratio, n_clients
        )
        if attack_type == "label_flip":
            poison_loader = PoisonedDataLoader(attack_type, num_classes)
        print(f"\n  POISONING ENABLED: {attack_type} attack (value={poison_value})")
        print(f"    Poisoned clients: {poisoned_clients} ({len(poisoned_clients)}/{n_clients})")
    
    context = PipelineContext(
        config=config,
        partition_label=partition_label,
        client_count=client_count,
        n_clients=n_clients,
        input_dim=input_dim,
        num_classes=num_classes,
        paths=paths_list,
        logger=logger,
        detailed_logger=detailed_logger,
        log_filename=log_filename,
        excel_filename=excel_filename,
        test_dataset=test_dataset,
        test_labels=test_labels,
        shared_state={"extra_log_tokens": extra_log_tokens},
        poisoned_clients=poisoned_clients,
        poison_loader=poison_loader,
    )
    
    log_timestamp(logger, "SIMULATION STARTED")
    log_timestamp(logger, f"Clients: {n_clients}, Rounds: {config.rounds}, Partition: {partition_label}")
    
    if context.shared_state.get("extra_log_tokens"):
        log_timestamp(logger, f"Extra config: {context.shared_state['extra_log_tokens']}")
    
    strategy.setup(context)
    
    for round_number in range(1, config.rounds + 1):
        logger.info(f"Round {round_number}/{config.rounds}")
        print(f"\n{COLORS.HEADER}Round {round_number}/{config.rounds}{COLORS.ENDC}")
        log_timestamp(logger, f"Round {round_number} started")
        
        round_metrics = strategy.run_round(context, round_number)
        
        if round_metrics:
            _record_metrics(context, round_number, round_metrics, excel_filename)
    
    strategy.finalize(context)
    
    total_time = context.shared_state.get("pipeline_elapsed_s")
    if total_time is not None:
        log_timestamp(logger, f"Pipeline completed in {total_time:.2f}s ({total_time/60:.2f}m)")
    else:
        log_timestamp(logger, "Pipeline completed")


def _extract_labels(dataset: tf.data.Dataset, num_classes: int) -> np.ndarray:
    labels: List[int] = []
    for _, batch_y in dataset:
        batch = batch_y.numpy()
        if batch.ndim == 1 or batch.shape[1] == 1:
            labels.extend(batch.astype(int).tolist())
        else:
            labels.extend(np.argmax(batch, axis=1).tolist())
    return np.asarray(labels, dtype=int)


def _record_metrics(context: PipelineContext, round_number: int, round_metrics: Dict[int, Dict[str, float]], excel_filename: str) -> None:
    for client_id, metrics in round_metrics.items():
        for key, value in metrics.items():
            metric_key = f"Round_{round_number}_{key}"
            context.results.setdefault(client_id, {})[metric_key] = value
    
    results_df = pd.DataFrame(context.results).T
    results_df.to_excel(excel_filename)
