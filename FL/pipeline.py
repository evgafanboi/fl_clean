import dataclasses
import logging
import os
import pickle
import shutil
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
from .decentralized import (
    braintorrent_select_server, log_braintorrent_selection,
    compute_model_similarity_scores, select_model_similarity_server, log_model_similarity_selection,
)
from .poison_utils import parse_poison_config, get_or_create_poisoned_clients, PoisonedDataLoader


def _record_round_weights(log_filename, round_num, global_weights=None, context=None):
    stem = os.path.splitext(os.path.basename(log_filename))[0]
    record_dir = os.path.join("temp_weights", f"{stem}_weight_record", f"round_{round_num}")
    os.makedirs(record_dir, exist_ok=True)

    if global_weights is not None:
        path = os.path.join(record_dir, "global_weight.bin")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(global_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return

    if context is None:
        return

    saved_any = False
    for state in getattr(context, "client_states", []):
        w = state.data.get("w")
        if w is not None:
            path = os.path.join(record_dir, f"client_{state.client_id}_weight.bin")
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(w, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            saved_any = True

    if saved_any or context.model_pool is None:
        return

    pool = context.model_pool
    for state in getattr(context, "client_states", []):
        model = pool.checkout(state.client_id)
        w = pool._get_weights(model)
        pool.release(model)
        path = os.path.join(record_dir, f"client_{state.client_id}_weight.bin")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(w, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)


def _config_fingerprint(config) -> str:
    import hashlib
    d = dataclasses.asdict(config)
    d.pop('checkpoint', None)
    d.pop('rounds', None)
    d.pop('skip_eval', None)
    raw = str(sorted(d.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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
    cleanup_interval: int = 10
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
    decentralized: Optional[str] = None
    checkpoint: int = 0
    skip_eval: bool = False

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

    def _load_class_metadata(self, partition_label: str, client_count: int) -> tuple[List[str], int]:
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
        dot_sum = 0.0
        norm1_sq = 0.0
        norm2_sq = 0.0
        for w1, w2 in zip(weights1, weights2):
            f1 = w1.ravel()
            f2 = w2.ravel()
            dot_sum += np.dot(f1, f2)
            norm1_sq += np.dot(f1, f1)
            norm2_sq += np.dot(f2, f2)
        norm1 = np.sqrt(norm1_sq)
        norm2 = np.sqrt(norm2_sq)
        if norm1 > 0 and norm2 > 0:
            return float(dot_sum / (norm1 * norm2))
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
        dot_sum = 0.0
        norm_c_sq = 0.0
        norm_g_sq = 0.0
        for c_w, g_w in zip(client_weights, global_weights):
            diff = c_w - g_w
            d_flat = diff.ravel()
            g_flat = g_w.ravel()
            dot_sum += np.dot(d_flat, g_flat)
            norm_c_sq += np.dot(d_flat, d_flat)
            norm_g_sq += np.dot(g_flat, g_flat)
        norm_c = np.sqrt(norm_c_sq)
        norm_g = np.sqrt(norm_g_sq)
        cos_sim = dot_sum / (norm_c * norm_g) if norm_c > 0 and norm_g > 0 else 0.0
        return max(0.0, float(cos_sim))

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
                cache=True,
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
            test_loss, accuracy, f1, precision, recall, _, _, _ = metrics
            
            self.logger.info(
                f"Client {client_idx} | PERSONALIZED | Acc: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Loss: {test_loss:.4f}"
            )
            print(
                f"{COLORS.OKGREEN}Client {client_idx}: Acc={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={test_loss:.4f}{COLORS.ENDC}"
            )
            
            del model, train_dataset, history
            aggressive_memory_cleanup()
        
        log_timestamp(self.logger, "Independent learning completed")
        print(f"{COLORS.OKGREEN}Independent learning completed!{COLORS.ENDC}")

    _WRAPPER_STRATEGIES = frozenset({'fedmlb', 'fedora'})

    def _checkpoint_dir(self) -> str:
        stem = os.path.splitext(os.path.basename(self.log_filename))[0]
        return os.path.join("checkpoint", stem)

    @staticmethod
    def _atomic_pickle(path, obj):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _save_strategy_state(self, ckpt_dir):
        strategy_obj = self.strategy_runtime.client_strategy
        if hasattr(strategy_obj, '_grad_L'):
            self._atomic_pickle(os.path.join(ckpt_dir, "strategy_state.bin"), {
                '_grad_L': strategy_obj._grad_L,
                '_prev_global': strategy_obj._prev_global,
                '_h': strategy_obj._h,
                '_n_clients_total': strategy_obj._n_clients_total,
                'round_num': strategy_obj.round_num,
            })

    def _save_client_progress(self, round_num, client_idx, n_clients, latest_weights,
                              stream, running_agg, total_samples_seen,
                              client_result, sample_sizes, client_losses,
                              participating_clients):
        ckpt_dir = self._checkpoint_dir()
        os.makedirs(ckpt_dir, exist_ok=True)

        gw_path = os.path.join(ckpt_dir, "global_weights.bin")
        if client_idx == 0 or not os.path.exists(gw_path):
            self._atomic_pickle(gw_path, latest_weights)

        if stream:
            self._atomic_pickle(os.path.join(ckpt_dir, "stream_state.bin"), {
                'running_agg': running_agg,
                'total_samples_seen': total_samples_seen,
            })
        else:
            results_dir = os.path.join(ckpt_dir, "client_results")
            os.makedirs(results_dir, exist_ok=True)
            self._atomic_pickle(os.path.join(results_dir, f"{client_idx}.bin"), client_result)

        self._atomic_pickle(os.path.join(ckpt_dir, "round_meta.bin"), {
            'sample_sizes': sample_sizes,
            'client_losses': client_losses,
            'participating_clients': participating_clients,
        })

        self._save_strategy_state(ckpt_dir)

        info_lines = [
            f"round: {round_num}",
            f"round_complete: false",
            f"last_client: {client_idx}",
            f"stream: {stream}",
            f"total_rounds: {self.config.rounds}",
            f"config_hash: {_config_fingerprint(self.config)}",
            f"strategy: {self.config.strategy}",
            f"n_clients: {n_clients}",
        ]
        with open(os.path.join(ckpt_dir, "info.txt"), "w") as f:
            f.write("\n".join(info_lines) + "\n")

    def _save_checkpoint(self, round_num, latest_weights, n_clients, round_times,
                         selected_server=None, ms_prev_scores=None):
        ckpt_dir = self._checkpoint_dir()
        os.makedirs(ckpt_dir, exist_ok=True)

        self._atomic_pickle(os.path.join(ckpt_dir, "global_weights.bin"), latest_weights)

        info_lines = [
            f"round: {round_num}",
            f"round_complete: true",
            f"total_rounds: {self.config.rounds}",
            f"config_hash: {_config_fingerprint(self.config)}",
            f"strategy: {self.config.strategy}",
            f"model: {self.config.model}",
            f"n_clients: {n_clients}",
            f"partition_type: {self.config.partition_type}",
            f"epochs: {self.config.epochs}",
            f"batch_size: {self.config.batch_size}",
            f"poisoned_clients: {self.poisoned_clients if self.poisoned_clients else 'none'}",
            f"poison_config: {self.config.poison or 'none'}",
            f"decentralized: {self.config.decentralized or 'none'}",
            f"selected_server: {selected_server if selected_server is not None else 'N/A'}",
            f"completed_rounds: {len(round_times)}",
            f"avg_round_time: {sum(round_times) / len(round_times):.2f}s" if round_times else "avg_round_time: N/A",
        ]
        with open(os.path.join(ckpt_dir, "info.txt"), "w") as f:
            f.write("\n".join(info_lines) + "\n")

        if ms_prev_scores is not None:
            self._atomic_pickle(os.path.join(ckpt_dir, "ms_prev_scores.bin"), ms_prev_scores)

        self._save_strategy_state(ckpt_dir)
        self.results_df.to_pickle(os.path.join(ckpt_dir, "results_df.pkl"))

        for artifact in ["round_meta.bin", "stream_state.bin"]:
            p = os.path.join(ckpt_dir, artifact)
            if os.path.exists(p):
                os.remove(p)
        results_dir = os.path.join(ckpt_dir, "client_results")
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)

        print(f"{COLORS.OKCYAN}Checkpoint saved (round {round_num}) -> {ckpt_dir}{COLORS.ENDC}")

    def _load_checkpoint(self):
        ckpt_dir = self._checkpoint_dir()
        info_path = os.path.join(ckpt_dir, "info.txt")
        if not os.path.exists(info_path):
            return None

        info = {}
        with open(info_path) as f:
            for line in f:
                key, _, val = line.strip().partition(": ")
                info[key] = val

        saved_hash = info.get("config_hash", "")
        current_hash = _config_fingerprint(self.config)
        if saved_hash and saved_hash != current_hash:
            print(f"{COLORS.WARNING}No checkpoint found, starting fresh{COLORS.ENDC}")
            return None

        with open(os.path.join(ckpt_dir, "global_weights.bin"), "rb") as f:
            latest_weights = pickle.load(f)

        ms_path = os.path.join(ckpt_dir, "ms_prev_scores.bin")
        ms_prev_scores = None
        if os.path.exists(ms_path):
            with open(ms_path, "rb") as f:
                ms_prev_scores = pickle.load(f)

        strategy_state_path = os.path.join(ckpt_dir, "strategy_state.bin")
        if os.path.exists(strategy_state_path):
            with open(strategy_state_path, "rb") as f:
                strategy_state = pickle.load(f)
            strategy_obj = self.strategy_runtime.client_strategy
            for key, val in strategy_state.items():
                setattr(strategy_obj, key, val)

        df_path = os.path.join(ckpt_dir, "results_df.pkl")
        if os.path.exists(df_path):
            self.results_df = pd.read_pickle(df_path)

        round_complete = info.get("round_complete", "true") == "true"

        if round_complete:
            resume_round = int(info["round"]) + 1
            print(f"{COLORS.OKGREEN}Resuming from checkpoint (completed round {info['round']}) -> starting round {resume_round}{COLORS.ENDC}")
            return {"resume_round": resume_round, "latest_weights": latest_weights, "ms_prev_scores": ms_prev_scores}

        round_num = int(info["round"])
        last_client = int(info["last_client"])
        stream = info.get("stream", "False") == "True"

        result = {
            "resume_round": round_num,
            "resume_client": last_client + 1,
            "latest_weights": latest_weights,
            "ms_prev_scores": ms_prev_scores,
            "stream": stream,
        }

        meta_path = os.path.join(ckpt_dir, "round_meta.bin")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            result['sample_sizes'] = meta['sample_sizes']
            result['client_losses'] = meta['client_losses']
            result['participating_clients'] = meta['participating_clients']

        if stream:
            ss_path = os.path.join(ckpt_dir, "stream_state.bin")
            if os.path.exists(ss_path):
                with open(ss_path, "rb") as f:
                    ss = pickle.load(f)
                result['running_agg'] = ss['running_agg']
                result['total_samples_seen'] = ss['total_samples_seen']
        else:
            results_dir = os.path.join(ckpt_dir, "client_results")
            client_results = []
            for i in range(last_client + 1):
                with open(os.path.join(results_dir, f"{i}.bin"), "rb") as f:
                    client_results.append(pickle.load(f))
            result['client_results'] = client_results

        print(f"{COLORS.OKGREEN}Resuming round {round_num} from client {last_client + 1}{COLORS.ENDC}")
        return result

    def _train_single_client(
        self,
        client_id: int,
        input_dim: int,
        num_classes: int,
        latest_weights,
        paths,
        reuse_model=None,
    ):
        print(f"\n{COLORS.BOLD}Client {client_id}{COLORS.ENDC}")

        client_start_time = time.time()
        log_timestamp(self.logger, f"Client {client_id} training started")

        if reuse_model is not None:
            model = reuse_model
        else:
            tf.keras.backend.clear_session()
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
            poison_loader=poison_loader,
            cache=True,
        )

        X_train_mmap = np.load(paths['train_X'], mmap_mode='r')
        sample_size = X_train_mmap.shape[0]
        del X_train_mmap

        print(f"Training client {client_id} for {self.config.epochs} epochs")
        
        if self.config.strategy in ["FedSSDexp"] and self.strategy_runtime and hasattr(self.strategy_runtime.client_strategy, 'train'):
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
            del global_model
            loss = 0.0
            history = None
        elif hasattr(self.strategy_runtime.client_strategy, 'train_client'):
            loss = self.strategy_runtime.client_strategy.train_client(
                model=model,
                dataset=train_dataset,
                epochs=self.config.epochs,
                client_id=client_id,
                global_weights=latest_weights,
            )
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
                verbose=2,
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

        if self.config.strategy == "SecureAggregation":
            poisoned_weights = self._apply_secure_aggregation_mask(client_id, poisoned_weights)

        if self.config.strategy == "FLTrust":
            if latest_weights is None:
                result_data = poisoned_weights
            else:
                result_data = [new_w - old_w for new_w, old_w in zip(poisoned_weights, latest_weights)]
        else:
            result_data = poisoned_weights

        del train_dataset, trained_weights
        if reuse_model is None:
            del model
        del history
        aggressive_memory_cleanup()

        return result_data, sample_size, loss

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

    _NEEDS_ALL_WEIGHTS = frozenset({'feddyn', 'fltrust', 'fedcomed', 'robustfilter'})

    def _can_stream_aggregate(self) -> bool:
        """Check if strategy supports incremental (streaming) aggregation."""
        strategy_name = getattr(self.strategy_runtime.client_strategy, 'name', '').lower()
        if strategy_name in self._NEEDS_ALL_WEIGHTS:
            return False
        if self.config.peer_trust:
            return False
        if self.config.decentralized == "ModelSimilarity":
            return False
        return True

    def _aggregate(self, weights_list, sample_sizes, participating_clients, global_update=None, models=None):
        aggregator = self.strategy_runtime.aggregator
        
        # FLTrust requires global_update parameter
        if self.config.strategy == "FLTrust":
            return aggregator.aggregate(weights_list, sample_sizes, global_update=global_update)
        elif self.strategy_runtime.requires_participant_ids:
            return aggregator.aggregate(weights_list, sample_sizes, participating_clients)
        else:
            return aggregator.aggregate(weights_list, sample_sizes)
    
    def _get_root_dataset(self, num_classes):
        if hasattr(self, '_root_dataset_cache'):
            return self._root_dataset_cache
        partition_path = os.path.join("data", "partitions", f"{self.n_clients}_client", self.partition_label)
        all_public_X = []
        all_public_y = []
        for client_idx in range(self.n_clients):
            all_public_X.append(np.load(os.path.join(partition_path, f"client_{client_idx}_X_public.npy")))
            all_public_y.append(np.load(os.path.join(partition_path, f"client_{client_idx}_y_public.npy")))
        combined_X = np.concatenate(all_public_X, axis=0).astype(np.float32)
        combined_y = np.concatenate(all_public_y, axis=0)
        combined_y = tf.keras.utils.to_categorical(combined_y.astype(np.int32), num_classes).astype(np.float32)
        self._root_dataset_cache = (combined_X, combined_y)
        return self._root_dataset_cache

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
        
        combined_X, combined_y = self._get_root_dataset(num_classes)
        
        root_dataset = tf.data.Dataset.from_tensor_slices((combined_X, combined_y))
        root_dataset = root_dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        print(f"  [SERVER] Training on root dataset ({len(combined_X)} samples) for {self.config.root_iterations} iterations")
        
        model.fit(root_dataset, epochs=self.config.root_iterations, verbose=0)
        
        new_weights = model.get_weights()
        if current_weights is None:
            global_update = new_weights
        else:
            global_update = [new_w - old_w for new_w, old_w in zip(new_weights, current_weights)]
        
        del model, root_dataset
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
            f"{COLORS.OKBLUE}Client {client_id} personalized eval -> Acc={accuracy:.4f}, F1={f1_value:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={test_loss:.4f}{COLORS.ENDC}"
        )

    def _evaluate_global_model(
        self,
        aggregated_weights,
        input_dim,
        num_classes,
        class_names,
        round_num,
        partition_label,
        eval_model=None,
    ):
        created_model = eval_model is None
        if created_model:
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
            f"{COLORS.OKGREEN}[GLOBAL] Acc={accuracy:.4f}, F1={f1_score_value:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={test_loss:.4f}{COLORS.ENDC}"
        )

        self.logger.info(
            f"Round {round_num} | GLOBAL | Acc: {accuracy:.4f} | F1: {f1_score_value:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Loss: {test_loss:.4f}"
        )
        if class_report and self.detailed_logger is not None:
            self.detailed_logger.info(f"Round {round_num}\n{class_report}")

        if created_model:
            del eval_model
        aggressive_memory_cleanup()

        return test_loss, accuracy, f1_score_value, precision, recall, per_class_metrics, confusion_mat

    def run(self):
        configure_gpu_memory()
        if not os.path.islink(self.config.weights_cache_dir):
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

        extra_tokens = [self.config.model, *list(extra_log_tokens.values())]
        if self.config.decentralized:
            extra_tokens.append(self.config.decentralized)

        self.logger, self.log_filename, self.detailed_logger = setup_logger(
            n_clients,
            partition_label,
            strategy_name=self.config.strategy,
            extra_tokens=extra_tokens,
            poison_suffix=poison_suffix,
            resume=self.config.checkpoint,
        )
        excel_filename = self.log_filename.replace('.log', '.xlsx')

        class_names, num_classes = self._load_class_metadata(partition_label, client_count)
        self.class_names = class_names
        input_dim = self._determine_input_dim()

        paths_list = self._prepare_paths(n_clients, partition_label, client_count)
        
        # Skip partition restoration for strategies that need separate public/validation data
        if self.config.strategy not in ["FLTrust"]:
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

        ms_prev_scores = None

        latest_weights = None
        start_round = 1
        _resume_client = 0
        _resume_accum = None
        pipeline_start_time = time.time()
        log_timestamp(self.logger, "=== FL PIPELINE STARTED ===")
        log_timestamp(self.logger, f"Strategy: {self.config.strategy}, Clients: {n_clients}, Rounds: {self.config.rounds}")
        round_times: List[float] = []

        if self.config.checkpoint:
            ckpt = self._load_checkpoint()
            if ckpt:
                latest_weights = ckpt["latest_weights"]
                start_round = ckpt["resume_round"]
                ms_prev_scores = ckpt.get("ms_prev_scores")
                if "resume_client" in ckpt:
                    _resume_client = ckpt["resume_client"]
                    _resume_accum = ckpt
                    log_timestamp(self.logger, f"Resuming round {start_round} from client {_resume_client}")
                else:
                    log_timestamp(self.logger, f"Resumed from checkpoint, starting at round {start_round}")

        # Check if final weight records already exist — skip training entirely
        stem = os.path.splitext(os.path.basename(self.log_filename))[0]
        record_base = os.path.join("temp_weights", f"{stem}_weight_record")
        final_weight = os.path.join(record_base, f"round_{self.config.rounds}", "global_weight.bin")
        if os.path.exists(final_weight):
            log_timestamp(self.logger, f"Weight records found (round {self.config.rounds}), skipping training")
            print(f"{COLORS.OKGREEN}Weight records found up to round {self.config.rounds} — skipping to evaluation{COLORS.ENDC}")
            self._run_eval_from_records(
                input_dim, num_classes, class_names, partition_label, excel_filename, record_base,
            )
            return

        # Special handling for None strategy (independent learning)
        if self.config.strategy == "None":
            self._run_independent_learning(n_clients, input_dim, num_classes, paths_list)
            return

        # Initialize global server model — reuse across clients if strategy allows
        print(f"\n{COLORS.OKCYAN}Initializing global server model{COLORS.ENDC}")
        log_timestamp(self.logger, "Initializing server model")
        reusable_model = create_model(
            architecture=self.config.model,
            input_dim=input_dim,
            num_classes=num_classes,
            batch_size=self.config.batch_size,
            strategy_runtime=self.strategy_runtime,
            client_id=None,
        )
        if latest_weights is None:
            latest_weights = reusable_model.get_weights()
        _strategy_key = getattr(self.strategy_runtime.client_strategy, 'name', '').lower()
        can_reuse = _strategy_key not in self._WRAPPER_STRATEGIES
        if not can_reuse:
            del reusable_model
            reusable_model = None
            aggressive_memory_cleanup()
        print(f"{COLORS.OKGREEN}Server model initialized (reuse={can_reuse}){COLORS.ENDC}")

        if self.config.strategy == "FedSSDexp":
            self.strategy_runtime.aggregator.M_class = np.ones(num_classes, dtype=np.float32)
            print(f"{COLORS.OKCYAN}M_class initialized to ones (will be computed from full training data in each round){COLORS.ENDC}")

        for round_num in range(start_round, self.config.rounds + 1):
            self.current_round = round_num
            round_start_time = time.time()

            self.logger.info(f"Round {round_num}/{self.config.rounds}")
            print(f"\n{COLORS.HEADER}Round {round_num}/{self.config.rounds}{COLORS.ENDC}")
            log_timestamp(self.logger, f"--- Round {round_num} started ---")

            # BrainTorrent: select server before each round
            if self.config.decentralized == "braintorrent":
                selected_server = braintorrent_select_server(latest_weights, n_clients)
                log_braintorrent_selection(round_num, selected_server, n_clients, self.logger)

            # ModelSimilarity: log server selection from previous round's scores
            if self.config.decentralized == "ModelSimilarity" and ms_prev_scores is not None:
                ms_server = select_model_similarity_server(ms_prev_scores)
                log_model_similarity_selection(round_num, ms_prev_scores, ms_server, self.logger)

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
                        
                        np.add.at(cm, (true_labels, pred_labels), 1)
                    
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

            # Refresh model at round boundaries to release CuDNN workspace to avoid DoRNNBackward OOM.
            if (can_reuse and reusable_model is not None
                    and round_num > start_round):
                del reusable_model
                tf.keras.backend.clear_session()
                aggressive_memory_cleanup()
                reusable_model = create_model(
                    architecture=self.config.model,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    batch_size=self.config.batch_size,
                    strategy_runtime=self.strategy_runtime,
                    client_id=None,
                )

            stream = self._can_stream_aggregate()
            sample_sizes: List[int] = []
            client_losses: List[float] = []
            participating_clients: List[int] = []
            running_agg = None
            total_samples_seen = 0
            client_results: List = [] if not stream else None

            first_client = 0
            if round_num == start_round and _resume_accum is not None:
                first_client = _resume_client
                sample_sizes = _resume_accum['sample_sizes']
                client_losses = _resume_accum['client_losses']
                participating_clients = _resume_accum['participating_clients']
                if stream:
                    running_agg = _resume_accum.get('running_agg')
                    total_samples_seen = _resume_accum.get('total_samples_seen', 0)
                else:
                    client_results = _resume_accum.get('client_results', [])
                _resume_accum = None
                print(f"{COLORS.OKGREEN}Resuming round from client {first_client}{COLORS.ENDC}")

            _REFRESH_EVERY = self.config.cleanup_interval

            for client_idx in range(first_client, n_clients):
                if (can_reuse and reusable_model is not None
                        and client_idx > 0 and client_idx % _REFRESH_EVERY == 0):
                    del reusable_model
                    tf.keras.backend.clear_session()
                    aggressive_memory_cleanup()
                    reusable_model = create_model(
                        architecture=self.config.model,
                        input_dim=input_dim,
                        num_classes=num_classes,
                        batch_size=self.config.batch_size,
                        strategy_runtime=self.strategy_runtime,
                        client_id=None,
                    )

                result_data, sample_size, loss = self._train_single_client(
                    client_id=client_idx,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    latest_weights=latest_weights,
                    paths=paths_list[client_idx],
                    reuse_model=reusable_model,
                )
                sample_sizes.append(sample_size)
                client_losses.append(loss)
                participating_clients.append(client_idx)

                if stream:
                    w = result_data['weights'] if isinstance(result_data, dict) and 'weights' in result_data else result_data
                    if running_agg is None:
                        running_agg = [layer_w.astype(np.float64) * sample_size for layer_w in w]
                    else:
                        for j, layer_w in enumerate(w):
                            running_agg[j] += layer_w.astype(np.float64) * sample_size
                    total_samples_seen += sample_size
                    del result_data, w
                else:
                    client_results.append(result_data)

                if self.config.checkpoint and (
                    client_idx == n_clients - 1
                    or (client_idx + 1) % self.config.checkpoint == 0
                ):
                    self._save_client_progress(
                        round_num, client_idx, n_clients, latest_weights,
                        stream, running_agg, total_samples_seen,
                        client_results[-1] if not stream else None,
                        sample_sizes, client_losses, participating_clients,
                    )

            if not participating_clients:
                break

            if stream:
                latest_weights = [
                    (layer / total_samples_seen).astype(np.float32)
                    for layer in running_agg
                ]
                del running_agg
            else:
                weights_list = [
                    d['weights'] if isinstance(d, dict) and 'weights' in d else d
                    for d in client_results
                ]
            
            # Compute peer trust scores if enabled (requires non-streamed weights_list)
            if not stream and self.config.peer_trust:
                peer_scores = self._compute_peer_trust_scores(weights_list, participating_clients)
                self.logger.info(f"\n=== Peer Trust Scores (Round {round_num}) ===")
                for client_id in participating_clients:
                    left_score, right_score = peer_scores[client_id]
                    left_str = f"{left_score:.4f}" if left_score is not None else "N/A"
                    right_str = f"{right_score:.4f}" if right_score is not None else "N/A"
                    self.logger.info(f"Client {client_id}: Left={left_str}, Right={right_str}")
                    print(f"{COLORS.OKCYAN}Client {client_id} PeerTrust: Left={left_str}, Right={right_str}{COLORS.ENDC}")
            
            if not stream:
                models_list = None
                
                # For FLTrust, compute server update on root dataset
                global_update = None
                if self.config.strategy == "FLTrust":
                    print(f"\n{COLORS.OKBLUE}Training server on root dataset{COLORS.ENDC}")
                    global_update = self._train_server_on_root_dataset(latest_weights, input_dim, num_classes)
                
                print(f"\n{COLORS.OKBLUE}Aggregating updates ({self.config.strategy}){COLORS.ENDC}")
                
                aggregated_result = self._aggregate(weights_list, sample_sizes, participating_clients, global_update, models_list)
                
                # For FLTrust, apply update to get new weights
                if self.config.strategy == "FLTrust":
                    if latest_weights is None:
                        latest_weights = aggregated_result
                    else:
                        latest_weights = [old_w + update for old_w, update in zip(latest_weights, aggregated_result)]
                else:
                    latest_weights = aggregated_result

                # ModelSimilarity: compute cosine similarity of each client's model vs global for next round
                if self.config.decentralized == "ModelSimilarity":
                    ms_prev_scores = compute_model_similarity_scores(weights_list, latest_weights, participating_clients)

                del weights_list, client_results

            aggressive_memory_cleanup()

            round_avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
            self.logger.info(
                f"Round {round_num} summary - ClientLossAvg: {round_avg_loss:.4f}"
            )
            print(
                f"{COLORS.OKGREEN}Round {round_num} completed - ClientLossAvg: {round_avg_loss:.4f}{COLORS.ENDC}"
            )

            round_time = time.time() - round_start_time
            round_times.append(round_time)
            log_timestamp(self.logger, f"--- Round {round_num} completed in {round_time:.2f}s ---")

            selected_server = None
            if self.config.decentralized == "braintorrent":
                selected_server = braintorrent_select_server(latest_weights, n_clients)
            elif self.config.decentralized == "ModelSimilarity" and ms_prev_scores is not None:
                selected_server = select_model_similarity_server(ms_prev_scores)
            if self.config.checkpoint:
                self._save_checkpoint(round_num, latest_weights, n_clients, round_times,
                                      selected_server=selected_server, ms_prev_scores=ms_prev_scores)

            _record_round_weights(self.log_filename, round_num, global_weights=latest_weights)

            aggressive_memory_cleanup()

        pipeline_end_time = time.time()
        total_time = pipeline_end_time - pipeline_start_time
        avg_round_time = sum(round_times) / len(round_times) if round_times else 0

        log_timestamp(self.logger, "=== TRAINING PHASE COMPLETED ===")
        self.logger.info(f"Training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        self.logger.info(f"Average time per round: {avg_round_time:.2f} seconds")
        self.logger.info(f"Total rounds completed: {len(round_times)}")

        print(f"{COLORS.OKGREEN}Training phase completed!{COLORS.ENDC}")
        print(f"{COLORS.OKGREEN}Training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes){COLORS.ENDC}")
        print(f"{COLORS.OKGREEN}Average time per round: {avg_round_time:.2f} seconds{COLORS.ENDC}")

        if self.config.checkpoint and os.path.isdir(self._checkpoint_dir()):
            shutil.rmtree(self._checkpoint_dir(), ignore_errors=True)
            print(f"{COLORS.OKCYAN}Checkpoint cleaned up{COLORS.ENDC}")

        if reusable_model is not None:
            del reusable_model
        tf.keras.backend.clear_session()
        aggressive_memory_cleanup()

        stem = os.path.splitext(os.path.basename(self.log_filename))[0]
        record_base = os.path.join("temp_weights", f"{stem}_weight_record")
        self._run_eval_from_records(
            input_dim, num_classes, class_names, partition_label, excel_filename, record_base,
        )

    def _run_eval_from_records(self, input_dim, num_classes, class_names, partition_label, excel_filename, record_base):
        log_timestamp(self.logger, "=== EVALUATION ===")
        print(f"\n{COLORS.HEADER}[EVALUATION]{COLORS.ENDC}")

        self.results_df = pd.DataFrame(columns=['Round', 'Loss', 'Accuracy', 'F1_Score', 'Precision', 'Recall'])
        eval_batch_size = min(self.config.batch_size, 2048)
        test_dataset = load_test_dataset(eval_batch_size, num_classes)

        eval_model = create_model(
            architecture=self.config.model,
            input_dim=input_dim,
            num_classes=num_classes,
            batch_size=self.config.batch_size,
            strategy_runtime=self.strategy_runtime,
            client_id=None,
        )

        for round_num in range(1, self.config.rounds + 1):
            weight_path = os.path.join(record_base, f"round_{round_num}", "global_weight.bin")
            if not os.path.exists(weight_path):
                self.logger.info(f"Round {round_num} | SKIPPED (no weight record)")
                print(f"{COLORS.WARNING}Round {round_num}: skipped (no weight record){COLORS.ENDC}")
                continue

            if self.config.skip_eval and round_num != self.config.rounds:
                continue

            with open(weight_path, "rb") as f:
                weights = pickle.load(f)
            eval_model.set_weights(weights)
            del weights

            collect_details = (round_num == self.config.rounds)
            result = evaluate_model_with_metrics(
                eval_model, test_dataset, num_classes,
                class_names, round_num, self.config.strategy,
                partition_label, collect_details=collect_details,
            )
            test_loss, accuracy, f1_value, precision, recall = result[:5]
            per_class_metrics = result[5]
            confusion_mat = result[6]
            class_report = result[7] if len(result) > 7 else ""

            self.logger.info(
                f"Round {round_num} | GLOBAL | Acc: {accuracy:.4f} | F1: {f1_value:.4f} | "
                f"Precision: {precision:.4f} | Recall: {recall:.4f} | Loss: {test_loss:.4f}"
            )
            print(
                f"{COLORS.OKGREEN}Round {round_num} | Acc={accuracy:.4f}, F1={f1_value:.4f}, "
                f"Precision={precision:.4f}, Recall={recall:.4f}, Loss={test_loss:.4f}{COLORS.ENDC}"
            )
            if class_report and self.detailed_logger is not None:
                self.detailed_logger.info(f"Round {round_num}\n{class_report}")

            new_row = pd.DataFrame({
                'Round': [round_num], 'Loss': [test_loss], 'Accuracy': [accuracy],
                'F1_Score': [f1_value], 'Precision': [precision], 'Recall': [recall],
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

            if round_num == self.config.rounds and per_class_metrics is not None:
                create_enhanced_excel_report(
                    excel_filename, self.results_df, per_class_metrics,
                    class_names, round_num, confusion_mat,
                )

        if not self.results_df.empty:
            self.results_df.to_excel(excel_filename, index=False)

        del eval_model, test_dataset
        aggressive_memory_cleanup()
        log_timestamp(self.logger, "=== SIMULATION COMPLETED ===")
        print(f"{COLORS.OKCYAN}Results saved to {excel_filename}{COLORS.ENDC}")
        print(f"{COLORS.OKGREEN}Simulation completed!{COLORS.ENDC}")


def run_pipeline(config: FLConfig) -> None:
    pipeline = FederatedLearningPipeline(config)
    pipeline.run()


def run_distillation_pipeline(config, strategy) -> None:
    from .config import FDConfig
    from .context import evaluate_model, ModelPool
    from .strategy.common import create_model as create_strategy_model
    
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
        extra_tokens=[config.model_type, *list(extra_log_tokens.values())],
        poison_suffix=poison_suffix,
        resume=config.checkpoint,
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
    strategies_without_public = ["FD", "FedProto", "FedDKD", "Exp1"]
    if strategy.name in strategies_without_public:
        from .data_utils import restore_full_partition
        restored_count = sum(restore_full_partition(paths) for paths in paths_list)
        if restored_count > 0:
            print(f"{COLORS.OKGREEN}Restored {restored_count} partitions for {strategy.name} (no public dataset){COLORS.ENDC}")
    
    # Defer test data loading to evaluation phase — pass None during training
    model_type = getattr(config, "model_type", "dense")
    model_pool = None
    if getattr(strategy, "use_model_pool", True):
        model_pool = ModelPool(
            pool_size=min(10, n_clients),
            factory_fn=lambda: create_strategy_model(input_dim, num_classes, config.batch_size, model_type=model_type),
            in_memory=bool(getattr(strategy, "prefer_in_memory_pool", False)),
        )
    
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
        test_dataset=None,
        test_labels=None,
        shared_state={"extra_log_tokens": extra_log_tokens},
        poisoned_clients=poisoned_clients,
        poison_loader=poison_loader,
        model_pool=model_pool,
    )
    
    log_timestamp(logger, "SIMULATION STARTED")
    log_timestamp(logger, f"Clients: {n_clients}, Rounds: {config.rounds}, Partition: {partition_label}")
    
    if context.shared_state.get("extra_log_tokens"):
        log_timestamp(logger, f"Extra config: {context.shared_state['extra_log_tokens']}")
    
    strategy.setup(context)
    
    start_round = 1
    ckpt_stem = os.path.splitext(os.path.basename(log_filename))[0]
    ckpt_dir = os.path.join("checkpoint", ckpt_stem)
    if config.checkpoint:
        ckpt_info_path = os.path.join(ckpt_dir, "info.txt")
        if os.path.exists(ckpt_info_path):
            ckpt_info = {}
            with open(ckpt_info_path) as f:
                for line in f:
                    key, _, val = line.strip().partition(": ")
                    ckpt_info[key] = val
            saved_hash = ckpt_info.get("config_hash", "")
            current_hash = _config_fingerprint(config)
            if saved_hash and saved_hash != current_hash:
                print(f"{COLORS.WARNING}Checkpoint config mismatch (saved={saved_hash}, current={current_hash}) \u2014 starting fresh{COLORS.ENDC}")
            else:
                start_round = int(ckpt_info["round"]) + 1
                shared_path = os.path.join(ckpt_dir, "shared_state.bin")
                if os.path.exists(shared_path) and os.path.getsize(shared_path) > 0:
                    with open(shared_path, "rb") as _f:
                        saved_shared = pickle.load(_f)
                    context.shared_state.update(saved_shared)
                results_path = os.path.join(ckpt_dir, "results.pkl")
                if os.path.exists(results_path):
                    context.results = pd.read_pickle(results_path)
                print(f"{COLORS.OKGREEN}Resuming from checkpoint (completed round {ckpt_info['round']}) -> starting round {start_round}{COLORS.ENDC}")
    
    # Check if final weight records already exist — skip training entirely
    stem = os.path.splitext(os.path.basename(log_filename))[0]
    record_base = os.path.join("temp_weights", f"{stem}_weight_record")
    final_round_dir = os.path.join(record_base, f"round_{config.rounds}")
    _global_model_strategy = getattr(strategy, "has_global_model", False)

    if _global_model_strategy:
        _final_exists = os.path.exists(os.path.join(final_round_dir, "global_weight.bin"))
    else:
        _final_exists = all(
            os.path.exists(os.path.join(final_round_dir, f"client_{i}_weight.bin"))
            for i in range(n_clients)
        )
    if _final_exists:
        log_timestamp(logger, f"Weight records found (round {config.rounds}), skipping training")
        print(f"{COLORS.OKGREEN}Weight records found up to round {config.rounds} — skipping to evaluation{COLORS.ENDC}")
        _run_distillation_eval(config, context, logger, log_filename, excel_filename, input_dim, num_classes, n_clients, model_type, record_base, _global_model_strategy)
        return

    for round_number in range(start_round, config.rounds + 1):
        logger.info(f"Round {round_number}/{config.rounds}")
        print(f"\n{COLORS.HEADER}Round {round_number}/{config.rounds}{COLORS.ENDC}")
        log_timestamp(logger, f"Round {round_number} started")
        
        round_metrics = strategy.run_round(context, round_number)
        
        if round_metrics:
            _record_metrics(context, round_number, round_metrics, excel_filename)

        if config.checkpoint:
            os.makedirs(ckpt_dir, exist_ok=True)
            info_lines = [
                f"round: {round_number}",
                f"total_rounds: {config.rounds}",
                f"config_hash: {_config_fingerprint(config)}",
                f"strategy: {strategy.name}",
                f"n_clients: {n_clients}",
                f"partition_type: {config.partition_type}",
                f"poisoned_clients: {poisoned_clients if poisoned_clients else 'none'}",
            ]
            with open(os.path.join(ckpt_dir, "info.txt"), "w") as _f:
                _f.write("\n".join(info_lines) + "\n")
            _SKIP_KEYS = {"extra_log_tokens", "global_model", "disc_pool"}
            saveable_shared = {}
            for k, v in context.shared_state.items():
                if k in _SKIP_KEYS:
                    continue
                if isinstance(v, (tf.data.Dataset, tf.Tensor)):
                    continue
                saveable_shared[k] = v
            _shared_tmp = os.path.join(ckpt_dir, "shared_state.bin.tmp")
            _shared_dst = os.path.join(ckpt_dir, "shared_state.bin")
            with open(_shared_tmp, "wb") as _f:
                pickle.dump(saveable_shared, _f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(_shared_tmp, _shared_dst)
            if context.results:
                pd.to_pickle(context.results, os.path.join(ckpt_dir, "results.pkl"))
            print(f"{COLORS.OKCYAN}Checkpoint saved (round {round_number}) -> {ckpt_dir}{COLORS.ENDC}")

        if _global_model_strategy:
            global_model = context.shared_state.get("global_model")
            if global_model is not None:
                _record_round_weights(log_filename, round_number, global_weights=global_model.get_weights())
        else:
            context.record_client_weights(round_number)
    
    strategy.finalize(context)
    
    total_time = context.shared_state.get("pipeline_elapsed_s")
    if total_time is not None:
        log_timestamp(logger, f"Training phase completed in {total_time:.2f}s ({total_time/60:.2f}m)")
    else:
        log_timestamp(logger, "Training phase completed")

    if config.checkpoint and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print(f"{COLORS.OKCYAN}Checkpoint cleaned up{COLORS.ENDC}")

    tf.keras.backend.clear_session()
    aggressive_memory_cleanup()

    stem = os.path.splitext(os.path.basename(log_filename))[0]
    record_base = os.path.join("temp_weights", f"{stem}_weight_record")
    _run_distillation_eval(config, context, logger, log_filename, excel_filename, input_dim, num_classes, n_clients, model_type, record_base, _global_model_strategy)


def _run_distillation_eval(config, context, logger, log_filename, excel_filename, input_dim, num_classes, n_clients, model_type, record_base, global_model_eval=False):
    from .context import evaluate_model
    from .strategy.common import create_model as create_strategy_model

    log_timestamp(logger, "=== EVALUATION ===")
    print(f"\n{COLORS.HEADER}[EVALUATION]{COLORS.ENDC}")

    context.results = {}
    test_dataset = load_test_dataset(config.batch_size, num_classes)
    test_labels = _extract_labels(test_dataset, num_classes)

    eval_model = create_strategy_model(input_dim, num_classes, config.batch_size, model_type=model_type)

    for round_number in range(1, config.rounds + 1):
        round_dir = os.path.join(record_base, f"round_{round_number}")

        if config.skip_eval and round_number != config.rounds:
            continue

        if global_model_eval:
            weight_path = os.path.join(round_dir, "global_weight.bin")
            if not os.path.exists(weight_path):
                logger.info(f"Round {round_number} | SKIPPED (no weight record)")
                print(f"{COLORS.WARNING}Round {round_number}: skipped (no weight record){COLORS.ENDC}")
                continue

            with open(weight_path, "rb") as _f:
                weights = pickle.load(_f)
            eval_model.set_weights(weights)
            del weights

            metrics = evaluate_model(eval_model, test_dataset, test_labels)
            round_metrics = {-1: metrics}
            _record_metrics(context, round_number, round_metrics, excel_filename)

            logger.info(
                "Round %s | GLOBAL | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Loss: %.4f",
                round_number, metrics["Acc"], metrics["F1"],
                metrics["Precision"], metrics["Recall"], metrics["Loss"],
            )
            print(
                f"{COLORS.OKGREEN}Round {round_number} | Acc={metrics['Acc']:.4f}, F1={metrics['F1']:.4f}, "
                f"Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, Loss={metrics['Loss']:.4f}{COLORS.ENDC}"
            )
        else:
            expected = [os.path.join(round_dir, f"client_{i}_weight.bin") for i in range(n_clients)]
            if not all(os.path.exists(p) for p in expected):
                logger.info(f"Round {round_number} | SKIPPED (incomplete weight records)")
                print(f"{COLORS.WARNING}Round {round_number}: skipped (incomplete weight records){COLORS.ENDC}")
                continue

            all_client_metrics = []
            for client_id in range(n_clients):
                with open(expected[client_id], "rb") as _f:
                    weights = pickle.load(_f)
                eval_model.set_weights(weights)
                del weights

                metrics = evaluate_model(eval_model, test_dataset, test_labels)
                all_client_metrics.append(metrics)
                logger.info(
                    "Round %s | Client %s | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Loss: %.4f",
                    round_number, client_id,
                    metrics["Acc"], metrics["F1"], metrics["Precision"], metrics["Recall"], metrics["Loss"],
                )
                print(
                    f"{COLORS.OKGREEN}Client {client_id}: Acc={metrics['Acc']:.4f}, F1={metrics['F1']:.4f}, "
                    f"Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, Loss={metrics['Loss']:.4f}{COLORS.ENDC}"
                )

            avg_metrics = {k: np.mean([m[k] for m in all_client_metrics]) for k in ("Acc", "F1", "Precision", "Recall", "Loss")}
            round_metrics = {i: all_client_metrics[i] for i in range(n_clients)}
            round_metrics[-1] = avg_metrics
            _record_metrics(context, round_number, round_metrics, excel_filename)

            logger.info(
                "Round %s | Avg | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Loss: %.4f",
                round_number, avg_metrics["Acc"], avg_metrics["F1"],
                avg_metrics["Precision"], avg_metrics["Recall"], avg_metrics["Loss"],
            )
            print(
                f"{COLORS.OKGREEN}Round {round_number} Avg | Acc={avg_metrics['Acc']:.4f}, F1={avg_metrics['F1']:.4f}, "
                f"Precision={avg_metrics['Precision']:.4f}, Recall={avg_metrics['Recall']:.4f}, Loss={avg_metrics['Loss']:.4f}{COLORS.ENDC}"
            )

    del eval_model, test_dataset, test_labels
    aggressive_memory_cleanup()
    log_timestamp(logger, "=== SIMULATION COMPLETED ===")
    print(f"{COLORS.OKCYAN}Results saved to {excel_filename}{COLORS.ENDC}")
    print(f"{COLORS.OKGREEN}Simulation completed!{COLORS.ENDC}")


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
