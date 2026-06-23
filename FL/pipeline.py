import dataclasses
import glob
import logging
import os
import pickle
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from .backend import get_torch_loader_kwargs as _torch_loader_kwargs, use_tf as _use_tf
if _use_tf():
    import tensorflow as tf

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
from .evaluation import compute_aurc, create_enhanced_excel_report, evaluate_model_with_metrics, plot_f1_threshold_curve
from .gpu import configure_gpu_memory, configure_gpu
from .logging_utils import log_timestamp, setup_logger
from .memory import aggressive_memory_cleanup, clear_session
from .model_factory import create_model
from .decentralized import (
    braintorrent_select_server, log_braintorrent_selection,
    compute_model_similarity_scores, select_model_similarity_server, log_model_similarity_selection,
)
from .poison_utils import parse_poison_config, get_or_create_poisoned_clients, PoisonedDataLoader, apply_gradient_scale_poison, PoisonedFLState


_STANDARD_EVAL_COLUMNS = ['Round', 'Loss', 'Accuracy', 'F1_Score', 'Precision', 'Recall', 'AUPRC', 'ECE']
_STANDARD_EVAL_REQUIRED_COLUMNS = ('Loss', 'Accuracy', 'F1_Score', 'Precision', 'Recall', 'AUPRC', 'ECE')
_DISTILL_EVAL_REQUIRED_SUFFIXES = ('Acc', 'F1', 'Precision', 'Recall', 'ECE', 'Loss')
_POISONEDFL_STATE_KEY = "__poisoned_fl_state__"


def _record_round_weights(log_filename, round_num, global_weights=None, context=None, keep_last_rounds=0):
    stem = os.path.splitext(os.path.basename(log_filename))[0]
    record_dir = os.path.join("temp_weights", f"{stem}_weight_record", f"round_{round_num}")
    os.makedirs(record_dir, exist_ok=True)

    if global_weights is not None:
        path = os.path.join(record_dir, "global_weight.bin")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(global_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        _cleanup_old_weight_rounds(os.path.join("temp_weights", f"{stem}_weight_record"), keep_last_rounds)
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
            continue

    if saved_any or context.model_pool is None:
        _cleanup_old_weight_rounds(os.path.join("temp_weights", f"{stem}_weight_record"), keep_last_rounds)
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

    _cleanup_old_weight_rounds(os.path.join("temp_weights", f"{stem}_weight_record"), keep_last_rounds)


def _cleanup_old_weight_rounds(record_base: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    round_dirs = []
    for entry in os.listdir(record_base):
        if entry.startswith("round_"):
            full = os.path.join(record_base, entry)
            if os.path.isdir(full):
                try:
                    round_num = int(entry.split("_")[1])
                    round_dirs.append((round_num, full))
                except (IndexError, ValueError):
                    continue
    round_dirs.sort(key=lambda x: x[0])
    for _, dirpath in round_dirs[:-keep_last]:
        shutil.rmtree(dirpath, ignore_errors=True)
        print(f"{COLORS.WARNING}Cleaned up {dirpath}{COLORS.ENDC}")


def _config_fingerprint(config) -> str:
    import hashlib
    d = dataclasses.asdict(config)
    d.pop('checkpoint', None)
    d.pop('rounds', None)
    d.pop('skip_eval', None)
    d.pop('fresh_run', None)
    d.pop('robust_workers', None)
    d.pop('f1_curve', None)
    d.pop('cache_test_set', None)
    raw = str(sorted(d.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _config_fingerprint_candidates(config):
    import hashlib
    d = dataclasses.asdict(config)
    for key in ('checkpoint', 'rounds', 'skip_eval', 'fresh_run', 'robust_workers', 'f1_curve', 'cache_test_set'):
        d.pop(key, None)
    candidates = {hashlib.sha256(str(sorted(d.items())).encode()).hexdigest()[:16]}
    if d.get('algorithm') == 'Ours' and str(d.get('robust_filter_reference', 'Blom')).lower() == 'blom':
        legacy = dict(d)
        legacy.pop('robust_filter_reference', None)
        legacy['robust_filter_v1'] = False
        legacy['robust_filter_v2'] = False
        legacy['robust_filter_v3'] = True
        legacy['robust_filter_v4'] = False
        candidates.add(hashlib.sha256(str(sorted(legacy.items())).encode()).hexdigest()[:16])
    return candidates


def _allow_eval_shortcut_from_records(ckpt_info_path: str, config) -> tuple[bool, Dict[str, str]]:
    if not os.path.exists(ckpt_info_path):
        return True, {}
    info = {}
    with open(ckpt_info_path) as f:
        for line in f:
            key, _, val = line.strip().partition(": ")
            if key:
                info[key] = val
    saved_hash = info.get("config_hash", "")
    current_hashes = _config_fingerprint_candidates(config)
    if saved_hash and saved_hash not in current_hashes:
        return False, info
    saved_backend = info.get("backend", "")
    current_backend = "tensorflow" if _use_tf() else "pytorch"
    if saved_backend and saved_backend != current_backend:
        return False, info
    try:
        saved_round = int(info.get("round", "0"))
    except ValueError:
        return False, info
    round_complete = info.get("round_complete", "true") == "true"
    return round_complete and saved_round >= config.rounds, info


@dataclass
class FLConfig:
    n_clients: int = 10
    partition_type: str = "iid-10"
    rounds: int = 10
    strategy: str = "FedAvg"
    feddyn_alpha: float = 0.1
    batch_size: int = 8192
    epochs: int = 5
    mu: float = 0.01

    model: str = "dense"
    robust_epsilon: float = 0.2
    robust_rm_budget: Optional[int] = None
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
    fresh_run: bool = False
    cache_test_set: bool = False
    flame_lambda: float = 0.001
    flame_passive_cluster: bool = False
    keep_last_rounds: int = 2
    save_weights: bool = True

    def to_strategy_params(self) -> Dict[str, object]:
        return {
            'mu': self.mu,

            'feddyn_alpha': self.feddyn_alpha,
            'epsilon': self.robust_epsilon,
            'robust_rm_budget': self.robust_rm_budget,
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
            'flame_lambda': self.flame_lambda,
            'flame_passive_cluster': self.flame_passive_cluster,
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
        self.per_client_loaders: Dict[int, Optional[PoisonedDataLoader]] = {}
        self.test_dataset = None
        self.eval_batch_size: Optional[int] = None
        self.class_names: List[str] = []
        self.partition_label: Optional[str] = None
        self.current_round = 0
        self.poison_attack: Optional[str] = None
        self.poison_value: Optional[float] = None
        self.poison_ratio: Optional[float] = None
        # In-memory round weights (avoids disk I/O when save_weights=False)
        self.round_weights_history: Dict[int, list] = {}
        # Secure Aggregation state
        self.dh_private_keys: Dict[int, int] = {}
        self.dh_public_keys: Dict[int, int] = {}
        self.poisoned_fl_state: Optional[PoisonedFLState] = None

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
        if not _use_tf():
            return {}
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
            clear_session()
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
            test_loss, accuracy, f1, precision, recall = metrics[:5]
            
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
        if self.poisoned_fl_state is not None:
            self._atomic_pickle(os.path.join(ckpt_dir, "poisoned_fl_state.bin"), self.poisoned_fl_state)

        info_lines = [
            f"round: {round_num}",
            f"round_complete: false",
            f"last_client: {client_idx}",
            f"stream: {stream}",
            f"total_rounds: {self.config.rounds}",
            f"config_hash: {_config_fingerprint(self.config)}",
            f"strategy: {self.config.strategy}",
            f"n_clients: {n_clients}",
            f"backend: {'tensorflow' if _use_tf() else 'pytorch'}",
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
            f"backend: {'tensorflow' if _use_tf() else 'pytorch'}",
        ]
        with open(os.path.join(ckpt_dir, "info.txt"), "w") as f:
            f.write("\n".join(info_lines) + "\n")

        if ms_prev_scores is not None:
            self._atomic_pickle(os.path.join(ckpt_dir, "ms_prev_scores.bin"), ms_prev_scores)
        if self.poisoned_fl_state is not None:
            self._atomic_pickle(os.path.join(ckpt_dir, "poisoned_fl_state.bin"), self.poisoned_fl_state)

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
        current_hashes = _config_fingerprint_candidates(self.config)
        if saved_hash and saved_hash not in current_hashes:
            print(f"{COLORS.WARNING}No checkpoint found, starting fresh{COLORS.ENDC}")
            return None

        saved_backend = info.get("backend", "")
        current_backend = "tensorflow" if _use_tf() else "pytorch"
        if saved_backend and saved_backend != current_backend:
            print(f"{COLORS.WARNING}Checkpoint backend mismatch (saved={saved_backend}, current={current_backend}) "
                  f"— weight arrays are incompatible across backends. Starting fresh{COLORS.ENDC}")
            return None

        with open(os.path.join(ckpt_dir, "global_weights.bin"), "rb") as f:
            latest_weights = pickle.load(f)

        ms_path = os.path.join(ckpt_dir, "ms_prev_scores.bin")
        ms_prev_scores = None
        if os.path.exists(ms_path):
            with open(ms_path, "rb") as f:
                ms_prev_scores = pickle.load(f)

        pfl_state_path = os.path.join(ckpt_dir, "poisoned_fl_state.bin")
        if os.path.exists(pfl_state_path):
            with open(pfl_state_path, "rb") as f:
                self.poisoned_fl_state = pickle.load(f)
        elif self.poison_attack == "poisonedfl":
            print(f"{COLORS.WARNING}Checkpoint missing PoisonedFL adaptive state; resume will restart attack adaptation{COLORS.ENDC}")

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
            clear_session()
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
        poison_loader = self.per_client_loaders.get(client_id, self.poison_loader if poisoned else None)
        if poisoned and self.poison_attack in ("label_flip", "targeted_flip"):
            print(f"  \u26a0\ufe0f  POISONED CLIENT - Labels flipped ({self.poison_attack})")
        if poisoned and self.poison_attack == "gradient_scale":
            scale = self.poison_value or 1.0
            print(f"  \u26a0\ufe0f  POISONED CLIENT - Scaling gradients x{scale:.1f}")

        if poisoned and self.poison_attack == "poisonedfl" and self.poisoned_fl_state is not None and latest_weights is not None:
            _mmap = np.load(paths['train_X'], mmap_mode='r')
            sample_size = _mmap.shape[0]
            del _mmap
            if self.poisoned_fl_state.cached_update is not None:
                poisoned_weights = self._apply_poison_to_weights(latest_weights, latest_weights, client_id)
            else:
                poisoned_weights = latest_weights
            if self.config.strategy == "FLTrust":
                result_data = [nw - ow for nw, ow in zip(poisoned_weights, latest_weights)]
            else:
                result_data = poisoned_weights
            if reuse_model is None:
                del model
            aggressive_memory_cleanup()
            print(f"{COLORS.WARNING}Client {client_id}, (poisonedfl attack) - skipping training and returning poisoned weights{COLORS.ENDC}")
            return result_data, sample_size, 0.0

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
            callbacks = []
            if _use_tf():
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor='loss',
                        patience=3,
                        restore_best_weights=True,
                        verbose=1
                    )
                )
            history = model.fit(
                train_dataset,
                epochs=self.config.epochs,
                callbacks=callbacks if callbacks else None,
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

    _NEEDS_ALL_WEIGHTS = frozenset({'feddyn', 'fltrust', 'fedcomed', 'robustfilter', 'flame'})

    def _can_stream_aggregate(self) -> bool:
        """Check if strategy supports incremental (streaming) aggregation."""
        strategy_name = getattr(self.strategy_runtime.client_strategy, 'name', '').lower()
        if strategy_name in self._NEEDS_ALL_WEIGHTS:
            return False
        if self.config.strategy.lower() in self._NEEDS_ALL_WEIGHTS:
            return False
        if self.config.peer_trust:
            return False
        if self.config.decentralized == "ModelSimilarity":
            return False
        return True

    def _aggregate(self, weights_list, sample_sizes, participating_clients, global_update=None, models=None, prev_global=None):
        aggregator = self.strategy_runtime.aggregator
        
        # FLTrust requires global_update parameter
        if self.config.strategy == "FLTrust":
            return aggregator.aggregate(weights_list, sample_sizes, global_update=global_update)
        elif self.config.strategy == "FLAME":
            return aggregator.aggregate(weights_list, sample_sizes, prev_global=prev_global, logger=self.logger, passive_cluster=self.config.flame_passive_cluster)
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
        y_oh = np.zeros((len(combined_y), num_classes), dtype=np.float32)
        y_oh[np.arange(len(combined_y)), combined_y.astype(np.int32)] = 1.0
        self._root_dataset_cache = (combined_X, y_oh)
        return self._root_dataset_cache

    def _train_server_on_root_dataset(self, current_weights, input_dim, num_classes):
        clear_session()
        
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
        
        root_dataset = None
        if _use_tf():
            root_dataset = (tf.data.Dataset.from_tensor_slices((combined_X, combined_y))
                            .batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE))
        else:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            ds = TensorDataset(torch.from_numpy(combined_X), torch.from_numpy(combined_y))
            root_dataset = DataLoader(ds, batch_size=self.config.batch_size, shuffle=False,
                                      **_torch_loader_kwargs())
        
        print(f"  [SERVER] Training on root dataset ({len(combined_X)} samples) for {self.config.root_iterations} iterations")
        
        model.fit(root_dataset, epochs=self.config.root_iterations, verbose=1)
        
        new_weights = model.get_weights()
        if current_weights is None:
            global_update = new_weights
        else:
            global_update = [new_w - old_w for new_w, old_w in zip(new_weights, current_weights)]
        
        del model, root_dataset
        aggressive_memory_cleanup()
        
        return global_update

    def _prepare_poisonedfl_round(self, latest_weights):
        state = self.poisoned_fl_state
        current_flat = np.concatenate([w.ravel() for w in latest_weights]).astype(np.float32)
        mal_update = state.compute_update(current_flat)
        state.cached_update = mal_update

    def _apply_poison_to_weights(self, weights, latest_weights, client_id):
        if self.poison_attack == "gradient_scale" and client_id in self.poisoned_clients:
            return apply_gradient_scale_poison(weights, self.poison_value or 1.0)
        if self.poison_attack == "poisonedfl" and client_id in self.poisoned_clients:
            state = self.poisoned_fl_state
            if state.cached_update is not None and latest_weights is not None:
                state.has_applied_poison = True
                offset, poisoned = 0, []
                for w in latest_weights:
                    n = w.size
                    poisoned.append((w.ravel() + state.cached_update[offset:offset + n]).reshape(w.shape).astype(w.dtype))
                    offset += n
                return poisoned
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
            clear_session()
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
        configure_gpu()

        if not getattr(self.config, 'save_weights', True):
            print(f"{COLORS.WARNING}⚠️  --no_save_weights enabled: round weights stored in memory only, crash recovery disabled for rounds{COLORS.ENDC}")

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
        if getattr(self.config, 'mixed_models', False):
            extra_tokens.append("mixed_models")
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
        if attack_type == "lma":
            raise ValueError("LMA is only supported for Ours, FedDistill, SSFL-IDS, and FedKD-IDS")
        if attack_type:
            self.poisoned_clients = get_or_create_poisoned_clients(
                partition_label, attack_type, poison_value, poison_ratio, n_clients
            )
            if attack_type == "label_flip":
                self.poison_loader = PoisonedDataLoader(attack_type, num_classes)
            elif attack_type == "targeted_flip":
                target_label = int(poison_value)
                for cid in self.poisoned_clients:
                    y = np.load(paths_list[cid]['train_y'], mmap_mode='r').astype(np.int32)
                    dominant = int(np.argmax(np.bincount(y, minlength=num_classes)))
                    del y
                    self.per_client_loaders[cid] = PoisonedDataLoader(
                        "targeted_flip", num_classes,
                        dominant_class=dominant, target_label=target_label,
                    )
            elif attack_type == "poisonedfl":
                self.poisoned_fl_state = PoisonedFLState(c0=poison_value)
            log_timestamp(self.logger, f"Poisoned clients: {self.poisoned_clients}")
            print(f"\n  POISONING ENABLED: {attack_type} attack (value={poison_value})")
            print(f"    Poisoned clients: {self.poisoned_clients} ({len(self.poisoned_clients)}/{n_clients})")
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

        if self.config.fresh_run:
            stem = os.path.splitext(os.path.basename(self.log_filename))[0]
            _wr_base = os.path.join("temp_weights", f"{stem}_weight_record")
            if os.path.isdir(_wr_base):
                shutil.rmtree(_wr_base)
            _ckpt_base = self._checkpoint_dir()
            if os.path.isdir(_ckpt_base):
                shutil.rmtree(_ckpt_base)
            _clear_eval_artifacts(self.log_filename.replace('.log', '.xlsx'))
            print(f"{COLORS.OKCYAN}fresh_run: cleared weight records, checkpoint, and eval artifacts{COLORS.ENDC}")

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
        final_round_dir = os.path.join(record_base, f"round_{self.config.rounds}")
        ckpt_info_path = os.path.join("checkpoint", stem, "info.txt")
        _allow_final_record_shortcut, _shortcut_info = _allow_eval_shortcut_from_records(ckpt_info_path, self.config)
        _final_exists = os.path.exists(os.path.join(final_round_dir, "global_weight.bin"))
        if _final_exists and _allow_final_record_shortcut:
            log_timestamp(self.logger, f"Weight records found (round {self.config.rounds}), skipping training")
            print(f"{COLORS.OKGREEN}Weight records found up to round {self.config.rounds} — skipping to evaluation{COLORS.ENDC}")
            self._run_eval_from_records(
                input_dim, num_classes, class_names, partition_label, excel_filename, record_base,
            )
            return
        if _final_exists and not _allow_final_record_shortcut:
            _stage = _shortcut_info.get("stage", "?")
            print(f"{COLORS.WARNING}Final-round weight records exist, but checkpoint is not fully complete (stage={_stage}) — resuming training{COLORS.ENDC}")

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

            if self.poison_attack == "poisonedfl" and latest_weights is not None:
                self._prepare_poisonedfl_round(latest_weights)

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
                        
                        predictions = global_model_for_mclass.predict(batch_X, batch_size=self.config.batch_size)
                        pred_labels = np.argmax(predictions, axis=1)
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
                clear_session()
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
                    clear_session()
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
                
                aggregated_result = self._aggregate(weights_list, sample_sizes, participating_clients, global_update, models_list, prev_global=latest_weights)
                
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

            # Store weights in memory (always, for evaluation)
            self.round_weights_history[round_num] = latest_weights

            # Conditionally write to disk (if save_weights is True)
            if getattr(self.config, 'save_weights', True):
                _record_round_weights(self.log_filename, round_num, global_weights=latest_weights, keep_last_rounds=self.config.keep_last_rounds)

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
        clear_session()
        aggressive_memory_cleanup()

        stem = os.path.splitext(os.path.basename(self.log_filename))[0]
        record_base = os.path.join("temp_weights", f"{stem}_weight_record")
        self._run_eval_from_records(
            input_dim, num_classes, class_names, partition_label, excel_filename, record_base,
        )

        # Conservative cleanup: only delete current run's temp_weights if save_weights=False
        if not getattr(self.config, 'save_weights', True) and os.path.isdir(record_base):
            shutil.rmtree(record_base, ignore_errors=True)
            print(f"{COLORS.OKCYAN}Cleaned up {record_base} (save_weights=False){COLORS.ENDC}")

    def _run_eval_from_records(self, input_dim, num_classes, class_names, partition_label, excel_filename, record_base):
        log_timestamp(self.logger, "=== EVALUATION ===")
        print(f"\n{COLORS.HEADER}[EVALUATION]{COLORS.ENDC}")
        progress_path = _eval_progress_path(excel_filename)

        # Resume: load any previously completed rounds
        evaluated_rounds: set = set()
        if self.config.fresh_run:
            self.results_df = pd.DataFrame(columns=_STANDARD_EVAL_COLUMNS)
        elif os.path.exists(progress_path):
            try:
                self.results_df = pd.read_pickle(progress_path)
            except Exception:
                self.results_df = pd.DataFrame(columns=_STANDARD_EVAL_COLUMNS)
        elif os.path.exists(excel_filename):
            try:
                try:
                    existing_df = pd.read_excel(excel_filename, sheet_name='Overall_Metrics')
                except Exception:
                    existing_df = pd.read_excel(excel_filename, sheet_name=0)
                self.results_df = existing_df
            except Exception:
                self.results_df = pd.DataFrame(columns=_STANDARD_EVAL_COLUMNS)
        else:
            self.results_df = pd.DataFrame(columns=_STANDARD_EVAL_COLUMNS)

        evaluated_rounds = _completed_standard_eval_rounds(self.results_df)
        if not self.results_df.empty and 'Round' in self.results_df.columns:
            keep_rows = self.results_df['Round'].astype(int).isin(evaluated_rounds)
            self.results_df = self.results_df.loc[keep_rows].copy()
        if evaluated_rounds:
            print(f"{COLORS.OKCYAN}Resuming eval — {len(evaluated_rounds)} rounds already done{COLORS.ENDC}")

        eval_batch_size = min(self.config.batch_size, 2048)
        test_dataset = load_test_dataset(eval_batch_size, num_classes)
        y_true_cache = np.load("data/y_test.npy").astype(np.int32).ravel()

        eval_model = create_model(
            architecture=self.config.model,
            input_dim=input_dim,
            num_classes=num_classes,
            batch_size=self.config.batch_size,
            strategy_runtime=self.strategy_runtime,
            client_id=None,
        )

        final_per_class_metrics = None
        final_confusion_mat = None

        # evaluate in reverse order (last round -> first round)
        for round_num in range(self.config.rounds, 0, -1):
            if round_num in evaluated_rounds:
                print(f"{COLORS.OKCYAN}Round {round_num}: already evaluated, skipping{COLORS.ENDC}")
                continue

            if self.config.skip_eval and round_num != self.config.rounds:
                continue

            # Try in-memory weights first (always populated during training)
            if round_num in self.round_weights_history:
                weights = self.round_weights_history[round_num]
            else:
                # Fallback: load from disk (for resumed runs or when save_weights=True)
                weight_path = os.path.join(record_base, f"round_{round_num}", "global_weight.bin")
                if not os.path.exists(weight_path):
                    self.logger.info(f"Round {round_num} | SKIPPED (no weight record)")
                    print(f"{COLORS.WARNING}Round {round_num}: skipped (no weight record){COLORS.ENDC}")
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
                y_true_cache=y_true_cache,
                compute_aurc_curve=collect_details,
            )
            test_loss, accuracy, f1_value, precision, recall = result[:5]
            per_class_metrics = result[5]
            confusion_mat = result[6]
            class_report = result[7] if len(result) > 7 else ""
            auprc = result[8] if len(result) > 8 else 0.0
            ece = result[9] if len(result) > 9 else 0.0
            aurc = result[10]
            tpr = result[11] if len(result) > 11 else 0.0
            fpr = result[12] if len(result) > 12 else 0.0

            _aurc_str = f" | AURC: {aurc:.4f}" if aurc is not None else ""
            self.logger.info(
                f"Round {round_num} | GLOBAL | Acc: {accuracy:.4f} | F1: {f1_value:.4f} | "
                f"Precision: {precision:.4f} | Recall: {recall:.4f} | TPR: {tpr:.4f} | FPR: {fpr:.4f} | AUPRC: {auprc:.4f} | ECE: {ece:.4f} | Loss: {test_loss:.4f}{_aurc_str}"
            )
            print(
                f"{COLORS.OKGREEN}Round {round_num} | Acc={accuracy:.4f}, F1={f1_value:.4f}, "
                f"Precision={precision:.4f}, Recall={recall:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}, AUPRC={auprc:.4f}, ECE={ece:.4f}, Loss={test_loss:.4f}{_aurc_str}{COLORS.ENDC}"
            )
            if class_report and self.detailed_logger is not None:
                self.detailed_logger.info(f"Round {round_num}\n{class_report}")

            if not self.results_df.empty and 'Round' in self.results_df.columns:
                self.results_df = self.results_df.loc[self.results_df['Round'].astype(int) != round_num].copy()
            new_row = pd.DataFrame({
                'Round': [round_num], 'Loss': [test_loss], 'Accuracy': [accuracy],
                'F1_Score': [f1_value], 'Precision': [precision], 'Recall': [recall],
                'AUPRC': [auprc], 'ECE': [ece], 'AURC': [aurc], 'TPR': [tpr], 'FPR': [fpr],
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

            if round_num == self.config.rounds and per_class_metrics is not None:
                final_per_class_metrics = per_class_metrics
                final_confusion_mat = confusion_mat

            self.results_df.to_pickle(progress_path)

        if getattr(self.config, 'f1_curve', False):
            _f1_log_stem = os.path.splitext(os.path.basename(self.log_filename))[0]
            plot_path = os.path.join("results", "plots", _f1_log_stem, "global_model.png")
            weight_path = os.path.join(record_base, f"round_{self.config.rounds}", "global_weight.bin")
            if os.path.exists(weight_path):
                from .context import get_model_proba
                X_test_np = np.load("data/X_test.npy").astype(np.float32)
                with open(weight_path, "rb") as _f:
                    weights = pickle.load(_f)
                eval_model.set_weights(weights)
                del weights
                proba = get_model_proba(eval_model, X_test_np, self.config.batch_size)
                del X_test_np
                _best_pt, _min_cov_pt = plot_f1_threshold_curve(y_true_cache, proba, plot_path, label=self.config.strategy, logger=self.logger)
                _global_aurc = compute_aurc(y_true_cache, proba)
                del proba
                self.logger.info(
                    "F1_CURVE | GLOBAL | Best θ=%.2f Acc=%.4f F1=%.4f P=%.4f R=%.4f Cov=%.4f | MinCov θ=%.2f Acc=%.4f F1=%.4f P=%.4f R=%.4f Cov=%.4f | AURC=%.4f",
                    _best_pt["theta"], _best_pt["Acc"], _best_pt["F1"], _best_pt["Precision"], _best_pt["Recall"], _best_pt["Coverage"],
                    _min_cov_pt["theta"], _min_cov_pt["Acc"], _min_cov_pt["F1"], _min_cov_pt["Precision"], _min_cov_pt["Recall"], _min_cov_pt["Coverage"],
                    _global_aurc,
                )
                print(
                    f"{COLORS.OKGREEN}F1_CURVE Global | Best θ={_best_pt['theta']:.2f}: Acc={_best_pt['Acc']:.4f} F1={_best_pt['F1']:.4f} "
                    f"P={_best_pt['Precision']:.4f} R={_best_pt['Recall']:.4f} Cov={_best_pt['Coverage']:.4f} | "
                    f"MinCov θ={_min_cov_pt['theta']:.2f}: Acc={_min_cov_pt['Acc']:.4f} F1={_min_cov_pt['F1']:.4f} "
                    f"P={_min_cov_pt['Precision']:.4f} R={_min_cov_pt['Recall']:.4f} Cov={_min_cov_pt['Coverage']:.4f} | AURC={_global_aurc:.4f}{COLORS.ENDC}"
                )

        if not self.results_df.empty:
            if final_per_class_metrics is not None:
                create_enhanced_excel_report(
                    excel_filename, self.results_df, final_per_class_metrics,
                    class_names, self.config.rounds, final_confusion_mat,
                )
            else:
                self.results_df.to_excel(excel_filename, index=False)
        if os.path.exists(progress_path):
            os.remove(progress_path)

        del eval_model, test_dataset, y_true_cache
        aggressive_memory_cleanup()
        log_timestamp(self.logger, "=== SIMULATION COMPLETED ===")
        print(f"{COLORS.OKCYAN}Results saved to {excel_filename}{COLORS.ENDC}")
        print(f"{COLORS.OKGREEN}Simulation completed!{COLORS.ENDC}")


def run_pipeline(config: FLConfig) -> None:
    pipeline = FederatedLearningPipeline(config)
    pipeline.run()


def _eval_progress_path(excel_filename: str) -> str:
    return os.path.splitext(excel_filename)[0] + '.progress.pkl'


def _clear_eval_artifacts(excel_filename: str) -> None:
    progress_path = _eval_progress_path(excel_filename)
    if os.path.exists(excel_filename):
        os.remove(excel_filename)
    if os.path.exists(progress_path):
        os.remove(progress_path)


def _completed_standard_eval_rounds(results_df: pd.DataFrame) -> set:
    if results_df is None or results_df.empty or 'Round' not in results_df.columns:
        return set()
    if any(col not in results_df.columns for col in _STANDARD_EVAL_REQUIRED_COLUMNS):
        return set()
    complete_rows = results_df.loc[:, list(_STANDARD_EVAL_REQUIRED_COLUMNS)].notna().all(axis=1)
    return set(results_df.loc[complete_rows, 'Round'].astype(int).tolist())


def _load_distill_results_from_excel(excel_filename: str) -> Dict[int, Dict[str, float]]:
    existing = pd.read_excel(excel_filename, sheet_name=0, index_col=0)
    loaded: Dict[int, Dict[str, float]] = {}
    for client_id, row in existing.iterrows():
        cid = client_id.item() if isinstance(client_id, np.generic) else client_id
        if isinstance(cid, float) and cid.is_integer():
            cid = int(cid)
        if isinstance(cid, str):
            try:
                cid = int(cid)
            except ValueError:
                pass
        metrics = {str(col): value for col, value in row.items() if not pd.isna(value)}
        loaded[cid] = metrics
    return loaded


def _round_has_distill_eval_metrics(metrics_dict: Dict[str, float], round_number: int) -> bool:
    prefix = f"Round_{round_number}_"
    return all(f"{prefix}{suffix}" in metrics_dict for suffix in _DISTILL_EVAL_REQUIRED_SUFFIXES)


def _completed_distill_eval_rounds(results: Dict[int, Dict[str, float]]) -> set:
    if not isinstance(results, dict) or not results:
        return set()

    candidate_rounds = set()
    for metrics_dict in results.values():
        if not isinstance(metrics_dict, dict):
            continue
        for key in metrics_dict:
            match = re.match(r'Round_(\d+)_', str(key))
            if match:
                candidate_rounds.add(int(match.group(1)))

    completed = set()
    summary_ids = (-3, -2, -1)
    for round_number in sorted(candidate_rounds):
        if any(
            isinstance(results.get(client_id), dict)
            and _round_has_distill_eval_metrics(results[client_id], round_number)
            for client_id in summary_ids
        ):
            completed.add(round_number)
            continue
        if any(
            isinstance(metrics_dict, dict) and _round_has_distill_eval_metrics(metrics_dict, round_number)
            for metrics_dict in results.values()
        ):
            completed.add(round_number)
    return completed


def _clear_distill_round_metrics(results: Dict[int, Dict[str, float]], round_number: int) -> None:
    if not isinstance(results, dict) or not results:
        return
    prefix = f"Round_{round_number}_"
    empty_ids = []
    for client_id, metrics_dict in list(results.items()):
        if not isinstance(metrics_dict, dict):
            continue
        stale_keys = [key for key in metrics_dict if str(key).startswith(prefix)]
        for key in stale_keys:
            del metrics_dict[key]
        if not metrics_dict:
            empty_ids.append(client_id)
    for client_id in empty_ids:
        del results[client_id]


def run_distillation_pipeline(config, strategy) -> None:
    configure_gpu()
    from .config import FDConfig
    from .context import evaluate_model, ModelPool
    from .strategy.common import create_model as create_strategy_model
    
    partition_label, client_count = parse_partition_type(config.partition_type)
    n_clients = min(config.n_clients, client_count)
    
    extra_log_tokens = strategy.extra_log_tokens()
    extra_tokens = [config.model_type, *list(extra_log_tokens.values())]
    if getattr(config, 'mixed_models', False):
        extra_tokens.insert(0, "mixed_models")

    poison_suffix = ""
    if config.poison:
        poison_suffix = config.poison.replace(" ", "_")

    logger, log_filename, detailed_logger = setup_logger(
        algorithm_name=strategy.name,
        n_clients=n_clients,
        partition_label=partition_label,
        extra_tokens=extra_tokens,
        poison_suffix=poison_suffix,
        resume=not getattr(config, 'fresh_run', False),
        create_detailed_log=False,
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
        if getattr(config, 'mixed_models', False):
            from .context import MixedModelPool
            from models.mixed_models import get_model_type_for_client
            archs = set(get_model_type_for_client(i, n_clients) for i in range(n_clients))
            factory_map = {a: (lambda _a=a: create_strategy_model(input_dim, num_classes, config.batch_size, model_type=_a)) for a in archs}
            model_pool = MixedModelPool(n_clients, factory_map)
        else:
            model_pool = ModelPool(
                pool_size=min(10, n_clients),
                factory_fn=lambda: create_strategy_model(input_dim, num_classes, config.batch_size, model_type=model_type),
                in_memory=True,
            )
    
    poisoned_clients = []
    poison_loader = None
    per_client_loaders = {}
    attack_type, poison_value, poison_ratio = parse_poison_config(config.poison)
    if attack_type == "lma" and strategy.name not in {"Ours", "FedDistill", "SSFL-IDS", "FedKD-IDS"}:
        raise ValueError("LMA is only supported for Ours, FedDistill, SSFL-IDS, and FedKD-IDS")
    if attack_type:
        poisoned_clients = get_or_create_poisoned_clients(
            partition_label, attack_type, poison_value, poison_ratio, n_clients
        )
        if attack_type == "label_flip":
            poison_loader = PoisonedDataLoader(attack_type, num_classes)
        elif attack_type == "targeted_flip":
            target_label = int(poison_value)
            for cid in poisoned_clients:
                y = np.load(paths_list[cid]['train_y'], mmap_mode='r').astype(np.int32)
                dominant = int(np.argmax(np.bincount(y, minlength=num_classes)))
                del y
                per_client_loaders[cid] = PoisonedDataLoader(
                    "targeted_flip", num_classes,
                    dominant_class=dominant, target_label=target_label,
                )
        _pc_display = poisoned_clients[:10] if len(poisoned_clients) > 10 else poisoned_clients
        _pc_str = str(_pc_display) + (" ..." if len(poisoned_clients) > 10 else "")
        print(f"\n  POISONING ENABLED: {attack_type} attack (value={poison_value})")
        print(f"    Poisoned clients ({len(poisoned_clients)}/{n_clients}): {_pc_str}")
        logger.info("POISONING: %s attack value=%s | %d/%d clients: %s",
                    attack_type, poison_value, len(poisoned_clients), n_clients, _pc_str)
    
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
        per_client_loaders=per_client_loaders,
        model_pool=model_pool,
        poisoned_fl_state=PoisonedFLState(c0=poison_value) if attack_type == "poisonedfl" else None,
    )
    
    log_timestamp(logger, "SIMULATION STARTED")
    log_timestamp(logger, f"Clients: {n_clients}, Rounds: {config.rounds}, Partition: {partition_label}")
    
    if context.shared_state.get("extra_log_tokens"):
        log_timestamp(logger, f"Extra config: {context.shared_state['extra_log_tokens']}")
    
    if not getattr(config, 'save_weights', True):
        print(f"{COLORS.WARNING}⚠️  --no_save_weights enabled: round weights stored in memory only, crash recovery disabled for rounds{COLORS.ENDC}")
        logger.info("--no_save_weights enabled: round weights stored in memory only, crash recovery disabled for rounds")
    
    strategy.setup(context)
    
    start_round = 1
    ckpt_stem = os.path.splitext(os.path.basename(log_filename))[0]
    ckpt_dir = os.path.join("checkpoint", ckpt_stem)
    ckpt_info_path = os.path.join(ckpt_dir, "info.txt")
    if getattr(config, 'fresh_run', False):
        _wr_stem = os.path.splitext(os.path.basename(log_filename))[0]
        _wr_base = os.path.join("temp_weights", f"{_wr_stem}_weight_record")
        if os.path.isdir(_wr_base):
            shutil.rmtree(_wr_base)
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        _clear_eval_artifacts(excel_filename)
        print(f"{COLORS.OKCYAN}fresh_run: cleared weight records, checkpoint, and eval artifacts{COLORS.ENDC}")
    if config.checkpoint:
        if os.path.exists(ckpt_info_path):
            ckpt_info = {}
            with open(ckpt_info_path) as f:
                for line in f:
                    key, _, val = line.strip().partition(": ")
                    ckpt_info[key] = val
            saved_hash = ckpt_info.get("config_hash", "")
            current_hashes = _config_fingerprint_candidates(config)
            saved_backend = ckpt_info.get("backend", "")
            current_backend = "tensorflow" if _use_tf() else "pytorch"
            if saved_backend and saved_backend != current_backend:
                print(f"{COLORS.WARNING}Checkpoint backend mismatch (saved={saved_backend}, current={current_backend}) "
                      f"— weight arrays are incompatible across backends. Starting fresh{COLORS.ENDC}")
            elif saved_hash and saved_hash not in current_hashes:
                print(f"{COLORS.WARNING}Checkpoint config mismatch (saved={saved_hash}, current={next(iter(current_hashes))}) \u2014 starting fresh{COLORS.ENDC}")
            else:
                _ckpt_round = int(ckpt_info["round"])
                _round_done = ckpt_info.get("round_complete", "true") == "true"
                start_round = _ckpt_round + 1 if _round_done else _ckpt_round
                shared_path = os.path.join(ckpt_dir, "shared_state.bin")
                _loaded_pfl_state = False
                if os.path.exists(shared_path) and os.path.getsize(shared_path) > 0:
                    with open(shared_path, "rb") as _f:
                        saved_shared = pickle.load(_f)
                    saved_pfl_state = saved_shared.pop(_POISONEDFL_STATE_KEY, None)
                    context.shared_state.update(saved_shared)
                    if saved_pfl_state is not None:
                        context.poisoned_fl_state = saved_pfl_state
                        _loaded_pfl_state = True
                results_path = os.path.join(ckpt_dir, "results.pkl")
                if os.path.exists(results_path):
                    context.results = pd.read_pickle(results_path)
                if attack_type == "poisonedfl" and start_round > 1 and not _loaded_pfl_state:
                    print(f"{COLORS.WARNING}Checkpoint missing PoisonedFL adaptive state; resume will restart attack adaptation{COLORS.ENDC}")
                if _round_done:
                    print(f"{COLORS.OKGREEN}Resuming from checkpoint (completed round {_ckpt_round}) -> starting round {start_round}{COLORS.ENDC}")
                else:
                    _stg = ckpt_info.get('stage', '?')
                    _sc = ckpt_info.get('stage_client', '?')
                    print(f"{COLORS.OKGREEN}Resuming incomplete round {start_round} (stage={_stg}, client={_sc}){COLORS.ENDC}")
    
    # Seed pool from weight records on resume
    # For incomplete rounds: try current round's mid-round records first
    #   (written by save_mid_round → _record_checkpoint_weights)
    # For completed rounds: use previous round's records
    if model_pool is not None:
        _wr_stem = os.path.splitext(os.path.basename(log_filename))[0]
        _wr_base = os.path.join("temp_weights", f"{_wr_stem}_weight_record")
        _seed_dir = os.path.join(_wr_base, f"round_{start_round}")
        if not os.path.isdir(_seed_dir) and start_round > 1:
            _seed_dir = os.path.join(_wr_base, f"round_{start_round - 1}")
        if os.path.isdir(_seed_dir):
            seeded = 0
            for i in range(n_clients):
                wp = os.path.join(_seed_dir, f"client_{i}_weight.bin")
                if os.path.exists(wp):
                    with open(wp, "rb") as _f:
                        model_pool._weights_cache[(i, "")] = model_pool._clone_weights(pickle.load(_f))
                    seeded += 1
            if seeded > 0:
                _seed_label = os.path.basename(_seed_dir)
                print(f"{COLORS.OKCYAN}Seeded pool from {_seed_label} weight records ({seeded}/{n_clients} clients){COLORS.ENDC}")

    # Check if final weight records already exist — skip training entirely
    stem = os.path.splitext(os.path.basename(log_filename))[0]
    record_base = os.path.join("temp_weights", f"{stem}_weight_record")
    final_round_dir = os.path.join(record_base, f"round_{config.rounds}")
    _global_model_strategy = getattr(strategy, "has_global_model", False)
    _allow_final_record_shortcut, _shortcut_info = _allow_eval_shortcut_from_records(ckpt_info_path, config)

    if _global_model_strategy:
        _final_exists = os.path.exists(os.path.join(final_round_dir, "global_weight.bin"))
    else:
        _final_exists = all(
            os.path.exists(os.path.join(final_round_dir, f"client_{i}_weight.bin"))
            for i in range(n_clients)
        )
    if _final_exists and _allow_final_record_shortcut:
        log_timestamp(logger, f"Weight records found (round {config.rounds}), skipping training")
        print(f"{COLORS.OKGREEN}Weight records found up to round {config.rounds} — skipping to evaluation{COLORS.ENDC}")
        _run_distillation_eval(config, context, logger, log_filename, excel_filename, input_dim, num_classes, n_clients, model_type, record_base, _global_model_strategy)
        return
    if _final_exists and not _allow_final_record_shortcut:
        _stage = _shortcut_info.get("stage", "?")
        print(f"{COLORS.WARNING}Final-round weight records exist, but checkpoint is not fully complete (stage={_stage}) — resuming training{COLORS.ENDC}")

    def _write_ckpt_info(rnd, complete):
        os.makedirs(ckpt_dir, exist_ok=True)
        _lines = [
            f"round: {rnd}",
            f"round_complete: {str(complete).lower()}",
            f"total_rounds: {config.rounds}",
            f"config_hash: {_config_fingerprint(config)}",
            f"strategy: {strategy.name}",
            f"n_clients: {n_clients}",
            f"partition_type: {config.partition_type}",
            f"poisoned_clients: {poisoned_clients if poisoned_clients else 'none'}",
            f"backend: {'tensorflow' if _use_tf() else 'pytorch'}",
        ]
        with open(os.path.join(ckpt_dir, "info.txt"), "w") as _f:
            _f.write("\n".join(_lines) + "\n")

    round_times: List[float] = []
    round_weights_history: Dict[int, list] = {}

    for round_number in range(start_round, config.rounds + 1):
        round_start_time = time.time()
        logger.info(f"Round {round_number}/{config.rounds}")
        print(f"\n{COLORS.HEADER}Round {round_number}/{config.rounds}{COLORS.ENDC}")
        log_timestamp(logger, f"Round {round_number} started")

        # Mark round in-progress so a crash is distinguishable from completion
        if config.checkpoint:
            _write_ckpt_info(round_number, False)
        
        round_metrics = strategy.run_round(context, round_number)
        
        if round_metrics:
            _record_metrics(context, round_number, round_metrics, excel_filename)

        if config.checkpoint:
            _write_ckpt_info(round_number, True)
            _SKIP_KEYS = {"extra_log_tokens", "global_model", "disc_pool"}
            saveable_shared = {}
            for k, v in context.shared_state.items():
                if k in _SKIP_KEYS:
                    continue
                skip = False
                if _use_tf() and isinstance(v, (tf.data.Dataset, tf.Tensor)):
                    skip = True
                if not skip:
                    try:
                        import torch
                        if isinstance(v, torch.Tensor):
                            skip = True
                    except Exception:
                        pass
                if not skip:
                    saveable_shared[k] = v
            if context.poisoned_fl_state is not None:
                saveable_shared[_POISONEDFL_STATE_KEY] = context.poisoned_fl_state
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
                gw = global_model.get_weights()
                round_weights_history[round_number] = gw
                if getattr(config, 'save_weights', True):
                    _record_round_weights(log_filename, round_number, global_weights=gw, keep_last_rounds=config.keep_last_rounds)
        else:
            context.record_client_weights(round_number)
            if config.keep_last_rounds > 0:
                stem = os.path.splitext(os.path.basename(log_filename))[0]
                _cleanup_old_weight_rounds(os.path.join("temp_weights", f"{stem}_weight_record"), config.keep_last_rounds)

        round_time = time.time() - round_start_time
        round_times.append(round_time)
        log_timestamp(logger, f"--- Round {round_number} completed in {round_time:.2f}s ---")
        print(f"{COLORS.OKGREEN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

    strategy.finalize(context)

    total_time = context.shared_state.get("pipeline_elapsed_s")
    if total_time is not None:
        log_timestamp(logger, f"Training phase completed in {total_time:.2f}s ({total_time/60:.2f}m)")
    else:
        log_timestamp(logger, "Training phase completed")
    if round_times:
        avg_round = sum(round_times) / len(round_times)
        logger.info(f"Average time per round: {avg_round:.2f}s")
        log_timestamp(logger, f"Average time per round: {avg_round:.2f}s")
        print(f"{COLORS.OKGREEN}Average time per round: {avg_round:.2f}s{COLORS.ENDC}")

    if config.checkpoint and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print(f"{COLORS.OKCYAN}Checkpoint cleaned up{COLORS.ENDC}")

    if _use_tf():
        tf.keras.backend.clear_session()  # noqa – TF-only teardown
    aggressive_memory_cleanup()

    stem = os.path.splitext(os.path.basename(log_filename))[0]
    record_base = os.path.join("temp_weights", f"{stem}_weight_record")
    _run_distillation_eval(config, context, logger, log_filename, excel_filename, input_dim, num_classes, n_clients, model_type, record_base, _global_model_strategy)

    # Conservative cleanup: only delete current run's temp_weights if save_weights=False
    if not getattr(config, 'save_weights', True) and os.path.isdir(record_base):
        shutil.rmtree(record_base, ignore_errors=True)
        print(f"{COLORS.OKCYAN}Cleaned up {record_base} (save_weights=False){COLORS.ENDC}")


def _run_distillation_eval(config, context, logger, log_filename, excel_filename, input_dim, num_classes, n_clients, model_type, record_base, global_model_eval=False):
    from .context import evaluate_model, get_model_proba
    from .evaluation import compute_aurc, plot_f1_threshold_curve
    from .strategy.common import create_model as create_strategy_model

    log_timestamp(logger, "=== EVALUATION ===")
    print(f"\n{COLORS.HEADER}[EVALUATION]{COLORS.ENDC}")
    progress_path = _eval_progress_path(excel_filename)

    if getattr(config, 'fresh_run', False):
        context.results = {}
    elif os.path.exists(progress_path):
        try:
            loaded = pd.read_pickle(progress_path)
            context.results = loaded if isinstance(loaded, dict) else {}
        except Exception:
            context.results = {}
    elif os.path.exists(excel_filename):
        try:
            context.results = _load_distill_results_from_excel(excel_filename)
        except Exception:
            context.results = {}

    if context.results is None:
        context.results = {}

    evaluated_rounds = _completed_distill_eval_rounds(context.results)
    if getattr(config, 'f1_curve', False):
        evaluated_rounds.discard(config.rounds)
    if evaluated_rounds:
        print(f"{COLORS.OKCYAN}Resuming eval — rounds already done: {sorted(evaluated_rounds)}{COLORS.ENDC}")

    X_test = np.load("data/X_test.npy").astype(np.float32)
    y_test = np.load("data/y_test.npy")
    test_labels = y_test.astype(np.int32).ravel() if y_test.ndim == 1 or y_test.shape[1] == 1 else np.argmax(y_test, axis=1).astype(np.int32)

    if not _use_tf():
        import torch
        torch.cuda.empty_cache()

    if getattr(config, 'cache_test_set', False) and not _use_tf():
        import torch
        X_test = torch.from_numpy(X_test).cuda()
        print(f"{COLORS.OKCYAN}X_test pinned to GPU ({X_test.element_size() * X_test.nelement() / 1e6:.0f} MB){COLORS.ENDC}")

    _mixed = getattr(config, 'mixed_models', False)
    _eval_model_arch = model_type
    eval_model = create_strategy_model(input_dim, num_classes, config.batch_size, model_type=model_type)
    attack_type, _, _ = parse_poison_config(getattr(config, 'poison', None))
    _poisonedfl_eval = attack_type == "poisonedfl" and bool(context.poisoned_clients)
    _lma_eval = attack_type == "lma" and bool(context.poisoned_clients)
    _skip_synthetic_byzantine_eval = _poisonedfl_eval or _lma_eval
    _f1_log_stem = os.path.splitext(os.path.basename(log_filename))[0]

    for round_number in range(config.rounds, 0, -1):
        if round_number in evaluated_rounds:
            print(f"{COLORS.OKCYAN}Round {round_number}: already evaluated, skipping{COLORS.ENDC}")
            continue

        if config.skip_eval and round_number != config.rounds:
            continue

        _clear_distill_round_metrics(context.results, round_number)

        round_dir = os.path.join(record_base, f"round_{round_number}")

        if global_model_eval:
            # Try in-memory weights first (always populated during training)
            if round_number in round_weights_history:
                weights = round_weights_history[round_number]
            else:
                # Fallback: load from disk (for resumed runs or when save_weights=True)
                weight_path = os.path.join(round_dir, "global_weight.bin")
                if not os.path.exists(weight_path):
                    logger.info(f"Round {round_number} | SKIPPED (no weight record)")
                    print(f"{COLORS.WARNING}Round {round_number}: skipped (no weight record){COLORS.ENDC}")
                    continue
                with open(weight_path, "rb") as _f:
                    weights = pickle.load(_f)
            if _eval_model_arch != model_type:
                del eval_model
                eval_model = create_strategy_model(input_dim, num_classes, config.batch_size, model_type=model_type)
                _eval_model_arch = model_type

            eval_model.set_weights(weights)
            del weights

            metrics = evaluate_model(eval_model, X_test, test_labels, config.batch_size, num_classes)
            round_metrics = {-1: metrics}
            _record_metrics(context, round_number, round_metrics, excel_filename)

            logger.info(
                "Round %s | GLOBAL | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | AUPRC: %.4f | ECE: %.4f | Loss: %.4f",
                round_number, metrics["Acc"], metrics["F1"],
                metrics["Precision"], metrics["Recall"], metrics["TPR"], metrics["FPR"], metrics["AUPRC"], metrics["ECE"], metrics["Loss"],
            )
            print(
                f"{COLORS.OKGREEN}Round {round_number} | Acc={metrics['Acc']:.4f}, F1={metrics['F1']:.4f}, "
                f"Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, TPR={metrics['TPR']:.4f}, FPR={metrics['FPR']:.4f}, AUPRC={metrics['AUPRC']:.4f}, ECE={metrics['ECE']:.4f}, Loss={metrics['Loss']:.4f}{COLORS.ENDC}"
            )
        else:
            global_weight_path = os.path.join(round_dir, "global_weight.bin")
            _is_final_client_eval = getattr(config, 'algorithm', '') in {"SSFL-IDS", "FedKD-IDS"}

            if not getattr(config, 'save_weights', True) and _is_final_client_eval:
                logger.info(f"Round {round_number} | SKIPPED (per-client eval requires save_weights=True)")
                print(f"{COLORS.WARNING}Round {round_number}: skipped (per-client eval requires --save_weights){COLORS.ENDC}")
                continue

            if _is_final_client_eval and round_number == config.rounds:
                client_bins = sorted(glob.glob(os.path.join(round_dir, "client_*_weight.bin")), key=lambda p: int(os.path.basename(p).split("_")[1]))
                if not client_bins and not os.path.exists(global_weight_path):
                    logger.info(f"Round {round_number} | SKIPPED (no weight record)")
                    print(f"{COLORS.WARNING}Round {round_number}: skipped (no weight record){COLORS.ENDC}")
                    continue
                if client_bins:
                    round_metrics: dict[int, dict] = {}
                    ghost_bin_path = None
                    ghost_cid = None
                    _f1_curve_results = []
                    _aurc_results = []
                    for bin_path in client_bins:
                        cid = int(os.path.basename(bin_path).split("_")[1])
                        if _skip_synthetic_byzantine_eval and cid in context.poisoned_clients:
                            if _poisonedfl_eval and ghost_bin_path is None:
                                ghost_bin_path = bin_path
                                ghost_cid = cid
                            logger.info("Round %s | Client %s | [byzantine, skipped]", round_number, cid)
                            print(f"{COLORS.WARNING}Client {cid}: [byzantine, skipped]{COLORS.ENDC}")
                            continue
                        if _mixed:
                            from models.mixed_models import get_model_type_for_client
                            arch = get_model_type_for_client(cid, n_clients)
                            if arch != _eval_model_arch:
                                del eval_model
                                eval_model = create_strategy_model(input_dim, num_classes, config.batch_size, model_type=arch)
                                _eval_model_arch = arch
                        with open(bin_path, "rb") as _f:
                            c_weights = pickle.load(_f)
                        eval_model.set_weights(c_weights)
                        del c_weights
                        _cap = getattr(config, 'f1_curve', False) and round_number == config.rounds
                        result = evaluate_model(eval_model, X_test, test_labels, config.batch_size, num_classes, compute_auprc=False, return_proba=_cap)
                        m, _proba = result if _cap else (result, None)
                        if _cap:
                            _c_plot = os.path.join("results", "plots", _f1_log_stem, f"client_{cid}.png")
                            _best_pt, _min_cov_pt = plot_f1_threshold_curve(test_labels, _proba, _c_plot, label=f"Client {cid}", logger=logger)
                            _aurc_val = compute_aurc(test_labels, _proba)
                            logger.info("Round %s | Client %s | F1_CURVE Best θ=%.2f Acc=%.4f F1=%.4f P=%.4f R=%.4f Cov=%.4f | AURC=%.4f",
                                round_number, cid, _best_pt["theta"], _best_pt["Acc"], _best_pt["F1"], _best_pt["Precision"], _best_pt["Recall"], _best_pt["Coverage"], _aurc_val)
                            _f1_curve_results.append((_best_pt, _min_cov_pt))
                            _aurc_results.append(_aurc_val)
                            del _proba
                        round_metrics[cid] = m
                        logger.info("Round %s | Client %s | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | ECE: %.4f | Loss: %.4f",
                                    round_number, cid, m["Acc"], m["F1"], m["Precision"], m["Recall"], m["TPR"], m["FPR"], m["ECE"], m["Loss"])
                        print(f"{COLORS.OKBLUE}Client {cid}: Acc={m['Acc']:.4f}, F1={m['F1']:.4f}, Precision={m['Precision']:.4f}, Recall={m['Recall']:.4f}, TPR={m['TPR']:.4f}, FPR={m['FPR']:.4f}, ECE={m['ECE']:.4f}, Loss={m['Loss']:.4f}{COLORS.ENDC}")
                    if round_metrics:
                        _record_metrics(context, round_number, round_metrics, excel_filename)
                        avg = {k: float(np.mean([v[k] for v in round_metrics.values()])) for k in next(iter(round_metrics.values()))}
                        _record_metrics(context, round_number, {(-3 if _skip_synthetic_byzantine_eval else -1): avg}, excel_filename)
                        _avg_label = "BENIGN_AVG" if _skip_synthetic_byzantine_eval else "AVG"
                        logger.info("Round %s | %s | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | ECE: %.4f | Loss: %.4f",
                            round_number, _avg_label, avg["Acc"], avg["F1"], avg["Precision"], avg["Recall"], avg["TPR"], avg["FPR"], avg["ECE"], avg["Loss"])
                        print(f"{COLORS.OKGREEN}Round {round_number} {_avg_label.replace('_', ' ')} | Acc={avg['Acc']:.4f}, F1={avg['F1']:.4f}, "
                              f"Precision={avg['Precision']:.4f}, Recall={avg['Recall']:.4f}, TPR={avg['TPR']:.4f}, FPR={avg['FPR']:.4f}, ECE={avg['ECE']:.4f}, Loss={avg['Loss']:.4f}{COLORS.ENDC}")
                        if _f1_curve_results:
                            _mk = ["Acc", "F1", "Precision", "Recall", "Coverage"]
                            _avg_best = {k: float(np.mean([r[0][k] for r in _f1_curve_results])) for k in _mk}
                            _avg_best_theta = float(np.mean([r[0]["theta"] for r in _f1_curve_results]))
                            _avg_mc = {k: float(np.mean([r[1][k] for r in _f1_curve_results])) for k in _mk}
                            _avg_mc_theta = float(np.mean([r[1]["theta"] for r in _f1_curve_results]))
                            logger.info("Round %s | %s | F1_CURVE Best θ=%.2f Acc=%.4f F1=%.4f P=%.4f R=%.4f Cov=%.4f | MinCov θ=%.2f Acc=%.4f F1=%.4f P=%.4f R=%.4f Cov=%.4f",
                                round_number, _avg_label,
                                _avg_best_theta, _avg_best["Acc"], _avg_best["F1"], _avg_best["Precision"], _avg_best["Recall"], _avg_best["Coverage"],
                                _avg_mc_theta, _avg_mc["Acc"], _avg_mc["F1"], _avg_mc["Precision"], _avg_mc["Recall"], _avg_mc["Coverage"])
                            print(f"{COLORS.OKGREEN}  Best θ={_avg_best_theta:.2f}: Acc={_avg_best['Acc']:.4f} F1={_avg_best['F1']:.4f} P={_avg_best['Precision']:.4f} R={_avg_best['Recall']:.4f} Cov={_avg_best['Coverage']:.4f} | MinCov θ={_avg_mc_theta:.2f}: Acc={_avg_mc['Acc']:.4f} F1={_avg_mc['F1']:.4f} P={_avg_mc['Precision']:.4f} R={_avg_mc['Recall']:.4f} Cov={_avg_mc['Coverage']:.4f}{COLORS.ENDC}")
                            _missing_counts = {}
                            for _best, _ in _f1_curve_results:
                                for _cls in _best.get("Missing_Classes", []):
                                    _missing_counts[int(_cls)] = _missing_counts.get(int(_cls), 0) + 1
                            if _missing_counts:
                                _missing_msg = " | ".join(f"Class {_cls}: ({100.0 * _count / len(_f1_curve_results):.1f} %)" for _cls, _count in sorted(_missing_counts.items()))
                                logger.info("Round %s | %s | F1_CURVE Best missing classes: %s", round_number, _avg_label, _missing_msg)
                                print(f"{COLORS.WARNING}  Best missing classes | {_missing_msg}{COLORS.ENDC}")
                        if _aurc_results:
                            _avg_aurc = float(np.mean(_aurc_results))
                            logger.info("Round %s | %s | AURC: %.4f", round_number, _avg_label, _avg_aurc)
                            print(f"{COLORS.OKGREEN}  AURC={_avg_aurc:.4f}{COLORS.ENDC}")
                    if context.poisoned_clients and not _skip_synthetic_byzantine_eval:
                        _benign_cids = [c for c in round_metrics if c >= 0 and c not in context.poisoned_clients]
                        if _benign_cids:
                            _b_avg = {k: float(np.mean([round_metrics[c][k] for c in _benign_cids])) for k in avg}
                            _record_metrics(context, round_number, {-3: _b_avg}, excel_filename)
                            logger.info("Round %s | BENIGN_AVG | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | ECE: %.4f | Loss: %.4f",
                                        round_number, _b_avg["Acc"], _b_avg["F1"], _b_avg["Precision"], _b_avg["Recall"], _b_avg["TPR"], _b_avg["FPR"], _b_avg["ECE"], _b_avg["Loss"])
                            print(f"{COLORS.OKGREEN}Round {round_number} Benign Avg | Acc={_b_avg['Acc']:.4f}, F1={_b_avg['F1']:.4f}, "
                                    f"Precision={_b_avg['Precision']:.4f}, Recall={_b_avg['Recall']:.4f}, TPR={_b_avg['TPR']:.4f}, FPR={_b_avg['FPR']:.4f}, ECE={_b_avg['ECE']:.4f}, Loss={_b_avg['Loss']:.4f}{COLORS.ENDC}")
                    if ghost_bin_path is not None:
                        if _mixed:
                            from models.mixed_models import get_model_type_for_client
                            ghost_arch = get_model_type_for_client(ghost_cid, n_clients)
                            if ghost_arch != _eval_model_arch:
                                del eval_model
                                eval_model = create_strategy_model(input_dim, num_classes, config.batch_size, model_type=ghost_arch)
                                _eval_model_arch = ghost_arch
                        with open(ghost_bin_path, "rb") as _f:
                            ghost_weights = pickle.load(_f)
                        eval_model.set_weights(ghost_weights)
                        del ghost_weights
                        ghost_metrics = evaluate_model(eval_model, X_test, test_labels, config.batch_size, num_classes, compute_auprc=False)
                        _record_metrics(context, round_number, {-1: ghost_metrics}, excel_filename)
                        logger.info("Round %s | Client -1 [POISONEDFL_GHOST] | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | ECE: %.4f | Loss: %.4f",
                                round_number, ghost_metrics["Acc"], ghost_metrics["F1"], ghost_metrics["Precision"], ghost_metrics["Recall"], ghost_metrics["TPR"], ghost_metrics["FPR"], ghost_metrics["ECE"], ghost_metrics["Loss"])
                        print(f"{COLORS.OKGREEN}Round {round_number} PoisonedFL ghost | Acc={ghost_metrics['Acc']:.4f}, F1={ghost_metrics['F1']:.4f}, "
                            f"Precision={ghost_metrics['Precision']:.4f}, Recall={ghost_metrics['Recall']:.4f}, TPR={ghost_metrics['TPR']:.4f}, FPR={ghost_metrics['FPR']:.4f}, ECE={ghost_metrics['ECE']:.4f}, Loss={ghost_metrics['Loss']:.4f}{COLORS.ENDC}")
                if os.path.exists(global_weight_path):
                    if _eval_model_arch != model_type:
                        del eval_model
                        eval_model = create_strategy_model(input_dim, num_classes, config.batch_size, model_type=model_type)
                        _eval_model_arch = model_type
                    with open(global_weight_path, "rb") as _f:
                        g_weights = pickle.load(_f)
                    eval_model.set_weights(g_weights)
                    del g_weights
                    g_metrics = evaluate_model(eval_model, X_test, test_labels, config.batch_size, num_classes)
                    _record_metrics(context, round_number, {-2: g_metrics}, excel_filename)
                    logger.info("Round %s | GLOBAL | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | AUPRC: %.4f | ECE: %.4f | Loss: %.4f",
                                round_number, g_metrics["Acc"], g_metrics["F1"],
                                g_metrics["Precision"], g_metrics["Recall"], g_metrics["TPR"], g_metrics["FPR"], g_metrics["AUPRC"], g_metrics["ECE"], g_metrics["Loss"])
                    print(f"{COLORS.OKGREEN}Round {round_number} Global | Acc={g_metrics['Acc']:.4f}, F1={g_metrics['F1']:.4f}, "
                          f"Precision={g_metrics['Precision']:.4f}, Recall={g_metrics['Recall']:.4f}, TPR={g_metrics['TPR']:.4f}, FPR={g_metrics['FPR']:.4f}, AUPRC={g_metrics['AUPRC']:.4f}, ECE={g_metrics['ECE']:.4f}, Loss={g_metrics['Loss']:.4f}{COLORS.ENDC}")
            elif os.path.exists(global_weight_path):
                if _eval_model_arch != model_type:
                    del eval_model
                    eval_model = create_strategy_model(input_dim, num_classes, config.batch_size, model_type=model_type)
                    _eval_model_arch = model_type
                with open(global_weight_path, "rb") as _f:
                    g_weights = pickle.load(_f)
                eval_model.set_weights(g_weights)
                del g_weights
                g_metrics = evaluate_model(eval_model, X_test, test_labels, config.batch_size, num_classes)
                _record_metrics(context, round_number, {-2: g_metrics}, excel_filename)
                logger.info(
                    "Round %s | GLOBAL | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | AUPRC: %.4f | ECE: %.4f | Loss: %.4f",
                    round_number, g_metrics["Acc"], g_metrics["F1"],
                    g_metrics["Precision"], g_metrics["Recall"], g_metrics["TPR"], g_metrics["FPR"], g_metrics["AUPRC"], g_metrics["ECE"], g_metrics["Loss"],
                )
                print(
                    f"{COLORS.OKGREEN}Round {round_number} Global | Acc={g_metrics['Acc']:.4f}, F1={g_metrics['F1']:.4f}, "
                    f"Precision={g_metrics['Precision']:.4f}, Recall={g_metrics['Recall']:.4f}, TPR={g_metrics['TPR']:.4f}, FPR={g_metrics['FPR']:.4f}, AUPRC={g_metrics['AUPRC']:.4f}, ECE={g_metrics['ECE']:.4f}, Loss={g_metrics['Loss']:.4f}{COLORS.ENDC}"
                )
            else:
                client_bins = sorted(glob.glob(os.path.join(round_dir, "client_*_weight.bin")), key=lambda p: int(os.path.basename(p).split("_")[1]))
                if not client_bins:
                    logger.info(f"Round {round_number} | SKIPPED (no weight record)")
                    print(f"{COLORS.WARNING}Round {round_number}: skipped (no weight record){COLORS.ENDC}")
                    continue
                round_metrics: dict[int, dict] = {}
                ghost_bin_path = None
                ghost_cid = None
                _f1_curve_results = []
                _aurc_results = []
                for bin_path in client_bins:
                    cid = int(os.path.basename(bin_path).split("_")[1])
                    if _skip_synthetic_byzantine_eval and cid in context.poisoned_clients:
                        if _poisonedfl_eval and ghost_bin_path is None:
                            ghost_bin_path = bin_path
                            ghost_cid = cid
                        logger.info("Round %s | Client %s | [byzantine, skipped]", round_number, cid)
                        print(f"{COLORS.WARNING}Client {cid}: [byzantine, skipped]{COLORS.ENDC}")
                        continue
                    if _mixed:
                        from models.mixed_models import get_model_type_for_client
                        arch = get_model_type_for_client(cid, n_clients)
                        if arch != _eval_model_arch:
                            del eval_model
                            eval_model = create_strategy_model(input_dim, num_classes, config.batch_size, model_type=arch)
                            _eval_model_arch = arch
                    with open(bin_path, "rb") as _f:
                        c_weights = pickle.load(_f)
                    eval_model.set_weights(c_weights)
                    del c_weights
                    _cap = getattr(config, 'f1_curve', False) and round_number == config.rounds
                    result = evaluate_model(eval_model, X_test, test_labels, config.batch_size, num_classes, compute_auprc=False, return_proba=_cap)
                    m, _proba = result if _cap else (result, None)
                    if _cap:
                        _c_plot = os.path.join("results", "plots", _f1_log_stem, f"client_{cid}.png")
                        _best_pt, _min_cov_pt = plot_f1_threshold_curve(test_labels, _proba, _c_plot, label=f"Client {cid}", logger=logger)
                        _aurc_val = compute_aurc(test_labels, _proba)
                        logger.info("Round %s | Client %s | F1_CURVE Best θ=%.2f Acc=%.4f F1=%.4f P=%.4f R=%.4f Cov=%.4f | AURC=%.4f",
                            round_number, cid, _best_pt["theta"], _best_pt["Acc"], _best_pt["F1"], _best_pt["Precision"], _best_pt["Recall"], _best_pt["Coverage"], _aurc_val)
                        _f1_curve_results.append((_best_pt, _min_cov_pt))
                        _aurc_results.append(_aurc_val)
                        del _proba
                    round_metrics[cid] = m
                    logger.info("Round %s | Client %s | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | ECE: %.4f | Loss: %.4f",
                                round_number, cid, m["Acc"], m["F1"], m["Precision"], m["Recall"], m["TPR"], m["FPR"], m["ECE"], m["Loss"])
                    print(f"{COLORS.OKBLUE}Client {cid}: Acc={m['Acc']:.4f}, F1={m['F1']:.4f}, Precision={m['Precision']:.4f}, Recall={m['Recall']:.4f}, TPR={m['TPR']:.4f}, FPR={m['FPR']:.4f}, ECE={m['ECE']:.4f}, Loss={m['Loss']:.4f}{COLORS.ENDC}")
                if round_metrics:
                    _record_metrics(context, round_number, round_metrics, excel_filename)
                    avg = {k: float(np.mean([m[k] for m in round_metrics.values()])) for k in next(iter(round_metrics.values()))}
                    _record_metrics(context, round_number, {(-3 if _skip_synthetic_byzantine_eval else -1): avg}, excel_filename)
                    _avg_label = "BENIGN_AVG" if _skip_synthetic_byzantine_eval else "AVG"
                    logger.info(
                        "Round %s | %s | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | ECE: %.4f | Loss: %.4f",
                        round_number, _avg_label, avg["Acc"], avg["F1"], avg["Precision"], avg["Recall"], avg["TPR"], avg["FPR"], avg["ECE"], avg["Loss"],
                    )
                    print(
                        f"{COLORS.OKGREEN}Round {round_number} {_avg_label.replace('_', ' ')} | Acc={avg['Acc']:.4f}, F1={avg['F1']:.4f}, "
                        f"Precision={avg['Precision']:.4f}, Recall={avg['Recall']:.4f}, TPR={avg['TPR']:.4f}, FPR={avg['FPR']:.4f}, ECE={avg['ECE']:.4f}, Loss={avg['Loss']:.4f}{COLORS.ENDC}"
                    )
                    if _f1_curve_results:
                        _mk = ["Acc", "F1", "Precision", "Recall", "Coverage"]
                        _avg_best = {k: float(np.mean([r[0][k] for r in _f1_curve_results])) for k in _mk}
                        _avg_best_theta = float(np.mean([r[0]["theta"] for r in _f1_curve_results]))
                        _avg_mc = {k: float(np.mean([r[1][k] for r in _f1_curve_results])) for k in _mk}
                        _avg_mc_theta = float(np.mean([r[1]["theta"] for r in _f1_curve_results]))
                        logger.info("Round %s | %s | F1_CURVE Best θ=%.2f Acc=%.4f F1=%.4f P=%.4f R=%.4f Cov=%.4f | MinCov θ=%.2f Acc=%.4f F1=%.4f P=%.4f R=%.4f Cov=%.4f",
                            round_number, _avg_label,
                            _avg_best_theta, _avg_best["Acc"], _avg_best["F1"], _avg_best["Precision"], _avg_best["Recall"], _avg_best["Coverage"],
                            _avg_mc_theta, _avg_mc["Acc"], _avg_mc["F1"], _avg_mc["Precision"], _avg_mc["Recall"], _avg_mc["Coverage"])
                        print(f"{COLORS.OKGREEN}  Best θ={_avg_best_theta:.2f}: Acc={_avg_best['Acc']:.4f} F1={_avg_best['F1']:.4f} P={_avg_best['Precision']:.4f} R={_avg_best['Recall']:.4f} Cov={_avg_best['Coverage']:.4f} | MinCov θ={_avg_mc_theta:.2f}: Acc={_avg_mc['Acc']:.4f} F1={_avg_mc['F1']:.4f} P={_avg_mc['Precision']:.4f} R={_avg_mc['Recall']:.4f} Cov={_avg_mc['Coverage']:.4f}{COLORS.ENDC}")
                        _missing_counts = {}
                        for _best, _ in _f1_curve_results:
                            for _cls in _best.get("Missing_Classes", []):
                                _missing_counts[int(_cls)] = _missing_counts.get(int(_cls), 0) + 1
                        if _missing_counts:
                            _missing_msg = " | ".join(f"Class {_cls}: ({100.0 * _count / len(_f1_curve_results):.1f} %)" for _cls, _count in sorted(_missing_counts.items()))
                            logger.info("Round %s | %s | F1_CURVE Best missing classes: %s", round_number, _avg_label, _missing_msg)
                            print(f"{COLORS.WARNING}  Best missing classes | {_missing_msg}{COLORS.ENDC}")
                    if _aurc_results:
                        _avg_aurc = float(np.mean(_aurc_results))
                        logger.info("Round %s | %s | AURC: %.4f", round_number, _avg_label, _avg_aurc)
                        print(f"{COLORS.OKGREEN}  AURC={_avg_aurc:.4f}{COLORS.ENDC}")
                if context.poisoned_clients and not _skip_synthetic_byzantine_eval:
                    _benign_cids = [c for c in round_metrics if c >= 0 and c not in context.poisoned_clients]
                    if _benign_cids:
                        _b_avg = {k: float(np.mean([round_metrics[c][k] for c in _benign_cids])) for k in avg}
                        _record_metrics(context, round_number, {-3: _b_avg}, excel_filename)
                        logger.info(
                            "Round %s | BENIGN_AVG | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | ECE: %.4f | Loss: %.4f",
                            round_number, _b_avg["Acc"], _b_avg["F1"], _b_avg["Precision"], _b_avg["Recall"], _b_avg["TPR"], _b_avg["FPR"], _b_avg["ECE"], _b_avg["Loss"],
                        )
                        print(
                            f"{COLORS.OKGREEN}Round {round_number} Benign Avg | Acc={_b_avg['Acc']:.4f}, F1={_b_avg['F1']:.4f}, "
                            f"Precision={_b_avg['Precision']:.4f}, Recall={_b_avg['Recall']:.4f}, TPR={_b_avg['TPR']:.4f}, FPR={_b_avg['FPR']:.4f}, ECE={_b_avg['ECE']:.4f}, Loss={_b_avg['Loss']:.4f}{COLORS.ENDC}"
                        )
                if ghost_bin_path is not None:
                    if _mixed:
                        from models.mixed_models import get_model_type_for_client
                        ghost_arch = get_model_type_for_client(ghost_cid, n_clients)
                        if ghost_arch != _eval_model_arch:
                            del eval_model
                            eval_model = create_strategy_model(input_dim, num_classes, config.batch_size, model_type=ghost_arch)
                            _eval_model_arch = ghost_arch
                    with open(ghost_bin_path, "rb") as _f:
                        ghost_weights = pickle.load(_f)
                    eval_model.set_weights(ghost_weights)
                    del ghost_weights
                    ghost_metrics = evaluate_model(eval_model, X_test, test_labels, config.batch_size, num_classes, compute_auprc=False)
                    _record_metrics(context, round_number, {-1: ghost_metrics}, excel_filename)
                    logger.info(
                        "Round %s | Client -1 [POISONEDFL_GHOST] | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | TPR: %.4f | FPR: %.4f | ECE: %.4f | Loss: %.4f",
                        round_number, ghost_metrics["Acc"], ghost_metrics["F1"], ghost_metrics["Precision"], ghost_metrics["Recall"], ghost_metrics["TPR"], ghost_metrics["FPR"], ghost_metrics["ECE"], ghost_metrics["Loss"],
                    )
                    print(
                        f"{COLORS.OKGREEN}Round {round_number} PoisonedFL ghost | Acc={ghost_metrics['Acc']:.4f}, F1={ghost_metrics['F1']:.4f}, "
                        f"Precision={ghost_metrics['Precision']:.4f}, Recall={ghost_metrics['Recall']:.4f}, TPR={ghost_metrics['TPR']:.4f}, FPR={ghost_metrics['FPR']:.4f}, ECE={ghost_metrics['ECE']:.4f}, Loss={ghost_metrics['Loss']:.4f}{COLORS.ENDC}"
                    )

    if getattr(config, 'f1_curve', False) and global_model_eval:
        plot_path = os.path.join("results", "plots", _f1_log_stem, "global_model.png")
        label = getattr(config, 'algorithm', 'model')
        weight_path = os.path.join(record_base, f"round_{config.rounds}", "global_weight.bin")
        if os.path.exists(weight_path):
            if _eval_model_arch != model_type:
                del eval_model
                eval_model = create_strategy_model(input_dim, num_classes, config.batch_size, model_type=model_type)
            with open(weight_path, "rb") as _f:
                weights = pickle.load(_f)
            eval_model.set_weights(weights)
            del weights
            proba = get_model_proba(eval_model, X_test, config.batch_size)
            _best_pt, _min_cov_pt = plot_f1_threshold_curve(test_labels, proba, plot_path, label=label, logger=logger)
            _global_aurc = compute_aurc(test_labels, proba)
            logger.info("F1_CURVE | GLOBAL | Best θ=%.2f Acc=%.4f F1=%.4f P=%.4f R=%.4f Cov=%.4f | MinCov θ=%.2f Acc=%.4f F1=%.4f P=%.4f R=%.4f Cov=%.4f | AURC=%.4f",
                _best_pt["theta"], _best_pt["Acc"], _best_pt["F1"], _best_pt["Precision"], _best_pt["Recall"], _best_pt["Coverage"],
                _min_cov_pt["theta"], _min_cov_pt["Acc"], _min_cov_pt["F1"], _min_cov_pt["Precision"], _min_cov_pt["Recall"], _min_cov_pt["Coverage"],
                _global_aurc)
            print(f"{COLORS.OKGREEN}F1_CURVE Global | Best θ={_best_pt['theta']:.2f}: Acc={_best_pt['Acc']:.4f} F1={_best_pt['F1']:.4f} P={_best_pt['Precision']:.4f} R={_best_pt['Recall']:.4f} Cov={_best_pt['Coverage']:.4f} | MinCov θ={_min_cov_pt['theta']:.2f}: Acc={_min_cov_pt['Acc']:.4f} F1={_min_cov_pt['F1']:.4f} P={_min_cov_pt['Precision']:.4f} R={_min_cov_pt['Recall']:.4f} Cov={_min_cov_pt['Coverage']:.4f} | AURC={_global_aurc:.4f}{COLORS.ENDC}")
            del proba

    del eval_model, X_test, y_test, test_labels
    if context.results:
        pd.DataFrame(context.results).T.to_excel(excel_filename)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    aggressive_memory_cleanup()
    log_timestamp(logger, "=== SIMULATION COMPLETED ===")
    print(f"{COLORS.OKCYAN}Results saved to {excel_filename}{COLORS.ENDC}")
    print(f"{COLORS.OKGREEN}Simulation completed!{COLORS.ENDC}")

    _keep = getattr(config, 'keep_last_rounds', 0)
    if _keep > 0:
        _cleanup_old_weight_rounds(record_base, _keep)


# def _extract_labels(dataset: tf.data.Dataset, num_classes: int) -> np.ndarray:
#     labels: List[int] = []
#     for _, batch_y in dataset:
#         batch = batch_y.numpy()
#         if batch.ndim == 1 or batch.shape[1] == 1:
#             labels.extend(batch.astype(int).tolist())
#         else:
#             labels.extend(np.argmax(batch, axis=1).tolist())
#     return np.asarray(labels, dtype=int)


def _record_metrics(context: PipelineContext, round_number: int, round_metrics: Dict[int, Dict[str, float]], excel_filename: str) -> None:
    for client_id, metrics in round_metrics.items():
        for key, value in metrics.items():
            metric_key = f"Round_{round_number}_{key}"
            context.results.setdefault(client_id, {})[metric_key] = value

    pd.to_pickle(context.results, _eval_progress_path(excel_filename))
