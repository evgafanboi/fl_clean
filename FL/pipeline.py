import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from .aggregators import StrategyRuntime, build_strategy
from .colors import COLORS
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
    adaptive_mu: bool = False
    model: str = "dense"
    robust_epsilon: float = 0.2
    robust_tau: float = 0.1
    poison: Optional[str] = None

    def to_strategy_params(self) -> Dict[str, object]:
        return {
            'mu': self.mu,
            'adaptive_mu': self.adaptive_mu,
            'feddyn_alpha': self.feddyn_alpha,
            'epsilon': self.robust_epsilon,
            'robust_tau': self.robust_tau,
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

        # Apply poisoning for poisoned clients
        poison_loader = self.poison_loader if client_id in self.poisoned_clients else None
        if poison_loader:
            print(f"  \u26a0\ufe0f  POISONED CLIENT - Labels will be flipped")

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

        client_train_time = time.time() - client_start_time
        log_timestamp(self.logger, f"Client {client_id} completed in {client_train_time:.2f}s")

        loss = float(history.history.get("loss", [0.0])[-1])
        self.logger.info(f"Client {client_id}: Loss={loss:.4f}")
        print(f"{COLORS.WARNING}Client {client_id}: Loss={loss:.4f}{COLORS.ENDC}")

        weights_file = os.path.join(
            self.config.weights_cache_dir,
            f"client_{client_id}_round_{self.current_round}_weights.pkl"
        )

        if self.config.strategy == "FedDyn":
            feddyn_data = model.get_feddyn_update()
            with open(weights_file, 'wb') as file_handler:
                pickle.dump(feddyn_data, file_handler)
        else:
            with open(weights_file, 'wb') as file_handler:
                pickle.dump(model.get_weights(), file_handler)

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

    def _aggregate(self, weights_list, sample_sizes, participating_clients):
        aggregator = self.strategy_runtime.aggregator
        if self.strategy_runtime.requires_participant_ids:
            return aggregator.aggregate(weights_list, sample_sizes, participating_clients)
        return aggregator.aggregate(weights_list, sample_sizes)

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

        eval_batch_size = min(self.config.batch_size, 2048)
        test_dataset = load_test_dataset(eval_batch_size, num_classes)

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
            poison_suffix = self.config.poison.replace("-", "_")

        self.logger, self.log_filename, self.detailed_logger = setup_logger(
            n_clients,
            partition_label,
            strategy_name=self.config.strategy,
            poison_suffix=poison_suffix,
        )
        excel_filename = self.log_filename.replace('.log', '.xlsx')

        class_names, num_classes = self._load_class_metadata(partition_label, client_count)
        input_dim = self._determine_input_dim()

        paths_list = self._prepare_paths(n_clients, partition_label, client_count)
        self._restore_partitions(paths_list)

        # Setup poisoning if configured
        attack_type, poison_ratio = parse_poison_config(self.config.poison)
        if attack_type:
            self.poisoned_clients = get_or_create_poisoned_clients(
                partition_label, attack_type, poison_ratio, n_clients
            )
            self.poison_loader = PoisonedDataLoader(attack_type, num_classes)
            log_timestamp(self.logger, f"POISONING ENABLED: {attack_type} attack on {len(self.poisoned_clients)} clients ({poison_ratio*100:.1f}%)")
            log_timestamp(self.logger, f"Poisoned clients: {self.poisoned_clients}")
            print(f"\n  POISONING ENABLED: {attack_type} attack")
            print(f"    Poisoned clients: {self.poisoned_clients} ({len(self.poisoned_clients)}/{n_clients})")

        latest_weights = None
        pipeline_start_time = time.time()
        log_timestamp(self.logger, "=== FL PIPELINE STARTED ===")
        log_timestamp(self.logger, f"Strategy: {self.config.strategy}, Clients: {n_clients}, Rounds: {self.config.rounds}")
        round_times: List[float] = []

        for round_num in range(1, self.config.rounds + 1):
            self.current_round = round_num
            round_start_time = time.time()

            self.logger.info(f"Round {round_num}/{self.config.rounds}")
            print(f"\n{COLORS.HEADER}Round {round_num}/{self.config.rounds}{COLORS.ENDC}")
            log_timestamp(self.logger, f"--- Round {round_num} started ---")

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
            print(f"\n{COLORS.OKBLUE}Aggregating weights ({self.config.strategy}){COLORS.ENDC}")
            aggregated_weights = self._aggregate(weights_list, sample_sizes, participating_clients)
            latest_weights = aggregated_weights

            del weights_list
            aggressive_memory_cleanup()

            eval_start_time = time.time()
            metrics = self._evaluate_global_model(
                aggregated_weights,
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

            del aggregated_weights
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
