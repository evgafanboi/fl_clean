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
    adaptive_mu: bool = False
    model: str = "dense"
    robust_epsilon: float = 0.2
    robust_tau: float = 0.1
    poison: Optional[str] = None
    personalized_eval: bool = False
    root_iterations: int = 1
    lambda1: float = 1.0
    lambda2: float = 1.0
    temperature: float = 1.0

    def to_strategy_params(self) -> Dict[str, object]:
        return {
            'mu': self.mu,
            'adaptive_mu': self.adaptive_mu,
            'feddyn_alpha': self.feddyn_alpha,
            'epsilon': self.robust_epsilon,
            'robust_tau': self.robust_tau,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'temperature': self.temperature,
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

        if self.config.personalized_eval:
            self._evaluate_personalized_client(model, client_id, num_classes)

        trained_weights = model.get_weights()
        poisoned_weights = self._apply_poison_to_weights(trained_weights, latest_weights, client_id)

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

    def _aggregate(self, weights_list, sample_sizes, participating_clients, global_update=None):
        aggregator = self.strategy_runtime.aggregator
        
        # FLTrust requires global_update parameter
        if self.config.strategy == "FLTrust":
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

        self.logger, self.log_filename, self.detailed_logger = setup_logger(
            n_clients,
            partition_label,
            strategy_name=self.config.strategy,
            poison_suffix=poison_suffix,
        )
        excel_filename = self.log_filename.replace('.log', '.xlsx')

        class_names, num_classes = self._load_class_metadata(partition_label, client_count)
        self.class_names = class_names
        self.eval_batch_size = min(self.config.batch_size, 2048)
        self.test_dataset = load_test_dataset(self.eval_batch_size, num_classes)
        input_dim = self._determine_input_dim()

        paths_list = self._prepare_paths(n_clients, partition_label, client_count)
        
        if self.config.strategy != "FLTrust":
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

        latest_weights = None
        pipeline_start_time = time.time()
        log_timestamp(self.logger, "=== FL PIPELINE STARTED ===")
        log_timestamp(self.logger, f"Strategy: {self.config.strategy}, Clients: {n_clients}, Rounds: {self.config.rounds}")
        round_times: List[float] = []
        
        if self.config.strategy == "FedMLB":
            print(f"\n{COLORS.OKCYAN}Initializing FedMLB server model (Round 0){COLORS.ENDC}")
            log_timestamp(self.logger, "Round 0 - Initializing server model")
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
            
            # For FLTrust, compute server update on root dataset
            global_update = None
            if self.config.strategy == "FLTrust":
                print(f"\n{COLORS.OKBLUE}Training server on root dataset{COLORS.ENDC}")
                global_update = self._train_server_on_root_dataset(latest_weights, input_dim, num_classes)
            
            print(f"\n{COLORS.OKBLUE}Aggregating updates ({self.config.strategy}){COLORS.ENDC}")
            aggregated_result = self._aggregate(weights_list, sample_sizes, participating_clients, global_update)
            
            # For FLTrust, apply update to get new weights
            if self.config.strategy == "FLTrust":
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
        extra_tokens=[f"gamma{config.gamma}", *extra_log_tokens.values()],
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
