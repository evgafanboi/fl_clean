from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

from .colors import COLORS
from .config import FDConfig
from .data_utils import load_test_dataset, parse_partition_type, setup_paths
from .logging_utils import log_timestamp, setup_logger
from .poison_utils import parse_poison_config, get_or_create_poisoned_clients, PoisonedDataLoader


@dataclass
class ClientState:
    client_id: int
    model: Any
    paths: Dict[str, str]
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    config: FDConfig
    partition_label: str
    client_count: int
    n_clients: int
    input_dim: int
    num_classes: int
    paths: List[Dict[str, str]]
    logger: Any
    detailed_logger: Any
    log_filename: str
    excel_filename: str
    test_dataset: tf.data.Dataset
    test_labels: np.ndarray
    results: Dict[int, Dict[str, float]] = field(default_factory=dict)
    client_states: List[ClientState] = field(default_factory=list)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    poisoned_clients: List[int] = field(default_factory=list)
    poison_loader: Any = None

    def add_client_state(self, client_id: int, model: Any, paths: Dict[str, str], **extras: Any) -> ClientState:
        state = ClientState(client_id=client_id, model=model, paths=paths, data=dict(extras))
        self.client_states.append(state)
        self.results.setdefault(client_id, {})
        return state


class DistillationPipeline:
    def __init__(self, config: FDConfig, algorithm: "DistillationAlgorithm") -> None:
        self.config = config
        self.algorithm = algorithm

    def run(self) -> None:
        context = self._prepare_context()

        log_timestamp(context.logger, "SIMULATION STARTED")
        log_timestamp(
            context.logger,
            f"Clients: {context.n_clients}, Rounds: {self.config.rounds}, Partition: {context.partition_label}"
        )

        if context.shared_state.get("extra_log_tokens"):
            log_timestamp(context.logger, f"Extra config: {context.shared_state['extra_log_tokens']}")

        self.algorithm.setup(context)

        for round_number in range(1, self.config.rounds + 1):
            context.logger.info(f"Round {round_number}/{self.config.rounds}")
            print(f"\n{COLORS.HEADER}Round {round_number}/{self.config.rounds}{COLORS.ENDC}")
            log_timestamp(context.logger, f"Round {round_number} started")

            round_metrics = self.algorithm.run_round(context, round_number)

            if round_metrics:
                self._record_metrics(context, round_number, round_metrics)

        self.algorithm.finalize(context)

        total_time = context.shared_state.get("pipeline_elapsed_s")
        if total_time is not None:
            log_timestamp(context.logger, f"Pipeline completed in {total_time:.2f}s ({total_time/60:.2f}m)")
        else:
            log_timestamp(context.logger, "Pipeline completed")

    # ------------------------------------------------------------------
    def _prepare_context(self) -> PipelineContext:
        partition_label, client_count = parse_partition_type(self.config.partition_type)
        n_clients = min(self.config.n_clients, client_count)

        extra_log_tokens = self.algorithm.extra_log_tokens()

        # Prepare poison suffix for logging
        poison_suffix = ""
        if self.config.poison:
            poison_suffix = self.config.poison.replace("-", "_")

        logger, log_filename, detailed_logger = setup_logger(
            algorithm_name=self.algorithm.name,
            n_clients=n_clients,
            partition_label=partition_label,
            extra_tokens=[f"gamma{self.config.gamma}", *extra_log_tokens.values()],
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

        test_dataset = load_test_dataset(self.config.batch_size, num_classes)
        test_labels = self._extract_labels(test_dataset, num_classes)

        # Setup poisoning if configured
        poisoned_clients = []
        poison_loader = None
        attack_type, poison_ratio = parse_poison_config(self.config.poison)
        if attack_type:
            poisoned_clients = get_or_create_poisoned_clients(
                partition_label, attack_type, poison_ratio, n_clients
            )
            poison_loader = PoisonedDataLoader(attack_type, num_classes)
            print(f"\n  POISONING ENABLED: {attack_type} attack")
            print(f"    Poisoned clients: {poisoned_clients} ({len(poisoned_clients)}/{n_clients})")

        return PipelineContext(
            config=self.config,
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

    def _record_metrics(
        self,
        context: PipelineContext,
        round_number: int,
        round_metrics: Dict[int, Dict[str, float]],
    ) -> None:
        for client_id, metrics in round_metrics.items():
            for key, value in metrics.items():
                metric_key = f"Round_{round_number}_{key}"
                context.results.setdefault(client_id, {})[metric_key] = value

        results_df = pd.DataFrame(context.results).T
        results_df.to_excel(context.excel_filename)

    @staticmethod
    def _extract_labels(dataset: tf.data.Dataset, num_classes: int) -> np.ndarray:
        labels: List[int] = []
        for _, batch_y in dataset:
            batch = batch_y.numpy()
            if batch.ndim == 1 or batch.shape[1] == 1:
                labels.extend(batch.astype(int).tolist())
            else:
                labels.extend(np.argmax(batch, axis=1).tolist())
        return np.asarray(labels, dtype=int)


def evaluate_model(model: Any, test_dataset: tf.data.Dataset, reference_labels: np.ndarray) -> Dict[str, float]:
    predictions = model.predict(test_dataset, verbose=1)
    if isinstance(predictions, list):
        predictions = predictions[0]
    pred_labels = np.argmax(predictions, axis=1)

    true_labels = reference_labels[: len(pred_labels)]

    accuracy = float(np.mean(pred_labels == true_labels))
    f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
    precision = float(precision_score(true_labels, pred_labels, average="macro", zero_division=0))
    recall = float(recall_score(true_labels, pred_labels, average="macro", zero_division=0))

    return {
        "Acc": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
    }
