import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

MODEL_WEIGHTS_DIR = os.path.join("temp_weights", "model_weights")


class ModelPool:
    def __init__(self, pool_size, factory_fn, weights_dir=MODEL_WEIGHTS_DIR, in_memory=False):
        self.weights_dir = weights_dir
        self.in_memory = in_memory
        self._available = [factory_fn() for _ in range(pool_size)]
        if self.in_memory:
            self._weights_cache = {}
            self._initial_weights = self._clone_weights(self._get_weights(self._available[0]))
            return

        if os.path.isdir(weights_dir):
            for f in os.listdir(weights_dir):
                os.remove(os.path.join(weights_dir, f))
        os.makedirs(weights_dir, exist_ok=True)
        self._init_path = os.path.join(weights_dir, "_init.weights.h5")
        self._keras_model(self._available[0]).save_weights(self._init_path)

    @staticmethod
    def _keras_model(model):
        return model.model if hasattr(model, "model") else model

    @staticmethod
    def _clone_weights(weights):
        return [np.array(weight, copy=True) for weight in weights]

    def _get_weights(self, model):
        if hasattr(model, "get_weights"):
            return model.get_weights()
        return self._keras_model(model).get_weights()

    def _set_weights(self, model, weights):
        copied_weights = self._clone_weights(weights)
        if hasattr(model, "set_weights"):
            model.set_weights(copied_weights)
            return
        self._keras_model(model).set_weights(copied_weights)

    def _path(self, client_id, tag=""):
        suffix = f"_{tag}" if tag else ""
        return os.path.join(self.weights_dir, f"c{client_id}{suffix}.weights.h5")

    def checkout(self, client_id, tag=""):
        model = self._available.pop()
        if self.in_memory:
            weights = self._weights_cache.get((client_id, tag), self._initial_weights)
            self._set_weights(model, weights)
            return model

        path = self._path(client_id, tag)
        weights_path = path if os.path.exists(path) else self._init_path
        self._keras_model(model).load_weights(weights_path)
        return model

    def checkin(self, client_id, model, tag=""):
        if self.in_memory:
            self._weights_cache[(client_id, tag)] = self._clone_weights(self._get_weights(model))
            self._available.append(model)
            return

        self._keras_model(model).save_weights(self._path(client_id, tag))
        self._available.append(model)

    def release(self, model):
        self._available.append(model)

    def save(self, client_id, model, tag=""):
        if self.in_memory:
            self._weights_cache[(client_id, tag)] = self._clone_weights(self._get_weights(model))
            return
        self._keras_model(model).save_weights(self._path(client_id, tag))


@dataclass
class ClientState:
    client_id: int
    model: Any
    paths: Dict[str, str]
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    config: Any
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
    model_pool: Any = None

    def add_client_state(self, client_id: int, model: Any, paths: Dict[str, str], **extras: Any) -> ClientState:
        state = ClientState(client_id=client_id, model=model, paths=paths, data=dict(extras))
        self.client_states.append(state)
        self.results.setdefault(client_id, {})
        return state


def evaluate_model(model: Any, test_dataset: tf.data.Dataset, reference_labels: np.ndarray) -> Dict[str, float]:
    predictions = model.predict(test_dataset, verbose=1)
    if isinstance(predictions, list):
        predictions = predictions[0]
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = reference_labels[:len(pred_labels)]
    
    y_true_onehot = tf.keras.utils.to_categorical(true_labels, predictions.shape[1])
    loss = float(-np.mean(np.sum(y_true_onehot * np.log(np.clip(predictions, 1e-7, 1.0)), axis=1)))
    
    accuracy = float(np.mean(pred_labels == true_labels))
    f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
    precision = float(precision_score(true_labels, pred_labels, average="macro", zero_division=0))
    recall = float(recall_score(true_labels, pred_labels, average="macro", zero_division=0))
    
    return {
        "Acc": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Loss": loss,
    }
