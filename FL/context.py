import os
import pickle
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
        self._factory_fn = factory_fn
        self._pool_size = pool_size
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

    def refresh(self):
        """Recreate pool models after tf.keras.backend.clear_session()."""
        self._available = [self._factory_fn() for _ in range(self._pool_size)]


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

    def _weight_record_dir(self, round_num: int) -> str:
        stem = os.path.splitext(os.path.basename(self.log_filename))[0]
        return os.path.join("temp_weights", f"{stem}_weight_record", f"round_{round_num}")

    def record_client_weight(self, round_num: int, client_id: int, weights) -> None:
        record_dir = self._weight_record_dir(round_num)
        os.makedirs(record_dir, exist_ok=True)
        path = os.path.join(record_dir, f"client_{client_id}_weight.bin")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def record_client_weights(self, round_num: int) -> None:
        for st in self.client_states:
            w = st.data.get("w")
            if w is not None:
                self.record_client_weight(round_num, st.client_id, w)
                continue
            if self.model_pool is None:
                continue
            model = self.model_pool.checkout(st.client_id)
            w = self.model_pool._get_weights(model)
            self.model_pool.release(model)
            self.record_client_weight(round_num, st.client_id, w)


def evaluate_model(model: Any, X_test: np.ndarray, y_labels: np.ndarray,
                   batch_size: int, num_classes: int) -> Dict[str, float]:
    base = model.base_model if hasattr(model, "base_model") else model
    if hasattr(base, "model"):
        base = base.model

    preds = base.predict(X_test, batch_size=batch_size, verbose=0)
    pred_labels = np.argmax(preds, axis=1).astype(np.int32)
    y_oh = tf.keras.utils.to_categorical(y_labels, num_classes).astype(np.float32)
    loss = float(-np.sum(y_oh * np.log(np.clip(preds, 1e-7, 1.0)))) / len(y_labels)
    del preds, y_oh

    accuracy = float(np.mean(pred_labels == y_labels))
    f1 = float(f1_score(y_labels, pred_labels, average="macro", zero_division=0))
    precision = float(precision_score(y_labels, pred_labels, average="macro", zero_division=0))
    recall = float(recall_score(y_labels, pred_labels, average="macro", zero_division=0))

    return {
        "Acc": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Loss": loss,
    }
