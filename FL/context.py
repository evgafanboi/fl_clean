import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from .backend import use_tf as _use_tf
from .evaluation import summarize_prob_predictions
if _use_tf():
    import tensorflow as tf


class ModelPool:
    """Pool of reusable Keras/PyTorch models with in-memory weight storage.

    Client weights are kept in ``_weights_cache``; no files are written to
    disk.  Persistence across rounds is handled by the weight-record layer
    (``record_client_weight`` / ``record_client_weights``).
    """

    def __init__(self, pool_size, factory_fn, *, in_memory=True, **_kw):
        self._factory_fn = factory_fn
        self._pool_size = pool_size
        self._available = [factory_fn() for _ in range(pool_size)]
        self.in_memory = True  # always in-memory; kept for back-compat checks
        self._weights_cache: Dict = {}
        self._initial_weights = self._clone_weights(
            self._get_weights(self._available[0])
        )

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

    def checkout(self, client_id, tag=""):
        model = self._available.pop()
        weights = self._weights_cache.get((client_id, tag), self._initial_weights)
        self._set_weights(model, weights)
        return model

    def checkin(self, client_id, model, tag=""):
        self._weights_cache[(client_id, tag)] = self._clone_weights(self._get_weights(model))
        self._available.append(model)

    def release(self, model):
        self._available.append(model)

    def save(self, client_id, model, tag=""):
        self._weights_cache[(client_id, tag)] = self._clone_weights(self._get_weights(model))

    def refresh(self):
        """Recreate pool models after tf.keras.backend.clear_session()."""
        self._available = [self._factory_fn() for _ in range(self._pool_size)]


class MixedModelPool:
    def __init__(self, n_clients, factory_map):
        self._n_clients = n_clients
        self._factory_map = factory_map
        self._weights_cache: Dict = {}
        self.in_memory = True

    @staticmethod
    def _clone_weights(weights):
        return [np.array(w, copy=True) for w in weights]

    def _get_arch(self, client_id):
        from models.mixed_models import get_model_type_for_client
        return get_model_type_for_client(client_id, self._n_clients)

    def _get_weights(self, model):
        return model.get_weights()

    def _set_weights(self, model, weights):
        model.set_weights(self._clone_weights(weights))

    def checkout(self, client_id, tag=""):
        arch = self._get_arch(client_id)
        model = self._factory_map[arch]()
        key = (client_id, tag)
        if key in self._weights_cache:
            model.set_weights(self._clone_weights(self._weights_cache[key]))
        return model

    def checkin(self, client_id, model, tag=""):
        self._weights_cache[(client_id, tag)] = self._clone_weights(model.get_weights())

    def release(self, model):
        pass

    def save(self, client_id, model, tag=""):
        self._weights_cache[(client_id, tag)] = self._clone_weights(model.get_weights())

    def get_cached_weights(self, client_id, tag=""):
        return self._weights_cache.get((client_id, tag))

    def refresh(self):
        pass


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
    test_dataset: Any
    test_labels: np.ndarray
    results: Dict[int, Dict[str, float]] = field(default_factory=dict)
    client_states: List[ClientState] = field(default_factory=list)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    poisoned_clients: List[int] = field(default_factory=list)
    poison_loader: Any = None
    per_client_loaders: Dict[int, Any] = field(default_factory=dict)
    model_pool: Any = None
    poisoned_fl_state: Any = None

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
        record_dir = self._weight_record_dir(round_num)
        for st in self.client_states:
            w = st.data.get("w")
            if w is not None:
                self.record_client_weight(round_num, st.client_id, w)
                continue
            if self.model_pool is None:
                continue
            if hasattr(self.model_pool, 'get_cached_weights'):
                w = self.model_pool.get_cached_weights(st.client_id)
                if w is not None:
                    self.record_client_weight(round_num, st.client_id, w)
                    continue
            model = self.model_pool.checkout(st.client_id)
            w = self.model_pool._get_weights(model)
            self.model_pool.release(model)
            self.record_client_weight(round_num, st.client_id, w)


def get_model_proba(model: Any, X_test: np.ndarray, batch_size: int) -> np.ndarray:
    base = model.base_model if hasattr(model, "base_model") else model
    if hasattr(base, "model"):
        base = base.model
    infer_bs = batch_size * 4 if not _use_tf() else batch_size
    return base.predict(X_test, batch_size=infer_bs, verbose=0)


def evaluate_model(model: Any, X_test: np.ndarray, y_labels: np.ndarray,
                   batch_size: int, num_classes: int, compute_auprc: bool = True,
                   return_proba: bool = False):
    base = model.base_model if hasattr(model, "base_model") else model
    if hasattr(base, "model"):
        base = base.model

    infer_bs = batch_size * 4 if not _use_tf() else batch_size
    preds = base.predict(X_test, batch_size=infer_bs, verbose=0)
    pred_labels, loss, auprc, ece = summarize_prob_predictions(y_labels, preds, num_classes, compute_auprc=compute_auprc)

    accuracy = float(np.mean(pred_labels == y_labels))
    f1 = float(f1_score(y_labels, pred_labels, average="macro", zero_division=0))
    precision = float(precision_score(y_labels, pred_labels, average="macro", zero_division=0))
    recall = float(recall_score(y_labels, pred_labels, average="macro", zero_division=0))

    is_attack_true = y_labels != 0
    is_attack_pred = pred_labels != 0
    n_attack = int(is_attack_true.sum())
    n_benign = int((~is_attack_true).sum())
    tpr = float((is_attack_true & is_attack_pred).sum() / n_attack) if n_attack > 0 else 0.0
    fpr = float(((~is_attack_true) & is_attack_pred).sum() / n_benign) if n_benign > 0 else 0.0

    metrics = {
        "Acc": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "TPR": tpr,
        "FPR": fpr,
        "ECE": ece,
        "Loss": loss,
    }
    if compute_auprc:
        metrics["AUPRC"] = auprc
    if return_proba:
        return metrics, preds
    del preds
    return metrics
