"""Base class for CIL algorithms"""

import numpy as np
from typing import Optional, Any


# TF model types — not safely picklable; excluded from checkpoint state
_TF_MODEL_ATTRS = frozenset({
    'old_model', 'old_logits_model', '_proxy_model', '_old_module', '_grad_encoder',
    '_generator', '_old_module_tf',
})


class CILMethod:
    """Base class for Continual/Incremental Learning methods"""
    
    def __init__(self, num_classes: int, **kwargs):
        self.num_classes = num_classes
        self.current_task = 0
        self.seen_classes = set()
    
    def before_task(self, task_id: int, task_classes: list):
        """Called before starting a new task"""
        self.current_task = task_id
        self.seen_classes.update(task_classes)
    
    def after_task(self, model, task_data: Optional[Any] = None):
        """Called after completing a task"""
        pass
    
    def get_train_step(self, model, optimizer, loss_fn):
        """
        Return a custom training step function.
        Default returns None (use standard training).
        """
        return None
    
    def extra_metrics(self) -> dict:
        """Return extra metrics for logging"""
        return {}

    def get_ckpt_state(self) -> dict:
        """Return serialisable checkpoint state.

        For PyTorch backend, all attributes are included (nn.Module is picklable).
        For TensorFlow backend, Keras model objects are excluded; they are
        reconstructed on resume via set_old_model().
        """
        from .base import _TF_MODEL_ATTRS
        from ..backend import use_tf as _use_tf
        if not _use_tf():
            return dict(self.__dict__)
        return {k: v for k, v in self.__dict__.items() if k not in _TF_MODEL_ATTRS}

    def set_ckpt_state(self, state: dict) -> None:
        """Restore checkpoint state produced by get_ckpt_state()."""
        self.__dict__.update(state)
