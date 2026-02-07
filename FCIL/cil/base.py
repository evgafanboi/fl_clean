"""Base class for CIL algorithms"""

import numpy as np
import tensorflow as tf
from typing import Optional


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
    
    def after_task(self, model, task_data: Optional[tf.data.Dataset] = None):
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
