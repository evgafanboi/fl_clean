"""Finetune CIL method - simply train on new task without catastrophic forgetting prevention"""

from .base import CILMethod


class Finetune(CILMethod):
    """Naive finetune on new task without any continual learning mechanism"""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes)
        self.name = "Finetune"
    
    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)
    
    def after_task(self, model, task_data=None):
        pass
