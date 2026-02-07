import numpy as np


class NoneStrategy:
    """
    Independent learning baseline - no federation.
    Each client trains independently for (rounds * local_epochs) total epochs.
    Only personalized evaluation is performed (no global model).
    """
    
    def __init__(self):
        self.name = "None"
        self.requires_updates = False
        self.requires_root_dataset = False
        self.independent_learning = True
    
    def aggregate(self, model_weights_list, sample_sizes=None):
        """
        No aggregation happens in independent learning.
        Return arbitrary weights (won't be used).
        """
        return model_weights_list[0] if model_weights_list else []
