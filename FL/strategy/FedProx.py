import numpy as np
from .FedAvg import FedAvg

class FedProx(FedAvg):
    def __init__(self, mu=0.01):
        super().__init__()
        self.name = "FedProx"
        self.mu = mu
        self.global_weights = None
        print(f"FedProx initialized with mu={mu}")
    
    def extra_log_tokens(self):
        return {"mu": self.mu}
    
    def aggregate(self, model_weights_list, sample_sizes=None):
        aggregated_weights = super().aggregate(model_weights_list, sample_sizes)
        self.global_weights = [w.copy() for w in aggregated_weights]
        return aggregated_weights
