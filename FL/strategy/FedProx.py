import numpy as np
from .FedAvg import FedAvg

class FedProx(FedAvg):
    def __init__(self, mu=0.01, adaptive_mu=False):
        super().__init__()
        self.name = "FedProx"
        self.base_mu = mu
        self.adaptive_mu = adaptive_mu
        self.current_round = 0
        self.global_weights = None
        print(f"FedProx initialized with base_mu={mu}, adaptive={adaptive_mu}")
    
    @property
    def mu(self):
        if not self.adaptive_mu:
            return self.base_mu
        
        growth_factor = 1 + 0.1 * self.current_round
        return min(self.base_mu * growth_factor, self.base_mu * 3.0)
    
    def aggregate(self, model_weights_list, sample_sizes=None):
        self.current_round += 1
        current_mu = self.mu
        print(f"FedProx Round {self.current_round}: Using μ = {current_mu:.4f}")
        
        aggregated_weights = super().aggregate(model_weights_list, sample_sizes)
        self.global_weights = [w.copy() for w in aggregated_weights]
        return aggregated_weights
