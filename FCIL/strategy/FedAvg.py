"""FedAvg strategy for FCIL"""

import numpy as np
from typing import List


class FedAvg:
    """Standard FedAvg aggregation"""
    
    def __init__(self):
        self.name = "FedAvg"
    
    def aggregate(self, client_weights_list: List, sample_sizes: List[int]) -> List[np.ndarray]:
        """Weighted average by number of samples"""
        total_samples = sum(sample_sizes)
        aggregated = [np.zeros_like(w) for w in client_weights_list[0]]
        
        for client_weights, n_samples in zip(client_weights_list, sample_sizes):
            weight = n_samples / total_samples
            for i, w in enumerate(client_weights):
                aggregated[i] += weight * w
        
        return aggregated
