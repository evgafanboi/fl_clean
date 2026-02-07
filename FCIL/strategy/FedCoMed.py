"""FedCoMed strategy for FCIL - Coordinate-wise Median aggregation"""

import numpy as np
from typing import List


class FedCoMed:
    """Coordinate-wise Median aggregation for Byzantine robustness"""
    
    def __init__(self):
        self.name = "FedCoMed"
    
    def aggregate(self, client_weights_list: List, sample_sizes: List[int]) -> List[np.ndarray]:
        """Aggregate using coordinate-wise median (robust to Byzantine clients)"""
        aggregated = []
        
        for layer_idx in range(len(client_weights_list[0])):
            layer_weights = [w[layer_idx] for w in client_weights_list]
            stacked = np.stack(layer_weights, axis=0)
            layer_median = np.median(stacked, axis=0)
            aggregated.append(layer_median)
        
        return aggregated
