"""Centralized learning baseline - merges all client data"""

import numpy as np
from typing import List


class Centralized:
    """Centralized baseline - no FL, trains on merged data from all clients"""
    
    def __init__(self):
        self.name = "Centralized"
    
    def aggregate(self, client_weights_list: List, sample_sizes: List[int]) -> List[np.ndarray]:
        """Return first client weights (only one 'client' with merged data)"""
        return client_weights_list[0]
