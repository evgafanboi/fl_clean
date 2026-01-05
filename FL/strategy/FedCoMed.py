"""
FedCoMed+: Robust Aggregation via ρ-smoothed Coordinate-wise Median

From the Fed+ framework, FedCoMed+ offers robust aggregation via the median with added 
flexibility in allowing different parameters for each coordinate.

Algorithm:
The aggregator computes w^t from {w_k^t : k ∈ S_t} using an iterative procedure:

1. Start with w̄ = w_mean := Mean{w_k^t : k ∈ S_t}
2. Iterate until w̄ converges:
   - For each client k: v_k ← max{0, w_k^t - w̄^t - ρ·sign(w_k^t - w̄^t)}
   - Update: w̄ ← w_mean - Mean{v_k : k ∈ S_t}

3. Final result: R(w̄^t, w_k^t) = (I - Λ_k^t)w_k^t + Λ_k^t w̄^t
   where Λ_k^t(i,i) := min{1, ρ/|w_k^t(i) - w̄^t(i)|}, i = 1,...,d

The ρ parameter controls the smoothness of the approximation.

Reference:
This is a ρ-smoothed approximation of the Coordinate-wise Median that provides
robustness to Byzantine attacks while maintaining smooth optimization properties.
"""

import numpy as np


class FedCoMed:
    """
    FedCoMed: Coordinate-wise Median aggregation.
    
    Provides robust aggregation through coordinate-wise median computation.
    """
    
    def __init__(self, name="FedCoMed"):
        """
        Initialize FedCoMed strategy.
        
        Args:
            name: Strategy name identifier
        """
        self.name = name
    
    def aggregate(self, weights_list, sample_sizes=None, **kwargs):
        """
        Aggregate client model weights using coordinate-wise median.
        
        Args:
            weights_list: List of model weights from clients
            sample_sizes: List of dataset sizes (ignored, median doesn't use weights)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Aggregated weights (coordinate-wise median across clients)
        """
        if not weights_list:
            raise ValueError("weights_list cannot be empty")
        
        if len(weights_list) == 1:
            return weights_list[0]
        
        aggregated_weights = []
        
        # For each layer in the model
        for layer_idx in range(len(weights_list[0])):
            # Extract weights for this layer from all clients
            layer_weights = [w[layer_idx] for w in weights_list]
            
            # Stack: shape (num_clients, *layer_shape)
            stacked = np.stack(layer_weights, axis=0)
            
            # Compute median along client axis (axis=0)
            layer_median = np.median(stacked, axis=0)
            
            aggregated_weights.append(layer_median)
        
        return aggregated_weights
    
    def get_aggregation_info(self):
        """
        Get information about the aggregation method.
        
        Returns:
            Dictionary with strategy details
        """
        return {
            'name': self.name,
            'type': 'coordinate_wise_median',
            'robustness': 'Byzantine-resistant (50% breakdown point)',
            'properties': [
                'Takes median of each parameter across clients',
                'Robust to up to 50% Byzantine clients',
                'No hyperparameters needed',
                'Ignores sample sizes (inherent property of median)',
                'Deterministic aggregation'
            ]
        }

