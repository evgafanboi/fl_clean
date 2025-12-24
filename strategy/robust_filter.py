import numpy as np
from typing import List, Tuple, Optional

class RobustFilter:
    """
    Robust Filter aggregation (Diakonikolas et al. 2017)
    
    Filter-based algorithm for robust mean estimation that handles
    Byzantine/poisoned clients by projecting onto the top eigenvector
    and removing outliers based on tail bounds.
    """
    
    def __init__(
        self,
        epsilon: float = 0.2,
        tau: float = 0.1,
        c1: float = 1.0,
        c2: float = 0.1,
        thres_multiplier: float = 10.0,
        preset: str = "weights"
    ):
        """
        Args:
            epsilon: Fraction of corrupted samples (Byzantine ratio)
            tau: Corruption tolerance parameter
            c1, c2: Constants for adaptive tail bounding
            thres_multiplier: Multiplier for spectral norm threshold
            preset: "weights" or "logits" for pre-configured parameters
        """
        self.epsilon = epsilon
        self.tau = tau
        self.c1 = c1
        self.c2 = c2
        self.thres_multiplier = thres_multiplier
        
        if preset == "logits":
            self.c2 = 0.5
            self.thres_multiplier = 5.0
        elif preset == "weights":
            self.c2 = 0.1
            self.thres_multiplier = 10.0
    
    def threshold(self, epsilon: float, d: int) -> float:
        """Threshold for spectral norm (Thres(ε))"""
        return self.thres_multiplier * np.sqrt(d) * epsilon
    
    def tail_bound(self, T: float, d: int, epsilon: float, delta: float, tau: float) -> float:
        """Adaptive tail bounding (exponential decay)"""
        return self.c1 * np.exp(-self.c2 * T)
    
    def slack_function(self, epsilon: float, spectral_norm: float) -> float:
        """Slack function δ(ε, s)"""
        return epsilon * spectral_norm
    
    def compute_robust_mean(
        self,
        samples: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Compute robust mean using the filter-based algorithm.
        
        Args:
            samples: List of vectors (model weights or logits)
            weights: Optional sample weights (for weighted averaging)
        
        Returns:
            Robust mean estimate
        """
        if len(samples) == 0:
            raise ValueError("Empty sample list")
        
        if len(samples) == 1:
            return samples[0]
        
        n = len(samples)
        d = samples[0].shape[0]
        
        S = np.vstack(samples)
        
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.array(weights) / np.sum(weights)
        
        mu_S = np.average(S, axis=0, weights=weights)
        
        centered = S - mu_S
        Sigma = np.cov(centered.T, aweights=weights)
        
        if d == 1:
            spectral_norm = np.abs(Sigma)
        else:
            spectral_norm = np.linalg.norm(Sigma, ord=2)
        
        threshold = self.threshold(self.epsilon, d)
        
        if spectral_norm <= threshold:
            return mu_S
        
        if d == 1:
            v_star = np.array([1.0])
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
            max_eigenvalue_idx = np.argmax(np.abs(eigenvalues))
            v_star = eigenvectors[:, max_eigenvalue_idx]
        
        projections = np.dot(centered, v_star)
        
        delta = self.slack_function(self.epsilon, spectral_norm)
        
        T = self._find_threshold(projections, weights, d, delta)
        
        filtered_indices = np.abs(projections) <= (T + delta)
        
        if np.sum(filtered_indices) == 0:
            return mu_S
        
        filtered_samples = S[filtered_indices]
        filtered_weights = weights[filtered_indices]
        filtered_weights = filtered_weights / np.sum(filtered_weights)
        
        robust_mean = np.average(filtered_samples, axis=0, weights=filtered_weights)
        
        return robust_mean
    
    def _find_threshold(
        self,
        projections: np.ndarray,
        weights: np.ndarray,
        d: int,
        delta: float
    ) -> float:
        """
        Binary search to find T such that tail bound is satisfied.
        """
        sorted_indices = np.argsort(np.abs(projections))
        sorted_projections = np.abs(projections[sorted_indices])
        sorted_weights = weights[sorted_indices]
        
        cumulative_weights = np.cumsum(sorted_weights)
        
        T_candidates = sorted_projections
        
        for i, T in enumerate(T_candidates):
            tail_mass = 1.0 - cumulative_weights[i]
            
            target_tail = self.tail_bound(T, d, self.epsilon, delta, self.tau)
            
            if tail_mass <= target_tail:
                return T
        
        return sorted_projections[-1]
    
    def aggregate(
        self,
        client_updates: List[np.ndarray],
        sample_sizes: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Aggregate client updates using robust filtering.
        
        Args:
            client_updates: List of client weight updates or gradients
            sample_sizes: Optional list of sample sizes for weighting
        
        Returns:
            Robust aggregated update
        """
        if sample_sizes is not None:
            weights = np.array(sample_sizes, dtype=np.float32)
            weights = weights / np.sum(weights)
        else:
            weights = None
        
        return self.compute_robust_mean(client_updates, weights)


class RobustFilterWeights(RobustFilter):
    """
    Robust Filter for model weights aggregation.
    
    Optimized for high-dimensional weight vectors.
    For a fully connected model with ~46K parameters.
    """
    
    def __init__(self, epsilon: float = 0.2, tau: float = 0.1):
        super().__init__(
            epsilon=epsilon,
            tau=tau,
            c1=1.0,
            c2=0.05,
            thres_multiplier=15.0,
            preset="weights"
        )
    
    def aggregate(
        self,
        client_weights: List[List[np.ndarray]],
        sample_sizes: List[int]
    ) -> List[np.ndarray]:
        """
        Aggregate model weights layer by layer.
        
        Args:
            client_weights: List of client weight lists (each client has multiple layers)
            sample_sizes: List of client sample sizes
        
        Returns:
            Aggregated weights (list of arrays, one per layer)
        """
        n_clients = len(client_weights)
        n_layers = len(client_weights[0])
        
        aggregated = []
        
        for layer_idx in range(n_layers):
            layer_updates = []
            
            for client_idx in range(n_clients):
                layer_weights = client_weights[client_idx][layer_idx]
                flattened = layer_weights.flatten()
                layer_updates.append(flattened)
            
            robust_mean = self.compute_robust_mean(layer_updates, sample_sizes)
            
            original_shape = client_weights[0][layer_idx].shape
            aggregated.append(robust_mean.reshape(original_shape))
        
        return aggregated


class RobustFilterLogits(RobustFilter):
    """
    Robust Filter for logits aggregation (FedMD, FedDistillation).
    
    Optimized for lower-dimensional logit vectors (num_classes).
    For CIC23: 34 classes.
    """
    
    def __init__(self, epsilon: float = 0.2, tau: float = 0.1):
        super().__init__(
            epsilon=epsilon,
            tau=tau,
            c1=1.0,
            c2=0.5,
            thres_multiplier=5.0,
            preset="logits"
        )
    
    def aggregate_per_class_logits(
        self,
        client_logits_list: List[np.ndarray],
        sample_sizes: List[int]
    ) -> np.ndarray:
        """
        Aggregate per-class logits from multiple clients.
        
        Args:
            client_logits_list: List of (num_classes, num_classes) logit matrices
            sample_sizes: List of client sample sizes
        
        Returns:
            Aggregated logits (num_classes, num_classes)
        """
        n_clients = len(client_logits_list)
        num_classes = client_logits_list[0].shape[0]
        
        aggregated_logits = np.zeros_like(client_logits_list[0])
        
        for class_idx in range(num_classes):
            class_logits_from_clients = [
                client_logits[class_idx, :] for client_logits in client_logits_list
            ]
            
            robust_mean = self.compute_robust_mean(
                class_logits_from_clients,
                sample_sizes
            )
            
            aggregated_logits[class_idx, :] = robust_mean
        
        return aggregated_logits
