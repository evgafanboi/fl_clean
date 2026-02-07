"""
FedSSD Experiment - Alternative Implementation

Uses full training data for M_class computation (like docs/fedssdexp.py)
Aggregates M_class values directly instead of confusion matrices
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict
from sklearn.metrics import confusion_matrix
from .FedSSD import train_with_ssd_loss
from .common import create_private_dataset


class FedSSDexp:
    """FedSSD experiment - compute M_class on full training data, aggregate M_class directly"""
    
    def __init__(self, m_max: float = 1.0, use_support: bool = False):
        self.name = "FedSSDexp"
        self.m_max = m_max
        self.use_support = use_support
        self.global_model = None
        self.M_class = None
        self.current_round = 0
    
    def extra_log_tokens(self) -> Dict[str, float]:
        return {
            "m_max": self.m_max,
            "support": 1.0 if self.use_support else 0.0,
        }
    
    def aggregate(self, model_weights_list: List, sample_sizes: List[int] = None, **kwargs):
        """Standard FedAvg aggregation"""
        self.current_round += 1
        
        if sample_sizes is None:
            sample_sizes = [1] * len(model_weights_list)
        
        total_samples = sum(sample_sizes)
        aggregated = [np.zeros_like(w) for w in model_weights_list[0]]
        
        for client_weights, n_samples in zip(model_weights_list, sample_sizes):
            weight = n_samples / total_samples
            for i, w in enumerate(client_weights):
                aggregated[i] += weight * w
        
        self.global_model = aggregated
        return aggregated
    
    def compute_M_class_from_cm(self, confusion_matrix: np.ndarray, num_classes: int) -> tuple:
        """
        Compute M_class from confusion matrix.
        Returns M_class and class counts.
        """
        M_class = np.zeros(num_classes, dtype=np.float32)
        class_counts = np.zeros(num_classes, dtype=np.int64)
        
        for k in range(num_classes):
            class_counts[k] = confusion_matrix[k, :].sum()
            if class_counts[k] > 0:
                A_k_k = confusion_matrix[k, k] / confusion_matrix[k, :].sum()
                confusion_rates = []
                for j in range(num_classes):
                    if j != k and confusion_matrix[j, :].sum() > 0:
                        A_j_k = confusion_matrix[j, k] / confusion_matrix[j, :].sum()
                        confusion_rates.append(A_j_k)
                
                max_confusion = max(confusion_rates) if confusion_rates else 0.0
                M_class[k] = A_k_k * (1.0 - max_confusion)
        
        return M_class, class_counts
    
    def aggregate_M_class_values(
        self,
        client_M_class_list: List[np.ndarray],
        client_class_counts_list: List[np.ndarray],
        num_classes: int,
    ) -> np.ndarray:
        """
        Aggregate M_class values directly (not confusion matrices).
        Uses support-weighted or uniform averaging.
        """
        aggregated_M = np.zeros(num_classes, dtype=np.float32)
        
        if self.use_support:
            # Support-weighted aggregation
            total_counts = np.zeros(num_classes, dtype=np.int64)
            
            for M_class, class_counts in zip(client_M_class_list, client_class_counts_list):
                for k in range(num_classes):
                    if class_counts[k] > 0:
                        aggregated_M[k] += M_class[k] * class_counts[k]
                        total_counts[k] += class_counts[k]
            
            for k in range(num_classes):
                if total_counts[k] > 0:
                    aggregated_M[k] /= total_counts[k]
        else:
            # Uniform aggregation (privacy-preserving)
            client_contributions = np.zeros(num_classes, dtype=np.int32)
            
            for M_class, class_counts in zip(client_M_class_list, client_class_counts_list):
                for k in range(num_classes):
                    if class_counts[k] > 0:
                        aggregated_M[k] += M_class[k]
                        client_contributions[k] += 1
            
            for k in range(num_classes):
                if client_contributions[k] > 0:
                    aggregated_M[k] /= client_contributions[k]
        
        self.M_class = aggregated_M
        return aggregated_M
    
    def train(
        self,
        model,
        global_model,
        train_X_path: str,
        train_y_path: str,
        input_dim: int,
        num_classes: int,
        batch_size: int,
        epochs: int,
    ):
        """Train with CE + SSD loss"""
        if self.M_class is None:
            raise ValueError("M_class not computed! Ensure Stage 1 (M_class aggregation) completed before training.")
        
        private_dataset = create_private_dataset(
            train_X_path, train_y_path, input_dim, num_classes, batch_size
        )
        
        print(f"  [FedSSDexp Stage 2] Training with M_class (min={self.M_class.min():.4f}, max={self.M_class.max():.4f})")
        
        train_with_ssd_loss(
            model_wrapper=model,
            private_dataset=private_dataset,
            global_model=global_model,
            M_class=self.M_class,
            m_max=self.m_max,
            num_classes=num_classes,
            epochs=epochs,
            pure_ssd=False,
        )
