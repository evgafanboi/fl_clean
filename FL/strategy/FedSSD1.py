"""
FedSSD without Public Dataset (V1)

Two-stage workflow using aggregated confusion matrices instead of public dataset:
Stage 1: Confusion matrix collection (forward pass on client validation data)
Stage 2: FedSSD training (CE + SSD loss with M_class from aggregated CMs)

Workflow per round:
1. Server sends global model to clients
2. Clients: Forward pass on validation set → confusion matrix (+ support counts)
3. Server: Aggregate confusion matrices with non-zero/support-weighted averaging
4. Server: Compute M_class from aggregated CM
5. Clients: FedSSD training (CE + SSD loss, distill from frozen global)
6. Server: FedAvg aggregation of client updates

Key difference from FedSSD:
- FedSSD: Uses public dataset to compute M_class
- FedSSD1: Uses aggregated client confusion matrices to compute M_class
- Training is identical: L_CE + L_SSD
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict
from .FedSSD import train_with_ssd_loss
from .common import create_private_dataset


class FedSSD1:
    """FedSSD without public dataset - uses aggregated confusion matrices for M_class"""
    
    def __init__(self, m_max: float = 1.0, use_support: bool = False):
        self.name = "FedSSD1"
        self.m_max = m_max
        self.use_support = use_support
        self.global_model = None
        self.global_confusion_matrix = None
        self.M_class = None
        self.current_round = 0
        self.stage = 1  # Stage 1: CM collection, Stage 2: SSD training
    
    def extra_log_tokens(self) -> Dict[str, float]:
        return {
            "m_max": self.m_max,
            "support": 1.0 if self.use_support else 0.0,
        }
    
    def should_collect_confusion_matrices(self) -> bool:
        """Check if this round requires CM collection (always Stage 1)"""
        return True
    
    def aggregate(self, model_weights_list: List, sample_sizes: List[int] = None, **kwargs):
        """Standard FedAvg aggregation after Stage 2 (SSD training)"""
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
    
    def aggregate_confusion_matrices(
        self,
        confusion_matrices: List[np.ndarray],
        support_counts: List[np.ndarray] = None,
        num_classes: int = 8,
    ) -> np.ndarray:
        """
        Aggregate confusion matrices with smart non-zero averaging.
        Only average per class if client has samples for that class.
        
        Args:
            confusion_matrices: List of client confusion matrices
            support_counts: Optional list of support arrays (samples per class per client)
            num_classes: Number of classes
        
        Returns:
            Aggregated global confusion matrix
        """
        n_clients = len(confusion_matrices)
        aggregated_cm = np.zeros((num_classes, num_classes), dtype=np.float32)
        
        for true_class in range(num_classes):
            for pred_class in range(num_classes):
                if self.use_support and support_counts is not None:
                    # Weighted averaging by support count
                    weighted_sum = 0.0
                    total_support = 0.0
                    
                    for client_idx in range(n_clients):
                        cm = confusion_matrices[client_idx]
                        support = support_counts[client_idx][true_class]
                        
                        # Only include if client has samples for this class AND non-zero metric
                        if support > 0 and cm[true_class, pred_class] > 0:
                            weighted_sum += cm[true_class, pred_class] * support
                            total_support += support
                    
                    if total_support > 0:
                        aggregated_cm[true_class, pred_class] = weighted_sum / total_support
                else:
                    # Simple non-zero averaging - abstain if no samples for class
                    non_zero_values = [
                        cm[true_class, pred_class] 
                        for cm in confusion_matrices 
                        if cm[true_class, pred_class] > 0
                    ]
                    if non_zero_values:
                        aggregated_cm[true_class, pred_class] = np.mean(non_zero_values)
        
        self.global_confusion_matrix = aggregated_cm
        return aggregated_cm
    
    def compute_M_class(self, confusion_matrix: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Compute M_class from confusion matrix: A_k_k * (1 - max_confusion_rate).
        Follows FedSSD approach.
        """
        M_class = np.zeros(num_classes, dtype=np.float32)
        
        for k in range(num_classes):
            class_total = confusion_matrix[k, :].sum()
            if class_total == 0:
                continue
            
            # Diagonal accuracy A_k_k
            A_k_k = confusion_matrix[k, k] / class_total
            
            # Max confusion rate from other classes
            confusion_rates = []
            for j in range(num_classes):
                if j == k:
                    continue
                row_sum = confusion_matrix[j, :].sum()
                if row_sum > 0:
                    confusion_rates.append(confusion_matrix[j, k] / row_sum)
            
            max_confusion = max(confusion_rates) if confusion_rates else 0.0
            M_class[k] = A_k_k * (1.0 - max_confusion)
        
        self.M_class = M_class
        return M_class
    
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
        """Train with CE + SSD loss (same as FedSSD)"""
        if self.M_class is None:
            raise ValueError("M_class not computed! Ensure Stage 1 (confusion matrix aggregation) completed before training.")
        
        private_dataset = create_private_dataset(
            train_X_path, train_y_path, input_dim, num_classes, batch_size
        )
        
        print(f"  [FedSSD1 Stage 2] Training with M_class (min={self.M_class.min():.4f}, max={self.M_class.max():.4f})")
        
        train_with_ssd_loss(
            model_wrapper=model,
            private_dataset=private_dataset,
            global_model=global_model,
            M_class=self.M_class,
            m_max=self.m_max,
            num_classes=num_classes,
            epochs=epochs,
        )
