"""
FedSSD2 - FedSSD with Public Unlabeled Dataset

Uses public unlabeled dataset (like SSFL-IDS):
- Stage 1: Clients predict on public data → server votes for pseudo-labels → 
  trains global model on pseudo-labeled public data → computes pseudo-confusion matrix
- Stage 2: Normal FedSSD training with pseudo-CM
- NO partition restoration needed (uses separate public dataset)
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict
from .FedSSD import train_with_ssd_loss


class FedSSD2:
    """FedSSD with public unlabeled dataset and pseudo-labeling"""
    
    def __init__(self, m_max: float = 1.0, use_support: bool = False, threshold: float = 0.3):
        self.name = "FedSSD2"
        self.m_max = m_max
        self.use_support = use_support  # Unused, kept for compatibility
        self.threshold = threshold
        self.global_model = None
        self.M_class = None
        self.current_round = 0
    
    def extra_log_tokens(self) -> Dict[str, float]:
        return {
            "m_max": self.m_max,
            "threshold": self.threshold,
        }
    
    def hard_label_vote(self, all_client_predictions: List[np.ndarray], num_classes: int, threshold: float = 0.3) -> np.ndarray:
        """
        Vote on pseudo-labels from client predictions with confidence threshold.
        Uses majority voting for each sample.
        
        Args:
            all_client_predictions: List of prediction arrays from clients (each is [n_samples, num_classes])
            num_classes: Number of classes
            threshold: Minimum vote ratio (votes/clients) to assign label (0-1)
                       If top_class_votes/n_clients < threshold, label as unlabeled (num_classes)
        
        Returns:
            Voted labels as 1D array [n_samples] (unlabeled samples marked as num_classes)
        """
        n_clients = len(all_client_predictions)
        n_samples = all_client_predictions[0].shape[0]
        
        voted_labels = np.zeros(n_samples, dtype=np.int32)
        
        for sample_idx in range(n_samples):
            votes = np.zeros(num_classes, dtype=np.int32)
            for client_preds in all_client_predictions:
                pred_label = np.argmax(client_preds[sample_idx])
                votes[pred_label] += 1
            
            top_class = np.argmax(votes)
            vote_ratio = votes[top_class] / n_clients
            
            # If top class doesn't exceed threshold, mark as unlabeled
            if vote_ratio >= threshold:
                voted_labels[sample_idx] = top_class
            else:
                voted_labels[sample_idx] = num_classes  # Unlabeled
        
        return voted_labels
    
    def compute_pseudo_confusion_matrix(
        self,
        model,
        X_public: np.ndarray,
        pseudo_labels: np.ndarray,
        num_classes: int,
        batch_size: int,
    ) -> np.ndarray:
        """
        Compute pseudo-confusion matrix from model predictions vs pseudo-labels.
        
        Args:
            model: Keras model (or wrapper)
            X_public: Public unlabeled data
            pseudo_labels: Voted pseudo-labels from clients
            num_classes: Number of classes
            batch_size: Batch size for prediction
        
        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        keras_model = model.model if hasattr(model, 'model') else model
        
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        
        for start_idx in range(0, len(X_public), batch_size):
            end_idx = min(start_idx + batch_size, len(X_public))
            batch_X = X_public[start_idx:end_idx]
            batch_pseudo_y = pseudo_labels[start_idx:end_idx]
            
            predictions = keras_model(batch_X, training=False)
            pred_labels = tf.argmax(predictions, axis=1).numpy()
            
            for true_label, pred_label in zip(batch_pseudo_y, pred_labels):
                cm[true_label, pred_label] += 1
        
        return cm
    
    def compute_M_class_from_cm(self, confusion_matrix: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Compute M_class from confusion matrix.
        
        M_class[k] = A_k_k * (1 - max(A_j_k for j != k))
        where A_k_k = cm[k, k] / sum(cm[k, :])
        """
        M_class = np.zeros(num_classes, dtype=np.float32)
        
        for k in range(num_classes):
            row_sum = confusion_matrix[k, :].sum()
            if row_sum > 0:
                A_k_k = confusion_matrix[k, k] / row_sum
                
                confusion_rates = []
                for j in range(num_classes):
                    if j != k and confusion_matrix[j, :].sum() > 0:
                        A_j_k = confusion_matrix[j, k] / confusion_matrix[j, :].sum()
                        confusion_rates.append(A_j_k)
                
                max_confusion = max(confusion_rates) if confusion_rates else 0.0
                M_class[k] = A_k_k * (1.0 - max_confusion)
        
        self.M_class = M_class
        return M_class
    
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
            raise ValueError("M_class not computed! Ensure Stage 1 (pseudo-CM computation) completed before training.")
        
        from .common import create_private_dataset
        
        private_dataset = create_private_dataset(
            train_X_path, train_y_path, input_dim, num_classes, batch_size
        )
        
        print(f"  [FedSSD2 Stage 2] Training with M_class (min={self.M_class.min():.4f}, max={self.M_class.max():.4f})")
        
        train_with_ssd_loss(
            model_wrapper=model,
            private_dataset=private_dataset,
            global_model=global_model,
            M_class=self.M_class,
            m_max=self.m_max,
            num_classes=num_classes,
            epochs=epochs,
        )
