"""Configuration for FCIL pipeline"""

from dataclasses import dataclass


@dataclass
class FCILConfig:
    """Configuration for Federated Continual Learning"""
    
    # FL settings
    strategy: str = "FedAvg"
    n_clients: int = 10
    rounds_per_task: int = 10
    
    # CIL settings
    cil_method: str = "finetune"
    ewc_lambda: float = 5000.0
    
    # Data settings
    partition_type: str = "label_skew"
    partition_root: str = "data/partitions"
    task_order_file: str = ""
    
    # Model settings
    model: str = "dense"
    batch_size: int = 256
    epochs_per_round: int = 1
    
    # Logging
    log_file: str = ""
