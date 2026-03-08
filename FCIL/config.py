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
    ewc_lambda: float = 50.0
    mas_lambda: float = 1.0
    lwf_alpha: float = 0.5
    lwf_temperature: float = 2.0
    icarl_memory: int = 2000
    icarl_bce: bool = False
    foster_beta1: float = 0.97
    foster_beta2: float = 0.97
    foster_lambda_okd: float = 1.0
    foster_compression_epochs: int = 50
    
    # Data settings
    partition_type: str = "label_skew_0.1"
    partition_root: str = "data/partitions"
    task_order_file: str = ""
    
    # Model settings
    model: str = "dense"
    batch_size: int = 256
    epochs_per_round: int = 1
    
    # Logging
    log_file: str = ""
    
    # BiC settings
    bic_val_split: float = 0.1

    # GLFC settings
    glfc_encoder_epochs: int = 50
    glfc_model_selection: bool = True
    glfc_grad_enc: str = "main"

    # SSD settings
    m_max: float = 1.0
