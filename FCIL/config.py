"""Configuration for FCIL pipeline"""

from dataclasses import dataclass


@dataclass
class FCILConfig:
    """Configuration for Federated Continual Learning"""
    
    # FL settings
    strategy: str = "FedAvg"
    n_clients: int = 10
    rounds_per_task: int = 5
    
    # CIL settings
    cil_method: str = "finetune"
    ewc_lambda: float = 50.0
    mas_lambda: float = 1.0
    lwf_alpha: float = 0.5
    lwf_temperature: float = 2.0
    icarl_memory: float = 1.0
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
    mixed_models: bool = False
    
    # Logging
    log_file: str = ""
    
    # BiC settings
    bic_val_split: float = 0.1

    # GLFC settings
    glfc_encoder_epochs: int = 50
    glfc_model_selection: bool = True
    glfc_grad_enc: str = "main"

    # CBKD settings
    cbkd_lambda: float = 10.0
    cbkd_alpha: float = 10.0
    cbkd_beta: float = 10.0
    cbkd_proto_size: int = 50

    # PASS settings (reuses cbkd_lambda, cbkd_proto_size)
    pass_gamma: float = 10.0

    # FEAT/EXP settings
    feat_lambda: float = 1.0
    feat_rho: float = 0.9
    feat_temp: float = 0.5
    ours_ekd_epochs: int = 2
    ours_ekd_lambda: float = 1.0
    ours_proto_lambda: float = 10.0
    ours_proto_rel_lambda: float = 1.0
    ours_encoder_lr_factor: float = 0.5
    # Ours-specific KD scaling (separate from PASS gamma)
    ours_kd_gamma: float = 1.0
    ours_drift_temp: float = 0.5
    replay_cap: bool = True
    replay_min_per_class: int = 128
    replay_balance: float = 1.0
    ours_entropy_beta: float = 0.0
    eva_quantile: float = 0.95
    robust_threshold: float = 0.3
    robust_workers: int = 8
    no_filter: bool = False

    # SSD settings
    m_max: float = 1.0

    # Backend
    use_tf: bool = False

    # Checkpointing
    checkpoint: int = 0
    fresh_run: bool = False

    # Eval scheduling
    last_eval: bool = False
    sweep_eval: bool = False
    keep_last_rounds: int = 2
