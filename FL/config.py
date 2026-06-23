from dataclasses import dataclass
from typing import Optional


@dataclass
class FDConfig:
    algorithm: str = "FD"
    n_clients: int = 10
    partition_type: str = "label_skew-10"
    rounds: int = 10
    epochs: int = 5
    batch_size: int = 8192
    gamma: float = 1.0
    data_calc: bool = False
    model_type: str = "dense"
    m_max: float = 1.0
    train_rounds: int = 3
    dis_rounds: int = 3
    dist_rounds: int = 2
    dkd_steps: int = 3
    dkd_lr: float = 0.001
    dkd_temp: float = 3.0
    personalized_eval: bool = False
    poison: str = None
    exp1_lambda: float = 1.0
    exp1_temperature: float = 3.0
    ours_ekd_lambda: float = 1.0
    ours_kd: str = "ekd"
    ab_alpha: float = 1.0
    ab_beta: float = 0.0
    ours_temperature: float = 4.0
    kd_epochs: int = 2
    robust_epsilon: float = 0.2
    robust_rm_budget: Optional[int] = None
    robust_threshold: float = 0.75
    robust_workers: int = 8
    robust_filter_reference: str = "Blom"
    robust_v: float = 4.0
    robust_filter_cronus: bool = False
    eva: bool = False
    eva2: bool = False
    evw: bool = False
    remove_dis: bool = False
    checkpoint: int = 0
    cleanup_interval: int = 10
    skip_eval: bool = False
    fresh_run: bool = False
    cache_test_set: bool = False
    mixed_models: bool = False
    hamming_tau: float = 0.5
    exp_rho: float = 0.05
    f1_curve: bool = False
    keep_last_rounds: int = 2
    no_disk_cache: bool = True
    save_weights: bool = True

    def to_algorithm_params(self) -> dict:
        return {
            'n_clients': self.n_clients,
            'partition_type': self.partition_type,
            'rounds': self.rounds,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'data_calc': self.data_calc,
            'model_type': self.model_type,
            'm_max': self.m_max,
            'train_rounds': self.train_rounds,
            'dis_rounds': self.dis_rounds,
            'dist_rounds': self.dist_rounds,
            'dkd_steps': self.dkd_steps,
            'dkd_lr': self.dkd_lr,
            'dkd_temp': self.dkd_temp,
            'personalized_eval': self.personalized_eval,
        }
