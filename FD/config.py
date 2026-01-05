from dataclasses import dataclass


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
    theta: float = -1.0
    dkd_steps: int = 3
    dkd_lr: float = 0.001
    personalized_eval: bool = False
    poison: str = None

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
            'theta': self.theta,
            'dkd_steps': self.dkd_steps,
            'dkd_lr': self.dkd_lr,
            'personalized_eval': self.personalized_eval,
        }
