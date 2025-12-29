import argparse

from .algorithms.fd import FederatedDistillation
from .algorithms.feddkd import FedDKD
from .algorithms.fedmd import FedMD
from .algorithms.fedproto import FedProto
from .algorithms.fedssd import FedSSD
from .algorithms.ssfl_ids import SSFLIDS
from .config import FDConfig


ALGORITHM_REGISTRY = {
    "FD": FederatedDistillation,
    "FedDKD": FedDKD,
    "FedProto": FedProto,
    "FedMD": FedMD,
    "FedSSD": FedSSD,
    "SSFL-IDS": SSFLIDS,
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Federated Distillation Pipeline")
    parser.add_argument("--algorithm", type=str, default="FD", choices=sorted(ALGORITHM_REGISTRY.keys()), help="Algorithm to run")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of participating clients")
    parser.add_argument("--partition_type", type=str, default="label_skew-10", help="Partition descriptor, e.g., label_skew-10")
    parser.add_argument("--rounds", type=int, default=10, help="Number of communication rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for training datasets")
    parser.add_argument("--gamma", type=float, default=1.0, help="Distillation temperature / weighting factor")
    parser.add_argument("--data_calc", action="store_true", help="Enable extra data calculations if supported by the algorithm")
    parser.add_argument("--model_type", type=str, default="dense", help="Model architecture identifier")
    parser.add_argument("--m_max", type=float, default=1.0, help="Maximum momentum value for selective sampling")
    parser.add_argument("--train_rounds", type=int, default=3, help="Training rounds for discriminator-based methods")
    parser.add_argument("--dis_rounds", type=int, default=3, help="Discriminator rounds for SSFL-IDS variants")
    parser.add_argument("--dist_rounds", type=int, default=2, help="Distillation rounds inside SSFL-IDS loop")
    parser.add_argument("--theta", type=float, default=-1.0, help="Threshold parameter for selective sharing")
    parser.add_argument("--dkd_steps", type=int, default=3, help="DKD gradient steps per round (for FedDKD)")
    parser.add_argument("--dkd_lr", type=float, default=0.001, help="Learning rate for DKD SGD updates (for FedDKD)")
    return parser


def main(argv=None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    config = FDConfig(
        algorithm=args.algorithm,
        n_clients=args.n_clients,
        partition_type=args.partition_type,
        rounds=args.rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        data_calc=args.data_calc,
        model_type=args.model_type,
        m_max=args.m_max,
        train_rounds=args.train_rounds,
        dis_rounds=args.dis_rounds,
        dist_rounds=args.dist_rounds,
        theta=args.theta,
        dkd_steps=args.dkd_steps,
        dkd_lr=args.dkd_lr,
    )

    algorithm_cls = ALGORITHM_REGISTRY.get(config.algorithm)
    if algorithm_cls is None:
        available = ", ".join(sorted(ALGORITHM_REGISTRY.keys()))
        parser.error(f"Unknown algorithm '{config.algorithm}'. Available algorithms: {available}")

    algorithm = algorithm_cls(config)
    algorithm.run()


if __name__ == "__main__":
    main()
