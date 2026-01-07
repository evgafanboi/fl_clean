import argparse

from .pipeline import FLConfig, run_pipeline


WEIGHT_AGGREGATION_STRATEGIES = {"FedAvg", "FedProx", "FedDyn", "FedCoMed", "RobustFilter", "DeepFed"}
DISTILLATION_STRATEGIES = {"FD", "FedDKD", "FedProto", "FedMD", "FedSSD", "SSFL-IDS"}
ALL_STRATEGIES = sorted(WEIGHT_AGGREGATION_STRATEGIES | DISTILLATION_STRATEGIES)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Federated Learning Pipeline")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients to simulate")
    parser.add_argument("--partition_type", type=str, default="iid-10", help="Partition descriptor, e.g., iid-10")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument(
        "--strategy",
        type=str,
        default="FedAvg",
        help=f"Strategy: {', '.join(ALL_STRATEGIES)}",
    )
    parser.add_argument("--feddyn_alpha", type=float, default=0.1, help="FedDyn alpha (paper best at 0.1)")
    parser.add_argument("--batch_size", type=int, default=8192, help="Minibatch size for local training")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--weights_cache_dir", type=str, default="temp_weights", help="Directory to cache client weights")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal term")
    parser.add_argument("--adaptive_mu", action="store_true", help="Enable adaptive FedProx mu schedule")
    parser.add_argument("--model", type=str, default="dense", help="Model architecture identifier (default: dense)")
    parser.add_argument("--robust_epsilon", type=float, default=0.2, help="RobustFilter epsilon (Byzantine ratio)")
    parser.add_argument("--robust_tau", type=float, default=0.1, help="RobustFilter tau parameter")
    parser.add_argument("--gamma", type=float, default=1.0, help="Distillation temperature / weighting factor (for distillation strategies)")
    parser.add_argument("--data_calc", action="store_true", help="Enable extra data calculations (for distillation strategies)")
    parser.add_argument("--m_max", type=float, default=1.0, help="Maximum momentum value for selective sampling")
    parser.add_argument("--train_rounds", type=int, default=3, help="Training rounds for discriminator-based methods")
    parser.add_argument("--dis_rounds", type=int, default=3, help="Discriminator rounds for SSFL-IDS")
    parser.add_argument("--dist_rounds", type=int, default=2, help="Distillation rounds inside SSFL-IDS loop")
    parser.add_argument("--theta", type=float, default=-1.0, help="Threshold parameter for selective sharing")
    parser.add_argument("--dkd_steps", type=int, default=3, help="DKD gradient steps per round (for FedDKD)")
    parser.add_argument("--dkd_lr", type=float, default=0.001, help="Learning rate for DKD SGD updates (for FedDKD)")
    parser.add_argument("--personalized_eval", action="store_true", help="Evaluate individual client models in addition to global model")
    parser.add_argument("--poison", type=str, default=None, help="Poison config: <attack_type>-<ratio>, e.g., label_flip-0.3")
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.strategy in DISTILLATION_STRATEGIES:
        from .strategy import FD, FedDKD, FedProto, FedMD, FedSSD, SSFLIDS
        from .config import FDConfig
        from .pipeline import run_distillation_pipeline
        
        strategy_registry = {
            "FD": FD.FederatedDistillation,
            "FedDKD": FedDKD.FedDKD,
            "FedProto": FedProto.FedProto,
            "FedMD": FedMD.FedMD,
            "FedSSD": FedSSD.FedSSD,
            "SSFL-IDS": SSFLIDS.SSFLIDS,
        }
        
        config = FDConfig(
            algorithm=args.strategy,
            n_clients=args.n_clients,
            partition_type=args.partition_type,
            rounds=args.rounds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            data_calc=args.data_calc,
            model_type=args.model,
            m_max=args.m_max,
            train_rounds=args.train_rounds,
            dis_rounds=args.dis_rounds,
            dist_rounds=args.dist_rounds,
            theta=args.theta,
            dkd_steps=args.dkd_steps,
            dkd_lr=args.dkd_lr,
            personalized_eval=args.personalized_eval,
            poison=args.poison,
        )
        
        strategy = strategy_registry[args.strategy](config)
        run_distillation_pipeline(config, strategy)
    else:
        config = FLConfig(
            n_clients=args.n_clients,
            partition_type=args.partition_type,
            rounds=args.rounds,
            strategy=args.strategy,
            feddyn_alpha=args.feddyn_alpha,
            batch_size=args.batch_size,
            epochs=args.epochs,
            weights_cache_dir=args.weights_cache_dir,
            mu=args.mu,
            adaptive_mu=args.adaptive_mu,
            model=args.model,
            robust_epsilon=args.robust_epsilon,
            robust_tau=args.robust_tau,
            poison=args.poison,
        )
        run_pipeline(config)


if __name__ == "__main__":
    main()
