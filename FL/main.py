import argparse

from .pipeline import FLConfig, run_pipeline


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Federated Learning Pipeline")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients to simulate")
    parser.add_argument("--partition_type", type=str, default="iid-10", help="Partition descriptor, e.g., iid-10")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument(
        "--strategy",
        type=str,
        default="FedAvg",
        help="Aggregation strategy: FedAvg, FedProx, FedDyn, FedCoMed, RobustFilter, DeepFed",
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
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

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
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
