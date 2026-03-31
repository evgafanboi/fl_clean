import argparse
import os
import warnings

# Suppress all warnings before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from .pipeline import FLConfig, run_pipeline


WEIGHT_AGGREGATION_STRATEGIES = {"FedAvg", "FedProx", "FedDyn", "FedCoMed", "RobustFilter", "DeepFed", "FLTrust", "SecureAggregation", "FedSSD1", "FedSSDexp", "FedSSD2", "None"}
DISTILLATION_STRATEGIES = {"FD", "FedDKD", "FedProto", "FedMD", "FedSSD", "SSFL-IDS", "Exp1", "Exp2", "Cronus"}
ALL_STRATEGIES = sorted(WEIGHT_AGGREGATION_STRATEGIES | DISTILLATION_STRATEGIES)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Federated Learning Pipeline")
    parser.add_argument(
        "--strategy",
        type=str,
        default="FedAvg",
        help=f"Strategy: {', '.join(ALL_STRATEGIES)}",
    )
    
    # feddyn
    parser.add_argument("--feddyn_alpha", type=float, default=0.1, help="FedDyn alpha (paper best at 0.1)")
    # fedprox
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal term")
    # feddkd
    parser.add_argument("--dkd_steps", type=int, default=3, help="DKD gradient steps per round (for FedDKD)")
    parser.add_argument("--dkd_lr", type=float, default=0.001, help="Learning rate for DKD SGD updates (for FedDKD)")
    # ssfl-ids
    parser.add_argument("--dis_rounds", type=int, default=3, help="Discriminator rounds (SSFL-IDS)")
    parser.add_argument("--dist_rounds", type=int, default=2, help="Distillation rounds (SSFL-IDS)")
    # robust filter
    parser.add_argument("--robust_epsilon", type=float, default=0.2, help="RobustFilter epsilon (Byzantine ratio)")
    # cronus
    parser.add_argument("--remove_dis", action="store_true", help="Cronus: remove discriminator, use plain softmax predictions")
    # fedssd
    parser.add_argument("--m_max", type=float, default=1.0, help="M_max value for FedSSD")
    parser.add_argument("--support", action="store_true", help="Use support-weighted aggregation (FedSSD1/FedSSDexp)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Voting threshold for FedSSD2 (0-1, votes/clients must exceed this), value should decrease along label skewness")
    # fltrust
    parser.add_argument("--root_iterations", type=int, default=1, help="Server training iterations on root dataset (for FLTrust)")
    # fedmlb
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight for hybrid CE loss (for FedMLB)")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Weight for KL divergence loss (for FedMLB)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for KL divergence (for FedMLB)")


    # exp1
    parser.add_argument("--exp1_lambda", type=float, default=1.0, help="Lambda for KLD loss in Exp1 (L = CE + lambda * KLD)")
    parser.add_argument("--exp1_temperature", type=float, default=3.0, help="Temperature for KL divergence in Exp1")
    # exp2
    parser.add_argument("--exp2_ekd_lambda", type=float, default=1.0, help="Lambda weighting L_2nd in EKD (Exp2)")
    parser.add_argument("--kd", type=str, default="ekd", choices=["ekd", "abkd"], help="KD method for Exp2 (ekd or abkd)")
    parser.add_argument("--ab_alpha", type=float, default=1.0, help="Alpha for ABKD divergence")
    parser.add_argument("--ab_beta", type=float, default=0.0, help="Beta for ABKD divergence")
    parser.add_argument("--exp2_temperature", type=float, default=4.0, help="Temperature for ABKD softmax scaling")

    # distillation hyperparameters
    parser.add_argument("--gamma", type=float, default=1.0, help="Distillation temperature / weighting factor (for FD, FedMD, FedProto)")
    parser.add_argument("--data_calc", action="store_true", help="Enable extra data calculations (for distillation strategies)")

    # global params
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients to simulate")
    parser.add_argument("--partition_type", type=str, default="iid-10", help="Partition descriptor, e.g., iid-10")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--model", type=str, default="dense", help="Model architecture identifier (default: dense)")
    parser.add_argument("--train_rounds", type=int, default=3, help="Training rounds for discriminator-based methods")
    parser.add_argument("--batch_size", type=int, default=8192, help="Minibatch size for local training")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--weights_cache_dir", type=str, default="temp_weights", help="Directory to cache client weights")

    # experimental info computation
    parser.add_argument("--trust_score", action="store_true", help="Enable trust score logging (FLTrust cosine similarity) after local training")
    parser.add_argument("--peer_trust", action="store_true", help="Enable peer trust score logging (cosine similarity with adjacent neighbors +-1)")
    parser.add_argument("--personalized_eval", action="store_true", help="Evaluate individual client models in addition to global model (for FedMD, FD, FedProto)")
    parser.add_argument("--skip_eval", action="store_true", help="Skip per-client evaluation on non-final rounds (FedProto, Cronus, Exp2, FedMD, SSFL-IDS)")
    
    # poisoning
    parser.add_argument(
        "--poison",
        nargs=3,
        metavar=("attack", "value", "ratio"),
        default=None,
        help="Poison config: <attack> <value> <ratio>, e.g., gradient_scale 10 0.5",
    )

    # decentralized
    parser.add_argument("--decentralized", type=str, default=None, help="Decentralized topology simulation (e.g. braintorrent)")

    # memory cleanup
    parser.add_argument("--cleanup_interval", type=int, default=25, help="Run tf.keras.backend.clear_session + gc.collect every N clients")

    # checkpointing
    parser.add_argument("--checkpoint", type=int, nargs='?', const=1, default=0, help="Checkpoint every N clients (0=disabled, bare flag=every client)")

    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.strategy in DISTILLATION_STRATEGIES:
        from .strategy import FD, FedDKD, FedProto, FedMD, FedSSD, SSFLIDS, Exp1, Exp2, Cronus
        from .config import FDConfig
        from .pipeline import run_distillation_pipeline
        
        strategy_registry = {
            "FD": FD.FederatedDistillation,
            "FedDKD": FedDKD.FedDKD,
            "FedProto": FedProto.FedProto,
            "FedMD": FedMD.FedMD,
            "FedSSD": FedSSD.FedSSD,
            "SSFL-IDS": SSFLIDS.SSFLIDS,
            "Exp1": Exp1.Exp1,
            "Exp2": Exp2.Exp2,
            "Cronus": Cronus.Cronus,
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

            dkd_steps=args.dkd_steps,
            dkd_lr=args.dkd_lr,
            personalized_eval=args.personalized_eval,
            poison=" ".join(args.poison) if args.poison else None,
            exp1_lambda=args.exp1_lambda,
            exp1_temperature=args.exp1_temperature,
            exp2_ekd_lambda=args.exp2_ekd_lambda,
            exp2_kd=args.kd,
            ab_alpha=args.ab_alpha,
            ab_beta=args.ab_beta,
            exp2_temperature=args.exp2_temperature,
            robust_epsilon=args.robust_epsilon,
            remove_dis=args.remove_dis,
            checkpoint=args.checkpoint,
            cleanup_interval=args.cleanup_interval,
            skip_eval=args.skip_eval,
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

            model=args.model,
            robust_epsilon=args.robust_epsilon,
            poison=" ".join(args.poison) if args.poison else None,
            root_iterations=args.root_iterations,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            temperature=args.temperature,
            personalized_eval=args.personalized_eval,
            trust_score=args.trust_score,
            peer_trust=args.peer_trust,
            m_max=args.m_max,
            support=args.support,
            threshold=args.threshold,
            decentralized=args.decentralized,
            checkpoint=args.checkpoint,
            cleanup_interval=args.cleanup_interval,
        )
        run_pipeline(config)


if __name__ == "__main__":
    main()
