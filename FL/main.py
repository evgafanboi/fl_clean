import argparse
import os
import warnings

# Suppress all warnings before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')


WEIGHT_AGGREGATION_STRATEGIES = {"FedAvg", "FedProx", "FedDyn", "FedCoMed", "RobustFilter", "DeepFed", "FLTrust", "SecureAggregation", "FedSSD1", "FedSSDexp", "FedSSD2", "None", "FLAME"}
DISTILLATION_STRATEGIES = {"FD", "FedDistill", "FedDKD", "FedProto", "FedMD", "FedSSD", "SSFL-IDS", "FedKD-IDS", "Exp1", "Ours", "Cronus"}
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
    # ssfl-ids & fedkd-ids
    parser.add_argument("--dis_rounds", type=int, default=3, help="Discriminator rounds (SSFL-IDS) & Verifier rounds (FedKD-IDS)")
    parser.add_argument("--dist_rounds", type=int, default=2, help="Distillation rounds (SSFL-IDS & FedKD-IDS)")
    parser.add_argument("--train_rounds", type=int, default=3, help="Training rounds for SSFL-IDS stage 1 and FedKD-IDS stage 1")
    # fedkd-ids
    parser.add_argument("--hamming_tau", type=float, default=1.0, help="FedKD-IDS: threshold = tau * mean(HM), range [0.5, 1.0]")
    # feddistill
    parser.add_argument("--exp_rho", type=float, default=0.05, help="ExpGuard step size (FedDistill)")
    # robust filter
    parser.add_argument("--robust_epsilon", type=float, default=0.2, help="RobustFilter epsilon (threshold sensitivity, not budget)")
    parser.add_argument("--robust_rm_budget", type=int, default=None, help="RobustFilter max removals (default: n_clients//2 - 1)")
    parser.add_argument("--robust_threshold", type=float, default=0.7, help="RobustFilterV3 tail score threshold")
    parser.add_argument("--robust_workers", type=int, default=8, help="Ours robust filter row-block worker threads")
    parser.add_argument("--robust_filter_reference", type=str, default="Blom", help="Ours: reference curve for per-sample spectral filter. 'Blom' (half-normal order stats) or 't' (Student-t, use --robust_v for df; more lenient)")
    parser.add_argument("--robust_v", type=float, default=4.0, help="Degrees of freedom for t-distribution reference (lower = more lenient, default 4.0)")
    parser.add_argument("--robust_filter_cronus", action="store_true", help="Ours: use pooled Cronus robust filter")
    # cronus
    parser.add_argument("--remove_dis", action="store_true", help="Cronus: use plain softmax predictions")
    # fedssd
    parser.add_argument("--m_max", type=float, default=1.0, help="M_max value for FedSSD")
    parser.add_argument("--support", action="store_true", help="Use support-weighted aggregation (FedSSD1/FedSSDexp)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Voting threshold for FedSSD2 (0-1, votes/clients must exceed this), value should decrease along label skewness")
    # fltrust
    parser.add_argument("--root_iterations", type=int, default=1, help="Server training iterations on root dataset (for FLTrust)")
    # flame
    parser.add_argument("--flame_lambda", type=float, default=0.001, help="FLAME DP noise coefficient (sigma = lambda * S_t)")
    parser.add_argument("--passive_cluster", action="store_true", help="FLAME: use unconstrained HDBSCAN and admit the largest cluster (ignores min_cluster_size majority requirement)")
    # fedmlb
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight for hybrid CE loss (for FedMLB)")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Weight for KL divergence loss (for FedMLB)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for KL divergence (for FedMLB)")
    
    # exp1
    parser.add_argument("--exp1_lambda", type=float, default=1.0, help="Lambda for KLD loss in Exp1 (L = CE + lambda * KLD)")
    parser.add_argument("--exp1_temperature", type=float, default=3.0, help="Temperature for KL divergence in Exp1")
    # ours
    parser.add_argument("--ours_ekd_lambda", type=float, default=1.0, help="Lambda weighting L_2nd in EKD (Ours)")
    parser.add_argument("--kd", type=str, default="ekd", choices=["ekd", "abkd"], help="KD method for Ours (ekd or abkd)")
    parser.add_argument("--ab_alpha", type=float, default=0.7, help="Alpha for ABKD divergence")
    parser.add_argument("--ab_beta", type=float, default=0.0, help="Beta for ABKD divergence")
    parser.add_argument("--ours_temperature", type=float, default=4.0, help="Temperature for ABKD softmax scaling")
    parser.add_argument("--kd_epochs", type=int, default=2, help="KD epochs for stage 1 in Ours strategy")
    eva_group = parser.add_mutually_exclusive_group()
    eva_group.add_argument("--eva", action="store_true", help="Ours: Energy-based Vote Abstention on survivor logits")
    eva_group.add_argument("--eva2", action="store_true", help="Ours: Alternate Energy-based Vote Abstention (unique counting)")
    eva_group.add_argument("--evw", action="store_true", help="Ours: Energy-based Vote Weighting on survivor logits")

    # distillation hyperparameters
    parser.add_argument("--gamma", type=float, default=1.0, help="Distillation temperature / weighting factor (for FD, FedMD, FedProto)")
    parser.add_argument("--data_calc", action="store_true", help="Enable extra data calculations (for distillation strategies)")

    # global params
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients to simulate")
    parser.add_argument("--partition_type", type=str, default="iid-10", help="Partition descriptor, e.g., iid-10")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--model", type=str, default="dense", help="Model architecture identifier (default: dense)")
    parser.add_argument("--batch_size", type=int, default=8192, help="Minibatch size for local training")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs per round")

    # experimental info computation
    parser.add_argument("--trust_score", action="store_true", help="Enable trust score logging (FLTrust cosine similarity) after local training")
    parser.add_argument("--peer_trust", action="store_true", help="Enable peer trust score logging (cosine similarity with adjacent neighbors +-1)")
    parser.add_argument("--personalized_eval", action="store_true", help="Evaluate individual client models in addition to global model (for FedMD, FD, FedProto)")
    parser.add_argument("--skip_eval", action="store_true", help="Only evaluate the final round's weight records (skip intermediate rounds)")
    
    # poisoning
    parser.add_argument(
        "--poison",
        nargs="+",
        metavar="TOKEN",
        default=None,
        help="Poison config tokens: <attack> [value] <ratio>, e.g., gradient_scale 10 0.5 or label_flip 0.2 or lma 0.2 or ilma 0.2",
    )
    # label_flip, ratio = fraction of clients to poison, e.g. "label_flip 0.2"
    # gradient_scale, value = multiplicative factor for weight updates (e.g. 10x), ratio = fraction of clients to poison, e.g. "gradient_scale 10 0.2"
    # targeted_flip, value = target label index (0 to num_classes-1), ratio = fraction of clients to poison, e.g. "targeted_flip 0 0.2"
    # poisonedfl, value (default 8) = c0 value, ratio = fraction of clients to poison, e.g. "poisonedfl 8 0.2"
    # lma/ilma, ratio = fraction of clients to poison, e.g. "lma 0.2" or "ilma 0.2"

    # decentralized
    parser.add_argument("--decentralized", type=str, default=None, help="Decentralized topology simulation (e.g. braintorrent)")

    # memory cleanup
    parser.add_argument("--cleanup_interval", type=int, default=25, help="Run tf.keras.backend.clear_session + gc.collect every N clients")

    # checkpointing
    parser.add_argument("--f1_curve", action="store_true", help="Plot F1 vs confidence-threshold curve for FedDistill/Ours at final round")
    parser.add_argument("--checkpoint", type=int, nargs='?', const=1, default=0, help="Checkpoint every N clients (0=disabled, bare flag=every client)")
    parser.add_argument("--fresh_run", action="store_true", help="Delete existing weight records for this run and start fresh")
    parser.add_argument("--cache_test_set", action="store_true", help="Pin X_test to GPU VRAM before evaluation (avoids repeated host↔device transfers; ~1 GB VRAM)")
    parser.add_argument("--use_tf", action="store_true", help="Use TensorFlow backend (default: PyTorch)")
    parser.add_argument("--mixed_models", action="store_true", help="Assign different model architectures to client quartiles (GRU/DCBLSTM/MLP/CNN)")
    parser.add_argument("--keep_last_rounds", type=int, default=2, help="Keep only last N round weight records (0=keep all)")
    parser.add_argument("--disk_cache", action="store_true", help="Write logit cache to disk instead of RAM")
    parser.add_argument("--save_weights", action="store_true", default=True, help="Save per-round model weights to disk (default: True)")
    parser.add_argument("--no_save_weights", action="store_false", dest="save_weights", help="Store round weights in memory only — disables crash recovery")
    parser.add_argument("--eval_only", action="store_true", help="Skip training, evaluate existing weight records and exit")

    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    from .backend import set_backend
    set_backend(args.use_tf)

    MIXED_COMPATIBLE = {"FedProto", "Cronus", "SSFL-IDS", "FedKD-IDS", "Ours"}
    if getattr(args, 'mixed_models', False) and args.strategy not in MIXED_COMPATIBLE:
        print(f"\033[95mIncompatible strategy\033[0m")
        return

    robust_rm_budget = args.robust_rm_budget if args.robust_rm_budget is not None else (args.n_clients // 2 - 1)

    if args.strategy in DISTILLATION_STRATEGIES:
        from .strategy import FD, FedDistill, FedDKD, FedProto, FedMD, FedSSD, SSFLIDS, FedKDIDS, Exp1, ours, Cronus
        from .config import FDConfig
        from .pipeline import run_distillation_pipeline
        
        strategy_registry = {
            "FD": FD.FederatedDistillation,
            "FedDKD": FedDKD.FedDKD,
            "FedProto": FedProto.FedProto,
            "FedMD": FedMD.FedMD,
            "FedSSD": FedSSD.FedSSD,
            "SSFL-IDS": SSFLIDS.SSFLIDS,
            "FedDistill": FedDistill.FedDistillExpGuard,
            "FedKD-IDS": FedKDIDS.FedKDIDS,
            "Exp1": Exp1.Exp1,
            "Ours": ours.Ours,
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
            ours_ekd_lambda=args.ours_ekd_lambda,
            ours_kd=args.kd,
            ab_alpha=args.ab_alpha,
            ab_beta=args.ab_beta,
            ours_temperature=args.ours_temperature,
            kd_epochs=args.kd_epochs,
            robust_epsilon=args.robust_epsilon,
            robust_rm_budget=robust_rm_budget,
            robust_threshold=args.robust_threshold,
            robust_workers=args.robust_workers,
            robust_filter_reference=args.robust_filter_reference,
            robust_v=args.robust_v,
            robust_filter_cronus=args.robust_filter_cronus,
            eva=args.eva,
            eva2=args.eva2,
            evw=args.evw,
            remove_dis=args.remove_dis,
            checkpoint=args.checkpoint,
            cleanup_interval=args.cleanup_interval,
            skip_eval=args.skip_eval,
            fresh_run=args.fresh_run,
            cache_test_set=args.cache_test_set,
            mixed_models=args.mixed_models,
            hamming_tau=args.hamming_tau,
            exp_rho=args.exp_rho,
            f1_curve=args.f1_curve,
            keep_last_rounds=args.keep_last_rounds,
            no_disk_cache=not args.disk_cache,
            save_weights=args.save_weights,
            eval_only=args.eval_only,
        )
        
        strategy = strategy_registry[args.strategy](config)
        run_distillation_pipeline(config, strategy)
    else:
        from .pipeline import FLConfig, run_pipeline
        config = FLConfig(
            n_clients=args.n_clients,
            partition_type=args.partition_type,
            rounds=args.rounds,
            strategy=args.strategy,
            feddyn_alpha=args.feddyn_alpha,
            batch_size=args.batch_size,
            epochs=args.epochs,
            mu=args.mu,

            model=args.model,
            robust_epsilon=args.robust_epsilon,
            robust_rm_budget=robust_rm_budget,
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
            skip_eval=args.skip_eval,
            fresh_run=args.fresh_run,
            cache_test_set=args.cache_test_set,
            flame_lambda=args.flame_lambda,
            flame_passive_cluster=args.passive_cluster,
            keep_last_rounds=args.keep_last_rounds,
            no_disk_cache=not args.disk_cache,
            save_weights=args.save_weights,
            eval_only=args.eval_only,
        )
        run_pipeline(config)


if __name__ == "__main__":
    main()
