"""Main entry point for FCIL experiments"""

import argparse
import sys
import os
import glob
from pathlib import Path

from .config import FCILConfig
from .pipeline import run_fcil_pipeline, run_no_global_pipeline


def find_task_order_file(partition_type: str, n_clients: int) -> str:
    """
    Find task order file based on partition_type and n_clients.
    Returns exact path or prompts user if multiple found.
    """
    keys = [partition_type]
    if partition_type.endswith("_cil"):
        keys.append(partition_type[:-4])
    if '-' in partition_type:
        base, client_count = partition_type.rsplit('-', 1)
        if client_count == str(n_clients):
            keys.append(base)
    patterns = [f"results/incremental_order/{key}_{n_clients}_client_*.json" for key in dict.fromkeys(keys)]
    matches = [path for pattern in patterns for path in glob.glob(pattern)]
    
    if len(matches) == 0:
        print(f"Error: No task order files found matching patterns: {patterns}")
        print(f"Run partition_task.py first to generate task splits.")
        sys.exit(1)
    elif len(matches) == 1:
        return matches[0]
    else:
        print(f"Error: Multiple task order files found for {partition_type} with {n_clients} clients:")
        for i, path in enumerate(matches, 1):
            filename = Path(path).name
            print(f"  {i}. {filename}")
        print(f"\nPlease specify which task configuration to use with --task_order")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Federated Continual/Incremental Learning')
    
    # FL settings
    parser.add_argument('--strategy', type=str, default='FedAvg',
                        choices=['FedAvg', 'Centralized', 'FedCoMed', 'FedSSD'],
                        help='FL aggregation strategy')
    parser.add_argument('--n_clients', type=int, default=10,
                        help='Number of clients')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Rounds per task')
    
    # CIL settings
    parser.add_argument('--cil', type=str, default='finetune',
                        choices=['finetune', 'ewc', 'mas', 'lwf', 'icarl', 'bic', 'foster', 'glfc', 'cbkd', 'pass', 'feat', 'exp', 'ours', 'ours2', 'ours3', 'ours4'],
                        help='CIL method')
    parser.add_argument('--ewc_lambda', type=float, default=10.0,
                        help='EWC regularization strength (normalized, typical range: 0.1-10)')
    parser.add_argument('--mas_lambda', type=float, default=1.0,
                        help='MAS regularization strength')
    parser.add_argument('--lwf_alpha', type=float, default=0.5,
                        help='LwF distillation weight')
    parser.add_argument('--lwf_temperature', type=float, default=2.0,
                        help='LwF distillation temperature')
    parser.add_argument('--memory', type=float, default=1.0,
                        help='iCaRL-style exemplar memory percentage added per client per task')
    parser.add_argument('--bce', action='store_true',
                        help='Use iCaRL BCE loss (paper default)')
    parser.add_argument('--beta1', type=float, default=0.97,
                        help='FOSTER beta1 for Stage 1 effective number')
    parser.add_argument('--beta2', type=float, default=0.97,
                        help='FOSTER beta2 for Stage 2 effective number')
    parser.add_argument('--lambda_okd', type=float, default=1.0,
                        help='FOSTER KD loss weight')
    parser.add_argument('--compression_epochs', type=int, default=50,
                        help='FOSTER Stage 2 compression epochs')
    parser.add_argument('--bic_val_split', type=float, default=0.4,
                        help='BiC fraction of each class reserved for bias correction validation')
    parser.add_argument('--cbkd_lambda', type=float, default=10.0,
                        help='CBKD prototype loss weight')
    parser.add_argument('--cbkd_alpha', type=float, default=10.0,
                        help='CBKD inter-class CFD weight')
    parser.add_argument('--cbkd_beta', type=float, default=10.0,
                        help='CBKD intra-class CFD weight')
    parser.add_argument('--proto_size', type=int, default=50,
                        help='CBKD/PASS augmented prototypes per class')
    parser.add_argument('--pass_gamma', type=float, default=10.0,
                        help='PASS: feature distillation weight')
    parser.add_argument('--feat_lambda', type=float, default=0.1,
                        help='FEAT/EXP: geometric structural alignment weight (FEAT default 0.1, EXP default 1.0)')
    parser.add_argument('--feat_rho', type=float, default=0.9,
                        help='FEAT: replay-energy EMA coefficient')
    parser.add_argument('--feat_temp', type=float, default=0.5,
                        help='FEAT: softmax temperature for ETF feature similarity')
    parser.add_argument('--ours_ekd_epochs', type=int, default=2,
                        help='Ours: per-task EKD epochs')
    parser.add_argument('--ours_ekd_lambda', type=float, default=1.0,
                        help='Ours: per-task EKD loss weight')
    parser.add_argument('--ours_proto_rel_lambda', type=float, default=1.0,
                        help='Ours: prototype-prototype relation loss weight')
    parser.add_argument('--ours_proto_lambda', type=float, default=10.0,
                        help='Ours: prototype augmentation weight for old-class prototypes')
    parser.add_argument('--ours_kd_gamma', type=float, default=1.0,
                        help='Ours: old-feature KD weight')
    parser.add_argument('--ours_encoder_lr_factor', type=float, default=0.5,
                        help='Ours: gradient scale for non-classifier parameters')
    parser.add_argument('--ours_drift_temp', type=float, default=0.5,
                        help='Ours: cosine-softmax temperature for old-class drift compensation')
    parser.add_argument('--replay_cap', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Ours: cap replay memory per client to the current task private length, spread evenly across old exemplar classes')
    parser.add_argument('--replay_min_per_class', type=int, default=128,
                        help='Ours: minimum replay exemplars per old class when replay_cap is enabled')
    parser.add_argument('--replay_balance', type=float, default=1.0,
                        help='Ours4: target old:new replay ratio knob (>1 more stability, <1 more plasticity)')
    parser.add_argument('--robust_threshold', type=float, default=0.9,
                        help='Ours: Blom robust-filter discard threshold')
    parser.add_argument('--robust_workers', type=int, default=8,
                        help='Ours: Blom robust-filter workers')
    parser.add_argument('--no_filter', action='store_true',
                        help='Ours: disable robust filtering and aggregate client logits by mean')
    parser.add_argument('--m_max', type=float, default=1.0,
                        help='FedSSD: maximum SSD distillation weight')
    parser.add_argument('--encoder_epochs', type=int, default=50,
                        help='GLFC: perturbation optimisation epochs for proto samples')
    parser.add_argument('--model_selection', action='store_true',
                        help='GLFC: enable proxy-server model selection via DLG')
    parser.add_argument('--grad_enc', type=str, default='small',
                        choices=['small', 'medium', 'main'],
                        help='GLFC: gradient encoder size (small=64, medium=128, main=mirror DenseModel)')
    
    # Data settings
    parser.add_argument('--partition_type', type=str, required=True,
                        help='Partition type (e.g., label_skew)')
    parser.add_argument('--task_order', type=str, default='',
                        help='Optional: Specific task order file path (auto-detected if not specified)')
    
    # Model settings
    parser.add_argument('--model', type=str, default='dense',
                        help='Model architecture')
    parser.add_argument('--mixed_models', action='store_true',
                        help='Heterogeneous client architectures (gru/mixed_dcblstm/dense/cnn by client id)')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per round')
    parser.add_argument('--use_tf', action='store_true',
                        help='Use TensorFlow backend (default: PyTorch)')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='Save checkpoint every N clients (0 = disabled)')
    parser.add_argument('--fresh_run', action='store_true',
                        help='Clear checkpoint and restart from scratch')
    parser.add_argument('--last_eval', action='store_true',
                        help='Evaluate only at the last round of each task')
    parser.add_argument('--sweep_eval', action='store_true',
                        help='Skip training: load saved weight records of the matching run and re-evaluate only')
    
    args = parser.parse_args()
    if args.feat_lambda is None:
        args.feat_lambda = 1.0 if args.cil == 'exp' else 0.1

    # GLFC is monolithic: warn and override incompatible strategies
    if args.cil == 'glfc' and args.strategy != 'FedAvg':
        from .colors import COLORS as _C
        print(f"{_C.FAIL}Incompatible command for GLFC: --strategy {args.strategy}. "
              f"GLFC is monolithic and uses its own FedAvg. Ignoring.{_C.ENDC}")
        args.strategy = 'FedAvg'

    # Auto-detect task order file if not specified
    if not args.task_order:
        task_order_file = find_task_order_file(args.partition_type, args.n_clients)
        print(f"Using task order: {task_order_file}")
    else:
        task_order_file = args.task_order
        if not os.path.exists(task_order_file):
            print(f"Error: Task order file not found: {task_order_file}")
            sys.exit(1)
    
    partition_root = "data/partitions"
    
    # Extract task size from task_order filename for logging
    task_size_token = Path(task_order_file).stem.split('_')[-2]  # e.g., "5.0" from "..._5.0_task"
    
    parts = [args.strategy, args.cil, f"{args.n_clients}client", args.partition_type, task_size_token, args.model]
    if args.mixed_models:
        parts.append("mixed")
    if args.cil == 'ewc':
        parts.append(f"lambda{args.ewc_lambda}")
    if args.cil == 'mas':
        parts.append(f"lambda{args.mas_lambda}")
    if args.cil == 'lwf':
        parts.append(f"alpha{args.lwf_alpha}")
        parts.append(f"temp{args.lwf_temperature}")
    if args.cil == 'icarl':
        parts.append(f"mem{args.memory}")
        parts.append(f"bce{int(args.bce)}")
    if args.cil == 'bic':
        parts.append(f"mem{args.memory}")
        parts.append(f"alpha{args.lwf_alpha}")
        parts.append(f"temp{args.lwf_temperature}")
        parts.append(f"val{args.bic_val_split}")
    if args.cil == 'foster':
        parts.append(f"mem{args.memory}")
        parts.append(f"b1_{args.beta1}")
        parts.append(f"b2_{args.beta2}")
        parts.append(f"okd{args.lambda_okd}")
        parts.append(f"ce{args.compression_epochs}")
    if args.cil == 'glfc':
        parts.append(f"mem{args.memory}")
        parts.append(f"enc{args.encoder_epochs}")
        if args.model_selection:
            parts.append("msel")
            parts.append(f"ge_{args.grad_enc}")
    if args.cil == 'cbkd':
        parts.append(f"lam{args.cbkd_lambda}")
        parts.append(f"a{args.cbkd_alpha}")
        parts.append(f"b{args.cbkd_beta}")
        parts.append(f"ps{args.proto_size}")
    if args.cil == 'pass':
        parts.append(f"lam{args.cbkd_lambda}")
        parts.append(f"g{args.pass_gamma}")
        parts.append(f"ps{args.proto_size}")
    if args.cil in ('feat', 'exp', 'ours'):
        parts.append(f"mem{args.memory}")
        parts.append(f"lam{args.feat_lambda}")
        parts.append(f"rho{args.feat_rho}")
        parts.append(f"temp{args.feat_temp}")
    if args.cil == 'ours':
        parts.append(f"plam{args.ours_proto_lambda}")
        parts.append(f"ekd{args.ours_ekd_epochs}")
        parts.append(f"ekdl{args.ours_ekd_lambda}")
        parts.append(f"kg{args.ours_kd_gamma}")
        parts.append(f"rel{args.ours_proto_rel_lambda}")
        parts.append(f"enc{args.ours_encoder_lr_factor}")
        parts.append("meanlogits" if args.no_filter else f"blom{args.robust_threshold}")
    if args.cil in ('ours2', 'ours3', 'ours4'):
        parts.append(f"mem{args.memory}")
        parts.append(f"kg{args.ours_kd_gamma}")
        parts.append("cap" if args.replay_cap else "nocap")
        if args.replay_cap:
            parts.append(f"rmin{args.replay_min_per_class}")
        if args.cil == 'ours3':
            parts.append("gated")
        if args.cil == 'ours4':
            parts.append(f"rbal{args.replay_balance}")
        parts.append(f"eva{0.95}_blom{args.robust_threshold}")
    if args.strategy == 'FedSSD':
        parts.append(f"ssd{args.m_max}")
    log_file = f"results_cil/{'_'.join(parts)}.log"
    
    config = FCILConfig(
        strategy=args.strategy,
        n_clients=args.n_clients,
        rounds_per_task=args.rounds,
        cil_method=args.cil,
        ewc_lambda=args.ewc_lambda,
        mas_lambda=args.mas_lambda,
        lwf_alpha=args.lwf_alpha,
        lwf_temperature=args.lwf_temperature,
        icarl_memory=args.memory,
        icarl_bce=args.bce,
        foster_beta1=args.beta1,
        foster_beta2=args.beta2,
        foster_lambda_okd=args.lambda_okd,
        foster_compression_epochs=args.compression_epochs,
        glfc_encoder_epochs=args.encoder_epochs,
        glfc_model_selection=args.model_selection,
        glfc_grad_enc=args.grad_enc,
        cbkd_lambda=args.cbkd_lambda,
        cbkd_alpha=args.cbkd_alpha,
        cbkd_beta=args.cbkd_beta,
        cbkd_proto_size=args.proto_size,
        pass_gamma=args.pass_gamma,
        feat_lambda=args.feat_lambda,
        feat_rho=args.feat_rho,
        feat_temp=args.feat_temp,
        ours_ekd_epochs=args.ours_ekd_epochs,
        ours_ekd_lambda=args.ours_ekd_lambda,
        ours_kd_gamma=args.ours_kd_gamma,
        ours_proto_lambda=args.ours_proto_lambda,
        ours_proto_rel_lambda=args.ours_proto_rel_lambda,
        ours_encoder_lr_factor=args.ours_encoder_lr_factor,
        ours_drift_temp=args.ours_drift_temp,
        replay_cap=args.replay_cap,
        replay_min_per_class=args.replay_min_per_class,
        replay_balance=args.replay_balance,
        robust_threshold=args.robust_threshold,
        robust_workers=args.robust_workers,
        no_filter=args.no_filter,
        partition_type=args.partition_type,
        partition_root=partition_root,
        task_order_file=task_order_file,
        model=args.model,
        mixed_models=args.mixed_models,
        batch_size=args.batch_size,
        epochs_per_round=args.epochs,
        log_file=log_file,
        bic_val_split=args.bic_val_split,
        m_max=args.m_max,
        use_tf=args.use_tf,
        checkpoint=args.checkpoint,
        fresh_run=args.fresh_run,
        last_eval=args.last_eval,
        sweep_eval=args.sweep_eval,
    )
    
    if config.sweep_eval:
        from .pipeline import run_sweep_eval_pipeline
        run_sweep_eval_pipeline(config)
    else:
        run_fcil_pipeline(config)


if __name__ == '__main__':
    main()
