"""Main entry point for FCIL experiments"""

import argparse
import sys
import os
import glob
from pathlib import Path

from .config import FCILConfig
from .pipeline import run_fcil_pipeline


def find_task_order_file(partition_type: str, n_clients: int) -> str:
    """
    Find task order file based on partition_type and n_clients.
    Returns exact path or prompts user if multiple found.
    """
    pattern = f"results/incremental_order/{partition_type}_{n_clients}_client_*.json"
    matches = glob.glob(pattern)
    
    if len(matches) == 0:
        print(f"Error: No task order files found matching pattern: {pattern}")
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
                        choices=['FedAvg', 'Centralized', 'FedCoMed'],
                        help='FL aggregation strategy')
    parser.add_argument('--n_clients', type=int, default=10,
                        help='Number of clients')
    parser.add_argument('--rounds', type=int, default=10,
                        help='Rounds per task')
    
    # CIL settings
    parser.add_argument('--cil', type=str, default='finetune',
                        choices=['finetune', 'ewc'],
                        help='CIL method')
    parser.add_argument('--ewc_lambda', type=float, default=1.0,
                        help='EWC regularization strength (normalized, typical range: 0.1-10)')
    
    # Data settings
    parser.add_argument('--partition_type', type=str, required=True,
                        help='Partition type (e.g., label_skew)')
    parser.add_argument('--task_order', type=str, default='',
                        help='Optional: Specific task order file path (auto-detected if not specified)')
    
    # Model settings
    parser.add_argument('--model', type=str, default='dense',
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per round')
    
    args = parser.parse_args()
    
    # Auto-detect task order file if not specified
    if not args.task_order:
        task_order_file = find_task_order_file(args.partition_type, args.n_clients)
        print(f"Using task order: {task_order_file}")
    else:
        task_order_file = args.task_order
        if not os.path.exists(task_order_file):
            print(f"Error: Task order file not found: {task_order_file}")
            sys.exit(1)
    
    # Partition root follows FL convention
    partition_root = "data/partitions"
    
    # Extract task size from task_order filename for logging
    task_size_token = Path(task_order_file).stem.split('_')[-2]  # e.g., "5.0" from "..._5.0_task"
    
    # Build log filename similar to FL
    parts = [args.strategy, args.cil, f"{args.n_clients}client", args.partition_type, task_size_token]
    if args.cil == 'ewc':
        parts.append(f"lambda{args.ewc_lambda}")
    log_file = f"results_cil/{'_'.join(parts)}.log"
    
    config = FCILConfig(
        strategy=args.strategy,
        n_clients=args.n_clients,
        rounds_per_task=args.rounds,
        cil_method=args.cil,
        ewc_lambda=args.ewc_lambda,
        partition_type=args.partition_type,
        partition_root=partition_root,
        task_order_file=task_order_file,
        model=args.model,
        batch_size=args.batch_size,
        epochs_per_round=args.epochs,
        log_file=log_file,
    )
    
    run_fcil_pipeline(config)


if __name__ == '__main__':
    main()
