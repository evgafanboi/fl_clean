import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

from .colors import COLORS


def log_timestamp(logger: logging.Logger, message: str) -> None:
    """Log a message with a timestamp to both logger and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {message}")
    print(f"{COLORS.OKCYAN}[{timestamp}] {message}{COLORS.ENDC}")


def setup_logger(
    n_clients: int,
    partition_type: str,
    strategy_name: str,
    results_dir: str = "results",
    poison_suffix: str = ""
) -> Tuple[logging.Logger, str, logging.Logger]:
    """Configure loggers for the federated learning pipeline."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Build filename with poison suffix if provided
    base_name = f"{strategy_name}_{n_clients}client_{partition_type}"
    if poison_suffix:
        base_name = f"{base_name}_{poison_suffix}"
    log_filename = f"{results_dir}/{base_name}.log"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        filemode='w'
    )

    print(f"{COLORS.OKCYAN}Logging to {log_filename}{COLORS.ENDC}")

    detailed_log_filename = log_filename.replace('.log', '_detailed_class_metrics.log')
    detailed_logger = logging.getLogger('detailed_metrics')
    detailed_logger.handlers.clear()
    detailed_logger.setLevel(logging.INFO)
    detailed_handler = logging.FileHandler(detailed_log_filename, mode='w')
    detailed_handler.setFormatter(logging.Formatter('%(message)s'))
    detailed_logger.addHandler(detailed_handler)
    detailed_logger.propagate = False

    return logging.getLogger(), log_filename, detailed_logger
