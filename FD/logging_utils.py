import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .colors import COLORS


def log_timestamp(logger: logging.Logger, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {message}")
    print(f"{COLORS.OKCYAN}[{timestamp}] {message}{COLORS.ENDC}")


def _build_log_filename(
    algorithm_name: str,
    n_clients: int,
    partition_label: str,
    extra_tokens: Optional[Iterable[str]],
    results_dir: str,
    poison_suffix: str = ""
) -> str:
    parts = [algorithm_name, f"{n_clients}client", partition_label]
    if extra_tokens:
        parts.extend(str(token) for token in extra_tokens if token)
    if poison_suffix:
        parts.append(poison_suffix)
    filename = "_".join(parts)
    return f"{results_dir}/{filename}.log"


def setup_logger(
    algorithm_name: str,
    n_clients: int,
    partition_label: str,
    extra_tokens: Optional[Iterable[str]] = None,
    results_dir: str = "results",
    poison_suffix: str = ""
) -> Tuple[logging.Logger, str, logging.Logger]:
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    log_filename = _build_log_filename(
        algorithm_name=algorithm_name,
        n_clients=n_clients,
        partition_label=partition_label,
        extra_tokens=extra_tokens,
        results_dir=results_dir,
        poison_suffix=poison_suffix
    )

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        filemode='w'
    )

    print(f"{COLORS.OKCYAN}Logging to {log_filename}{COLORS.ENDC}")

    detailed_log_filename = log_filename.replace('.log', '_detailed_class_metrics.log')
    detailed_logger = logging.getLogger('fd_detailed_metrics')
    detailed_logger.handlers.clear()
    detailed_logger.setLevel(logging.INFO)
    detailed_handler = logging.FileHandler(detailed_log_filename, mode='w')
    detailed_handler.setFormatter(logging.Formatter('%(message)s'))
    detailed_logger.addHandler(detailed_handler)
    detailed_logger.propagate = False

    return logging.getLogger(), log_filename, detailed_logger
