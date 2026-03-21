import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .colors import COLORS


def _truncate_after_last_round(filepath: str) -> None:
    """Truncate a log file after the last 'Round N completed' line, removing partial-round leftovers."""
    p = Path(filepath)
    if not p.exists():
        return
    data = p.read_bytes()
    last_pos = -1
    marker = b"Round "
    suffix = b"completed"
    search_start = 0
    while True:
        idx = data.find(marker, search_start)
        if idx == -1:
            break
        line_end = data.find(b"\n", idx)
        if line_end == -1:
            line_end = len(data)
        line = data[idx:line_end]
        if suffix in line:
            last_pos = line_end + 1
        search_start = line_end + 1
    if last_pos > 0 and last_pos < len(data):
        p.write_bytes(data[:last_pos])


def log_timestamp(logger: logging.Logger, message: str) -> None:
    """Log a message with a timestamp to both logger and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {message}")
    print(f"{COLORS.OKCYAN}[{timestamp}] {message}{COLORS.ENDC}")


def setup_logger(
    n_clients: int = None,
    partition_type: str = None,
    strategy_name: str = None,
    algorithm_name: str = None,
    partition_label: str = None,
    extra_tokens: Optional[Iterable[str]] = None,
    results_dir: str = "results",
    poison_suffix: str = "",
    resume: bool = False,
) -> Tuple[logging.Logger, str, logging.Logger]:
    """Configure loggers for the federated learning pipeline. Supports both FL and FD style arguments."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Handle both FL and FD argument styles
    name = algorithm_name or strategy_name
    partition = partition_label or partition_type
    
    # Build filename with optional extra tokens and poison suffix
    parts = [name, f"{n_clients}client", partition]
    if extra_tokens:
        parts.extend(str(token) for token in extra_tokens if token)
    if poison_suffix:
        parts.append(poison_suffix)
    log_filename = f"{results_dir}/{'_'.join(parts)}.log"

    if resume:
        _truncate_after_last_round(log_filename)
        detailed_log_filename = log_filename.replace('.log', '_detailed_class_metrics.log')
        _truncate_after_last_round(detailed_log_filename)
    else:
        detailed_log_filename = log_filename.replace('.log', '_detailed_class_metrics.log')

    mode = 'a' if resume else 'w'
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        filemode=mode
    )

    print(f"{COLORS.OKCYAN}Logging to {log_filename}{COLORS.ENDC}")

    detailed_logger = logging.getLogger('detailed_metrics')
    detailed_logger.handlers.clear()
    detailed_logger.setLevel(logging.INFO)
    detailed_handler = logging.FileHandler(detailed_log_filename, mode=mode)
    detailed_handler.setFormatter(logging.Formatter('%(message)s'))
    detailed_logger.addHandler(detailed_handler)
    detailed_logger.propagate = False

    return logging.getLogger(), log_filename, detailed_logger
