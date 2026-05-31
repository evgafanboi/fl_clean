import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

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


def _checkpoint_stem_for_log(filepath: str) -> str:
    stem = Path(filepath).stem
    detailed_suffix = "_detailed_class_metrics"
    if stem.endswith(detailed_suffix):
        stem = stem[:-len(detailed_suffix)]
    return stem


def _load_checkpoint_info_for_log(filepath: str) -> Dict[str, str]:
    stem = _checkpoint_stem_for_log(filepath)
    info_path = Path("checkpoint") / stem / "info.txt"
    if not info_path.exists():
        return {}
    info: Dict[str, str] = {}
    for line in info_path.read_text().splitlines():
        key, _, val = line.partition(": ")
        if key:
            info[key] = val
    return info


def _truncate_after_checkpoint_client(filepath: str, round_number: int, client_id: int) -> bool:
    p = Path(filepath)
    if not p.exists():
        return False
    data = p.read_bytes()
    prefix = f"Round {round_number} | Client {client_id}".encode()
    safe_end = -1
    offset = 0
    for line in data.splitlines(keepends=True):
        if prefix in line:
            safe_end = offset + len(line)
        offset += len(line)
    if safe_end > 0 and safe_end < len(data):
        p.write_bytes(data[:safe_end])
        return True
    return False


def _truncate_at_evaluation(filepath: str) -> bool:
    p = Path(filepath)
    if not p.exists():
        return False
    data = p.read_bytes()
    marker = b"=== EVALUATION ==="
    idx = data.rfind(marker)
    if idx == -1:
        return False
    line_start = data.rfind(b"\n", 0, idx)
    cut = line_start + 1 if line_start >= 0 else 0
    p.write_bytes(data[:cut])
    return True


def _truncate_for_resume(filepath: str) -> None:
    info = _load_checkpoint_info_for_log(filepath)
    if not info:
        if not _truncate_at_evaluation(filepath):
            _truncate_after_last_round(filepath)
        return
    if info.get("round_complete", "true") == "true":
        if not _truncate_at_evaluation(filepath):
            _truncate_after_last_round(filepath)
        return
    try:
        round_number = int(info.get("round", "0"))
        client_id = int(info.get("stage_client", "-1"))
    except ValueError:
        return
    if client_id >= 0 and _truncate_after_checkpoint_client(filepath, round_number, client_id):
        return


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
    create_detailed_log: bool = True,
) -> Tuple[logging.Logger, str, logging.Logger]:
    """Configure loggers for the federated learning pipeline. Supports both FL and FD style arguments."""
    if not os.path.islink(results_dir):
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

    mode = 'a' if resume else 'w'
    if resume:
        _truncate_for_resume(log_filename)
        if create_detailed_log:
            detailed_log_filename = log_filename.replace('.log', '_detailed_class_metrics.log')
            _truncate_for_resume(detailed_log_filename)

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
    if create_detailed_log:
        detailed_log_filename = log_filename.replace('.log', '_detailed_class_metrics.log')
        detailed_handler = logging.FileHandler(detailed_log_filename, mode=mode)
        detailed_handler.setFormatter(logging.Formatter('%(message)s'))
        detailed_logger.addHandler(detailed_handler)
    else:
        detailed_logger.addHandler(logging.NullHandler())
    detailed_logger.propagate = False

    return logging.getLogger(), log_filename, detailed_logger
