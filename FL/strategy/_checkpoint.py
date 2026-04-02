"""Shared mid-round checkpoint helpers for distillation strategies.

Every distillation strategy that iterates over clients inside ``run_round``
can use these three helpers to support ``--checkpoint N`` (save every *N*
clients so a crash mid-round doesn't lose all work).

Usage pattern (see Cronus.py for the reference implementation)::

    _ckpt = getattr(cfg, "checkpoint", 0)

    # At the top of run_round — attempt resume
    if _ckpt:
        mid = load_mid_round(context, "mystrategy", round_number)
        if mid is not None:
            first_client = mid["last_client_idx"] + 1
            ...restore accumulated state from mid...

    # Inside the per-client loop
    if _ckpt and (idx == last_idx or (idx + 1) % _ckpt == 0):
        save_mid_round(context, "mystrategy", {
            "round": round_number,
            "last_client_idx": idx,
            ...strategy-specific accumulated state...
        })

    # After the loop completes successfully
    if _ckpt:
        clear_mid_round(context, "mystrategy")
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

from ..colors import COLORS

# Avoid circular import — callers pass PipelineContext; we only need .log_filename
# so we accept Any.


def _ckpt_dir(context: Any) -> str:
    """Return the checkpoint directory for the current pipeline run."""
    stem = os.path.splitext(os.path.basename(context.log_filename))[0]
    return os.path.join("checkpoint", stem)


def save_mid_round(context: Any, tag: str, payload: Dict[str, Any]) -> None:
    """Atomically write *payload* to ``<ckpt_dir>/<tag>_mid.bin``."""
    d = _ckpt_dir(context)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f"{tag}_mid.bin.tmp")
    dst = os.path.join(d, f"{tag}_mid.bin")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst)
    last = payload.get("last_client_idx", "?")
    print(f"{COLORS.OKCYAN}  Mid-round checkpoint saved ({tag}, client {last}){COLORS.ENDC}")


def load_mid_round(context: Any, tag: str, round_number: int) -> Optional[Dict[str, Any]]:
    """Load a mid-round checkpoint if it exists and matches *round_number*."""
    d = _ckpt_dir(context)
    path = os.path.join(d, f"{tag}_mid.bin")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("round") != round_number:
        return None
    return payload


def clear_mid_round(context: Any, tag: str) -> None:
    """Remove the mid-round checkpoint after a successful round."""
    d = _ckpt_dir(context)
    path = os.path.join(d, f"{tag}_mid.bin")
    if os.path.exists(path):
        os.remove(path)
