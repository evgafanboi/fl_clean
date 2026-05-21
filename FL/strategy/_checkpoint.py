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

_POISONEDFL_STATE_KEY = "__poisoned_fl_state__"

# Avoid circular import — callers pass PipelineContext; we only need .log_filename
# so we accept Any.


def _ckpt_dir(context: Any) -> str:
    """Return the checkpoint directory for the current pipeline run."""
    stem = os.path.splitext(os.path.basename(context.log_filename))[0]
    return os.path.join("checkpoint", stem)


def save_mid_round(context: Any, tag: str, payload: Dict[str, Any]) -> None:
    """Atomically write *payload* to ``<ckpt_dir>/<tag>_mid.bin``.

    Also records client weights for eval replay up to last_client_idx
    and updates ``info.txt`` with stage progress so a crash mid-round
    can be resumed correctly.
    """
    d = _ckpt_dir(context)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f"{tag}_mid.bin.tmp")
    dst = os.path.join(d, f"{tag}_mid.bin")
    if getattr(context, "poisoned_fl_state", None) is not None and _POISONEDFL_STATE_KEY not in payload:
        payload = dict(payload)
        payload[_POISONEDFL_STATE_KEY] = context.poisoned_fl_state

    # Determine first_idx: only write clients that are new since the last checkpoint
    # (same stage = incremental; stage transition = full rewrite from 0)
    first_idx = 0
    if os.path.exists(dst):
        try:
            with open(dst, "rb") as f:
                prev = pickle.load(f)
            if prev.get("stage") == payload.get("stage"):
                first_idx = prev.get("last_client_idx", -1) + 1
        except Exception:
            pass

    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst)
    last = payload.get("last_client_idx", "?")
    print(f"{COLORS.OKCYAN}  Mid-round checkpoint saved ({tag}, client {last}){COLORS.ENDC}")

    # Update info.txt with stage progress
    _update_info_stage(d, payload)

    round_num = payload.get("round")
    last_idx = payload.get("last_client_idx")
    if round_num is not None and last_idx is not None:
        _record_checkpoint_weights(context, round_num, first_idx, last_idx)


def _record_checkpoint_weights(context: Any, round_num: int, first_client_idx: int, last_client_idx: int) -> None:
    """Record client weights for clients from first_client_idx to last_client_idx."""
    for st in context.client_states[first_client_idx:last_client_idx + 1]:
        w = st.data.get("w")
        if w is not None:
            context.record_client_weight(round_num, st.client_id, w)
            continue
        if context.model_pool is None:
            continue
        pool = context.model_pool
        if hasattr(pool, 'get_cached_weights'):
            w = pool.get_cached_weights(st.client_id)
            if w is not None:
                context.record_client_weight(round_num, st.client_id, w)
                continue
        model = pool.checkout(st.client_id)
        w = pool._get_weights(model)
        pool.release(model)
        context.record_client_weight(round_num, st.client_id, w)


def _update_info_stage(ckpt_dir: str, payload: Dict[str, Any]) -> None:
    """Merge stage progress into the existing ``info.txt``."""
    info_path = os.path.join(ckpt_dir, "info.txt")
    if not os.path.exists(info_path):
        return
    info: Dict[str, str] = {}
    with open(info_path) as f:
        for line in f:
            key, _, val = line.strip().partition(": ")
            if key:
                info[key] = val
    if "stage" in payload:
        info["stage"] = str(payload["stage"])
    if "last_client_idx" in payload:
        info["stage_client"] = str(payload["last_client_idx"])
    with open(info_path, "w") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in info.items()) + "\n")


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
    saved_pfl_state = payload.pop(_POISONEDFL_STATE_KEY, None)
    if saved_pfl_state is not None:
        context.poisoned_fl_state = saved_pfl_state
    return payload


def clear_mid_round(context: Any, tag: str) -> None:
    """Remove the mid-round checkpoint after a successful round."""
    d = _ckpt_dir(context)
    path = os.path.join(d, f"{tag}_mid.bin")
    if os.path.exists(path):
        os.remove(path)
