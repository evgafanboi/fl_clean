from __future__ import annotations

import gc
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from ..backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf
from tqdm import tqdm

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext
from ..poison_utils import parse_poison_config, apply_gradient_scale_poison, poisonedfl_log_values, poisonedfl_store_round_weights, poisonedfl_unified_weights, poisonedfl_warmstart_weights
from .base import DistillationStrategy
from ._checkpoint import save_mid_round, load_mid_round, clear_mid_round
from .robust_filter import CronusRobustFilter, RobustFilterV3
from .common import (
    create_model,
    create_private_dataset,
    load_public_dataset_from_clients,
    lma_logits as _lma_logits,
    numpy_from_dataset,
    poisonedfl_ghost_model_type,
)

LOGIT_ABS_CAP = 1e6
EKD_LOGIT_CLIP = 30.0
EVA_PRIVATE_CHUNK = 100_000
EVA_PUBLIC_CHUNK = 500_000
EVA_EPS = 1e-6
EVA_PRIVATE_QUANTILE = 0.95 # energy-based vote abstention quantile
EVW_TAU1 = 0.0
EVW_TAU2 = 1.0


def _sanitize_logits(logits: np.ndarray, cap: float = LOGIT_ABS_CAP) -> np.ndarray:
    return np.nan_to_num(logits, nan=0.0, posinf=cap, neginf=-cap).astype(np.float32, copy=False)


def _format_v3_discard_ratios(discard_frac: np.ndarray, survivor: np.ndarray, limit: int = 10) -> List[Tuple[int, float]]:
    return sorted(
        [
            (client_id, round(float(discard_frac[client_id]), 4))
            for client_id in range(len(discard_frac))
            if not survivor[client_id]
        ],
        key=lambda item: -item[1],
    )[:limit]


def _format_removal_summary(robust_filter, removal_counts: Dict[int, int]) -> Tuple[str, object]:
    if type(robust_filter) is RobustFilterV3:
        removed_client_ids = sorted(client_id for client_id, count in removal_counts.items() if count > 0)
        return "removed client_ids", removed_client_ids
    top_removed = sorted(removal_counts.items(), key=lambda item: -item[1])
    return "top removals (client_idx: count)", top_removed[:5]


def _eva_mode(config) -> Optional[str]:
    if getattr(config, "evw", False):
        return "evw"
    elif getattr(config, "eva", False):
        return "eva"
    elif getattr(config, "eva2", False):
        return "eva2"
    return None


def _support_path(cache_dir: str, client_id: int) -> str:
    return os.path.join(cache_dir, f"client_{client_id}_support.bin")


def _client_id_from_logit_path(path: str) -> int:
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def _energy_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    row_max = np.max(scaled, axis=1, keepdims=True)
    lse = row_max[:, 0] + np.log(np.exp(scaled - row_max).sum(axis=1) + 1e-12)
    return (-temperature * lse).astype(np.float32)


def _support_weights_from_scores(scores: np.ndarray, mode: Optional[str]) -> np.ndarray:
    if mode == "evw":
        return np.clip((EVW_TAU2 - scores) / max(EVW_TAU2 - EVW_TAU1, EVA_EPS), 0.0, 1.0).astype(np.float32)
    return np.ones_like(scores, dtype=np.float32)


def _write_support_values(output_path: str, values: np.ndarray) -> None:
    with open(output_path, "wb") as fp:
        fp.write(np.asarray(values, dtype=np.float32).tobytes())


def _load_private_energies(model_wrapper, private_X_path: str, batch_size: int) -> np.ndarray:
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else model_wrapper
    X_mmap = np.load(private_X_path, mmap_mode="r")
    energies = []
    for start in range(0, len(X_mmap), EVA_PRIVATE_CHUNK):
        chunk = np.asarray(X_mmap[start:start + EVA_PRIVATE_CHUNK], dtype=np.float32)
        logits = _sanitize_logits(logits_model.predict(chunk, batch_size=batch_size, verbose=0))
        energies.append(_energy_from_logits(logits))
        del chunk, logits
    del X_mmap
    if not energies:
        return np.empty(0, dtype=np.float32)
    energy = np.concatenate(energies, axis=0)
    del energies
    return energy


def _compute_private_energy_quantile(model_wrapper, private_X_path: str, batch_size: int) -> float:
    energy = _load_private_energies(model_wrapper, private_X_path, batch_size)
    if energy.size == 0:
        return 0.0
    threshold = float(np.quantile(energy, EVA_PRIVATE_QUANTILE))
    del energy
    return threshold


def _compute_private_energy_eva2(model_wrapper, private_X_path: str, batch_size: int) -> float:
    energy = _load_private_energies(model_wrapper, private_X_path, batch_size)
    if energy.size == 0:
        return 0.0
    unique_energy, counts = np.unique(energy, return_counts=True)
    unique_energy = unique_energy[::-1]
    counts = counts[::-1]
    
    total_energy = len(energy)
    num_unique = len(unique_energy)
    threshold = float(unique_energy[0])
    
    threshold_ratio = 1.0 / num_unique
    
    for i in range(num_unique):
        c_i = counts[i]
        if c_i / total_energy < threshold_ratio:
            continue
        else:
            threshold = float(unique_energy[i])
            break
            
    del energy
    return threshold


def _compute_private_energy_stats(model_wrapper, private_X_path: str, batch_size: int) -> Tuple[float, float]:
    energy = _load_private_energies(model_wrapper, private_X_path, batch_size)
    if energy.size == 0:
        return 0.0, 1.0
    mu = float(np.median(energy))
    sigma = float(np.median(np.abs(energy - mu)) * 1.4826 + EVA_EPS)
    del energy
    return mu, sigma


def generate_logits_and_support_to_file(
    model_wrapper,
    private_X_path: str,
    public_features: np.ndarray,
    batch_size: int,
    output_path: str,
    support_path: str,
    mode: str,
) -> tuple:
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else model_wrapper
    q95 = None
    mu = None
    sigma = None
    if mode == "eva":
        q95 = _compute_private_energy_quantile(model_wrapper, private_X_path, batch_size)
    elif mode == "eva2":
        q95 = _compute_private_energy_eva2(model_wrapper, private_X_path, batch_size)
    else:
        mu, sigma = _compute_private_energy_stats(model_wrapper, private_X_path, batch_size)
    total_rows = 0
    num_classes = None
    support_sum = 0.0
    support_nonzero = 0
    with open(output_path, "wb") as log_fp, open(support_path, "wb") as support_fp:
        for start in range(0, len(public_features), EVA_PUBLIC_CHUNK):
            chunk = public_features[start:start + EVA_PUBLIC_CHUNK]
            logits = _sanitize_logits(logits_model.predict(chunk, batch_size=batch_size, verbose=0))
            if num_classes is None:
                num_classes = logits.shape[1]
            log_fp.write(logits.tobytes())
            energies = _energy_from_logits(logits)
            if mode == "eva" or mode == "eva2":
                support = (energies <= q95).astype(np.float32)
            else:
                scores = (energies - mu) / sigma
                support = _support_weights_from_scores(scores, mode)
            support_fp.write(np.asarray(support, dtype=np.float32).tobytes())
            total_rows += logits.shape[0]
            support_sum += float(support.sum())
            support_nonzero += int((support > 0).sum())
            del logits, energies, support, chunk
            if mode != "eva" and mode != "eva2":
                del scores
    stats = {
        "support_mean": support_sum / max(total_rows, 1),
        "support_rate": support_nonzero / max(total_rows, 1),
    }
    if mode == "eva" or mode == "eva2":
        stats["q95"] = q95
    else:
        stats["mu"] = mu
        stats["sigma"] = sigma
    return output_path, (total_rows, num_classes), stats


# ── Logit generation (chunked to disk) ──────────────────────────────────────

def generate_logits_to_file(model_wrapper, public_features: np.ndarray, batch_size: int, output_path: str) -> tuple:
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else model_wrapper
    chunk_size = 500_000
    total_rows = 0
    num_classes = None
    with open(output_path, "wb") as fp:
        for start in range(0, len(public_features), chunk_size):
            chunk = public_features[start : start + chunk_size]
            logits = _sanitize_logits(logits_model.predict(chunk, batch_size=batch_size, verbose=0))
            if num_classes is None:
                num_classes = logits.shape[1]
            fp.write(logits.tobytes())
            total_rows += logits.shape[0]
            del logits, chunk
    return output_path, (total_rows, num_classes)


# ── Consensus ────────────────────────────────────────────────────────────────

def compute_consensus_from_files(logit_files: List[str], shape: tuple) -> np.ndarray:
    n_clients = len(logit_files)
    n_samples, n_classes = shape
    consensus = np.zeros(shape, dtype=np.float32)
    chunk_rows = 100_000
    row_bytes = n_classes * 4
    for fpath in logit_files:
        with open(fpath, "rb") as f:
            offset = 0
            while offset < n_samples:
                rows = min(chunk_rows, n_samples - offset)
                raw = f.read(rows * row_bytes)
                chunk = np.frombuffer(raw, dtype=np.float32).reshape(rows, n_classes)
                consensus[offset : offset + rows] += chunk
                offset += rows
                del chunk
    consensus /= n_clients
    return consensus


def compute_robust_consensus_from_files(
    logit_files: List[str], shape: tuple, robust_filter,
    poisoned_client_ids=None, logger=None, round_number=None,
) -> tuple:
    n_clients = len(logit_files)
    n_samples, n_classes = shape
    consensus = np.zeros(shape, dtype=np.float32)
    chunk_rows = 100_000
    row_bytes = n_classes * 4
    max_eig = None
    max_ratio = None
    removal_counts = {c: 0 for c in range(n_clients)}  # samples excluded per client
    samplewise_cronus = isinstance(robust_filter, CronusRobustFilter)
    eig_chunks = [] if samplewise_cronus else None

    if isinstance(robust_filter, RobustFilterV3):
        total_counts = np.zeros(n_clients, dtype=np.int64)
        total_rows = 0
        chunk_masks = []
        handles = [open(fpath, "rb") for fpath in logit_files]
        pbar = tqdm(total=n_samples, desc="Scoring (v3)", unit="sample")
        for offset in range(0, n_samples, chunk_rows):
            rows = min(chunk_rows, n_samples - offset)
            client_chunks = [
                _sanitize_logits(np.frombuffer(h.read(rows * row_bytes), dtype=np.float32).reshape(rows, n_classes))
                for h in handles
            ]
            S_batch = np.stack(client_chunks, axis=1)
            chunk_mask, chunk_max = robust_filter.count_discards_mask(S_batch)
            total_counts += chunk_mask.sum(axis=0)
            chunk_masks.append(chunk_mask)
            total_rows += rows
            if chunk_max is not None and (max_eig is None or chunk_max > max_eig):
                max_eig = chunk_max
            pbar.update(rows)
            del client_chunks, S_batch
        pbar.close()
        for h in handles:
            h.close()
        discard_frac = total_counts.astype(np.float64) / max(total_rows, 1)
        survivor = discard_frac <= robust_filter.robust_threshold
        for c in range(n_clients):
            if not survivor[c]:
                removal_counts[c] = total_rows
        n_failed = int((~survivor).sum())
        failed_ratios = _format_v3_discard_ratios(discard_frac, survivor)
        msg = f"  V3 filter: threshold={robust_filter.robust_threshold:.2f} | {n_failed}/{n_clients} clients failed | top discard ratios: {failed_ratios}"
        print(msg)
        if logger is not None:
            logger.info(msg)
        v3_supported = np.zeros(n_samples, dtype=bool)
        handles = [open(fpath, "rb") for fpath in logit_files]
        pbar = tqdm(total=n_samples, desc="Mean pass (v3)", unit="sample")
        for chunk_i, offset in enumerate(range(0, n_samples, chunk_rows)):
            rows = min(chunk_rows, n_samples - offset)
            client_chunks = [
                _sanitize_logits(np.frombuffer(h.read(rows * row_bytes), dtype=np.float32).reshape(rows, n_classes))
                for h in handles
            ]
            S_batch = np.stack(client_chunks, axis=1)
            per_sample_alive = (~chunk_masks[chunk_i]) & survivor[np.newaxis, :]
            alive_f = per_sample_alive.astype(np.float32)
            n_alive = alive_f.sum(axis=1, keepdims=True)
            has_alive = (n_alive[:, 0] > 0)
            v3_supported[offset:offset + rows] = has_alive
            n_alive = n_alive.clip(min=1)
            consensus[offset:offset + rows] = (np.einsum('rkc,rk->rc', S_batch, alive_f, optimize=True) / n_alive).astype(np.float32)
            pbar.update(rows)
            del client_chunks, S_batch, per_sample_alive, alive_f, n_alive
        pbar.close()
        for h in handles:
            h.close()
        return consensus, max_eig, max_ratio, removal_counts, eig_report, v3_supported
    else:
        handles = [open(fpath, "rb") for fpath in logit_files]
        pbar = tqdm(total=n_samples, desc="Robust consensus", unit="sample")
        for offset in range(0, n_samples, chunk_rows):
            rows = min(chunk_rows, n_samples - offset)
            client_chunks = [
                _sanitize_logits(np.frombuffer(h.read(rows * row_bytes), dtype=np.float32).reshape(rows, n_classes))
                for h in handles
            ]
            S_batch = np.stack(client_chunks, axis=1)
            if samplewise_cronus:
                means, batch_max_eig, removal_delta, _ = robust_filter.filter_batch(S_batch)
                batch_max_ratio = None
                eig_stats = getattr(robust_filter, "last_filter_stats", None)
                if eig_stats is not None:
                    eig_chunks.append(np.asarray(eig_stats["pre_top_eigs"], dtype=np.float64))
            else:
                means, batch_max_eig, removal_delta, batch_max_ratio = robust_filter.compute_robust_mean_batch(S_batch)
            consensus[offset : offset + rows] = means
            if batch_max_eig is not None and (max_eig is None or batch_max_eig > max_eig):
                max_eig = batch_max_eig
            if batch_max_ratio is not None and (max_ratio is None or batch_max_ratio > max_ratio):
                max_ratio = batch_max_ratio
            for c in range(n_clients):
                removal_counts[c] += int(removal_delta[c]) if samplewise_cronus else rows * int(removal_delta[c] > 0)
            pbar.update(rows)
            del client_chunks, S_batch, means
        pbar.close()
        for h in handles:
            h.close()

    if poisoned_client_ids is not None and logger is not None:
        poisoned_set = set(poisoned_client_ids)
        predicted_removed = {c for c, v in removal_counts.items() if v > 0}
        TP = len(predicted_removed & poisoned_set)
        FP = len(predicted_removed - poisoned_set)
        FN = len(poisoned_set - predicted_removed)
        TN = n_clients - TP - FP - FN
        TPR = TP / max(1, TP + FN)
        FPR = FP / max(1, FP + TN)
        TNR = TN / max(1, TN + FP)
        FNR = FN / max(1, FN + TP)
        logger.info(
            "Round %s | FilterMetrics | TP=%d FP=%d TN=%d FN=%d | TPR=%.3f FPR=%.3f TNR=%.3f FNR=%.3f",
            round_number, TP, FP, TN, FN, TPR, FPR, TNR, FNR,
        )

    eig_report = None
    if samplewise_cronus and eig_chunks:
        top_eigs = np.concatenate(eig_chunks)
        eig_report = {
            "n_samples": int(top_eigs.size),
            "min_eig": float(top_eigs.min()),
            "mean_eig": float(top_eigs.mean()),
            "med_eig": float(np.median(top_eigs)),
            "max_eig": float(top_eigs.max()),
        }

    return consensus, max_eig, max_ratio, removal_counts, eig_report, None


def compute_supported_consensus_from_files(
    logit_files: List[str],
    support_files: List[str],
    shape: tuple,
    mode: str,
    robust_filter=None,
    poisoned_client_ids=None,
    logger=None,
    round_number=None,
) -> tuple:
    n_clients = len(logit_files)
    n_samples, n_classes = shape
    consensus = np.zeros(shape, dtype=np.float32)
    supported_mask = np.zeros(n_samples, dtype=bool)
    chunk_rows = 100_000
    row_bytes = n_classes * 4
    support_bytes = 4
    max_eig = None
    max_ratio = None
    removal_counts = {c: 0 for c in range(n_clients)}
    samplewise_cronus = isinstance(robust_filter, CronusRobustFilter)
    eig_chunks = [] if samplewise_cronus else None

    budget = getattr(robust_filter, "budget", 0) if robust_filter is not None else 0
    global_v3 = isinstance(robust_filter, RobustFilterV3) if robust_filter is not None else False
    if global_v3:
        total_counts = np.zeros(n_clients, dtype=np.int64)
        total_rows_v3 = 0
        chunk_masks = []
        handles = [open(fpath, "rb") for fpath in logit_files]
        score_pbar = tqdm(total=n_samples, desc="Scoring (v3)", unit="sample")
        for offset in range(0, n_samples, chunk_rows):
            rows = min(chunk_rows, n_samples - offset)
            client_chunks = [
                _sanitize_logits(np.frombuffer(h.read(rows * row_bytes), dtype=np.float32).reshape(rows, n_classes))
                for h in handles
            ]
            S_batch = np.stack(client_chunks, axis=1)
            chunk_mask, chunk_max = robust_filter.count_discards_mask(S_batch)
            total_counts += chunk_mask.sum(axis=0)
            chunk_masks.append(chunk_mask)
            total_rows_v3 += rows
            if chunk_max is not None and (max_eig is None or chunk_max > max_eig):
                max_eig = chunk_max
            score_pbar.update(rows)
            del client_chunks, S_batch
        score_pbar.close()
        for h in handles:
            h.close()
        discard_frac = total_counts.astype(np.float64) / max(total_rows_v3, 1)
        survivor = discard_frac <= robust_filter.robust_threshold
        for c in range(n_clients):
            if not survivor[c]:
                removal_counts[c] = total_rows_v3
        n_failed = int((~survivor).sum())
        failed_ratios = _format_v3_discard_ratios(discard_frac, survivor)
        msg = f"  V3 filter: threshold={robust_filter.robust_threshold:.2f} | {n_failed}/{n_clients} clients failed | top discard ratios: {failed_ratios}"
        print(msg)
        if logger is not None:
            logger.info(msg)
        log_handles = [open(fpath, "rb") for fpath in logit_files]
        support_handles = [open(fpath, "rb") for fpath in support_files]
        mean_pbar = tqdm(total=n_samples, desc="Mean pass (v3)", unit="sample")
        for chunk_i, offset in enumerate(range(0, n_samples, chunk_rows)):
            rows = min(chunk_rows, n_samples - offset)
            client_chunks = [
                _sanitize_logits(np.frombuffer(h.read(rows * row_bytes), dtype=np.float32).reshape(rows, n_classes))
                for h in log_handles
            ]
            support_chunks = [
                np.frombuffer(h.read(rows * support_bytes), dtype=np.float32)
                for h in support_handles
            ]
            S_batch = np.stack(client_chunks, axis=1)
            Q_batch = np.stack(support_chunks, axis=1).astype(np.float32)
            chunk_consensus = consensus[offset:offset + rows]
            per_sample_alive = (~chunk_masks[chunk_i]) & survivor[np.newaxis, :]
            if mode == "eva" or mode == "eva2":
                Q_survivors = (Q_batch > 0.0).astype(np.float32) * per_sample_alive.astype(np.float32)
            else:
                Q_survivors = Q_batch * per_sample_alive.astype(np.float32)
            support_mass = Q_survivors.sum(axis=1)
            valid = support_mass > 0.0
            if valid.any():
                weighted_sum = np.einsum("rk,rkc->rc", Q_survivors[valid], S_batch[valid], optimize=True)
                chunk_consensus[valid] = (weighted_sum / support_mass[valid, np.newaxis]).astype(np.float32)
                supported_mask[offset:offset + rows] = valid
                del weighted_sum
            del Q_survivors, support_mass, valid, per_sample_alive
            mean_pbar.update(rows)
            del client_chunks, support_chunks, S_batch, Q_batch
        mean_pbar.close()
        for handle in log_handles:
            handle.close()
        for handle in support_handles:
            handle.close()
    else:
        log_handles = [open(fpath, "rb") for fpath in logit_files]
        support_handles = [open(fpath, "rb") for fpath in support_files]
        pbar = tqdm(total=n_samples, desc="Robust consensus", unit="sample")
        for offset in range(0, n_samples, chunk_rows):
            rows = min(chunk_rows, n_samples - offset)
            client_chunks = [
                _sanitize_logits(np.frombuffer(h.read(rows * row_bytes), dtype=np.float32).reshape(rows, n_classes))
                for h in log_handles
            ]
            support_chunks = [
                np.frombuffer(h.read(rows * support_bytes), dtype=np.float32)
                for h in support_handles
            ]
            S_batch = np.stack(client_chunks, axis=1)
            Q_batch = np.stack(support_chunks, axis=1).astype(np.float32)
            chunk_consensus = consensus[offset:offset + rows]

            if mode == "eva" or mode == "eva2":
                if budget > 0:
                    if samplewise_cronus:
                        _, batch_max_eig, removal_delta, survivor_mask = robust_filter.filter_batch(S_batch)
                        batch_max_ratio = None
                        survivor_weights = survivor_mask.astype(np.float32)
                        eig_stats = getattr(robust_filter, "last_filter_stats", None)
                        if eig_stats is not None:
                            eig_chunks.append(np.asarray(eig_stats["pre_top_eigs"], dtype=np.float64))
                    else:
                        _, batch_max_eig, removal_delta, batch_max_ratio = robust_filter.compute_robust_mean_batch(S_batch)
                        survivor_weights = np.broadcast_to((removal_delta == 0).astype(np.float32), (rows, n_clients))
                    if batch_max_eig is not None and (max_eig is None or batch_max_eig > max_eig):
                        max_eig = batch_max_eig
                    if batch_max_ratio is not None and (max_ratio is None or batch_max_ratio > max_ratio):
                        max_ratio = batch_max_ratio
                    for c in range(n_clients):
                        removal_counts[c] += int(removal_delta[c]) if samplewise_cronus else rows * int(removal_delta[c] > 0)
                else:
                    survivor_weights = np.ones((rows, n_clients), dtype=np.float32)

                Q_survivors = (Q_batch > 0.0).astype(np.float32) * survivor_weights
                support_mass = Q_survivors.sum(axis=1)
                valid = support_mass > 0.0
                if valid.any():
                    masked_sum = np.einsum("rk,rkc->rc", Q_survivors[valid], S_batch[valid], optimize=True)
                    chunk_consensus[valid] = (masked_sum / support_mass[valid, np.newaxis]).astype(np.float32)
                    supported_mask[offset:offset + rows] = valid
                    del masked_sum
                del Q_survivors, support_mass, valid
            else:
                if budget > 0:
                    if samplewise_cronus:
                        _, batch_max_eig, removal_delta, survivor_mask = robust_filter.filter_batch(S_batch)
                        batch_max_ratio = None
                        survivor_weights = survivor_mask.astype(np.float32)
                        eig_stats = getattr(robust_filter, "last_filter_stats", None)
                        if eig_stats is not None:
                            eig_chunks.append(np.asarray(eig_stats["pre_top_eigs"], dtype=np.float64))
                    else:
                        _, batch_max_eig, removal_delta, batch_max_ratio = robust_filter.compute_robust_mean_batch(S_batch)
                        survivor_weights = np.broadcast_to((removal_delta == 0).astype(np.float32), (rows, n_clients))
                    if batch_max_eig is not None and (max_eig is None or batch_max_eig > max_eig):
                        max_eig = batch_max_eig
                    if batch_max_ratio is not None and (max_ratio is None or batch_max_ratio > max_ratio):
                        max_ratio = batch_max_ratio
                    for c in range(n_clients):
                        removal_counts[c] += int(removal_delta[c]) if samplewise_cronus else rows * int(removal_delta[c] > 0)
                else:
                    survivor_weights = np.ones((rows, n_clients), dtype=np.float32)

                Q_survivors = Q_batch * survivor_weights
                support_mass = Q_survivors.sum(axis=1)
                valid = support_mass > 0.0
                if valid.any():
                    weighted_sum = np.einsum("rk,rkc->rc", Q_survivors[valid], S_batch[valid], optimize=True)
                    chunk_consensus[valid] = (weighted_sum / support_mass[valid, np.newaxis]).astype(np.float32)
                    supported_mask[offset:offset + rows] = valid
                del Q_survivors, support_mass, valid

            pbar.update(rows)
            del client_chunks, support_chunks, S_batch, Q_batch
        pbar.close()
        for handle in log_handles:
            handle.close()
        for handle in support_handles:
            handle.close()

    if poisoned_client_ids is not None and logger is not None and budget > 0:
        poisoned_set = set(poisoned_client_ids)
        predicted_removed = {c for c, v in removal_counts.items() if v > 0}
        TP = len(predicted_removed & poisoned_set)
        FP = len(predicted_removed - poisoned_set)
        FN = len(poisoned_set - predicted_removed)
        TN = n_clients - TP - FP - FN
        TPR = TP / max(1, TP + FN)
        FPR = FP / max(1, FP + TN)
        TNR = TN / max(1, TN + FP)
        FNR = FN / max(1, FN + TP)
        logger.info(
            "Round %s | FilterMetrics | TP=%d FP=%d TN=%d FN=%d | TPR=%.3f FPR=%.3f TNR=%.3f FNR=%.3f",
            round_number, TP, FP, TN, FN, TPR, FPR, TNR, FNR,
        )

    eig_report = None
    if samplewise_cronus and eig_chunks:
        top_eigs = np.concatenate(eig_chunks)
        eig_report = {
            "n_samples": int(top_eigs.size),
            "min_eig": float(top_eigs.min()),
            "mean_eig": float(top_eigs.mean()),
            "med_eig": float(np.median(top_eigs)),
            "max_eig": float(top_eigs.max()),
        }

    return consensus, supported_mask, max_eig, max_ratio, removal_counts, eig_report


# ── Stage 2: KD (pure distillation, no CE) ──────────────────────────────────

def ekd_stage(
    model_wrapper, consensus_logits: np.ndarray, public_features: np.ndarray,
    batch_size: int, epochs: int, ekd_lambda: float,
) -> float | None:
    from ..backend import use_tf
    if not use_tf():
        return _ekd_stage_pt(model_wrapper, consensus_logits, public_features, batch_size, epochs, ekd_lambda)
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else keras_model

    optimizer = keras_model.optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model.optimizer = optimizer
    lam = tf.constant(ekd_lambda, dtype=tf.float32)

    if not hasattr(model_wrapper, '_ekd_train_step'):
        @tf.function
        def train_step(batch_X, teacher_logits):
            with tf.GradientTape() as tape:
                student_logits = tf.clip_by_value(logits_model(batch_X, training=True), -EKD_LOGIT_CLIP, EKD_LOGIT_CLIP)
                teacher_logits = tf.clip_by_value(teacher_logits, -EKD_LOGIT_CLIP, EKD_LOGIT_CLIP)

                alpha_T = tf.exp(teacher_logits) + 1.0
                alpha_S = tf.exp(student_logits) + 1.0
                alpha0_T = tf.reduce_sum(alpha_T, axis=-1, keepdims=True)
                alpha0_S = tf.reduce_sum(alpha_S, axis=-1, keepdims=True)

                L_1st = tf.reduce_sum(
                    (alpha_T / alpha0_T) * (tf.math.log(alpha_T / alpha0_T + 1e-8) - tf.math.log(alpha_S / alpha0_S + 1e-8)),
                    axis=-1,
                )

                L_2nd = (
                    tf.math.lgamma(alpha0_T[:, 0]) - tf.math.lgamma(alpha0_S[:, 0])
                    - tf.reduce_sum(tf.math.lgamma(alpha_T) - tf.math.lgamma(alpha_S), axis=-1)
                    + tf.reduce_sum(
                        (alpha_T - alpha_S) * (tf.math.digamma(alpha_T) - tf.math.digamma(alpha0_T)),
                        axis=-1,
                    )
                )

                loss = tf.reduce_mean(L_1st + lam * L_2nd)

            gradients = tape.gradient(loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return loss

        model_wrapper._ekd_train_step = train_step

    train_step = model_wrapper._ekd_train_step

    n_samples = len(public_features)
    final_loss = None
    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]
            loss = train_step(public_features[idx], consensus_logits[idx])
            epoch_loss += float(loss)
            batches += 1
        if batches > 0:
            final_loss = epoch_loss / batches
            print(f"    EKD epoch {epoch + 1}/{epochs} — loss {final_loss:.4f}")
    return final_loss


def _ekd_stage_pt(
    model_wrapper, consensus_logits: np.ndarray, public_features: np.ndarray,
    batch_size: int, epochs: int, ekd_lambda: float,
) -> float | None:
    import torch
    from ..backend import get_torch_device
    dev = get_torch_device()
    logits_model = model_wrapper.get_logits_model()
    net = logits_model._w.nn
    net.to(dev)
    net.train()
    opt = model_wrapper.optimizer
    lam = ekd_lambda
    n_samples = len(public_features)
    feat_t = torch.from_numpy(public_features).to(dev)
    logits_t = torch.from_numpy(consensus_logits).to(dev)
    final_loss = None
    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            bX = feat_t[idx]
            bT = torch.clamp(logits_t[idx], -EKD_LOGIT_CLIP, EKD_LOGIT_CLIP)
            s_logits = torch.clamp(net(bX, return_logits=True), -EKD_LOGIT_CLIP, EKD_LOGIT_CLIP)
            alpha_T = torch.exp(bT) + 1.0
            alpha_S = torch.exp(s_logits) + 1.0
            a0_T = alpha_T.sum(-1, keepdim=True)
            a0_S = alpha_S.sum(-1, keepdim=True)
            L1 = ((alpha_T / a0_T) * (torch.log(alpha_T / a0_T + 1e-8) - torch.log(alpha_S / a0_S + 1e-8))).sum(-1)
            L2 = (torch.lgamma(a0_T[:, 0]) - torch.lgamma(a0_S[:, 0])
                  - (torch.lgamma(alpha_T) - torch.lgamma(alpha_S)).sum(-1)
                  + ((alpha_T - alpha_S) * (torch.digamma(alpha_T) - torch.digamma(a0_T))).sum(-1))
            loss = (L1 + lam * L2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            epoch_loss += loss.item()
            batches += 1
        if batches > 0:
            final_loss = epoch_loss / batches
            print(f"    EKD epoch {epoch + 1}/{epochs} — loss {final_loss:.4f}")
    return final_loss


def abkd_stage(
    model_wrapper, consensus_logits: np.ndarray, public_features: np.ndarray,
    batch_size: int, epochs: int, alpha: float, beta: float, temperature: float,
) -> float | None:
    from ..backend import use_tf
    if not use_tf():
        return _abkd_stage_pt(model_wrapper, consensus_logits, public_features,
                               batch_size, epochs, alpha, beta, temperature)
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else keras_model

    optimizer = keras_model.optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model.optimizer = optimizer
    a = tf.constant(alpha, dtype=tf.float32)
    b = tf.constant(beta, dtype=tf.float32)
    T = tf.constant(temperature, dtype=tf.float32)

    if not hasattr(model_wrapper, '_abkd_train_step'):
        @tf.function
        def train_step(batch_X, teacher_logits):
            with tf.GradientTape() as tape:
                student_logits = logits_model(batch_X, training=True)
                p = tf.nn.softmax(teacher_logits / T)
                q = tf.nn.softmax(student_logits / T)

                ab_sum = a + b
                if alpha == 0.0 and beta == 0.0:
                    log_diff = tf.math.log(q + 1e-10) - tf.math.log(p + 1e-10)
                    divergence = 0.5 * tf.reduce_sum(log_diff ** 2, axis=1)
                elif alpha == 0.0:
                    q_b = tf.pow(q, b)
                    p_b = tf.pow(p, b)
                    ratio = q_b / (p_b + 1e-10)
                    divergence = (1.0 / b) * tf.reduce_sum(
                        q_b * tf.math.log(ratio + 1e-10) - q_b + p_b, axis=1,
                    )
                elif beta == 0.0:
                    p_a = tf.pow(p, a)
                    q_a = tf.pow(q, a)
                    divergence = (1.0 / a) * tf.reduce_sum(
                        p_a * tf.math.log(p_a / (q_a + 1e-10) + 1e-10) - p_a + q_a, axis=1,
                    )
                elif alpha + beta == 0.0:
                    p_a = tf.pow(p, a)
                    q_a = tf.pow(q, a)
                    divergence = tf.reduce_sum(
                        (1.0 / a) * (tf.math.log(q_a / (p_a + 1e-10) + 1e-10) + p_a / (q_a + 1e-10) - 1.0),
                        axis=1,
                    )
                else:
                    p_a = tf.pow(p, a)
                    q_b = tf.pow(q, b)
                    p_ab = tf.pow(p, ab_sum)
                    q_ab = tf.pow(q, ab_sum)
                    divergence = -tf.reduce_sum(
                        p_a * q_b - (a / ab_sum) * p_ab - (b / ab_sum) * q_ab,
                        axis=1,
                    ) / (a * b)

                loss = tf.reduce_mean(divergence)

            gradients = tape.gradient(loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            return loss

        model_wrapper._abkd_train_step = train_step

    train_step = model_wrapper._abkd_train_step

    n_samples = len(public_features)
    final_loss = None
    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]
            loss = train_step(public_features[idx], consensus_logits[idx])
            epoch_loss += float(loss)
            batches += 1
        if batches > 0:
            final_loss = epoch_loss / batches
            print(f"    ABKD epoch {epoch + 1}/{epochs} — loss {final_loss:.4f}")
    return final_loss


def _abkd_stage_pt(
    model_wrapper, consensus_logits: np.ndarray, public_features: np.ndarray,
    batch_size: int, epochs: int, alpha: float, beta: float, temperature: float,
) -> float | None:
    import torch
    from ..backend import get_torch_device
    dev = get_torch_device()
    logits_model = model_wrapper.get_logits_model()
    net = logits_model._w.nn
    net.to(dev)
    net.train()
    opt = model_wrapper.optimizer
    T = temperature
    a = alpha
    b = beta
    ab_sum = a + b
    n_samples = len(public_features)
    feat_t = torch.from_numpy(public_features).to(dev)
    teacher_t = torch.from_numpy(consensus_logits).to(dev)
    final_loss = None
    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            bX = feat_t[idx]
            bT_logits = teacher_t[idx]
            s_logits = net(bX, return_logits=True)
            p = torch.softmax(bT_logits / T, dim=-1)
            q = torch.softmax(s_logits / T, dim=-1)
            if a == 0.0 and b == 0.0:
                log_diff = torch.log(q + 1e-10) - torch.log(p + 1e-10)
                divergence = 0.5 * (log_diff ** 2).sum(1)
            elif a == 0.0:
                q_b = q.pow(b)
                p_b = p.pow(b)
                divergence = (1.0 / b) * (q_b * torch.log(q_b / (p_b + 1e-10) + 1e-10) - q_b + p_b).sum(1)
            elif b == 0.0:
                p_a = p.pow(a)
                q_a = q.pow(a)
                divergence = (1.0 / a) * (p_a * torch.log(p_a / (q_a + 1e-10) + 1e-10) - p_a + q_a).sum(1)
            elif ab_sum == 0.0:
                p_a = p.pow(a)
                q_a = q.pow(a)
                divergence = ((1.0 / a) * (torch.log(q_a / (p_a + 1e-10) + 1e-10) + p_a / (q_a + 1e-10) - 1.0)).sum(1)
            else:
                p_a = p.pow(a)
                q_b = q.pow(b)
                p_ab = p.pow(ab_sum)
                q_ab = q.pow(ab_sum)
                divergence = (-(p_a * q_b - (a / ab_sum) * p_ab - (b / ab_sum) * q_ab) / (a * b)).sum(1)
            loss = divergence.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            epoch_loss += loss.item()
            batches += 1
        if batches > 0:
            final_loss = epoch_loss / batches
            print(f"    ABKD epoch {epoch + 1}/{epochs} — loss {final_loss:.4f}")
    return final_loss


# ── Stage 1: CE training on private data ─────────────────────────────────────

def ce_stage(model_wrapper, private_dataset, epochs: int) -> None:
    print(f"    CE training for {epochs} epochs")
    history = model_wrapper.fit(private_dataset, epochs=epochs, verbose=1)
    if hasattr(history, "history") and "loss" in history.history:
        print(f"      final CE loss: {history.history['loss'][-1]:.4f}")


# ── Strategy ─────────────────────────────────────────────────────────────────

class Ours(DistillationStrategy):
    name = "Ours"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.kd_epochs = getattr(config, "kd_epochs", 2)
        self.ce_epochs = config.epochs
        self.eva_mode = _eva_mode(config)
        if getattr(config, "robust_filter_cronus", False):
            self.robust_filter = CronusRobustFilter(
                budget=getattr(config, "robust_rm_budget", None) or 0,
                workers=getattr(config, "robust_workers", 8),
            )
        else:
            self.robust_filter = RobustFilterV3(
                robust_threshold=getattr(config, "robust_threshold", 0.3),
                reference=getattr(config, "robust_filter_reference", "Blom"),
                t_df=getattr(config, "robust_v", 4.0),
                workers=getattr(config, "robust_workers", 8),
            )

    def extra_log_tokens(self) -> Dict[str, float]:
        budget = getattr(self.config, "robust_rm_budget", 0)
        tokens = {"kd": self.config.ours_kd, "ekd_lambda": self.config.ours_ekd_lambda, "budget": budget}
        if self.eva_mode is not None:
            tokens["eva_mode"] = self.eva_mode
        if getattr(self.config, "robust_filter_cronus", False):
            tokens["robust_filter"] = "cronus"
        else:
            ref = getattr(self.config, "robust_filter_reference", "Blom").lower()
            tokens["robust_filter"] = "v3" if ref == "blom" else ref
            tokens["robust_threshold"] = getattr(self.config, "robust_threshold", 0.3)
            if ref == "t":
                tokens["robust_v"] = getattr(self.config, "robust_v", 4.0)
        if self.config.ours_kd == "abkd":
            tokens.update({"ab_alpha": self.config.ab_alpha, "ab_beta": self.config.ab_beta, "temperature": self.config.ours_temperature})
        return tokens

    def setup(self, context: PipelineContext) -> None:
        config = context.config
        kd_method = getattr(config, "ours_kd", "ekd")

        print(f"{COLORS.OKGREEN}Preparing Ours ({kd_method.upper()}){COLORS.ENDC}")

        stem = os.path.splitext(os.path.basename(context.log_filename))[0]
        no_cache = getattr(config, 'no_disk_cache', False)
        print(f"{COLORS.WARNING}[DEBUG] Ours.setup: no_disk_cache={no_cache}, stem={stem}{COLORS.ENDC}")
        if no_cache:
            self.cache_dir = tempfile.mkdtemp(prefix=f"ours_{stem}_", dir="/dev/shm")
            print(f"{COLORS.OKGREEN}[DEBUG] Created cache_dir in /dev/shm: {self.cache_dir}{COLORS.ENDC}")
        else:
            self.cache_dir = os.path.join("temp_weights", f"{stem}_weight_record", "ours_cache")
            print(f"{COLORS.WARNING}[DEBUG] Using disk cache_dir: {self.cache_dir}{COLORS.ENDC}")

        public_unlabeled_ds, total_public = load_public_dataset_from_clients(
            context.paths, batch_size=config.batch_size, num_classes=context.num_classes,
            shuffle=False, return_labels=False,
        )
        public_features = numpy_from_dataset(public_unlabeled_ds)
        del public_unlabeled_ds

        if no_cache:
            context.shared_state.update({
                "public_features": public_features,
                "public_sample_count": total_public,
            })
        else:
            os.makedirs(self.cache_dir, exist_ok=True)
            pub_path = os.path.join(self.cache_dir, "public_features.npy")
            np.save(pub_path, public_features)
            del public_features
            context.shared_state.update({
                "public_features_path": pub_path,
                "public_sample_count": total_public,
            })

        for client_id, paths in enumerate(context.paths):
            context.add_client_state(client_id, None, paths)
            aggressive_memory_cleanup()

    def _generate_logits(self, context, public_features, first_client=0):
        config = context.config
        pool = context.model_pool
        logit_files = []
        logit_shape = None
        eva_mode = self.eva_mode
        attack_type, poison_value, _ = parse_poison_config(getattr(config, "poison", None))
        cleanup_interval = min(getattr(config, 'cleanup_interval', 10), len(context.client_states))

        for skipped_idx in range(first_client):
            fpath = os.path.join(self.cache_dir, f"client_{context.client_states[skipped_idx].client_id}.bin")
            if os.path.exists(fpath):
                logit_files.append(fpath)

        _pfl_gen = getattr(context, 'poisoned_fl_state', None)
        _ghost_bytes = None
        _ghost_shape = None
        _ghost_warm = poisonedfl_warmstart_weights(context.shared_state, _pfl_gen, fallback=context.shared_state.get("init_w"), prefer_poisoned=True) if _pfl_gen is not None else None
        if _pfl_gen is not None and context.poisoned_clients and _ghost_warm is not None:
            print(f"{COLORS.WARNING}  [PoisonedFL] Ghost model generating logits for {len(context.poisoned_clients)} byzantine clients{COLORS.ENDC}")
            _gm = create_model(context.input_dim, context.num_classes, config.batch_size, model_type=poisonedfl_ghost_model_type(config))
            _gm.set_weights(_ghost_warm)
            _lm = _gm.get_logits_model() if hasattr(_gm, "get_logits_model") else _gm
            _garr = _sanitize_logits(_lm.predict(public_features, batch_size=config.batch_size, verbose=0))
            _ghost_bytes = _garr.tobytes()
            _ghost_shape = _garr.shape
            del _gm, _lm, _garr
            aggressive_memory_cleanup()

        _lma_adv_bytes = None
        _lma_adv_shape = None
        _lma_adv_log = None
        if attack_type == "lma" and context.poisoned_clients:
            stale = context.shared_state.get("lma_stale_consensus")
            if stale is not None:
                _adv = _lma_logits(stale, raw=True)
                _lma_adv_bytes = _adv.astype(np.float32).tobytes()
                _lma_adv_shape = _adv.shape
                _hcounts = np.bincount(np.argmax(_adv, axis=1), minlength=context.num_classes)
                _htop3 = np.argsort(_hcounts)[::-1][:3]
                _lma_adv_log = " | ".join(f"cls{c}:{100*_hcounts[c]/max(len(_adv),1):.1f}%" for c in _htop3 if _hcounts[c] > 0)
                del _adv, _hcounts

        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_client:
                continue
            fpath = os.path.join(self.cache_dir, f"client_{state.client_id}.bin")
            if attack_type == "lma" and state.client_id in context.poisoned_clients:
                if _lma_adv_bytes is not None:
                    with open(fpath, "wb") as _hf:
                        _hf.write(_lma_adv_bytes)
                    if eva_mode is not None:
                        _write_support_values(_support_path(self.cache_dir, state.client_id), np.ones(_lma_adv_shape[0], dtype=np.float32))
                    logit_files.append(fpath)
                    if logit_shape is None:
                        logit_shape = _lma_adv_shape
                    context.logger.info("Round %s | Client %s [LMA] | top3 adv-argmax: %s", self._cur_round, state.client_id, _lma_adv_log)
                    continue
                # Round 1: no stale yet — skip entirely
                continue
            if _pfl_gen is not None and state.client_id in context.poisoned_clients:
                if _ghost_bytes is not None:
                    with open(fpath, 'wb') as _gf:
                        _gf.write(_ghost_bytes)
                    if eva_mode is not None:
                        _write_support_values(_support_path(self.cache_dir, state.client_id), np.ones(_ghost_shape[0], dtype=np.float32))
                    logit_files.append(fpath)
                    if logit_shape is None:
                        logit_shape = _ghost_shape
                    _garr2 = np.frombuffer(_ghost_bytes, dtype=np.float32).reshape(_ghost_shape)
                    _gcounts = np.bincount(np.argmax(_garr2, axis=1), minlength=context.num_classes)
                    _gtop3 = np.argsort(_gcounts)[::-1][:3]
                    _gtop = " | ".join(f"cls{c}:{100*_gcounts[c]/max(_ghost_shape[0],1):.1f}%" for c in _gtop3 if _gcounts[c] > 0)
                    context.logger.info("Round %s | Client %s [POISONED] | top3 logit-argmax: %s", self._cur_round, state.client_id, _gtop)
                    del _garr2, _gcounts
                continue
            model = pool.checkout(state.client_id)
            if state.client_id in context.poisoned_clients and attack_type == "gradient_scale":
                poisoned_w = apply_gradient_scale_poison(model.get_weights(), poison_value)
                model.set_weights(poisoned_w)
                print(f"{COLORS.WARNING}  [POISON] Client {state.client_id}: gradient_scale applied (\u00d7{poison_value}){COLORS.ENDC}")
                context.logger.info("Round %s | Client %s [POISON] gradient_scale \u00d7%s applied", self._cur_round, state.client_id, poison_value)
            if eva_mode is not None:
                _, shape, support_stats = generate_logits_and_support_to_file(
                    model, state.paths["train_X"], public_features, config.batch_size, fpath, _support_path(self.cache_dir, state.client_id), eva_mode,
                )
                if eva_mode == "eva" or eva_mode == "eva2":
                    context.logger.info(
                        "Round %s | Client %s | %s | q95=%.4f support_rate=%.3f support_mean=%.3f",
                        self._cur_round,
                        state.client_id,
                        eva_mode.upper(),
                        support_stats["q95"],
                        support_stats["support_rate"],
                        support_stats["support_mean"],
                    )
                else:
                    context.logger.info(
                        "Round %s | Client %s | EVW | mu=%.4f sigma=%.4f support_rate=%.3f support_mean=%.3f",
                        self._cur_round,
                        state.client_id,
                        support_stats["mu"],
                        support_stats["sigma"],
                        support_stats["support_rate"],
                        support_stats["support_mean"],
                    )
            else:
                _, shape = generate_logits_to_file(
                    model, public_features, config.batch_size, fpath,
                )

            pool.release(model)
            logit_files.append(fpath)
            logit_shape = shape

            _n_classes = context.num_classes
            _row_bytes = _n_classes * 4
            with open(fpath, "rb") as _f:
                _raw = _f.read()
            _logits = np.frombuffer(_raw, dtype=np.float32).reshape(-1, _n_classes)
            _argmax = np.argmax(_logits, axis=1)
            _counts = np.bincount(_argmax, minlength=_n_classes)
            _top3 = np.argsort(_counts)[::-1][:3]
            _top = " | ".join(f"cls{c}:{100*_counts[c]/max(len(_argmax),1):.1f}%" for c in _top3 if _counts[c] > 0)
            _poison_tag = " [POISONED]" if state.client_id in context.poisoned_clients else ""
            context.logger.info("Round %s | Client %s%s | top3 logit-argmax: %s", self._cur_round, state.client_id, _poison_tag, _top)
            del _raw, _logits, _argmax, _counts

            print(f"  Client {state.client_id}: logits {shape} -> {fpath}")
            if (client_idx + 1) % cleanup_interval == 0:
                aggressive_memory_cleanup()

            if self._ckpt and (
                client_idx == len(context.client_states) - 1
                or (client_idx + 1) % self._ckpt == 0
            ):
                save_mid_round(context, "ours", {
                    "round": self._cur_round,
                    "stage": "logits",
                    "last_client_idx": client_idx,
                    "logit_files": logit_files,
                })

        # If all clients were already done (checkpoint resume), derive shape
        # from the first recovered file so callers never receive None.
        if logit_shape is None and logit_files:
            n_classes = context.num_classes
            row_bytes = os.path.getsize(logit_files[0])
            logit_shape = (row_bytes // (n_classes * 4), n_classes)

        return logit_files, logit_shape

    def _generate_logits_mean(self, context, public_features):
        """Accumulate mean consensus in-memory during logit generation (no file I/O)."""
        config = context.config
        pool = context.model_pool
        n_clients = len(context.client_states)
        _pfl_m = getattr(context, 'poisoned_fl_state', None)
        attack_type, poison_value, _ = parse_poison_config(getattr(config, "poison", None))
        _lma = attack_type == "lma"
        accumulator = None
        n_benign = 0
        cleanup_interval = min(getattr(config, 'cleanup_interval', 10), n_clients)

        for client_idx, state in enumerate(context.client_states):
            if _pfl_m is not None and state.client_id in context.poisoned_clients:
                continue
            if _lma and state.client_id in context.poisoned_clients:
                continue
            model = pool.checkout(state.client_id)
            logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else model
            logits = _sanitize_logits(logits_model.predict(public_features, batch_size=config.batch_size, verbose=0))
            if accumulator is None:
                accumulator = logits.astype(np.float64)
            else:
                accumulator += logits
            n_benign += 1
            pool.release(model)
            del logits
            print(f"  Client {state.client_id}: logits accumulated ({n_benign})")
            if (client_idx + 1) % cleanup_interval == 0:
                aggressive_memory_cleanup()

        _ghost_warm = poisonedfl_warmstart_weights(context.shared_state, _pfl_m, fallback=context.shared_state.get("init_w"), prefer_poisoned=True) if _pfl_m is not None else None
        if _pfl_m is not None and context.poisoned_clients and _ghost_warm is not None:
            _gm = create_model(context.input_dim, context.num_classes, config.batch_size, model_type=poisonedfl_ghost_model_type(config))
            _gm.set_weights(_ghost_warm)
            _lm = _gm.get_logits_model() if hasattr(_gm, "get_logits_model") else _gm
            _gl = _sanitize_logits(_lm.predict(public_features, batch_size=config.batch_size, verbose=0)).astype(np.float64)
            n_byz = len(context.poisoned_clients)
            if accumulator is None:
                accumulator = _gl * n_byz
            else:
                accumulator += _gl * n_byz
            n_benign += n_byz
            del _gm, _lm, _gl
        if _lma and context.poisoned_clients:
            stale = context.shared_state.get("lma_stale_consensus")
            if stale is not None:
                n_byz = len(context.poisoned_clients)
                adv = _lma_logits(stale, raw=True)
                if accumulator is None:
                    accumulator = adv.astype(np.float64) * n_byz
                else:
                    accumulator += adv.astype(np.float64) * n_byz
                n_benign += n_byz
                del adv
        accumulator /= max(n_benign, 1)
        return accumulator.astype(np.float32)

    def _run_kd_stage(self, context, consensus_logits, public_features, first_client=0):
        config = context.config
        kd_method = getattr(config, "ours_kd", "ekd")
        pool = context.model_pool
        cleanup_interval = min(getattr(config, 'cleanup_interval', 10), len(context.client_states))

        if len(public_features) == 0:
            context.logger.info("Round %s | KD skipped (0 supported public samples)", self._cur_round)
            print(f"{COLORS.WARNING}Round {self._cur_round}: KD skipped (0 supported public samples){COLORS.ENDC}")
            return

        _pfl_kd = getattr(context, 'poisoned_fl_state', None)
        attack_type, _, _ = parse_poison_config(getattr(config, "poison", None))
        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_client:
                continue
            _poisoned = state.client_id in context.poisoned_clients
            if _poisoned and (_pfl_kd is not None or attack_type == "lma"):
                _tag = "PoisonedFL" if _pfl_kd is not None else "LMA"
                context.logger.info("Round %s | Client %s [%s] KD skipped", self._cur_round, state.client_id, _tag)
                continue
            print(f"\n{COLORS.BOLD}Client {state.client_id} — Stage 2 ({kd_method.upper()}){COLORS.ENDC}")
            model = pool.checkout(state.client_id)
            if kd_method == "abkd":
                abkd_stage(
                    model, consensus_logits, public_features,
                    config.batch_size, self.kd_epochs,
                    config.ab_alpha, config.ab_beta, config.ours_temperature,
                )
            else:
                ekd_stage(
                    model, consensus_logits, public_features,
                    config.batch_size, self.kd_epochs,
                    getattr(config, "ours_ekd_lambda", 1.0),
                )
            pool.checkin(state.client_id, model)
            if (client_idx + 1) % cleanup_interval == 0:
                aggressive_memory_cleanup()

            if self._ckpt and (
                client_idx == len(context.client_states) - 1
                or (client_idx + 1) % self._ckpt == 0
            ):
                save_mid_round(context, "ours", {
                    "round": self._cur_round,
                    "stage": "kd",
                    "last_client_idx": client_idx,
                })

    def _run_ce_stage(self, context, first_client=0):
        config = context.config
        pool = context.model_pool
        _pfl_ce = getattr(context, 'poisoned_fl_state', None)
        attack_type, _, _ = parse_poison_config(getattr(config, "poison", None))
        cleanup_interval = min(getattr(config, 'cleanup_interval', 10), len(context.client_states))
        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_client:
                continue
            print(f"\n{COLORS.BOLD}Client {state.client_id} — Stage 1 (CE){COLORS.ENDC}")
            _poisoned = state.client_id in context.poisoned_clients
            if _poisoned and (_pfl_ce is not None or attack_type == "lma"):
                _tag = "PoisonedFL" if _pfl_ce is not None else "LMA"
                context.logger.info("Round %s | Client %s [%s] CE skipped", self._cur_round, state.client_id, _tag)
                continue
            model = pool.checkout(state.client_id)
            poison_loader = context.per_client_loaders.get(state.client_id, context.poison_loader) if _poisoned else None
            private_dataset = create_private_dataset(
                state.paths["train_X"], state.paths["train_y"],
                context.input_dim, context.num_classes, config.batch_size,
                poison_loader=poison_loader,
            )
            ce_stage(model, private_dataset, self.ce_epochs)
            pool.checkin(state.client_id, model)
            del private_dataset
            if (client_idx + 1) % cleanup_interval == 0:
                aggressive_memory_cleanup()

            if self._ckpt and (
                client_idx == len(context.client_states) - 1
                or (client_idx + 1) % self._ckpt == 0
            ):
                save_mid_round(context, "ours", {
                    "round": self._cur_round,
                    "stage": "ce",
                    "last_client_idx": client_idx,
                })

    def _run_poisonedfl_ghost_stage(self, context, consensus_logits, public_features, round_number: int) -> None:
        _pfl = getattr(context, 'poisoned_fl_state', None)
        if _pfl is None or not context.poisoned_clients:
            return
        config = context.config
        _kd_m = getattr(config, "ours_kd", "ekd")
        ghost = create_model(context.input_dim, context.num_classes, config.batch_size, model_type=poisonedfl_ghost_model_type(config))
        ghost_start = poisonedfl_warmstart_weights(context.shared_state, _pfl, fallback=context.shared_state.get("init_w"))
        if ghost_start is not None:
            ghost.set_weights(ghost_start)
        if len(public_features) > 0:
            if _kd_m == "abkd":
                ghost_distill_loss = abkd_stage(ghost, consensus_logits, public_features, config.batch_size, self.kd_epochs,
                                                config.ab_alpha, config.ab_beta, config.ours_temperature)
            else:
                ghost_distill_loss = ekd_stage(ghost, consensus_logits, public_features, config.batch_size, self.kd_epochs,
                                               getattr(config, "ours_ekd_lambda", 1.0))
            ghost_w = ghost.get_weights()
        else:
            ghost_distill_loss = None
            ghost_w = ghost.get_weights()
            context.logger.info("Round %s | EVA | PoisonedFL ghost KD skipped (0 supported public samples)", round_number)
            print(f"{COLORS.WARNING}Round {round_number}: PoisonedFL ghost KD skipped (0 supported public samples){COLORS.ENDC}")
        del ghost
        poisoned_w = poisonedfl_unified_weights([ghost_w], _pfl)
        poisonedfl_store_round_weights(context.shared_state, ghost_w, poisoned_w, _pfl)
        _distill_loss, _c0, _c, _mal_norm, _alignment = poisonedfl_log_values(_pfl, ghost_distill_loss)
        context.logger.info("Round %s | PoisonedFL | distill_loss=%s c0=%.4f c=%.4f mal_norm=%.4e aligned=%s | Ghost KD'd, injected into %d byzantine clients", round_number, _distill_loss, _c0, _c, _mal_norm, _alignment, len(context.poisoned_clients))
        if not getattr(config, "mixed_models", False):
            _pool_g = context.model_pool
            for st in context.client_states:
                if st.client_id in context.poisoned_clients:
                    m = _pool_g.checkout(st.client_id)
                    m.set_weights(poisoned_w)
                    _pool_g.checkin(st.client_id, m)
        del ghost_w, poisoned_w
        aggressive_memory_cleanup()

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        no_cache = getattr(config, 'no_disk_cache', False)
        if no_cache:
            public_features = context.shared_state["public_features"]
        else:
            public_features = np.load(context.shared_state["public_features_path"], mmap_mode="r")
        eva_mode = self.eva_mode
        no_filter = not isinstance(self.robust_filter, RobustFilterV3) and getattr(config, "robust_rm_budget", 0) == 0 and eva_mode is None
        _pfl = getattr(context, 'poisoned_fl_state', None)

        os.makedirs(self.cache_dir, exist_ok=True)

        self._ckpt = getattr(config, "checkpoint", 0)
        self._cur_round = round_number

        skip_ce = False
        skip_logits = False
        skip_kd = False
        first_ce = 0
        first_logit = 0
        first_kd = 0
        saved_logit_files = None

        if self._ckpt:
            mid = load_mid_round(context, "ours", round_number)
            if mid is not None:
                stage = mid.get("stage", "ce")
                last = mid["last_client_idx"]
                if stage == "ce":
                    first_ce = last + 1
                elif stage == "logits":
                    skip_ce = True
                    first_logit = last + 1
                    saved_logit_files = mid.get("logit_files", [])
                elif stage == "kd":
                    skip_ce = True
                    skip_logits = True
                    first_kd = last + 1
                print(f"{COLORS.OKGREEN}Resuming round {round_number} stage={stage} from client {last + 1}{COLORS.ENDC}")

        if not skip_ce:
            self._run_ce_stage(context, first_client=first_ce)

        if no_filter:
            consensus_path = os.path.join(self.cache_dir, f"r{round_number}_consensus.npy")

            # Stage 1b — generate or load mean consensus
            if not skip_logits:
                print(f"\n{COLORS.OKCYAN}Generating logits + mean consensus in-memory (no filtering){COLORS.ENDC}")
                consensus_logits = self._generate_logits_mean(context, public_features)
                print(f"  Consensus shape: {consensus_logits.shape}")
                context.logger.info("Round %s | MeanConsensus (no filter) | shape=%s", round_number, consensus_logits.shape)
                np.save(consensus_path, consensus_logits)
            elif os.path.exists(consensus_path):
                print(f"\n{COLORS.OKCYAN}Loading cached consensus logits for KD resume{COLORS.ENDC}")
                consensus_logits = np.load(consensus_path)
            else:
                # Cache lost — regenerate from pool (post-CE weights still intact)
                print(f"\n{COLORS.OKCYAN}Regenerating mean consensus for KD resume{COLORS.ENDC}")
                consensus_logits = self._generate_logits_mean(context, public_features)

            context.shared_state["lma_stale_consensus"] = consensus_logits.copy()

            # Stage 2 — KD distillation
            self._run_kd_stage(context, consensus_logits, public_features, first_client=first_kd)
            if _pfl is not None and context.poisoned_clients:
                _kd_m = getattr(config, "ours_kd", "ekd")
                ghost = create_model(context.input_dim, context.num_classes, config.batch_size, model_type=poisonedfl_ghost_model_type(config))
                ghost_start = poisonedfl_warmstart_weights(context.shared_state, _pfl, fallback=context.shared_state.get("init_w"))
                if ghost_start is not None:
                    ghost.set_weights(ghost_start)
                if _kd_m == "abkd":
                    ghost_distill_loss = abkd_stage(ghost, consensus_logits, public_features, config.batch_size, self.kd_epochs,
                                                    config.ab_alpha, config.ab_beta, config.ours_temperature)
                else:
                    ghost_distill_loss = ekd_stage(ghost, consensus_logits, public_features, config.batch_size, self.kd_epochs,
                                                   getattr(config, "ours_ekd_lambda", 1.0))
                ghost_w = ghost.get_weights()
                del ghost
                poisoned_w = poisonedfl_unified_weights([ghost_w], _pfl)
                poisonedfl_store_round_weights(context.shared_state, ghost_w, poisoned_w, _pfl)
                _distill_loss, _c0, _c, _mal_norm, _alignment = poisonedfl_log_values(_pfl, ghost_distill_loss)
                context.logger.info("Round %s | PoisonedFL | distill_loss=%s c0=%.4f c=%.4f mal_norm=%.4e aligned=%s | Ghost KD'd, injected into %d byzantine clients", round_number, _distill_loss, _c0, _c, _mal_norm, _alignment, len(context.poisoned_clients))
                if not getattr(config, "mixed_models", False):
                    _pool_g = context.model_pool
                    for st in context.client_states:
                        if st.client_id in context.poisoned_clients:
                            m = _pool_g.checkout(st.client_id)
                            m.set_weights(poisoned_w)
                            _pool_g.checkin(st.client_id, m)
                del ghost_w, poisoned_w
                aggressive_memory_cleanup()
            del consensus_logits
            if os.path.exists(consensus_path):
                os.remove(consensus_path)
        else:
            if not skip_logits:
                print(f"\n{COLORS.OKCYAN}Generating public logits{COLORS.ENDC}")
                logit_files, logit_shape = self._generate_logits(context, public_features, first_client=first_logit)
            else:
                logit_files = saved_logit_files or []
                logit_shape = None
                if logit_files:
                    row_bytes_test = os.path.getsize(logit_files[0])
                    n_classes = context.num_classes
                    logit_shape = (row_bytes_test // (n_classes * 4), n_classes)

            if not skip_kd:
                consensus_path = os.path.join(self.cache_dir, f"r{round_number}_consensus.npy")
                support_mask_path = os.path.join(self.cache_dir, f"r{round_number}_supported_mask.npy")
                if logit_shape is None and not os.path.exists(consensus_path):
                    raise RuntimeError(
                        "logit_shape is None — no logit files were produced or recovered. "
                        "Check that client logit .bin files exist in the cache directory."
                    )
                if skip_logits and os.path.exists(consensus_path):
                    print(f"\n{COLORS.OKCYAN}Loading cached consensus logits{COLORS.ENDC}")
                    consensus_logits = np.load(consensus_path)
                    if eva_mode is not None and os.path.exists(support_mask_path):
                        supported_mask = np.load(support_mask_path).astype(bool)
                    else:
                        supported_mask = np.ones(len(consensus_logits), dtype=bool)
                else:
                    budget = getattr(config, "robust_rm_budget", 0)
                    if eva_mode is not None:
                        support_files = [_support_path(self.cache_dir, _client_id_from_logit_path(path)) for path in logit_files]
                        robust_filter = None if budget == 0 else self.robust_filter
                        label = "weighted" if eva_mode == "evw" else "abstention"
                        print(f"\n{COLORS.OKCYAN}Computing {label} consensus logits ({eva_mode.upper()}){COLORS.ENDC}")
                        consensus_logits, supported_mask, max_eig, max_ratio, removal_counts, eig_report = compute_supported_consensus_from_files(
                            logit_files, support_files, logit_shape, eva_mode, robust_filter,
                            poisoned_client_ids=(None if _pfl is not None else context.poisoned_clients),
                            logger=context.logger,
                            round_number=round_number,
                        )
                        unsupported = int((~supported_mask).sum())
                        print(f"  Consensus shape: {consensus_logits.shape}")
                        print(f"  unsupported rows={unsupported}/{len(supported_mask)} ({100 * unsupported / max(len(supported_mask), 1):.1f}%)")
                        if budget > 0:
                            eig_s = f"{max_eig:.6f}" if max_eig is not None else "N/A"
                            ratio_s = f"{max_ratio:.4f}" if max_ratio is not None else "N/A"
                            removal_label, removal_value = _format_removal_summary(self.robust_filter, removal_counts)
                            print(f"  max_eig={eig_s}  max_ratio={ratio_s}  {removal_label}: {removal_value}")
                            context.logger.info(
                                "Round %s | EVAConsensus | mode=%s | max_eig=%s | max_ratio=%s | removals=%s | unsupported=%d/%d",
                                round_number, eva_mode, eig_s, ratio_s, removal_value, unsupported, len(supported_mask),
                            )
                            if eig_report is not None:
                                print(f"  pre-filter eigs: min={eig_report['min_eig']:.6f} mean={eig_report['mean_eig']:.6f} med={eig_report['med_eig']:.6f} max={eig_report['max_eig']:.6f}")
                                context.logger.info(
                                    "Round %s | EVAConsensus | mode=%s | pre_filter_eigs | n_samples=%d | min_eig=%.6f | mean_eig=%.6f | med_eig=%.6f | max_eig=%.6f",
                                    round_number, eva_mode, eig_report["n_samples"], eig_report["min_eig"], eig_report["mean_eig"], eig_report["med_eig"], eig_report["max_eig"],
                                )
                        else:
                            context.logger.info(
                                "Round %s | EVAConsensus | mode=%s | budget=0 | unsupported=%d/%d",
                                round_number, eva_mode, unsupported, len(supported_mask),
                            )
                    elif budget == 0:
                        print(f"\n{COLORS.OKCYAN}Computing mean consensus logits (no filtering){COLORS.ENDC}")
                        consensus_logits = compute_consensus_from_files(logit_files, logit_shape)
                        supported_mask = np.ones(len(consensus_logits), dtype=bool)
                        print(f"  Consensus shape: {consensus_logits.shape}")
                        context.logger.info(
                            "Round %s | MeanConsensus | budget=0 (no filtering)",
                            round_number,
                        )
                    else:
                        print(f"\n{COLORS.OKCYAN}Computing robust consensus logits (budget={budget}){COLORS.ENDC}")
                        consensus_logits, max_eig, max_ratio, removal_counts, eig_report, v3_mask = compute_robust_consensus_from_files(
                            logit_files, logit_shape, self.robust_filter,
                            poisoned_client_ids=(None if _pfl is not None else context.poisoned_clients),
                            logger=context.logger,
                            round_number=round_number,
                        )
                        supported_mask = v3_mask if v3_mask is not None else np.ones(len(consensus_logits), dtype=bool)
                        print(f"  Consensus shape: {consensus_logits.shape}")
                        eig_s = f"{max_eig:.6f}" if max_eig is not None else "N/A"
                        ratio_s = f"{max_ratio:.4f}" if max_ratio is not None else "N/A"
                        removal_label, removal_value = _format_removal_summary(self.robust_filter, removal_counts)
                        print(f"  max_eig={eig_s}  max_ratio={ratio_s}  {removal_label}: {removal_value}")
                        context.logger.info(
                            "Round %s | RobustConsensus | max_eig=%s | max_ratio=%s | removals=%s",
                            round_number, eig_s, ratio_s, removal_value,
                        )
                        if eig_report is not None:
                            print(f"  pre-filter eigs: min={eig_report['min_eig']:.6f} mean={eig_report['mean_eig']:.6f} med={eig_report['med_eig']:.6f} max={eig_report['max_eig']:.6f}")
                            context.logger.info(
                                "Round %s | RobustConsensus | pre_filter_eigs | n_samples=%d | min_eig=%.6f | mean_eig=%.6f | med_eig=%.6f | max_eig=%.6f",
                                round_number, eig_report["n_samples"], eig_report["min_eig"], eig_report["mean_eig"], eig_report["med_eig"], eig_report["max_eig"],
                            )
                    np.save(consensus_path, consensus_logits)
                    if eva_mode is not None:
                        np.save(support_mask_path, supported_mask.astype(np.uint8))

                context.shared_state["lma_stale_consensus"] = consensus_logits.copy()

                for fpath in logit_files:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                if eva_mode is not None:
                    for fpath in [_support_path(self.cache_dir, _client_id_from_logit_path(path)) for path in logit_files]:
                        if os.path.exists(fpath):
                            os.remove(fpath)

                kd_public_features = np.asarray(public_features[supported_mask], dtype=np.float32) if not supported_mask.all() else np.asarray(public_features, dtype=np.float32)
                kd_consensus_logits = consensus_logits[supported_mask]
                self._run_kd_stage(context, kd_consensus_logits, kd_public_features, first_client=first_kd)
                self._run_poisonedfl_ghost_stage(context, kd_consensus_logits, kd_public_features, round_number)

                del consensus_logits
                del kd_public_features, kd_consensus_logits, supported_mask
                if os.path.exists(consensus_path):
                    os.remove(consensus_path)
                if eva_mode is not None and os.path.exists(support_mask_path):
                    os.remove(support_mask_path)

        del public_features
        aggressive_memory_cleanup()

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        if self._ckpt:
            clear_mid_round(context, "ours")

        if no_cache and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            print(f"{COLORS.WARNING}Cleaned up {self.cache_dir}{COLORS.ENDC}")

        return {}

    def finalize(self, context) -> None:
        """Clean up cache directory on experiment finalization."""
        if hasattr(self, 'cache_dir') and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)
