from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import time
from typing import Dict, List

import numpy as np
from ..backend import get_torch_loader_kwargs as _torch_loader_kwargs, use_tf as _use_tf
if _use_tf():
    import tensorflow as tf

from ..colors import COLORS
from ..data_utils import create_client_dataset
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext
from ..poison_utils import parse_poison_config, apply_gradient_scale_poison, poisonedfl_unified_weights
from .base import DistillationStrategy
from ._checkpoint import save_mid_round, load_mid_round, clear_mid_round
from .common import create_model, load_public_dataset_from_clients, lma_logits as _lma_logits, numpy_from_dataset

FEDEXPGUARD_CACHE_DIR = os.path.join("temp_weights", "fedexpguard_cache")


def _base_public_path():
    return os.path.join(FEDEXPGUARD_CACHE_DIR, "public_features.npy")


def _round_public_path(round_number):
    return os.path.join(FEDEXPGUARD_CACHE_DIR, f"r{round_number}_public_X.npy")


def _soft_pack_path(round_number):
    return os.path.join(FEDEXPGUARD_CACHE_DIR, f"r{round_number}_soft_pack.bin")


def _soft_stride(n_public, n_classes):
    return n_public * n_classes * 4


def _init_soft_pack(path, n_clients, n_public, n_classes):
    total_bytes = n_clients * _soft_stride(n_public, n_classes)
    mode = "r+b" if os.path.exists(path) else "w+b"
    with open(path, mode) as fp:
        fp.truncate(total_bytes)


def _read_soft_rows(fp, client_id, row_offset, rows, n_public, n_classes):
    row_bytes = n_classes * 4
    fp.seek(client_id * _soft_stride(n_public, n_classes) + row_offset * row_bytes)
    raw = fp.read(rows * row_bytes)
    return np.frombuffer(raw, dtype=np.float32).reshape(rows, n_classes)


def _predict_soft_labels(model, X, batch_size, output_path, client_id=None, n_public=None):
    src = model.get_logits_model() if hasattr(model, "get_logits_model") else model
    logits = src.predict(X, batch_size=batch_size, verbose=0)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    soft = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    if client_id is None:
        with open(output_path, "wb") as _f:
            _f.write(soft.tobytes())
        return
    with open(output_path, "r+b") as _f:
        _f.seek(client_id * _soft_stride(n_public, soft.shape[1]))
        _f.write(soft.tobytes())


def _top_eigenvector(cov, n_iter=100, eps=1e-10):
    """Power iteration for leading eigenvector. cov: [N, C, C] -> [N, C, 1]"""
    v = np.ones((cov.shape[0], cov.shape[1], 1), dtype=cov.dtype)
    for _ in range(n_iter):
        m = cov @ v
        nrm = np.linalg.norm(m, axis=1, keepdims=True).clip(min=eps)
        v = m / nrm
    return v


def _filter_outlier_scores(preds):
    """preds: [K, N, C]. Returns scores [N, K, 1] (projection onto top eigenvector)."""
    K, N, C = preds.shape
    p = preds.transpose(1, 2, 0)  # [N, C, K]
    mean_p = p.mean(axis=2, keepdims=True)  # [N, C, 1]
    centered = (p - mean_p).transpose(0, 2, 1)  # [N, K, C]
    cov = np.einsum('nkc,nkd->ncd', centered.astype(np.float64), centered.astype(np.float64)) / max(K - 1, 1)  # [N, C, C]
    ev = _top_eigenvector(cov).astype(preds.dtype)  # [N, C, 1]
    return centered @ ev  # [N, K, 1]


def _filtered_mean(preds, scores, threshold):
    """preds: [K, N, C], scores: [N, K, 1], threshold: [N]. Returns [N, C]."""
    p = preds.transpose(1, 2, 0)  # [N, C, K]
    mask = (np.abs(scores.squeeze(2)) <= threshold[:, None]).astype(preds.dtype)  # [N, K]
    n_kept = mask.sum(axis=1, keepdims=True).clip(min=1)  # [N, 1]
    filt = (p * mask[:, None, :]).sum(axis=2) / n_kept  # [N, C]
    all_zero = (mask.sum(axis=1) == 0)
    if all_zero.any():
        filt[all_zero] = p[all_zero].mean(axis=2)
    return filt  # [N, C]


_EXPGUARD_CHUNK = 10_000
_EXPGUARD_ROW_BLOCK = 2_048


def _expguard_pass1_block(chunks, quantile, k_keep):
    scores1 = _filter_outlier_scores(chunks)
    abs_s1 = np.abs(scores1.squeeze(2))
    rank = np.argsort(abs_s1, axis=1)
    keep_idx = rank[:, :k_keep]
    nkc = chunks.transpose(1, 0, 2)
    reduced = nkc[np.arange(chunks.shape[1])[:, None], keep_idx, :].transpose(1, 0, 2)
    scores2 = _filter_outlier_scores(reduced)
    abs_s2 = np.abs(scores2.squeeze(2))
    threshold = np.quantile(abs_s2, quantile, axis=1)
    con = _filtered_mean(reduced, scores2, threshold)
    return np.linalg.norm(chunks - con[None], axis=2).sum(axis=1).astype(np.float64)


def _expguard_pass2_block(chunks, w_norm):
    return np.einsum("krc,k->rc", chunks, w_norm)


def _expguard_aggregate(
    soft_pack_path: str,
    client_ids: List[int],
    exp_w: np.ndarray,
    rho: float,
    N_pub: int,
    n_classes: int,
    workers: int = 8,
    logger=None,
    round_number=None,
):
    K = len(client_ids)
    max_byz = (K // 2 - 1) if K % 2 == 0 else K // 2
    quantile = 1.0 - max(max_byz, 0) / (2.0 * K)
    k_keep = max(int(quantile * K), 1)
    row_bytes = n_classes * 4

    # ── Pass 1: streaming Cronus filter → Cronus consensus + per-client rho ──
    rho_sum = np.zeros(K, dtype=np.float64)
    fp = open(soft_pack_path, "rb")
    for off in range(0, N_pub, _EXPGUARD_CHUNK):
        rows = min(_EXPGUARD_CHUNK, N_pub - off)
        chunks = np.stack([
            _read_soft_rows(fp, cid, off, rows, N_pub, n_classes)
            for cid in client_ids
        ], axis=0)  # [K, rows, C]

        if workers > 1 and rows > _EXPGUARD_ROW_BLOCK:
            blocks = [(start, min(start + _EXPGUARD_ROW_BLOCK, rows)) for start in range(0, rows, _EXPGUARD_ROW_BLOCK)]
            with ThreadPoolExecutor(max_workers=min(workers, len(blocks))) as pool:
                futures = [
                    pool.submit(_expguard_pass1_block, chunks[:, start:end, :], quantile, k_keep)
                    for start, end in blocks
                ]
                for future in futures:
                    rho_sum += future.result()
        else:
            rho_sum += _expguard_pass1_block(chunks, quantile, k_keep)
        del chunks
    fp.close()

    mean_rho = rho_sum / N_pub  # [K]

    # ── Update exponential weights ────────────────────────────────────────────
    new_w = exp_w.copy()
    for i, cid in enumerate(client_ids):
        new_w[cid] = exp_w[cid] * np.exp(-rho * mean_rho[i])
        if logger is not None:
            logger.info(
                "Round %s | ExpGuard | client=%d | outlier=%.4f | w %.4e -> %.4e",
                round_number, cid, mean_rho[i], float(exp_w[cid]), float(new_w[cid]),
            )

    # ── Pass 2: streaming weighted aggregation ────────────────────────────────
    w_sub = np.array([new_w[cid] for cid in client_ids], dtype=np.float32)
    w_norm = w_sub / w_sub.sum()  # [K]
    consensus = np.zeros((N_pub, n_classes), dtype=np.float32)
    fp = open(soft_pack_path, "rb")
    for off in range(0, N_pub, _EXPGUARD_CHUNK):
        rows = min(_EXPGUARD_CHUNK, N_pub - off)
        chunks = np.stack([
            _read_soft_rows(fp, cid, off, rows, N_pub, n_classes)
            for cid in client_ids
        ], axis=0)  # [K, rows, C]

        if workers > 1 and rows > _EXPGUARD_ROW_BLOCK:
            blocks = [(start, min(start + _EXPGUARD_ROW_BLOCK, rows)) for start in range(0, rows, _EXPGUARD_ROW_BLOCK)]
            with ThreadPoolExecutor(max_workers=min(workers, len(blocks))) as pool:
                futures = [
                    (start, end, pool.submit(_expguard_pass2_block, chunks[:, start:end, :], w_norm))
                    for start, end in blocks
                ]
                for start, end, future in futures:
                    consensus[off + start:off + end] = future.result()
        else:
            consensus[off:off + rows] = _expguard_pass2_block(chunks, w_norm)
        del chunks
    fp.close()

    return consensus, new_w


def _train_on_soft_labels_pt(model_wrapper, X, soft_y, batch_size, epochs):
    import torch
    import torch.nn.functional as F
    from ..backend import get_torch_device
    from torch.utils.data import DataLoader, TensorDataset
    dev = get_torch_device()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(soft_y)),
        batch_size=batch_size, shuffle=True, **_torch_loader_kwargs(),
    )
    net = model_wrapper.nn
    opt = model_wrapper.optimizer
    net.to(dev).train()
    for ep in range(epochs):
        total, n = 0.0, 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(dev), y_b.to(dev)
            logits = net(X_b, return_logits=True)
            loss = (-y_b * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"    epoch {ep + 1}/{epochs}  loss={total / max(n, 1):.4f}")


def _train_on_soft_labels(model_wrapper, X, soft_y, batch_size, epochs):
    if not _use_tf():
        return _train_on_soft_labels_pt(model_wrapper, X, soft_y, batch_size, epochs)
    ds = (tf.data.Dataset.from_tensor_slices((X, soft_y))
          .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    model_wrapper.fit(ds, epochs=epochs, verbose=0)


class FedDistillExpGuard(DistillationStrategy):
    name = "FedDistill"
    has_global_model = True
    use_model_pool = False

    def extra_log_tokens(self) -> Dict:
        return {"exp_rho": getattr(self.config, "exp_rho", 0.05)}

    def setup(self, context: PipelineContext) -> None:
        config = context.config
        os.makedirs(FEDEXPGUARD_CACHE_DIR, exist_ok=True)

        pub_ds, total_public = load_public_dataset_from_clients(
            context.paths, batch_size=config.batch_size,
            num_classes=context.num_classes, shuffle=False, return_labels=False,
        )
        pub_feats = numpy_from_dataset(pub_ds)
        del pub_ds
        np.save(_base_public_path(), pub_feats)
        context.shared_state["public_sample_count"] = pub_feats.shape[0]
        del pub_feats

        from ..memory import clear_session
        clear_session()

        global_model = create_model(context.input_dim, context.num_classes,
                                    config.batch_size, model_type=config.model_type)
        context.shared_state["global_w"] = global_model.get_weights()
        context.shared_state["global_model"] = global_model

        n = len(context.paths)
        if "exp_weights" not in context.shared_state:
            context.shared_state["exp_weights"] = np.ones(n, dtype=np.float64)

        for client_id, paths in enumerate(context.paths):
            st = context.add_client_state(client_id, None, paths)
            st.data["w"] = None

        print(f"{COLORS.OKGREEN}FedDistill+ExpGuard initialized ({n} clients, rho={getattr(config, 'exp_rho', 0.05)}){COLORS.ENDC}")

    def run_round(self, context: PipelineContext, round_number: int) -> Dict:
        round_start = time.time()
        config = context.config
        n_public = context.shared_state["public_sample_count"]
        exp_w = context.shared_state["exp_weights"]
        rho = getattr(config, "exp_rho", 0.05)
        global_w = context.shared_state["global_w"]
        attack_type, poison_value, _ = parse_poison_config(getattr(config, "poison", None))
        _ckpt = getattr(config, "checkpoint", 0)
        _REFRESH_EVERY = getattr(config, "cleanup_interval", 25)

        # ---- public data (consistent within the round) ----
        pub_X_path = _round_public_path(round_number)
        if os.path.exists(pub_X_path):
            pub_X = np.array(np.load(pub_X_path, mmap_mode="r"), dtype=np.float32)
        else:
            base_mmap = np.load(_base_public_path(), mmap_mode="r")
            perm = np.random.permutation(n_public)
            pub_X = np.array(base_mmap[perm], dtype=np.float32)
            del base_mmap
            np.save(pub_X_path, pub_X)

        # ---- checkpoint resume ----
        soft_pack_path = _soft_pack_path(round_number)
        soft_client_ids: List[int] = []
        first_client = 0
        resume_pack = False
        if _ckpt:
            mid = load_mid_round(context, "fedexpguard", round_number)
            if mid is not None:
                candidate_path = mid.get("soft_pack_path", soft_pack_path)
                if os.path.exists(candidate_path):
                    first_client = mid["last_client_idx"] + 1
                    soft_pack_path = candidate_path
                    soft_client_ids = mid.get("soft_client_ids", [])
                    resume_pack = True
                    print(f"{COLORS.OKGREEN}Resuming round {round_number} from client {first_client}{COLORS.ENDC}")
        if not resume_pack:
            _init_soft_pack(soft_pack_path, context.n_clients, n_public, context.num_classes)

        print(f"\n{COLORS.HEADER}Round {round_number} Stage I (local training + soft labels){COLORS.ENDC}")

        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_client:
                continue

            if client_idx > 0 and client_idx % _REFRESH_EVERY == 0:
                from ..memory import clear_session
                clear_session()
                aggressive_memory_cleanup()
                pub_X = np.array(np.load(pub_X_path, mmap_mode="r"), dtype=np.float32)

            cid = state.client_id
            _poisoned = cid in context.poisoned_clients
            print(f"\n{COLORS.BOLD}Client {cid}{COLORS.ENDC}")

            client_model = create_model(context.input_dim, context.num_classes,
                                        config.batch_size, model_type=config.model_type)
            client_model.set_weights(global_w)

            if _poisoned and attack_type in ("poisonedfl", "lma"):
                _tag = "PoisonedFL" if attack_type == "poisonedfl" else "LMA"
                context.logger.info("Round %s | Client %s [%s] CE skipped", round_number, cid, _tag)
                del client_model
                continue

            _poison_loader = context.per_client_loaders.get(cid, context.poison_loader) if _poisoned else None
            ds = create_client_dataset(
                state.paths["train_X"], state.paths["train_y"],
                context.input_dim, context.num_classes, config.batch_size,
                poison_loader=_poison_loader,
            )
            client_model.fit(ds, epochs=config.epochs, verbose=0)
            del ds
            new_w = client_model.get_weights()

            if _poisoned and attack_type == "gradient_scale":
                new_w = apply_gradient_scale_poison(new_w, poison_value)
                client_model.set_weights(new_w)
                print(f"{COLORS.WARNING}  [POISON] Client {cid}: gradient_scale ×{poison_value}{COLORS.ENDC}")
                context.logger.info("Round %s | Client %s [POISON] gradient_scale ×%s", round_number, cid, poison_value)

            state.data["w"] = new_w

            _predict_soft_labels(client_model, pub_X, config.batch_size, soft_pack_path, client_id=cid, n_public=n_public)
            del client_model
            if cid not in soft_client_ids:
                soft_client_ids.append(cid)

            if _ckpt and (
                client_idx == len(context.client_states) - 1
                or (client_idx + 1) % _ckpt == 0
            ):
                save_mid_round(context, "fedexpguard", {
                    "round": round_number,
                    "last_client_idx": client_idx,
                    "soft_pack_path": soft_pack_path,
                    "soft_client_ids": soft_client_ids,
                })

        # ---- LMA: ghost soft labels (compute once, copy to all byzantine clients) ----
        if attack_type == "lma" and context.poisoned_clients:
            stale = context.shared_state.get("lma_stale_consensus")
            if stale is not None:
                _lma_adv = _lma_logits(stale)
                _lma_bytes = _lma_adv.astype(np.float32).tobytes()
                for st in context.client_states:
                    if st.client_id not in context.poisoned_clients:
                        continue
                    with open(soft_pack_path, "r+b") as _cf:
                        _cf.seek(st.client_id * _soft_stride(n_public, context.num_classes))
                        _cf.write(_lma_bytes)
                    if st.client_id not in soft_client_ids:
                        soft_client_ids.append(st.client_id)
                context.logger.info(
                    "Round %s | LMA | Byzantine soft labels generated for %d clients",
                    round_number, len(context.poisoned_clients),
                )
                del _lma_adv, _lma_bytes

        # ---- PoisonedFL: poison the current global model once, then copy logits ----
        _pfl = getattr(context, "poisoned_fl_state", None)
        if _pfl is not None and context.poisoned_clients:
            poisoned_w = poisonedfl_unified_weights([global_w], _pfl)
            context.shared_state["poisonedfl_ghost_w"] = [w.copy() for w in poisoned_w]
            poison_model = create_model(context.input_dim, context.num_classes,
                                        config.batch_size, model_type=config.model_type)
            poison_model.set_weights(poisoned_w)
            for st in context.client_states:
                if st.client_id not in context.poisoned_clients:
                    continue
                _predict_soft_labels(poison_model, pub_X, config.batch_size, soft_pack_path, client_id=st.client_id, n_public=n_public)
                if st.client_id not in soft_client_ids:
                    soft_client_ids.append(st.client_id)
            del poison_model
            _mal_norm = float(np.linalg.norm(_pfl.cached_update)) if _pfl.cached_update is not None else 0.0
            context.logger.info(
                "Round %s | PoisonedFL | c=%.4f mal_norm=%.4e | Ghost global model generated soft labels for %d clients",
                round_number, _pfl.scaling_factor, _mal_norm, len(context.poisoned_clients),
            )

        from ..memory import clear_session
        clear_session()
        aggressive_memory_cleanup()

        # ---- ExpGuard aggregation ----
        print(f"\n{COLORS.HEADER}Round {round_number} ExpGuard (rho={rho}){COLORS.ENDC}")
        consensus, new_exp_w = _expguard_aggregate(
            soft_pack_path, soft_client_ids, exp_w, rho, n_public, context.num_classes,
            workers=getattr(config, "robust_workers", 8),
            logger=context.logger, round_number=round_number,
        )
        context.shared_state["exp_weights"] = new_exp_w
        context.shared_state["lma_stale_consensus"] = consensus.copy()

        if context.poisoned_clients:
            poisoned_set = set(context.poisoned_clients)
            rejected = [cid for cid in soft_client_ids if new_exp_w[cid] < 1e-10]
            TP = len(set(rejected) & poisoned_set)
            FP = len(set(rejected) - poisoned_set)
            context.logger.info(
                "Round %s | ExpGuard | effective_zero_weight: TP=%d FP=%d (threshold=1e-10)",
                round_number, TP, FP,
            )

        # ---- train global model on consensus ----
        print(f"\n{COLORS.HEADER}Round {round_number} Global model training{COLORS.ENDC}")
        global_model = create_model(context.input_dim, context.num_classes,
                                    config.batch_size, model_type=config.model_type)
        global_model.set_weights(global_w)
        _train_on_soft_labels(global_model, pub_X, consensus, config.batch_size, config.dist_rounds)

        context.shared_state["global_w"] = global_model.get_weights()
        context.shared_state["global_model"] = global_model  # pipeline records global_weight.bin from this

        del pub_X, consensus
        for name in os.listdir(FEDEXPGUARD_CACHE_DIR):
            if name.startswith(f"r{round_number}_"):
                os.remove(os.path.join(FEDEXPGUARD_CACHE_DIR, name))
        aggressive_memory_cleanup()

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        if _ckpt:
            clear_mid_round(context, "fedexpguard")

        return {}

    def finalize(self, context: PipelineContext) -> None:
        from ..memory import clear_session
        clear_session()
        if os.path.isdir(FEDEXPGUARD_CACHE_DIR):
            for name in os.listdir(FEDEXPGUARD_CACHE_DIR):
                os.remove(os.path.join(FEDEXPGUARD_CACHE_DIR, name))
