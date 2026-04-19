from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from ..backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf

from ..colors import COLORS
from ..data_utils import create_client_dataset
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext
from ..poison_utils import parse_poison_config, apply_gaussian_noise_scale
from .base import DistillationStrategy
from .common import create_model, load_public_dataset_from_clients, numpy_from_dataset
from .robust_filter import RobustFilter
from tqdm import tqdm

CACHE_DIR = os.path.join("temp_weights", "cronus_cache")


# ── file-path helpers ──────────────────────────────────────────────────

def _pred_path(cid: int, rnd: int) -> str:
    return os.path.join(CACHE_DIR, f"r{rnd}_c{cid}.bin")


def _pub_path(rnd: int | None = None) -> str:
    if rnd is None:
        return os.path.join(CACHE_DIR, "public_X.npy")
    return os.path.join(CACHE_DIR, f"r{rnd}_pub_X.npy")


def _pseudo_path(rnd: int) -> str:
    return os.path.join(CACHE_DIR, f"r{rnd}_pseudo_y.npy")


# ── predict / filter ───────────────────────────────────────────────────

def _predict_to_file(model, X: np.ndarray, num_classes: int,
                     batch_size: int, path: str) -> None:
    preds = model.predict(X, verbose=0, batch_size=batch_size)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fp:
        fp.write(preds.astype(np.float32).tobytes())


def _robust_filter(pred_files: List[str], n_samples: int,
                   num_classes: int, epsilon: float,
                   ) -> Tuple[np.ndarray, Optional[float], Optional[float], Set[int]]:
    rf = RobustFilter(epsilon=epsilon)
    row_bytes = num_classes * 4
    CHUNK = 50_000
    pseudo = np.empty(n_samples, dtype=np.int32)
    max_eig: Optional[float] = None
    max_ratio: Optional[float] = None
    removal_counts = np.zeros(len(pred_files), dtype=np.int64)
    removed: Set[int] = set()
    handles = [open(f, "rb") for f in pred_files]
    boundary = 1.0 / num_classes
    off = 0
    pbar = tqdm(total=n_samples, desc="Robust filter", unit="sample")
    while off < n_samples:
        rows = min(CHUNK, n_samples - off)
        S_batch = np.stack([
            np.frombuffer(h.read(rows * row_bytes), dtype=np.float32)
              .reshape(rows, num_classes)
            for h in handles
        ], axis=1)
        means, batch_eig, batch_removals, batch_ratio = rf.compute_robust_mean_batch(S_batch)
        if batch_eig is not None and (max_eig is None or batch_eig > max_eig):
            max_eig = batch_eig
        if batch_ratio is not None and (max_ratio is None or batch_ratio > max_ratio):
            max_ratio = batch_ratio
        removal_counts += batch_removals
        for c in range(len(pred_files)):
            if batch_removals[c] > 0:
                removed.add(c)
        max_vals = means.max(axis=1)
        labels = means.argmax(axis=1).astype(np.int32)
        labels[max_vals <= boundary] = -1
        pseudo[off : off + rows] = labels
        pbar.update(rows)
        off += rows
        del S_batch, means
    pbar.close()
    for h in handles:
        h.close()
    return pseudo, max_eig, max_ratio, removed


# ── merged dataset (round 2+) ─────────────────────────────────────────

def _make_merged_dataset(priv_X_path, priv_y_path, pub_X_path, pseudo_y_path,
                         input_dim, num_classes, batch_size, poison_loader=None):
    from ..backend import use_tf
    priv_X = np.array(np.load(priv_X_path, mmap_mode="r"), dtype=np.float32)
    priv_y = np.array(np.load(priv_y_path, mmap_mode="r"), dtype=np.int32)
    if poison_loader:
        priv_y = poison_loader.poison_labels(priv_y)
    priv_y_oh = np.zeros((len(priv_y), num_classes), dtype=np.float32)
    priv_y_oh[np.arange(len(priv_y)), priv_y] = 1.0

    pseudo = np.array(np.load(pseudo_y_path, mmap_mode="r"), dtype=np.int32)
    valid = pseudo >= 0
    pub_X = np.array(np.load(pub_X_path, mmap_mode="r")[valid], dtype=np.float32)
    pub_y_oh = np.zeros((int(valid.sum()), num_classes), dtype=np.float32)
    pub_y_oh[np.arange(int(valid.sum())), pseudo[valid]] = 1.0
    del pseudo

    X = np.concatenate([priv_X, pub_X])
    y = np.concatenate([priv_y_oh, pub_y_oh])
    del priv_X, priv_y_oh, pub_X, pub_y_oh
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    del perm

    if not use_tf():
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          pin_memory=False, num_workers=0)
    return (tf.data.Dataset.from_tensor_slices((X, y))
              .batch(batch_size).prefetch(tf.data.AUTOTUNE))


# ── checkpoint helpers ──────────────────────────────────────────────────

def _ckpt_dir(context: PipelineContext) -> str:
    stem = os.path.splitext(os.path.basename(context.log_filename))[0]
    return os.path.join("checkpoint", stem)


def _save_mid_round(context: PipelineContext, round_number: int,
                    last_client_idx: int, pred_files: List[str],
                    pred_cids: List[int], pub_X_file: str) -> None:
    d = _ckpt_dir(context)
    os.makedirs(d, exist_ok=True)
    payload = {
        "round": round_number,
        "last_client_idx": last_client_idx,
        "pred_files": pred_files,
        "pred_cids": pred_cids,
        "pub_X_file": pub_X_file,
        "client_weights": {
            st.client_id: st.data["w"]
            for st in context.client_states
            if st.data.get("w") is not None
        },
    }
    tmp = os.path.join(d, "cronus_mid.bin.tmp")
    dst = os.path.join(d, "cronus_mid.bin")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst)
    print(f"{COLORS.OKCYAN}  Mid-round checkpoint saved (client {last_client_idx}){COLORS.ENDC}")

    for st in context.client_states[:last_client_idx + 1]:
        w = st.data.get("w")
        if w is not None:
            context.record_client_weight(round_number, st.client_id, w)


def _load_mid_round(context: PipelineContext, round_number: int):
    d = _ckpt_dir(context)
    path = os.path.join(d, "cronus_mid.bin")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("round") != round_number:
        return None
    # Restore per-client weights
    for st in context.client_states:
        w = payload["client_weights"].get(st.client_id)
        if w is not None:
            st.data["w"] = w
    return payload


def _clear_mid_round(context: PipelineContext) -> None:
    d = _ckpt_dir(context)
    path = os.path.join(d, "cronus_mid.bin")
    if os.path.exists(path):
        os.remove(path)


# ── strategy ───────────────────────────────────────────────────────────

class Cronus(DistillationStrategy):
    name = "Cronus"
    use_model_pool = False

    # ── setup ──────────────────────────────────────────────────────────

    def setup(self, context: PipelineContext) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cfg = context.config

        pub_ds, _ = load_public_dataset_from_clients(
            context.paths, batch_size=cfg.batch_size,
            num_classes=context.num_classes, shuffle=False, return_labels=False,
        )
        pub_X = numpy_from_dataset(pub_ds)
        del pub_ds
        np.save(_pub_path(), pub_X)
        context.shared_state["n_public"] = pub_X.shape[0]
        del pub_X

        from ..memory import clear_session
        clear_session()

        _mixed = getattr(cfg, 'mixed_models', False)
        if not _mixed:
            reusable = create_model(context.input_dim, context.num_classes,
                                    cfg.batch_size, model_type=cfg.model_type)
            init_w = reusable.get_weights()
            context.shared_state["init_w"] = init_w
            self._model = reusable
        else:
            self._model = None

        for cid, paths in enumerate(context.paths):
            st = context.add_client_state(cid, None, paths)
            st.data["w"] = None

        print(f"{COLORS.OKGREEN}Cronus initialized ({context.n_clients} clients){COLORS.ENDC}")

    # ── round ──────────────────────────────────────────────────────────

    def run_round(self, context: PipelineContext, round_number: int
                  ) -> Dict[int, Dict[str, float]]:
        t0 = time.time()
        cfg = context.config
        n_pub = context.shared_state["n_public"]
        _mixed = getattr(cfg, 'mixed_models', False)
        model = self._model
        _REFRESH_EVERY = getattr(cfg, "cleanup_interval", 25)

        # Round-boundary refresh
        if round_number > 1 and not _mixed:
            del model
            from ..memory import clear_session
            clear_session()
            aggressive_memory_cleanup()
            model = create_model(context.input_dim, context.num_classes,
                                 cfg.batch_size, model_type=cfg.model_type)
            self._model = model
        elif round_number > 1:
            from ..memory import clear_session
            clear_session()
            aggressive_memory_cleanup()

        base = np.load(_pub_path(), mmap_mode="r")
        perm = np.random.permutation(n_pub)
        open_X = np.array(base[perm], dtype=np.float32)
        del base
        pub_X_file = _pub_path(round_number)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(pub_X_file, open_X)

        attack_type, poison_value, _ = parse_poison_config(
            getattr(cfg, "poison", None))

        is_init = round_number == 1
        prev_pseudo = context.shared_state.get("pseudo_path")
        prev_pub   = context.shared_state.get("pub_X_path")

        tag = "Init (private only)" if is_init else "Merged (private + public)"
        print(f"\n{COLORS.HEADER}Round {round_number} — {tag}{COLORS.ENDC}")

        pred_files: List[str] = []
        pred_cids:  List[int] = []
        first_client = 0

        # ---- mid-round resume ----
        _ckpt_interval = getattr(cfg, "checkpoint", 0)
        if _ckpt_interval:
            mid = _load_mid_round(context, round_number)
            if mid is not None:
                first_client = mid["last_client_idx"] + 1
                pred_files = mid["pred_files"]
                pred_cids = mid["pred_cids"]
                print(f"{COLORS.OKGREEN}Resuming round {round_number} from client {first_client}{COLORS.ENDC}")

        # ---- per-client: reuse model → set_weights → train → predict ----
        for idx, st in enumerate(context.client_states):
            if idx < first_client:
                continue
            cid = st.client_id
            poisoned = cid in context.poisoned_clients

            if _mixed:
                if idx > 0 and idx % _REFRESH_EVERY == 0:
                    from ..memory import clear_session
                    clear_session()
                    aggressive_memory_cleanup()
                from models.mixed_models import get_model_type_for_client
                arch = get_model_type_for_client(cid, context.n_clients)
                model = create_model(context.input_dim, context.num_classes,
                                     cfg.batch_size, model_type=arch)
                if st.data["w"]:
                    model.set_weights(st.data["w"])
            else:
                # Periodic refresh to defrag GPU memory (same as FedAvg)
                if idx > 0 and idx % _REFRESH_EVERY == 0:
                    del model
                    from ..memory import clear_session
                    clear_session()
                    aggressive_memory_cleanup()
                    model = create_model(context.input_dim, context.num_classes,
                                         cfg.batch_size, model_type=cfg.model_type)
                    self._model = model
                model.set_weights(st.data["w"] or context.shared_state["init_w"])

            print(f"\n{COLORS.BOLD}Client {cid}{COLORS.ENDC}")

            p_loader = (context.poison_loader
                        if poisoned and context.poison_loader else None)

            if is_init:
                ds = create_client_dataset(
                    st.paths["train_X"], st.paths["train_y"],
                    context.input_dim, context.num_classes,
                    cfg.batch_size, poison_loader=p_loader)
            else:
                ds = _make_merged_dataset(
                    st.paths["train_X"], st.paths["train_y"],
                    prev_pub, prev_pseudo,
                    context.input_dim, context.num_classes,
                    cfg.batch_size, poison_loader=p_loader)

            model.fit(ds, epochs=cfg.epochs)
            st.data["w"] = model.get_weights()

            if poisoned and attack_type == "gradient_scale":
                apply_gaussian_noise_scale(model, poison_value)

            pf = _pred_path(cid, round_number)
            _predict_to_file(model, open_X, context.num_classes,
                             cfg.batch_size, pf)
            pred_files.append(pf)
            pred_cids.append(cid)

            del ds
            if _mixed:
                del model
                model = None
            aggressive_memory_cleanup()

            # Per-client checkpoint (matches FedAvg pipeline pattern)
            if _ckpt_interval and (
                idx == len(context.client_states) - 1
                or (idx + 1) % _ckpt_interval == 0
            ):
                _save_mid_round(context, round_number, idx,
                                pred_files, pred_cids, pub_X_file)

        del open_X
        aggressive_memory_cleanup()

        # ---- robust filter ----
        print(f"\n{COLORS.HEADER}Robust filtering{COLORS.ENDC}")
        eps = getattr(cfg, "robust_epsilon", 0.2)
        pseudo_file = _pseudo_path(round_number)
        if not os.path.exists(pseudo_file):
            pseudo, max_eig, max_ratio, removed_idx = _robust_filter(
                pred_files, n_pub, context.num_classes, eps)

            removed_cids = sorted({pred_cids[i] for i in removed_idx})
            eig_s = f"{max_eig:.6f}" if max_eig is not None else "N/A"
            ratio_s = f"{max_ratio:.4f}" if max_ratio is not None else "N/A"
            valid_n = int(np.sum(pseudo >= 0))
            print(f"  epsilon={eps}  max_eig={eig_s}  max_ratio={ratio_s}  removed={removed_cids or 'none'}")
            print(f"  {valid_n}/{n_pub} pseudo-labeled")
            context.logger.info(
                "Round %s | RobustFilter | eps=%.4f | max_eig=%s | max_ratio=%s | removed=%s | valid=%d/%d",
                round_number, eps, eig_s, ratio_s, removed_cids or "none", valid_n, n_pub)

            np.save(pseudo_file, pseudo)
            del pseudo

            for f in pred_files:
                if os.path.exists(f):
                    os.remove(f)
        else:
            print(f"  Pseudo labels already exist, skipping robust filter")
        if round_number > 1:
            self._cleanup(round_number - 1)

        context.shared_state["pseudo_path"] = pseudo_file
        context.shared_state["pub_X_path"] = pub_X_file

        # Clear mid-round checkpoint now that the round completed successfully
        if getattr(cfg, "checkpoint", 0):
            _clear_mid_round(context)

        dt = time.time() - t0
        context.shared_state["pipeline_elapsed_s"] = (
            context.shared_state.get("pipeline_elapsed_s", 0.0) + dt)
        context.logger.info("Round %s completed in %.2fs", round_number, dt)
        print(f"{COLORS.OKCYAN}Round {round_number} done in {dt:.1f}s{COLORS.ENDC}")
        return {}

    @staticmethod
    def _cleanup(rnd: int):
        for name in os.listdir(CACHE_DIR):
            if name.startswith(f"r{rnd}_"):
                os.remove(os.path.join(CACHE_DIR, name))

    def finalize(self, context: PipelineContext) -> None:
        del self._model
        from ..memory import clear_session
        clear_session()
        for name in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, name))
