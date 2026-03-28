from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..data_utils import create_client_dataset
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext, evaluate_model
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
    with open(path, "wb") as fp:
        fp.write(preds.astype(np.float32).tobytes())


def _robust_filter(pred_files: List[str], n_samples: int,
                   num_classes: int, epsilon: float,
                   ) -> Tuple[np.ndarray, Optional[float], Set[int]]:
    rf = RobustFilter(epsilon=epsilon, tau=0.1, preset="logits")
    row_bytes = num_classes * 4
    CHUNK = 200_000
    pseudo = np.empty(n_samples, dtype=np.int32)
    max_eig: Optional[float] = None
    removed: Set[int] = set()
    handles = [open(f, "rb") for f in pred_files]
    boundary = 1.0 / num_classes
    off = 0
    pbar = tqdm(total=n_samples, desc="Robust filter", unit="sample")
    while off < n_samples:
        rows = min(CHUNK, n_samples - off)
        chunks = np.stack([
            np.frombuffer(h.read(rows * row_bytes), dtype=np.float32)
              .reshape(rows, num_classes)
            for h in handles
        ])
        for i in range(rows):
            mean, eig, rm = rf.compute_robust_mean_debug(chunks[:, i, :])
            if eig is not None and (max_eig is None or eig > max_eig):
                max_eig = eig
            removed.update(rm)
            pseudo[off + i] = int(np.argmax(mean)) if float(np.max(mean)) > boundary else -1
        pbar.update(rows)
        off += rows
    pbar.close()
    for h in handles:
        h.close()
    return pseudo, max_eig, removed


# ── merged dataset (round 2+) ─────────────────────────────────────────

def _make_merged_dataset(priv_X_path, priv_y_path, pub_X_path, pseudo_y_path,
                         input_dim, num_classes, batch_size, poison_loader=None):
    priv_X = np.array(np.load(priv_X_path, mmap_mode="r"), dtype=np.float32)
    priv_y = np.array(np.load(priv_y_path, mmap_mode="r"), dtype=np.int32)
    if poison_loader:
        priv_y = poison_loader.poison_labels(priv_y)
    priv_y = tf.keras.utils.to_categorical(priv_y, num_classes).astype(np.float32)

    pseudo = np.array(np.load(pseudo_y_path, mmap_mode="r"), dtype=np.int32)
    valid = pseudo >= 0
    pub_X = np.array(np.load(pub_X_path, mmap_mode="r")[valid], dtype=np.float32)
    pub_y = tf.keras.utils.to_categorical(pseudo[valid], num_classes).astype(np.float32)
    del pseudo

    X = np.concatenate([priv_X, pub_X])
    y = np.concatenate([priv_y, pub_y])
    del priv_X, priv_y, pub_X, pub_y
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    del perm

    return (tf.data.Dataset.from_tensor_slices((X, y))
              .batch(batch_size).prefetch(tf.data.AUTOTUNE))


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

        tf.keras.backend.clear_session()
        reusable = create_model(context.input_dim, context.num_classes,
                                cfg.batch_size, model_type=cfg.model_type)
        init_w = reusable.get_weights()

        context.shared_state["init_w"] = init_w
        context.shared_state["global_w"] = [np.array(w, copy=True) for w in init_w]
        self._model = reusable

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
        model = self._model
        _REFRESH_EVERY = getattr(cfg, "cleanup_interval", 25)

        # Round-boundary refresh
        if round_number > 1:
            del model
            tf.keras.backend.clear_session()
            aggressive_memory_cleanup()
            model = create_model(context.input_dim, context.num_classes,
                                 cfg.batch_size, model_type=cfg.model_type)
            self._model = model

        base = np.load(_pub_path(), mmap_mode="r")
        perm = np.random.permutation(n_pub)
        open_X = np.array(base[perm], dtype=np.float32)
        del base
        pub_X_file = _pub_path(round_number)
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

        # ---- per-client: reuse model → set_weights → train → predict ----
        for idx, st in enumerate(context.client_states):
            cid = st.client_id
            poisoned = cid in context.poisoned_clients

            # Periodic refresh to defrag GPU memory (same as FedAvg)
            if idx > 0 and idx % _REFRESH_EVERY == 0:
                del model
                tf.keras.backend.clear_session()
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
            aggressive_memory_cleanup()

        del open_X
        aggressive_memory_cleanup()

        # ---- robust filter ----
        print(f"\n{COLORS.HEADER}Robust filtering{COLORS.ENDC}")
        eps = getattr(cfg, "robust_epsilon", 0.2)
        pseudo, max_eig, removed_idx = _robust_filter(
            pred_files, n_pub, context.num_classes, eps)

        removed_cids = sorted({pred_cids[i] for i in removed_idx})
        eig_s = f"{max_eig:.6f}" if max_eig is not None else "N/A"
        valid_n = int(np.sum(pseudo >= 0))
        print(f"  epsilon={eps}  max_eig={eig_s}  removed={removed_cids or 'none'}")
        print(f"  {valid_n}/{n_pub} pseudo-labeled")
        context.logger.info(
            "Round %s | RobustFilter | eps=%.4f | max_eig=%s | removed=%s | valid=%d/%d",
            round_number, eps, eig_s, removed_cids or "none", valid_n, n_pub)

        pseudo_file = _pseudo_path(round_number)
        np.save(pseudo_file, pseudo)
        del pseudo

        for f in pred_files:
            os.remove(f)
        if round_number > 1:
            self._cleanup(round_number - 1)

        context.shared_state["pseudo_path"] = pseudo_file
        context.shared_state["pub_X_path"] = pub_X_file

        # ---- global model on pseudo-labeled public ----
        print(f"\n{COLORS.HEADER}Global model training{COLORS.ENDC}")
        context.shared_state["global_w"] = self._train_global(
            context, model, pub_X_file, pseudo_file, cfg.batch_size, cfg.epochs)

        # ---- evaluation ----
        metrics: Dict[int, Dict[str, float]] = {}

        if round_number == cfg.rounds:
            print(f"\n{COLORS.HEADER}Per-client evaluation{COLORS.ENDC}")
            for st in context.client_states:
                model.set_weights(st.data["w"] or context.shared_state["init_w"])
                ev = evaluate_model(model, context.test_dataset, context.test_labels)
                metrics[st.client_id] = ev
                context.logger.info(
                    "Round %s | Client %s | Acc: %.4f | F1: %.4f",
                    round_number, st.client_id, ev["Acc"], ev["F1"])
                print(f"  Client {st.client_id}: Acc={ev['Acc']:.4f} F1={ev['F1']:.4f}")

        model.set_weights(context.shared_state["global_w"])
        gev = evaluate_model(model, context.test_dataset, context.test_labels)
        metrics[-1] = gev

        context.logger.info(
            "Round %s | GLOBAL | Acc: %.4f | F1: %.4f | Loss: %.4f",
            round_number, gev["Acc"], gev["F1"], gev["Loss"])
        print(f"{COLORS.OKGREEN}Global: Acc={gev['Acc']:.4f} F1={gev['F1']:.4f} Loss={gev['Loss']:.4f}{COLORS.ENDC}")

        dt = time.time() - t0
        context.shared_state["pipeline_elapsed_s"] = (
            context.shared_state.get("pipeline_elapsed_s", 0.0) + dt)
        context.logger.info("Round %s completed in %.2fs", round_number, dt)
        print(f"{COLORS.OKCYAN}Round {round_number} done in {dt:.1f}s{COLORS.ENDC}")
        return metrics

    # ── helpers ────────────────────────────────────────────────────────

    def _train_global(self, ctx, model, pub_X_path, pseudo_path, batch_size, epochs):
        pseudo = np.array(np.load(pseudo_path, mmap_mode="r"), dtype=np.int32)
        valid = pseudo >= 0
        if not np.any(valid):
            print("  No valid pseudo-labels — skipping")
            return ctx.shared_state["global_w"]

        model.set_weights(ctx.shared_state["global_w"])

        X = np.array(np.load(pub_X_path, mmap_mode="r")[valid], dtype=np.float32)
        y = tf.keras.utils.to_categorical(pseudo[valid], ctx.num_classes).astype(np.float32)
        del pseudo
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]
        del perm

        input_dim = X.shape[1]
        num_classes = ctx.num_classes

        ds = (tf.data.Dataset.from_tensor_slices((X, y))
              .batch(batch_size).prefetch(tf.data.AUTOTUNE))

        model.fit(ds, epochs=epochs)
        w = model.get_weights()
        del ds, X, y
        return w

    @staticmethod
    def _cleanup(rnd: int):
        for name in os.listdir(CACHE_DIR):
            if name.startswith(f"r{rnd}_"):
                os.remove(os.path.join(CACHE_DIR, name))

    def finalize(self, context: PipelineContext) -> None:
        del self._model
        tf.keras.backend.clear_session()
        for name in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, name))
