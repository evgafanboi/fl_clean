from __future__ import annotations

import gc
import os
import time
from typing import Dict, List

import numpy as np
from ..backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf
from tqdm import tqdm

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext
from ..poison_utils import parse_poison_config, apply_gradient_scale_poison, poisonedfl_apply_cached_weights, poisonedfl_log_values, poisonedfl_store_round_weights, poisonedfl_unified_weights, poisonedfl_warmstart_weights
from .base import DistillationStrategy
from ._checkpoint import save_mid_round, load_mid_round, clear_mid_round
from .robust_filter import AdaptiveRobustFilter, IterativeRobustFilter
from .common import (
    create_model,
    create_private_dataset,
    load_public_dataset_from_clients,
    lma_logits as _lma_logits,
    numpy_from_dataset,
    poisonedfl_ghost_model_type,
)

LOGITS_CACHE_DIR = os.path.join("temp_weights", "ours_cache")
LOGIT_ABS_CAP = 1e6
EKD_LOGIT_CLIP = 30.0


def _sanitize_logits(logits: np.ndarray, cap: float = LOGIT_ABS_CAP) -> np.ndarray:
    return np.nan_to_num(logits, nan=0.0, posinf=cap, neginf=-cap).astype(np.float32, copy=False)


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
    logit_files: List[str], shape: tuple, robust_filter: AdaptiveRobustFilter,
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

    handles = [open(fpath, "rb") for fpath in logit_files]
    pbar = tqdm(total=n_samples, desc="Robust consensus", unit="sample")
    for offset in range(0, n_samples, chunk_rows):
        rows = min(chunk_rows, n_samples - offset)
        client_chunks = [
            _sanitize_logits(np.frombuffer(h.read(rows * row_bytes), dtype=np.float32).reshape(rows, n_classes))
            for h in handles
        ]
        S_batch = np.stack(client_chunks, axis=1)  # (rows, n_clients, n_classes)
        means, batch_max_eig, removal_delta, batch_max_ratio = robust_filter.compute_robust_mean_batch(S_batch)
        consensus[offset : offset + rows] = means
        if batch_max_eig is not None and (max_eig is None or batch_max_eig > max_eig):
            max_eig = batch_max_eig
        if batch_max_ratio is not None and (max_ratio is None or batch_max_ratio > max_ratio):
            max_ratio = batch_max_ratio
        for c in range(n_clients):
            if removal_delta[c] > 0:
                removal_counts[c] += rows
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

    return consensus, max_eig, max_ratio, removal_counts


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
        self.kd_epochs = getattr(config, "kd_epochs", 1)
        self.ce_epochs = config.epochs
        filter_cls = IterativeRobustFilter if getattr(config, "robust_filter_v2", False) else AdaptiveRobustFilter
        self.robust_filter = filter_cls(
            budget=getattr(config, "robust_rm_budget", None) or 0,
            tail_threshold=getattr(config, "robust_threshold", 0.75),
            workers=getattr(config, "robust_workers", 8),
        )

    def extra_log_tokens(self) -> Dict[str, float]:
        budget = getattr(self.config, "robust_rm_budget", 0)
        tokens = {"kd": self.config.ours_kd, "ekd_lambda": self.config.ours_ekd_lambda, "budget": budget}
        if getattr(self.config, "robust_filter_v2", False):
            tokens["robust_filter"] = "v2"
        if self.config.ours_kd == "abkd":
            tokens.update({"ab_alpha": self.config.ab_alpha, "ab_beta": self.config.ab_beta, "temperature": self.config.ours_temperature})
        return tokens

    def setup(self, context: PipelineContext) -> None:
        config = context.config
        kd_method = getattr(config, "ours_kd", "ekd")

        print(f"{COLORS.OKGREEN}Preparing Ours ({kd_method.upper()}){COLORS.ENDC}")

        public_unlabeled_ds, total_public = load_public_dataset_from_clients(
            context.paths, batch_size=config.batch_size, num_classes=context.num_classes,
            shuffle=False, return_labels=False,
        )
        public_features = numpy_from_dataset(public_unlabeled_ds)
        del public_unlabeled_ds

        os.makedirs(LOGITS_CACHE_DIR, exist_ok=True)
        pub_path = os.path.join(LOGITS_CACHE_DIR, "public_features.npy")
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
        attack_type, poison_value, _ = parse_poison_config(getattr(config, "poison", None))
        cleanup_interval = min(getattr(config, 'cleanup_interval', 10), len(context.client_states))

        for skipped_idx in range(first_client):
            fpath = os.path.join(LOGITS_CACHE_DIR, f"client_{context.client_states[skipped_idx].client_id}.bin")
            if os.path.exists(fpath):
                logit_files.append(fpath)

        _pfl_gen = getattr(context, 'poisoned_fl_state', None)
        _ghost_bytes = None
        _ghost_shape = None
        _ghost_warm = poisonedfl_warmstart_weights(context.shared_state, _pfl_gen, fallback=context.shared_state.get("init_w")) if _pfl_gen is not None else None
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
            fpath = os.path.join(LOGITS_CACHE_DIR, f"client_{state.client_id}.bin")
            if attack_type == "lma" and state.client_id in context.poisoned_clients:
                if _lma_adv_bytes is not None:
                    with open(fpath, "wb") as _hf:
                        _hf.write(_lma_adv_bytes)
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

        _ghost_warm = poisonedfl_warmstart_weights(context.shared_state, _pfl_m, fallback=context.shared_state.get("init_w")) if _pfl_m is not None else None
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

        _pfl_kd = getattr(context, 'poisoned_fl_state', None)
        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_client:
                continue
            if _pfl_kd is not None and state.client_id in context.poisoned_clients:
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

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        public_features = np.load(context.shared_state["public_features_path"], mmap_mode="r")
        no_filter = getattr(config, "robust_rm_budget", 0) == 0
        _pfl = getattr(context, 'poisoned_fl_state', None)

        os.makedirs(LOGITS_CACHE_DIR, exist_ok=True)

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
            consensus_path = os.path.join(LOGITS_CACHE_DIR, f"r{round_number}_consensus.npy")

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
                retry_proxy = context.shared_state.get("poisonedfl_proxy_w")
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
                if (
                    _pfl.last_hypothesis_success is False
                    and _pfl.last_poisoned is not None
                    and retry_proxy is not None
                ):
                    context.logger.info("Round %s | PoisonedFL | H0 -> retry ghost KD from previous unpoisoned proxy", round_number)
                    ghost = create_model(context.input_dim, context.num_classes, config.batch_size, model_type=poisonedfl_ghost_model_type(config))
                    ghost.set_weights([w.copy() for w in retry_proxy])
                    if _kd_m == "abkd":
                        abkd_stage(ghost, consensus_logits, public_features, config.batch_size, self.kd_epochs,
                                   config.ab_alpha, config.ab_beta, config.ours_temperature)
                    else:
                        ekd_stage(ghost, consensus_logits, public_features, config.batch_size, self.kd_epochs,
                                  getattr(config, "ours_ekd_lambda", 1.0))
                    ghost_w = ghost.get_weights()
                    del ghost
                    poisoned_w = poisonedfl_apply_cached_weights(ghost_w, _pfl, track_as_prev=True, previous_proxy=retry_proxy)
                poisonedfl_store_round_weights(context.shared_state, ghost_w, poisoned_w)
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
                consensus_path = os.path.join(LOGITS_CACHE_DIR, f"r{round_number}_consensus.npy")
                if logit_shape is None and not os.path.exists(consensus_path):
                    raise RuntimeError(
                        "logit_shape is None — no logit files were produced or recovered. "
                        "Check that client logit .bin files exist in the cache directory."
                    )
                if skip_kd and os.path.exists(consensus_path):
                    print(f"\n{COLORS.OKCYAN}Loading cached consensus logits{COLORS.ENDC}")
                    consensus_logits = np.load(consensus_path)
                else:
                    budget = getattr(config, "robust_rm_budget", 0)
                    if budget == 0:
                        print(f"\n{COLORS.OKCYAN}Computing mean consensus logits (no filtering){COLORS.ENDC}")
                        consensus_logits = compute_consensus_from_files(logit_files, logit_shape)
                        print(f"  Consensus shape: {consensus_logits.shape}")
                        context.logger.info(
                            "Round %s | MeanConsensus | budget=0 (no filtering)",
                            round_number,
                        )
                    else:
                        print(f"\n{COLORS.OKCYAN}Computing robust consensus logits (budget={budget}){COLORS.ENDC}")
                        consensus_logits, max_eig, max_ratio, removal_counts = compute_robust_consensus_from_files(
                            logit_files, logit_shape, self.robust_filter,
                            poisoned_client_ids=(None if _pfl is not None else context.poisoned_clients),
                            logger=context.logger,
                            round_number=round_number,
                        )
                        print(f"  Consensus shape: {consensus_logits.shape}")
                        eig_s = f"{max_eig:.6f}" if max_eig is not None else "N/A"
                        ratio_s = f"{max_ratio:.4f}" if max_ratio is not None else "N/A"
                        top_removed = sorted(removal_counts.items(), key=lambda x: -x[1])
                        print(f"  max_eig={eig_s}  max_ratio={ratio_s}  top removals (client_idx: count): {top_removed[:5]}")
                        context.logger.info(
                            "Round %s | RobustConsensus | max_eig=%s | max_ratio=%s | removals=%s",
                            round_number, eig_s, ratio_s, top_removed,
                        )
                    np.save(consensus_path, consensus_logits)
                    context.shared_state["lma_stale_consensus"] = consensus_logits.copy()

                for fpath in logit_files:
                    if os.path.exists(fpath):
                        os.remove(fpath)

                self._run_kd_stage(context, consensus_logits, public_features, first_client=first_kd)
                if _pfl is not None and context.poisoned_clients:
                    _kd_m = getattr(config, "ours_kd", "ekd")
                    retry_proxy = context.shared_state.get("poisonedfl_proxy_w")
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
                    if (
                        _pfl.last_hypothesis_success is False
                        and _pfl.last_poisoned is not None
                        and retry_proxy is not None
                    ):
                        context.logger.info("Round %s | PoisonedFL | H0 -> retry ghost KD from previous unpoisoned proxy", round_number)
                        ghost = create_model(context.input_dim, context.num_classes, config.batch_size, model_type=poisonedfl_ghost_model_type(config))
                        ghost.set_weights([w.copy() for w in retry_proxy])
                        if _kd_m == "abkd":
                            abkd_stage(ghost, consensus_logits, public_features, config.batch_size, self.kd_epochs,
                                       config.ab_alpha, config.ab_beta, config.ours_temperature)
                        else:
                            ekd_stage(ghost, consensus_logits, public_features, config.batch_size, self.kd_epochs,
                                      getattr(config, "ours_ekd_lambda", 1.0))
                        ghost_w = ghost.get_weights()
                        del ghost
                        poisoned_w = poisonedfl_apply_cached_weights(ghost_w, _pfl, track_as_prev=True, previous_proxy=retry_proxy)
                    poisonedfl_store_round_weights(context.shared_state, ghost_w, poisoned_w)
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

        del public_features
        aggressive_memory_cleanup()

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        if self._ckpt:
            clear_mid_round(context, "ours")

        return {}
