from __future__ import annotations

import gc
import os
import time
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext
from .base import DistillationStrategy
from ._checkpoint import save_mid_round, load_mid_round, clear_mid_round
from .robust_filter import RobustFilter
from .common import (
    create_model,
    create_private_dataset,
    load_public_dataset_from_clients,
    numpy_from_dataset,
)

LOGITS_CACHE_DIR = os.path.join("temp_weights", "ours_cache")


# ── Logit generation (chunked to disk) ──────────────────────────────────────

def generate_logits_to_file(model_wrapper, public_features: np.ndarray, batch_size: int, output_path: str) -> tuple:
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else model_wrapper
    chunk_size = 500_000
    total_rows = 0
    num_classes = None
    with open(output_path, "wb") as fp:
        for start in range(0, len(public_features), chunk_size):
            chunk = public_features[start : start + chunk_size]
            logits = logits_model.predict(chunk, batch_size=batch_size, verbose=0)
            if num_classes is None:
                num_classes = logits.shape[1]
            fp.write(logits.astype(np.float32).tobytes())
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
    logit_files: List[str], shape: tuple, robust_filter: RobustFilter,
) -> tuple:
    n_clients = len(logit_files)
    n_samples, n_classes = shape
    consensus = np.zeros(shape, dtype=np.float32)
    # edit chunk_rows to increase filtering speed at the cost of mem
    chunk_rows = 50_000
    row_bytes = n_classes * 4
    max_eig = None
    max_ratio = None
    removal_counts = {c: 0 for c in range(n_clients)}

    handles = [open(fpath, "rb") for fpath in logit_files]
    pbar = tqdm(total=n_samples, desc="Robust consensus", unit="sample")
    for offset in range(0, n_samples, chunk_rows):
        rows = min(chunk_rows, n_samples - offset)
        client_chunks = [
            np.frombuffer(h.read(rows * row_bytes), dtype=np.float32).reshape(rows, n_classes)
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
            removal_counts[c] += int(removal_delta[c])
        pbar.update(rows)
        del client_chunks, S_batch, means
    pbar.close()
    for h in handles:
        h.close()

    return consensus, max_eig, max_ratio, removal_counts


# ── Stage 2: KD (pure distillation, no CE) ──────────────────────────────────

def ekd_stage(
    model_wrapper, consensus_logits: np.ndarray, public_features: np.ndarray,
    batch_size: int, epochs: int, ekd_lambda: float,
) -> None:
    keras_model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else keras_model

    optimizer = keras_model.optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model.optimizer = optimizer
    lam = tf.constant(ekd_lambda, dtype=tf.float32)

    if not hasattr(model_wrapper, '_ekd_train_step'):
        @tf.function
        def train_step(batch_X, teacher_logits):
            with tf.GradientTape() as tape:
                student_logits = logits_model(batch_X, training=True)

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
            print(f"    EKD epoch {epoch + 1}/{epochs} — loss {epoch_loss / batches:.4f}")


def abkd_stage(
    model_wrapper, consensus_logits: np.ndarray, public_features: np.ndarray,
    batch_size: int, epochs: int, alpha: float, beta: float, temperature: float,
) -> None:
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
            print(f"    ABKD epoch {epoch + 1}/{epochs} — loss {epoch_loss / batches:.4f}")


# ── Stage 1: CE training on private data ─────────────────────────────────────

def ce_stage(model_wrapper, private_dataset: tf.data.Dataset, epochs: int) -> None:
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
        self.robust_filter = RobustFilter(
            epsilon=config.robust_epsilon,
        )

    def extra_log_tokens(self) -> Dict[str, float]:
        tokens = {"kd": self.config.ours_kd, "ekd_lambda": self.config.ours_ekd_lambda,
                  "eps": self.config.robust_epsilon}
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
        cleanup_interval = min(getattr(config, 'cleanup_interval', 10), len(context.client_states))

        for skipped_idx in range(first_client):
            fpath = os.path.join(LOGITS_CACHE_DIR, f"client_{context.client_states[skipped_idx].client_id}.bin")
            if os.path.exists(fpath):
                logit_files.append(fpath)

        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_client:
                continue
            fpath = os.path.join(LOGITS_CACHE_DIR, f"client_{state.client_id}.bin")
            model = pool.checkout(state.client_id)

            _, shape = generate_logits_to_file(
                model, public_features, config.batch_size, fpath,
            )

            pool.release(model)
            logit_files.append(fpath)
            logit_shape = shape
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

    def _run_kd_stage(self, context, consensus_logits, public_features, first_client=0):
        config = context.config
        kd_method = getattr(config, "ours_kd", "ekd")
        pool = context.model_pool
        cleanup_interval = min(getattr(config, 'cleanup_interval', 10), len(context.client_states))

        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_client:
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
        cleanup_interval = min(getattr(config, 'cleanup_interval', 10), len(context.client_states))
        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_client:
                continue
            print(f"\n{COLORS.BOLD}Client {state.client_id} — Stage 1 (CE){COLORS.ENDC}")
            model = pool.checkout(state.client_id)
            private_dataset = create_private_dataset(
                state.paths["train_X"], state.paths["train_y"],
                context.input_dim, context.num_classes, config.batch_size,
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
            if os.path.exists(consensus_path):
                print(f"\n{COLORS.OKCYAN}Loading cached consensus logits{COLORS.ENDC}")
                consensus_logits = np.load(consensus_path)
            else:
                print(f"\n{COLORS.OKCYAN}Computing robust consensus logits (eps={config.robust_epsilon}){COLORS.ENDC}")
                consensus_logits, max_eig, max_ratio, removal_counts = compute_robust_consensus_from_files(
                    logit_files, logit_shape, self.robust_filter,
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

            for fpath in logit_files:
                if os.path.exists(fpath):
                    os.remove(fpath)

            self._run_kd_stage(context, consensus_logits, public_features, first_client=first_kd)

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
