from __future__ import annotations

import gc
import os
import time
from typing import Dict, List

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext, evaluate_model, ModelPool
from .base import DistillationStrategy
from .common import (
    create_model,
    create_private_dataset,
    load_public_dataset_from_clients,
    numpy_from_dataset,
)
from models.dense_discri import create_discriminator as create_disc_model

LOGITS_CACHE_DIR = os.path.join("temp_weights", "exp2_cache")
DISC_WEIGHTS_DIR = os.path.join("temp_weights", "exp2_disc_weights")


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


def generate_logits_to_file_with_disc(
    model_wrapper, disc_model, public_features: np.ndarray,
    batch_size: int, num_classes: int, output_path: str,
) -> tuple:
    logits_model = model_wrapper.get_logits_model() if hasattr(model_wrapper, "get_logits_model") else model_wrapper
    disc_net = disc_model.model if hasattr(disc_model, "model") else disc_model
    chunk_size = 500_000
    total_rows = 0
    with open(output_path, "wb") as fp:
        for start in range(0, len(public_features), chunk_size):
            chunk = public_features[start : start + chunk_size]
            logits = logits_model.predict(chunk, batch_size=batch_size, verbose=0)
            dis_pred = disc_net.predict(chunk, batch_size=batch_size, verbose=0).reshape(-1)
            uncertain_mask = dis_pred > 0.5
            if np.any(uncertain_mask):
                logits[uncertain_mask] = 0.0
            fp.write(logits.astype(np.float32).tobytes())
            total_rows += logits.shape[0]
            del logits, dis_pred, chunk
    return output_path, (total_rows, num_classes)


def train_discriminator(
    classify_model, disc_model, public_features: np.ndarray,
    dis_rounds: int, batch_size: int, num_classes: int, private_X_path: str,
) -> bool:
    logits_model = classify_model.get_logits_model() if hasattr(classify_model, "get_logits_model") else classify_model

    chunk_size = 500_000
    max_probs = np.empty(len(public_features), dtype=np.float32)
    for start in range(0, len(public_features), chunk_size):
        end = min(start + chunk_size, len(public_features))
        logits = logits_model.predict(public_features[start:end], batch_size=batch_size, verbose=0)
        probs = tf.nn.softmax(logits).numpy()
        max_probs[start:end] = np.max(probs, axis=1)
        del logits, probs

    theta = float(np.median(max_probs))
    sure_unknown_mask = max_probs < theta
    sure_unknown_feature = public_features[sure_unknown_mask]
    del max_probs

    if sure_unknown_feature.size == 0:
        return False

    X_mmap = np.load(private_X_path, mmap_mode="r")
    sure_known = np.array(X_mmap[: len(sure_unknown_feature)], dtype=np.float32)
    del X_mmap

    dis_X = np.vstack([sure_known, sure_unknown_feature])
    dis_y = np.concatenate([
        np.zeros(len(sure_known), dtype=np.float32),
        np.ones(len(sure_unknown_feature), dtype=np.float32),
    ])
    del sure_known, sure_unknown_feature

    indices = np.random.permutation(len(dis_X))
    dis_X, dis_y = dis_X[indices], dis_y[indices]

    dataset = tf.data.Dataset.from_tensor_slices((dis_X, dis_y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    for _ in range(dis_rounds):
        disc_model.fit(dataset, epochs=1, verbose=0)

    del dis_X, dis_y, dataset
    return True


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


# ── Stage 1: KD (pure distillation, no CE) ──────────────────────────────────

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


# ── Stage 2: CE training on private data ─────────────────────────────────────

def ce_stage(model_wrapper, private_dataset: tf.data.Dataset, epochs: int) -> None:
    print(f"    CE training for {epochs} epochs")
    history = model_wrapper.fit(private_dataset, epochs=epochs, verbose=1)
    if hasattr(history, "history") and "loss" in history.history:
        print(f"      final CE loss: {history.history['loss'][-1]:.4f}")


# ── Strategy ─────────────────────────────────────────────────────────────────

class Exp2(DistillationStrategy):
    name = "Exp2"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.kd_epochs = 1
        self.ce_epochs = config.epochs

    def extra_log_tokens(self) -> Dict[str, float]:
        tokens = {"kd": self.config.exp2_kd, "ekd_lambda": self.config.exp2_ekd_lambda}
        if self.config.exp2_kd == "abkd":
            tokens.update({"ab_alpha": self.config.ab_alpha, "ab_beta": self.config.ab_beta, "temperature": self.config.exp2_temperature})
        if not getattr(self.config, "remove_dis", False):
            tokens["dis_rounds"] = self.config.dis_rounds
        return tokens

    def setup(self, context: PipelineContext) -> None:
        config = context.config
        is_sequence = config.model_type.lower() == "gru"
        use_dis = not getattr(config, "remove_dis", False)
        kd_method = getattr(config, "exp2_kd", "ekd")

        print(f"{COLORS.OKGREEN}Preparing Exp2 ({kd_method.upper()} + discriminator){COLORS.ENDC}")

        public_unlabeled_ds, total_public = load_public_dataset_from_clients(
            context.paths, batch_size=config.batch_size, num_classes=context.num_classes,
            shuffle=False, return_labels=False, is_sequence=is_sequence,
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
            "is_sequence": is_sequence,
            "use_dis": use_dis,
        })

        if use_dis:
            disc_pool = ModelPool(
                pool_size=min(5, context.n_clients),
                factory_fn=lambda: create_disc_model(context.input_dim),
                weights_dir=DISC_WEIGHTS_DIR,
            )
            context.shared_state["disc_pool"] = disc_pool

        for client_id, paths in enumerate(context.paths):
            context.add_client_state(client_id, None, paths)
            aggressive_memory_cleanup()

    def _generate_logits(self, context, public_features):
        config = context.config
        use_dis = context.shared_state["use_dis"]
        disc_pool = context.shared_state.get("disc_pool")
        pool = context.model_pool
        logit_files = []
        logit_shape = None

        for state in context.client_states:
            fpath = os.path.join(LOGITS_CACHE_DIR, f"client_{state.client_id}.bin")
            model = pool.checkout(state.client_id)

            if use_dis:
                disc = disc_pool.checkout(state.client_id)
                train_discriminator(
                    model, disc, public_features,
                    config.dis_rounds, config.batch_size, context.num_classes,
                    state.paths["train_X"],
                )
                _, shape = generate_logits_to_file_with_disc(
                    model, disc, public_features,
                    config.batch_size, context.num_classes, fpath,
                )
                disc_pool.checkin(state.client_id, disc)
            else:
                _, shape = generate_logits_to_file(
                    model, public_features, config.batch_size, fpath,
                )

            pool.release(model)
            logit_files.append(fpath)
            logit_shape = shape
            print(f"  Client {state.client_id}: logits {shape} -> {fpath}")
            aggressive_memory_cleanup()

        return logit_files, logit_shape

    def _run_kd_stage(self, context, consensus_logits, public_features):
        config = context.config
        kd_method = getattr(config, "exp2_kd", "ekd")
        pool = context.model_pool

        for state in context.client_states:
            print(f"\n{COLORS.BOLD}Client {state.client_id} — Stage 1 ({kd_method.upper()}){COLORS.ENDC}")
            model = pool.checkout(state.client_id)
            if kd_method == "abkd":
                abkd_stage(
                    model, consensus_logits, public_features,
                    config.batch_size, self.kd_epochs,
                    config.ab_alpha, config.ab_beta, config.exp2_temperature,
                )
            else:
                ekd_stage(
                    model, consensus_logits, public_features,
                    config.batch_size, self.kd_epochs,
                    getattr(config, "exp2_ekd_lambda", 1.0),
                )
            pool.checkin(state.client_id, model)
            aggressive_memory_cleanup()

    def _run_ce_stage(self, context):
        config = context.config
        pool = context.model_pool
        for state in context.client_states:
            print(f"\n{COLORS.BOLD}Client {state.client_id} — Stage 2 (CE){COLORS.ENDC}")
            model = pool.checkout(state.client_id)
            private_dataset = create_private_dataset(
                state.paths["train_X"], state.paths["train_y"],
                context.input_dim, context.num_classes, config.batch_size,
                is_sequence=context.shared_state["is_sequence"],
            )
            ce_stage(model, private_dataset, self.ce_epochs)
            pool.checkin(state.client_id, model)
            del private_dataset
            aggressive_memory_cleanup()

    def _evaluate(self, context, round_number):
        pool = context.model_pool
        all_client_metrics = []
        round_metrics: Dict[int, Dict[str, float]] = {}

        for state in context.client_states:
            model = pool.checkout(state.client_id)
            metrics = evaluate_model(model, context.test_dataset, context.test_labels)
            pool.release(model)
            all_client_metrics.append(metrics)
            round_metrics[state.client_id] = metrics
            context.logger.info(
                "Round %s | Client %s | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Loss: %.4f",
                round_number, state.client_id,
                metrics["Acc"], metrics["F1"], metrics["Precision"], metrics["Recall"], metrics["Loss"],
            )
            print(
                f"{COLORS.OKGREEN}Client {state.client_id}: Acc={metrics['Acc']:.4f}, F1={metrics['F1']:.4f}, "
                f"Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, Loss={metrics['Loss']:.4f}{COLORS.ENDC}"
            )
            del metrics
            gc.collect()

        avg_metrics = {
            "Acc": np.mean([m["Acc"] for m in all_client_metrics]),
            "F1": np.mean([m["F1"] for m in all_client_metrics]),
            "Precision": np.mean([m["Precision"] for m in all_client_metrics]),
            "Recall": np.mean([m["Recall"] for m in all_client_metrics]),
            "Loss": np.mean([m["Loss"] for m in all_client_metrics]),
        }
        round_metrics[-1] = avg_metrics

        print(
            f"{COLORS.OKGREEN}Round {round_number} - Avg Acc={avg_metrics['Acc']:.4f}, "
            f"F1={avg_metrics['F1']:.4f}, Precision={avg_metrics['Precision']:.4f}, "
            f"Recall={avg_metrics['Recall']:.4f}, Loss={avg_metrics['Loss']:.4f}{COLORS.ENDC}"
        )
        context.logger.info(
            "Round %s | Avg | Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Loss: %.4f",
            round_number, avg_metrics["Acc"], avg_metrics["F1"],
            avg_metrics["Precision"], avg_metrics["Recall"], avg_metrics["Loss"],
        )
        return round_metrics

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        public_features = np.load(context.shared_state["public_features_path"], mmap_mode="r")

        os.makedirs(LOGITS_CACHE_DIR, exist_ok=True)

        # CE -> Logits -> KD -> Eval (every round)
        self._run_ce_stage(context)

        print(f"\n{COLORS.OKCYAN}Generating public logits{COLORS.ENDC}")
        logit_files, logit_shape = self._generate_logits(context, public_features)

        print(f"\n{COLORS.OKCYAN}Computing consensus logits{COLORS.ENDC}")
        consensus_logits = compute_consensus_from_files(logit_files, logit_shape)
        print(f"  Consensus shape: {consensus_logits.shape}")

        for fpath in logit_files:
            os.remove(fpath)

        self._run_kd_stage(context, consensus_logits, public_features)

        del consensus_logits, public_features
        aggressive_memory_cleanup()

        round_metrics = self._evaluate(context, round_number)

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        return round_metrics
