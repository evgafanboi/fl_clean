from __future__ import annotations

import gc
import time
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from ..colors import COLORS
from ..context import PipelineContext
from .base import DistillationStrategy
from ._checkpoint import save_mid_round, load_mid_round, clear_mid_round
from .common import create_model, create_private_dataset


def compute_client_logits(
    model_wrapper, X_path: str, y_path: str, num_classes: int
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    logits_model = model_wrapper.get_logits_model()
    X_mmap = np.load(X_path, mmap_mode='r')
    y_mmap = np.load(y_path, mmap_mode='r')
    total = X_mmap.shape[0]

    logits_sum = np.zeros((num_classes, num_classes), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)

    chunk_size = 100_000
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        X_chunk = np.array(X_mmap[start:end], dtype=np.float32)
        y_chunk = np.array(y_mmap[start:end], dtype=np.int32)
        logits = logits_model.predict(X_chunk, batch_size=8192, verbose=0).astype(np.float64)
        np.add.at(logits_sum, y_chunk, logits)
        np.add.at(counts, y_chunk, 1)
        del X_chunk, y_chunk, logits

    per_class_logits = {}
    sample_counts = {}
    for c in range(num_classes):
        if counts[c] == 0:
            continue
        sample_counts[c] = int(counts[c])
        per_class_logits[c] = (logits_sum[c] / float(counts[c])).astype(np.float32)

    return per_class_logits, sample_counts


def aggregate_logits(
    all_stats: Dict[int, Tuple[Dict[int, np.ndarray], Dict[int, int]]], num_classes: int
) -> Dict[int, np.ndarray]:
    global_logits = {}
    for c in range(num_classes):
        c_logits = []
        c_counts = []
        for cid, (logits, counts) in all_stats.items():
            if c not in counts:
                continue
            c_logits.append(logits[c])
            c_counts.append(counts[c])
        if not c_logits:
            continue
        weights = np.array(c_counts, dtype=np.float64)
        weights /= weights.sum()
        global_logits[c] = np.average(np.stack(c_logits), axis=0, weights=weights).astype(np.float32)
    return global_logits


class Exp1(DistillationStrategy):
    name = "Exp1"

    def extra_log_tokens(self) -> Dict[str, float]:
        return {
            "lambda": self.config.exp1_lambda,
            "temperature": self.config.exp1_temperature,
        }

    def _build_training_infra(self, context: PipelineContext):
        self._shared_wrapper = create_model(context.input_dim, context.num_classes, context.config.batch_size)
        keras_model = self._shared_wrapper.model if hasattr(self._shared_wrapper, 'model') else self._shared_wrapper
        self._shared_keras = keras_model

        logits_layer = keras_model.get_layer('logits')
        self._logits_model = tf.keras.Model(
            inputs=keras_model.input,
            outputs=logits_layer.output,
        )

        nc = context.num_classes
        self._gl_var = tf.Variable(tf.zeros([nc, nc]), trainable=False)
        self._hl_var = tf.Variable(tf.zeros([nc], dtype=tf.bool), trainable=False)

        self._optimizer = tf.keras.optimizers.Adam(0.001)
        ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        lambda_val = tf.constant(self.config.exp1_lambda, dtype=tf.float32)
        T = tf.constant(self.config.exp1_temperature, dtype=tf.float32)
        all_vars = keras_model.trainable_variables
        logits_model = self._logits_model
        gl_var = self._gl_var
        hl_var = self._hl_var
        optimizer = self._optimizer

        @tf.function
        def train_step(X_batch, y_batch):
            with tf.GradientTape() as tape:
                logits = logits_model(X_batch, training=True)
                ce = ce_loss_fn(y_batch, logits)

                real_labels = tf.argmax(y_batch, axis=1)
                target_logits = tf.gather(gl_var, real_labels)
                mask = tf.gather(hl_var, real_labels)
                v_logits = tf.boolean_mask(logits, mask)
                v_targets = tf.boolean_mask(target_logits, mask)

                kl = tf.cond(
                    tf.shape(v_logits)[0] > 0,
                    lambda: tf.reduce_mean(
                        tf.reduce_sum(
                            tf.nn.softmax(v_targets / T)
                            * (tf.nn.log_softmax(v_targets / T) - tf.nn.log_softmax(v_logits / T)),
                            axis=-1,
                        )
                    ) * (T * T),
                    lambda: 0.0,
                )

                loss = ce + lambda_val * kl

            grads = tape.gradient(loss, all_vars)
            optimizer.apply_gradients(zip(grads, all_vars))
            return loss, ce, kl

        self._train_step = train_step

    def setup(self, context: PipelineContext) -> None:
        for client_id, paths in enumerate(context.paths):
            context.add_client_state(client_id, None, paths)
        self._infra_built = False

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        global_logits = context.shared_state.get("global_logits", {})
        has_global = len(global_logits) > 0

        if has_global and not self._infra_built:
            self._build_training_infra(context)
            self._infra_built = True

        if has_global:
            gl_arr = np.zeros((context.num_classes, context.num_classes), dtype=np.float32)
            hl_arr = np.zeros(context.num_classes, dtype=bool)
            for c, l in global_logits.items():
                gl_arr[c] = l
                hl_arr[c] = True
            self._gl_var.assign(gl_arr)
            self._hl_var.assign(hl_arr)

        print(f"\n{COLORS.OKCYAN}[STEP 1/3] Local training{COLORS.ENDC}")
        pool = context.model_pool
        all_stats = {}
        all_client_metrics = []
        round_metrics: Dict[int, Dict[str, float]] = {}

        _ckpt = getattr(config, "checkpoint", 0)
        first_client = 0
        if _ckpt:
            mid = load_mid_round(context, "exp1", round_number)
            if mid is not None:
                first_client = mid["last_client_idx"] + 1
                all_stats = mid.get("all_stats", {})
                all_client_metrics = mid.get("all_client_metrics", [])
                round_metrics = mid.get("round_metrics", {})
                print(f"{COLORS.OKGREEN}Resuming round {round_number} from client {first_client}{COLORS.ENDC}")

        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_client:
                continue
            print(f"\n{COLORS.BOLD}Client {state.client_id}{COLORS.ENDC}")
            poison_loader = context.poison_loader if state.client_id in context.poisoned_clients else None

            model = pool.checkout(state.client_id)

            if not has_global:
                print("    Round 1: CE-only training")
                dataset = create_private_dataset(
                    state.paths["train_X"], state.paths["train_y"],
                    context.input_dim, context.num_classes, config.batch_size,
                    poison_loader=poison_loader,
                )
                model.fit(dataset, epochs=config.epochs)
                del dataset
            else:
                client_keras = model.model if hasattr(model, 'model') else model
                self._shared_keras.set_weights(client_keras.get_weights())

                dataset = create_private_dataset(
                    state.paths["train_X"], state.paths["train_y"],
                    context.input_dim, context.num_classes, config.batch_size,
                    poison_loader=poison_loader,
                )

                print(f"    [KLD] Training ({config.epochs} epochs)")
                for epoch in range(config.epochs):
                    e_loss = e_ce = e_kl = 0.0
                    n_batches = 0
                    for X_batch, y_batch in dataset:
                        loss, ce, kl = self._train_step(X_batch, y_batch)
                        e_loss += float(loss)
                        e_ce += float(ce)
                        e_kl += float(kl)
                        n_batches += 1
                    if n_batches > 0:
                        print(
                            f"      Epoch {epoch+1}/{config.epochs} - Loss: {e_loss/n_batches:.4f} "
                            f"(CE: {e_ce/n_batches:.4f}, KL: {e_kl/n_batches:.4f})"
                        )
                del dataset

                client_keras.set_weights(self._shared_keras.get_weights())

            print(f"  Client {state.client_id}...", end="", flush=True)
            stats = compute_client_logits(
                model, state.paths["train_X"], state.paths["train_y"],
                context.num_classes,
            )
            all_stats[state.client_id] = stats
            print(" done")

            pool.checkin(state.client_id, model)

            gc.collect()

            # Per-client checkpoint
            if _ckpt and (
                client_idx == len(context.client_states) - 1
                or (client_idx + 1) % _ckpt == 0
            ):
                save_mid_round(context, "exp1", {
                    "round": round_number,
                    "last_client_idx": client_idx,
                    "all_stats": all_stats,
                    "all_client_metrics": all_client_metrics,
                    "round_metrics": round_metrics,
                })

        print(f"\n{COLORS.OKCYAN}[STEP 2/3] Aggregating logits{COLORS.ENDC}")

        global_logits = aggregate_logits(all_stats, context.num_classes)
        context.shared_state["global_logits"] = global_logits

        print(f"  Global: {len(global_logits)} class logits")
        del all_stats

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = (
            context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        )
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        if _ckpt:
            clear_mid_round(context, "exp1")

        return {}
