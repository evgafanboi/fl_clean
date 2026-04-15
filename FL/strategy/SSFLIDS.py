from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
from ..backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf
    from models.dense_discri import create_discriminator
else:
    from models.pt_dense_discri import create_discriminator

from ..colors import COLORS
from ..data_utils import create_client_dataset
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext, ModelPool
from .base import DistillationStrategy
from ._checkpoint import save_mid_round, load_mid_round, clear_mid_round
from .common import create_model, load_public_dataset_from_clients, numpy_from_dataset

SSFLIDS_DISC_DIR = os.path.join("temp_weights", "ssflids_disc_weights")
SSFLIDS_CACHE_DIR = os.path.join("temp_weights", "ssflids_cache")


def _softmax_np(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _make_public_ds(X, y, batch_size):
    if _use_tf():
        return (tf.data.Dataset.from_tensor_slices((X, y))
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      pin_memory=False, num_workers=0)


def _base_public_path() -> str:
    return os.path.join(SSFLIDS_CACHE_DIR, "public_features.npy")


def _round_public_path(round_number: int) -> str:
    return os.path.join(SSFLIDS_CACHE_DIR, f"r{round_number}_public_X.npy")


def _pred_path(client_id: int, round_number: int) -> str:
    return os.path.join(SSFLIDS_CACHE_DIR, f"r{round_number}_c{client_id}_hard.npy")


def train_discriminator(
    classify_model,
    discri_model,
    open_feature: np.ndarray,
    dis_rounds: int,
    batch_size: int,
    private_X_path: str,
) -> bool:
    logits_model = classify_model.get_logits_model() if hasattr(classify_model, "get_logits_model") else classify_model

    chunk_size = 500_000
    max_probs_list = []
    for start in range(0, len(open_feature), chunk_size):
        logits = logits_model.predict(open_feature[start:start + chunk_size], batch_size=batch_size, verbose=0)
        probs = _softmax_np(logits)
        max_probs_list.append(np.max(probs, axis=1))
        del logits, probs
    max_probs = np.concatenate(max_probs_list)
    del max_probs_list

    theta = float(np.median(max_probs))
    sure_unknown_mask = max_probs < theta
    sure_unknown_feature = open_feature[sure_unknown_mask]
    del max_probs

    if sure_unknown_feature.size == 0:
        return False

    X_mmap = np.load(private_X_path, mmap_mode="r")
    sure_known_feature = np.array(X_mmap[: len(sure_unknown_feature)], dtype=np.float32)
    del X_mmap

    dis_X = np.vstack([sure_known_feature, sure_unknown_feature])
    dis_y = np.concatenate([
        np.zeros(len(sure_known_feature), dtype=np.float32),
        np.ones(len(sure_unknown_feature), dtype=np.float32),
    ])
    del sure_known_feature, sure_unknown_feature

    indices = np.random.permutation(len(dis_X))
    dis_X, dis_y = dis_X[indices], dis_y[indices]
    del indices

    if _use_tf():
        dataset = (tf.data.Dataset.from_tensor_slices((dis_X, dis_y))
                   .batch(batch_size).prefetch(tf.data.AUTOTUNE))
        discriminator_net = discri_model.model if hasattr(discri_model, "model") else discri_model
        for _ in range(dis_rounds):
            discriminator_net.fit(dataset, epochs=1, verbose=0)
    else:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(torch.from_numpy(dis_X), torch.from_numpy(dis_y))
        dataset = DataLoader(ds, batch_size=batch_size, shuffle=False,
                             pin_memory=False, num_workers=0)
        for _ in range(dis_rounds):
            discri_model.fit(dataset, epochs=1, verbose=0)

    del dataset, dis_X, dis_y
    return True


def predict_with_discriminator(
    classify_model,
    discri_model,
    X_open: np.ndarray,
    num_classes: int,
    batch_size: int,
    output_path: str,
) -> None:
    logits_model = classify_model.get_logits_model() if hasattr(classify_model, "get_logits_model") else classify_model

    boundary = 1.0 / num_classes
    hard_labels: List[int] = []

    for start in range(0, len(X_open), batch_size):
        X_batch = X_open[start:start + batch_size]
        logits = logits_model.predict(X_batch, batch_size=batch_size, verbose=0)
        probs = _softmax_np(logits)
        dis_pred = discri_model.predict(X_batch, batch_size=batch_size, verbose=0).reshape(-1)
        probs[dis_pred > 0.5] = 1.0 / num_classes
        for row in probs:
            max_p = float(np.max(row))
            hard_labels.append(int(np.argmax(row)) if max_p > boundary else num_classes)
        del X_batch, probs, dis_pred

    np.save(output_path, np.array(hard_labels, dtype=np.int32))
    del hard_labels


def hard_label_vote(pred_files: List[str], num_classes: int) -> np.ndarray:
    mmaps = [np.load(f, mmap_mode="r") for f in pred_files]
    sample_cnt = mmaps[0].shape[0]
    voted = np.empty(sample_cnt, dtype=np.int32)

    chunk = 50_000
    for s in range(0, sample_cnt, chunk):
        e = min(s + chunk, sample_cnt)
        label_votes = np.zeros((e - s, num_classes), dtype=np.int32)
        for m in mmaps:
            labels_chunk = np.array(m[s:e], dtype=np.int32)
            valid = labels_chunk < num_classes
            rows = np.arange(e - s)
            label_votes[rows[valid], labels_chunk[valid]] += 1
        voted[s:e] = np.argmax(label_votes, axis=1)
        del label_votes

    del mmaps
    return voted


class SSFLIDS(DistillationStrategy):
    name = "SSFL-IDS"
    use_model_pool = False

    def extra_log_tokens(self) -> Dict[str, float]:
        return {"dis_rounds": self.config.dis_rounds, "train_rounds": self.config.train_rounds, "dist_rounds": self.config.dist_rounds}

    def setup(self, context: PipelineContext) -> None:
        config = context.config
        os.makedirs(SSFLIDS_CACHE_DIR, exist_ok=True)

        public_dataset, total_public = load_public_dataset_from_clients(
            context.paths,
            batch_size=config.batch_size,
            num_classes=context.num_classes,
            shuffle=False,
            return_labels=False,
        )
        public_features = numpy_from_dataset(public_dataset)
        del public_dataset

        np.save(_base_public_path(), public_features)
        context.shared_state["public_sample_count"] = public_features.shape[0]
        del public_features

        from ..memory import clear_session
        clear_session()
        reusable = create_model(context.input_dim, context.num_classes,
                                config.batch_size, model_type=config.model_type)
        context.shared_state["init_w"] = reusable.get_weights()
        self._model = reusable

        disc_pool = ModelPool(
            pool_size=min(5, len(context.paths)),
            factory_fn=lambda: create_discriminator(context.input_dim),
            weights_dir=SSFLIDS_DISC_DIR,
        )
        context.shared_state["disc_pool"] = disc_pool

        for client_id, paths in enumerate(context.paths):
            st = context.add_client_state(client_id, None, paths)
            st.data["w"] = None
            y_mmap = np.load(paths["train_y"], mmap_mode="r")
            st.data["class_counts"] = np.bincount(y_mmap.astype(np.int32), minlength=context.num_classes)
            del y_mmap

        print(f"{COLORS.OKGREEN}SSFL-IDS initialized ({len(context.paths)} clients){COLORS.ENDC}")

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        n_public = context.shared_state["public_sample_count"]
        disc_pool = context.shared_state["disc_pool"]
        model = self._model
        _REFRESH_EVERY = getattr(config, "cleanup_interval", 25)

        if round_number > 1:
            del model
            from ..memory import clear_session
            clear_session()
            aggressive_memory_cleanup()
            model = create_model(context.input_dim, context.num_classes,
                                 config.batch_size, model_type=config.model_type)
            self._model = model
            disc_pool.refresh()

        base_mmap = np.load(_base_public_path(), mmap_mode="r")
        permutation = np.random.permutation(n_public)
        open_feature = np.array(base_mmap[permutation], dtype=np.float32)
        del base_mmap

        os.makedirs(SSFLIDS_CACHE_DIR, exist_ok=True)
        pub_X_path = _round_public_path(round_number)

        print(f"\n{COLORS.HEADER}Round {round_number} Stage I{COLORS.ENDC}")
        pred_files: List[str] = []

        _ckpt = getattr(config, "checkpoint", 0)
        first_s1 = 0
        first_s2 = 0
        skip_stage2_init = False
        if _ckpt:
            mid = load_mid_round(context, "ssflids", round_number)
            if mid is not None:
                stage = mid.get("stage", 1)
                pred_files = mid.get("pred_files", [])
                for st in context.client_states:
                    w = mid.get("client_weights", {}).get(st.client_id)
                    if w is not None:
                        st.data["w"] = w
                if stage >= 2:
                    first_s1 = len(context.client_states)
                    first_s2 = mid["last_client_idx"] + 1
                    skip_stage2_init = True
                else:
                    first_s1 = mid["last_client_idx"] + 1
                print(f"{COLORS.OKGREEN}Resuming round {round_number} stage {stage} from client {mid['last_client_idx'] + 1}{COLORS.ENDC}")

        if not os.path.exists(pub_X_path):
            np.save(pub_X_path, open_feature)

        for client_idx, state in enumerate(context.client_states):
            if client_idx < first_s1:
                continue

            if client_idx > 0 and client_idx % _REFRESH_EVERY == 0:
                del model
                from ..memory import clear_session
                clear_session()
                aggressive_memory_cleanup()
                model = create_model(context.input_dim, context.num_classes,
                                     config.batch_size, model_type=config.model_type)
                self._model = model
                disc_pool.refresh()

            cid = state.client_id
            model.set_weights(state.data["w"] or context.shared_state["init_w"])

            print(f"\n{COLORS.BOLD}Client {cid} Stage I{COLORS.ENDC}")
            ds = create_client_dataset(
                state.paths["train_X"], state.paths["train_y"],
                context.input_dim, context.num_classes, config.batch_size,
            )
            model.fit(ds, epochs=config.train_rounds, verbose=0)
            del ds

            if np.sum(state.data["class_counts"] > 0) <= 1:
                print("  Skipping discriminator (insufficient classes)")
                state.data["w"] = model.get_weights()
                aggressive_memory_cleanup()
                continue

            disc = disc_pool.checkout(cid)
            trained = train_discriminator(
                model, disc, open_feature,
                config.dis_rounds, config.batch_size,
                state.paths["train_X"],
            )
            if not trained:
                print("  Discriminator training skipped (no uncertain samples)")
                disc_pool.checkin(cid, disc)
                state.data["w"] = model.get_weights()
                aggressive_memory_cleanup()
                continue

            pred_path = _pred_path(cid, round_number)
            predict_with_discriminator(
                model, disc, open_feature,
                context.num_classes, config.batch_size, pred_path,
            )
            disc_pool.checkin(cid, disc)
            pred_files.append(pred_path)

            state.data["w"] = model.get_weights()

            if _ckpt and (
                client_idx == len(context.client_states) - 1
                or (client_idx + 1) % _ckpt == 0
            ):
                save_mid_round(context, "ssflids", {
                    "round": round_number,
                    "stage": 1,
                    "last_client_idx": client_idx,
                    "pred_files": pred_files,
                    "client_weights": {
                        st.client_id: st.data["w"]
                        for st in context.client_states
                        if st.data.get("w") is not None
                    },
                })

        del open_feature
        aggressive_memory_cleanup()

        print(f"\n{COLORS.HEADER}Round {round_number} Stage II{COLORS.ENDC}")
        pseudo_y_path = os.path.join(SSFLIDS_CACHE_DIR, f"r{round_number}_pseudo_y.npy")
        if not skip_stage2_init and not os.path.exists(pseudo_y_path):
            global_labels_np = hard_label_vote(pred_files, context.num_classes)
            np.save(pseudo_y_path, global_labels_np)
            del global_labels_np
        aggressive_memory_cleanup()

        pub_X = np.array(np.load(pub_X_path, mmap_mode="r"), dtype=np.float32)
        pseudo_y = np.array(np.load(pseudo_y_path, mmap_mode="r"), dtype=np.int32)
        valid_mask = pseudo_y >= 0
        pub_X = pub_X[valid_mask]
        pseudo_y = pseudo_y[valid_mask]
        print(f"  Pseudo-labeled samples: {len(pseudo_y)} / {int(valid_mask.size)} ({100*len(pseudo_y)/max(valid_mask.size,1):.1f}%)")
        y_cat = np.zeros((len(pseudo_y), context.num_classes), dtype=np.float32)
        y_cat[np.arange(len(pseudo_y)), pseudo_y] = 1.0
        perm = np.random.permutation(len(pub_X))
        public_ds = _make_public_ds(pub_X[perm], y_cat[perm], config.batch_size)
        del pub_X, pseudo_y, y_cat, perm, valid_mask

        for s2_idx, state in enumerate(context.client_states):
            if s2_idx < first_s2:
                continue

            if s2_idx > 0 and s2_idx % _REFRESH_EVERY == 0:
                del model
                from ..memory import clear_session
                clear_session()
                aggressive_memory_cleanup()
                model = create_model(context.input_dim, context.num_classes,
                                     config.batch_size, model_type=config.model_type)
                self._model = model
                pub_X = np.array(np.load(pub_X_path, mmap_mode="r"), dtype=np.float32)
                pseudo_y_arr = np.array(np.load(pseudo_y_path, mmap_mode="r"), dtype=np.int32)
                valid_mask = pseudo_y_arr >= 0
                pub_X = pub_X[valid_mask]
                pseudo_y_arr = pseudo_y_arr[valid_mask]
                y_cat = np.zeros((len(pseudo_y_arr), context.num_classes), dtype=np.float32)
                y_cat[np.arange(len(pseudo_y_arr)), pseudo_y_arr] = 1.0
                p = np.random.permutation(len(pub_X))
                public_ds = _make_public_ds(pub_X[p], y_cat[p], config.batch_size)
                del pub_X, pseudo_y_arr, y_cat, p, valid_mask

            model.set_weights(state.data["w"])
            print(f"Client {state.client_id}: training on pseudo-labeled public data")
            model.fit(public_ds, epochs=config.dist_rounds, verbose=0)
            state.data["w"] = model.get_weights()

            if _ckpt and (
                s2_idx == len(context.client_states) - 1
                or (s2_idx + 1) % _ckpt == 0
            ):
                save_mid_round(context, "ssflids", {
                    "round": round_number,
                    "stage": 2,
                    "last_client_idx": s2_idx,
                    "pred_files": [],
                    "client_weights": {
                        st.client_id: st.data["w"]
                        for st in context.client_states
                        if st.data.get("w") is not None
                    },
                })

        del public_ds
        for name in os.listdir(SSFLIDS_CACHE_DIR):
            if name.startswith(f"r{round_number}_"):
                os.remove(os.path.join(SSFLIDS_CACHE_DIR, name))
        aggressive_memory_cleanup()

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        if _ckpt:
            clear_mid_round(context, "ssflids")

        return {}

    def finalize(self, context: PipelineContext) -> None:
        from ..memory import clear_session
        clear_session()
        if os.path.isdir(SSFLIDS_CACHE_DIR):
            for name in os.listdir(SSFLIDS_CACHE_DIR):
                os.remove(os.path.join(SSFLIDS_CACHE_DIR, name))
        if os.path.isdir(SSFLIDS_DISC_DIR):
            for name in os.listdir(SSFLIDS_DISC_DIR):
                os.remove(os.path.join(SSFLIDS_DISC_DIR, name))
