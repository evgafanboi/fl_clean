from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List

import numpy as np
from ..backend import get_torch_loader_kwargs as _torch_loader_kwargs, use_tf as _use_tf
if _use_tf():
    import tensorflow as tf

from ..colors import COLORS
from ..data_utils import create_client_dataset
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext, ModelPool
from ..poison_utils import parse_poison_config, apply_gradient_scale_poison, poisonedfl_log_values, poisonedfl_store_round_weights, poisonedfl_unified_weights, poisonedfl_warmstart_weights
from .base import DistillationStrategy
from ._checkpoint import save_mid_round, load_mid_round, clear_mid_round
from .common import create_model, lma_targets, load_public_dataset_from_clients, numpy_from_dataset, poisonedfl_ghost_model_type

FEDKDIDS_CACHE_DIR = os.path.join("temp_weights", "fedkdids_cache")


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
                      **_torch_loader_kwargs())


def _base_public_path() -> str:
    return os.path.join(FEDKDIDS_CACHE_DIR, "public_features.npy")


def _round_public_path(round_number: int) -> str:
    return os.path.join(FEDKDIDS_CACHE_DIR, f"r{round_number}_public_X.npy")


def _pred_path(client_id: int, round_number: int) -> str:
    return os.path.join(FEDKDIDS_CACHE_DIR, f"r{round_number}_c{client_id}_hard.npy")


def _predict_hard_labels(
    model,
    X_open: np.ndarray,
    num_classes: int,
    batch_size: int,
    output_path: str,
) -> None:
    logits_model = model.get_logits_model() if hasattr(model, "get_logits_model") else model
    logits = logits_model.predict(X_open, batch_size=batch_size, verbose=0)
    hard = np.argmax(logits, axis=1).astype(np.int32)
    del logits
    np.save(output_path, hard)


def _hamming_filter(pred_files: List[str], tau: float, logger=None, round_number=None, client_ids=None) -> List[int]:
    """
    Compute per-client harmonic mean of pairwise Hamming distances.
    Threshold = tau * mean(HM_i), range [0.5, 1.0].
    Returns indices (into pred_files) that pass the filter (HM_i < threshold).
    """
    K = len(pred_files)
    mmaps = [np.load(f, mmap_mode="r") for f in pred_files]
    D = len(mmaps[0])
    chunk = 50_000

    hd = np.zeros((K, K), dtype=np.float64)
    for s in range(0, D, chunk):
        e = min(s + chunk, D)
        block = np.stack([np.array(m[s:e], dtype=np.int32) for m in mmaps], axis=0)
        for i in range(K):
            diff = (block[i] != block).astype(np.float64).mean(axis=1)
            hd[i] += diff * (e - s)
    hd /= D
    del mmaps

    hm_scores = np.empty(K, dtype=np.float64)
    for i in range(K):
        dists = np.concatenate([hd[i, :i], hd[i, i + 1:]])
        pos = dists[dists > 0]
        hm_scores[i] = len(pos) / np.sum(1.0 / pos) if len(pos) > 0 else 0.0

    threshold = tau * float(hm_scores.mean())

    survivors = []
    for i in range(K):
        tag = "PASS" if hm_scores[i] < threshold else "REJECT"
        if logger is not None:
            client_id = client_ids[i] if client_ids is not None and i < len(client_ids) else i
            logger.info("Round %s | HammingFilter | client_file_idx=%d | client_id=%s | HM=%.4f | %s", round_number, i, client_id, hm_scores[i], tag)
        if hm_scores[i] < threshold:
            survivors.append(i)

    if logger is not None:
        logger.info("Round %s | HammingFilter | threshold=%.4f (tau=%.2f * mean_HM=%.4f)", round_number, threshold, tau, float(hm_scores.mean()))

    if not survivors:
        survivors = list(range(K))
        if logger is not None:
            logger.info("Round %s | HammingFilter | all rejected — falling back to full set", round_number)

    return survivors


def hard_label_vote(pred_files: List[str], num_classes: int, logger=None) -> np.ndarray:
    mmaps = [np.load(f, mmap_mode="r") for f in pred_files]
    sample_cnt = mmaps[0].shape[0]
    voted = np.empty(sample_cnt, dtype=np.int32)
    zero_vote_defaults = 0

    chunk = 50_000
    for s in range(0, sample_cnt, chunk):
        e = min(s + chunk, sample_cnt)
        label_votes = np.zeros((e - s, num_classes), dtype=np.int32)
        for m in mmaps:
            labels_chunk = np.array(m[s:e], dtype=np.int32)
            valid = labels_chunk < num_classes
            rows = np.arange(e - s)
            label_votes[rows[valid], labels_chunk[valid]] += 1
        zero_vote_defaults += int((label_votes.sum(axis=1) == 0).sum())
        voted[s:e] = np.argmax(label_votes, axis=1)
        del label_votes

    del mmaps

    total = max(len(voted), 1)
    counts = np.bincount(voted, minlength=num_classes)[:num_classes]
    parts = [f"{i} ({100 * c / total:.1f} %)" for i, c in enumerate(counts)]
    lines = [" | ".join(parts[i:i + 10]) for i in range(0, len(parts), 10)]
    zero_vote_msg = f"Zero-vote defaults to class 0: {zero_vote_defaults}/{total} ({100 * zero_vote_defaults / total:.1f}%)"
    msg = "Vote: " + "\n      ".join(lines) + "\n      " + zero_vote_msg
    print(msg)
    if logger is not None:
        logger.info("Vote: %s", " | ".join(parts))
        logger.info("Vote: %s", zero_vote_msg)

    return voted


class FedKDIDS(DistillationStrategy):
    name = "FedKD-IDS"
    use_model_pool = False

    def extra_log_tokens(self) -> Dict[str, float]:
        return {
            "train_rounds": self.config.train_rounds,
            "dist_rounds": self.config.dist_rounds,
            "hamming_tau": getattr(self.config, "hamming_tau", 0.5),
        }

    def setup(self, context: PipelineContext) -> None:
        config = context.config
        os.makedirs(FEDKDIDS_CACHE_DIR, exist_ok=True)

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

        _mixed = getattr(config, 'mixed_models', False)
        if not _mixed:
            reusable = create_model(context.input_dim, context.num_classes,
                                    config.batch_size, model_type=config.model_type)
            context.shared_state["init_w"] = reusable.get_weights()
            self._model = reusable
        else:
            self._model = None

        for client_id, paths in enumerate(context.paths):
            st = context.add_client_state(client_id, None, paths)
            st.data["w"] = None

        print(f"{COLORS.OKGREEN}FedKD-IDS initialized ({len(context.paths)} clients){COLORS.ENDC}")

    def run_round(self, context: PipelineContext, round_number: int) -> Dict[int, Dict[str, float]]:
        round_start = time.time()
        config = context.config
        n_public = context.shared_state["public_sample_count"]
        _mixed = getattr(config, 'mixed_models', False)
        model = self._model
        _REFRESH_EVERY = getattr(config, "cleanup_interval", 25)
        attack_type, poison_value, _ = parse_poison_config(getattr(config, "poison", None))
        tau = getattr(config, "hamming_tau", 0.5)
        _pfl_stage1 = getattr(context, 'poisoned_fl_state', None)
        _lma_stage1 = attack_type == "lma"

        if round_number > 1 and not _mixed:
            del model
            from ..memory import clear_session
            clear_session()
            aggressive_memory_cleanup()
            model = create_model(context.input_dim, context.num_classes,
                                 config.batch_size, model_type=config.model_type)
            self._model = model
        elif round_number > 1:
            from ..memory import clear_session
            clear_session()
            aggressive_memory_cleanup()

        base_mmap = np.load(_base_public_path(), mmap_mode="r")
        permutation = np.random.permutation(n_public)
        open_feature = np.array(base_mmap[permutation], dtype=np.float32)
        del base_mmap

        os.makedirs(FEDKDIDS_CACHE_DIR, exist_ok=True)
        pub_X_path = _round_public_path(round_number)

        print(f"\n{COLORS.HEADER}Round {round_number} Stage I (Verifier){COLORS.ENDC}")
        pred_files: List[str] = []
        pred_client_ids: List[int] = []

        _ckpt = getattr(config, "checkpoint", 0)
        first_s1 = 0
        first_s2 = 0
        skip_stage2_init = False
        if _ckpt:
            mid = load_mid_round(context, "fedkdids", round_number)
            if mid is not None:
                stage = mid.get("stage", 1)
                pred_files = mid.get("pred_files", [])
                pred_client_ids = mid.get("pred_client_ids", [])
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

            if _mixed:
                if client_idx > 0 and client_idx % _REFRESH_EVERY == 0:
                    from ..memory import clear_session
                    clear_session()
                    aggressive_memory_cleanup()
                from models.mixed_models import get_model_type_for_client
                arch = get_model_type_for_client(state.client_id, context.n_clients)
                model = create_model(context.input_dim, context.num_classes,
                                     config.batch_size, model_type=arch)
                if state.data["w"]:
                    model.set_weights(state.data["w"])
            else:
                if client_idx > 0 and client_idx % _REFRESH_EVERY == 0:
                    del model
                    from ..memory import clear_session
                    clear_session()
                    aggressive_memory_cleanup()
                    model = create_model(context.input_dim, context.num_classes,
                                         config.batch_size, model_type=config.model_type)
                    self._model = model
                model.set_weights(state.data["w"] or context.shared_state["init_w"])

            cid = state.client_id
            print(f"\n{COLORS.BOLD}Client {cid} Stage I{COLORS.ENDC}")
            _poisoned = cid in context.poisoned_clients

            if _poisoned and (_pfl_stage1 is not None or _lma_stage1):
                _tag = "PoisonedFL" if _pfl_stage1 is not None else "LMA"
                context.logger.info("Round %s | Client %s [%s] Stage I skipped", round_number, cid, _tag)
                if _mixed:
                    del model
                aggressive_memory_cleanup()
                continue

            _s1_poison_loader = context.per_client_loaders.get(cid, context.poison_loader) if _poisoned else None
            ds = create_client_dataset(
                state.paths["train_X"], state.paths["train_y"],
                context.input_dim, context.num_classes, config.batch_size,
                poison_loader=_s1_poison_loader,
            )
            model.fit(ds, epochs=config.train_rounds, verbose=0)
            del ds
            new_w = model.get_weights()

            poisoned_w = None
            if _poisoned and attack_type == "gradient_scale":
                poisoned_w = apply_gradient_scale_poison(new_w, poison_value)
                model.set_weights(poisoned_w)
                print(f"{COLORS.WARNING}  [POISON] Client {cid}: gradient_scale applied (×{poison_value}){COLORS.ENDC}")
                context.logger.info("Round %s | Client %s [POISON] gradient_scale ×%s applied", round_number, cid, poison_value)

            pred_path = _pred_path(cid, round_number)
            _predict_hard_labels(model, open_feature, context.num_classes, config.batch_size, pred_path)
            pred_files.append(pred_path)
            pred_client_ids.append(cid)

            _preds = np.load(pred_path, mmap_mode="r")
            _counts = np.bincount(np.array(_preds, dtype=np.int32), minlength=context.num_classes)
            _top3 = np.argsort(_counts)[::-1][:3]
            _top = " | ".join(f"cls{c}:{100*_counts[c]/max(len(_preds),1):.1f}%" for c in _top3 if _counts[c] > 0)
            _poison_tag = " [POISONED]" if _poisoned else ""
            context.logger.info("Round %s | Client %s%s | top3: %s", round_number, cid, _poison_tag, _top)
            del _preds

            state.data["w"] = poisoned_w if poisoned_w is not None else new_w

            if _ckpt and (
                client_idx == len(context.client_states) - 1
                or (client_idx + 1) % _ckpt == 0
            ):
                save_mid_round(context, "fedkdids", {
                    "round": round_number,
                    "stage": 1,
                    "last_client_idx": client_idx,
                    "pred_files": pred_files,
                    "pred_client_ids": pred_client_ids,
                    "client_weights": {
                        st.client_id: st.data["w"]
                        for st in context.client_states
                        if st.data.get("w") is not None
                    },
                })

            if _mixed:
                del model

        if _lma_stage1 and context.poisoned_clients:
            stale = context.shared_state.get("lma_stale_consensus")
            if stale is not None:
                lma_hard = lma_targets(stale, context.num_classes)
                lma_counts = np.bincount(lma_hard, minlength=context.num_classes)
                lma_top3 = np.argsort(lma_counts)[::-1][:3]
                lma_top = " | ".join(f"cls{c}:{100*lma_counts[c]/max(len(lma_hard),1):.1f}%" for c in lma_top3 if lma_counts[c] > 0)
                for cid in context.poisoned_clients:
                    pred_path = _pred_path(cid, round_number)
                    np.save(pred_path, lma_hard)
                    if cid not in pred_client_ids:
                        pred_files.append(pred_path)
                        pred_client_ids.append(cid)
                    context.logger.info("Round %s | Client %s [LMA] | top3 pred-argmax: %s", round_number, cid, lma_top)
                del lma_hard, lma_counts

        _ghost_stage1_w = poisonedfl_warmstart_weights(context.shared_state, _pfl_stage1, fallback=context.shared_state.get("init_w"), prefer_poisoned=True) if _pfl_stage1 is not None else None
        if _pfl_stage1 is not None and context.poisoned_clients and _ghost_stage1_w is not None:
            ghost = create_model(context.input_dim, context.num_classes,
                                 config.batch_size, model_type=poisonedfl_ghost_model_type(config))
            ghost.set_weights(_ghost_stage1_w)
            logits_model = ghost.get_logits_model() if hasattr(ghost, "get_logits_model") else ghost
            ghost_logits = logits_model.predict(open_feature, batch_size=config.batch_size, verbose=0)
            ghost_hard = np.argmax(ghost_logits, axis=1).astype(np.int32)
            del ghost, logits_model, ghost_logits
            ghost_counts = np.bincount(ghost_hard, minlength=context.num_classes)
            ghost_top3 = np.argsort(ghost_counts)[::-1][:3]
            ghost_top = " | ".join(f"cls{c}:{100*ghost_counts[c]/max(len(ghost_hard),1):.1f}%" for c in ghost_top3 if ghost_counts[c] > 0)
            for cid in context.poisoned_clients:
                pred_path = _pred_path(cid, round_number)
                np.save(pred_path, ghost_hard)
                if cid not in pred_client_ids:
                    pred_files.append(pred_path)
                    pred_client_ids.append(cid)
                context.logger.info("Round %s | Client %s [POISONED] | top3 pred-argmax: %s", round_number, cid, ghost_top)
            del ghost_hard, ghost_counts

        if pred_files and len(pred_files) == len(pred_client_ids):
            ordered = sorted(zip(pred_client_ids, pred_files), key=lambda item: item[0])
            pred_client_ids = [cid for cid, _ in ordered]
            pred_files = [path for _, path in ordered]

        context.logger.info("Round %s | StageI | prediction files=%d clients=%d", round_number, len(pred_files), len(pred_client_ids))

        del open_feature
        aggressive_memory_cleanup()

        # ---- Hamming filter (Verifier) ----
        print(f"\n{COLORS.HEADER}Round {round_number} Verifier (Hamming filter, tau={tau}){COLORS.ENDC}")
        if not skip_stage2_init and pred_files:
            survivor_local_idx = _hamming_filter(pred_files, tau, logger=context.logger, round_number=round_number, client_ids=pred_client_ids)
            filtered_pred_files = [pred_files[i] for i in survivor_local_idx]
            filtered_cids = [pred_client_ids[i] for i in survivor_local_idx]
            rejected_cids = [pred_client_ids[i] for i in range(len(pred_files)) if i not in set(survivor_local_idx)]
            if rejected_cids:
                context.logger.info("Round %s | HammingFilter | rejected client_ids: %s", round_number, rejected_cids)
            if context.poisoned_clients:
                poisoned_set = set(context.poisoned_clients)
                TP = len(set(rejected_cids) & poisoned_set)
                FP = len(set(rejected_cids) - poisoned_set)
                FN = len(poisoned_set - set(rejected_cids))
                TN = len(pred_client_ids) - TP - FP - FN
                context.logger.info(
                    "Round %s | HammingFilterMetrics | TP=%d FP=%d TN=%d FN=%d",
                    round_number, TP, FP, TN, FN,
                )
        else:
            filtered_pred_files = pred_files
            filtered_cids = pred_client_ids

        print(f"\n{COLORS.HEADER}Round {round_number} Stage II{COLORS.ENDC}")
        pseudo_y_path = os.path.join(FEDKDIDS_CACHE_DIR, f"r{round_number}_pseudo_y.npy")
        if not skip_stage2_init and not os.path.exists(pseudo_y_path):
            active_files = filtered_pred_files if filtered_pred_files else pred_files
            global_labels_np = hard_label_vote(active_files, context.num_classes, logger=context.logger)
            np.save(pseudo_y_path, global_labels_np)
            del global_labels_np
        aggressive_memory_cleanup()

        pub_X = np.array(np.load(pub_X_path, mmap_mode="r"), dtype=np.float32)
        pseudo_y = np.array(np.load(pseudo_y_path, mmap_mode="r"), dtype=np.int32)
        context.shared_state["lma_stale_consensus"] = np.array(pseudo_y, copy=True)
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

            _pfl_s2 = getattr(context, 'poisoned_fl_state', None)
            if state.client_id in context.poisoned_clients and (_pfl_s2 is not None or attack_type == "lma"):
                continue

            if s2_idx > 0 and s2_idx % _REFRESH_EVERY == 0:
                if not _mixed:
                    del model
                from ..memory import clear_session
                clear_session()
                aggressive_memory_cleanup()
                if not _mixed:
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

            if _mixed:
                from models.mixed_models import get_model_type_for_client
                arch = get_model_type_for_client(state.client_id, context.n_clients)
                model = create_model(context.input_dim, context.num_classes,
                                     config.batch_size, model_type=arch)

            if state.data["w"] is not None:
                model.set_weights(state.data["w"])
            elif not _mixed:
                model.set_weights(context.shared_state["init_w"])
            print(f"Client {state.client_id}: training on pseudo-labeled public data")
            model.fit(public_ds, epochs=config.dist_rounds, verbose=0)
            state.data["w"] = model.get_weights()

            if _ckpt and (
                s2_idx == len(context.client_states) - 1
                or (s2_idx + 1) % _ckpt == 0
            ):
                save_mid_round(context, "fedkdids", {
                    "round": round_number,
                    "stage": 2,
                    "last_client_idx": s2_idx,
                    "pred_files": [],
                    "pred_client_ids": [],
                    "client_weights": {
                        st.client_id: st.data["w"]
                        for st in context.client_states
                        if st.data.get("w") is not None
                    },
                })

            if _mixed:
                del model

        # ---- PoisonedFL: ghost model trained on public_ds ----
        _pfl = getattr(context, 'poisoned_fl_state', None)
        if _pfl is not None and context.poisoned_clients:
            ghost = create_model(context.input_dim, context.num_classes,
                                 config.batch_size, model_type=poisonedfl_ghost_model_type(config))
            ghost_start = poisonedfl_warmstart_weights(context.shared_state, _pfl, fallback=context.shared_state.get("init_w"))
            if ghost_start is not None:
                ghost.set_weights(ghost_start)
            print(f"{COLORS.WARNING}  [PoisonedFL] Ghost model generating logits for {len(context.poisoned_clients)} byzantine clients{COLORS.ENDC}")
            ghost_distill_loss = ghost.fit(public_ds, epochs=config.dist_rounds, verbose=0)
            ghost_w = ghost.get_weights()
            del ghost
            poisoned_w = poisonedfl_unified_weights([ghost_w], _pfl)
            poisonedfl_store_round_weights(context.shared_state, ghost_w, poisoned_w, _pfl)
            if not _mixed:
                for st in context.client_states:
                    if st.client_id in context.poisoned_clients:
                        st.data["w"] = poisoned_w
            _distill_loss, _c0, _c, _mal_norm, _alignment = poisonedfl_log_values(_pfl, ghost_distill_loss)
            context.logger.info(
                "Round %s | PoisonedFL | distill_loss=%s c0=%.4f c=%.4f mal_norm=%.4e aligned=%s | Ghost trained on public_ds, injected into %d byzantine clients",
                round_number, _distill_loss, _c0, _c, _mal_norm, _alignment, len(context.poisoned_clients),
            )
            del ghost_w, poisoned_w
            aggressive_memory_cleanup()

        del public_ds

        print(f"\n{COLORS.HEADER}Round {round_number} Global model{COLORS.ENDC}")
        pub_X = np.array(np.load(pub_X_path, mmap_mode="r"), dtype=np.float32)
        pseudo_y_arr = np.array(np.load(pseudo_y_path, mmap_mode="r"), dtype=np.int32)
        valid_mask = pseudo_y_arr >= 0
        pub_X = pub_X[valid_mask]
        pseudo_y_arr = pseudo_y_arr[valid_mask]
        y_cat = np.zeros((len(pseudo_y_arr), context.num_classes), dtype=np.float32)
        y_cat[np.arange(len(pseudo_y_arr)), pseudo_y_arr] = 1.0
        p = np.random.permutation(len(pub_X))
        global_ds = _make_public_ds(pub_X[p], y_cat[p], config.batch_size)
        del pub_X, pseudo_y_arr, y_cat, p, valid_mask

        if not _mixed:
            del model
        from ..memory import clear_session
        clear_session()
        aggressive_memory_cleanup()
        global_model = create_model(context.input_dim, context.num_classes,
                                    config.batch_size, model_type=config.model_type)
        if not _mixed:
            global_model.set_weights(context.shared_state["init_w"])
        global_model.fit(global_ds, epochs=config.dist_rounds, verbose=0)
        del global_ds

        stem = os.path.splitext(os.path.basename(context.log_filename))[0]
        record_dir = os.path.join("temp_weights", f"{stem}_weight_record", f"round_{round_number}")
        os.makedirs(record_dir, exist_ok=True)
        g_path = os.path.join(record_dir, "global_weight.bin")
        tmp = g_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(global_model.get_weights(), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, g_path)
        print(f"  Global model saved to {g_path}")

        del global_model
        clear_session()
        aggressive_memory_cleanup()
        if not _mixed:
            model = create_model(context.input_dim, context.num_classes,
                                 config.batch_size, model_type=config.model_type)
            self._model = model

        for name in os.listdir(FEDKDIDS_CACHE_DIR):
            if name.startswith(f"r{round_number}_"):
                os.remove(os.path.join(FEDKDIDS_CACHE_DIR, name))
        aggressive_memory_cleanup()

        round_time = time.time() - round_start
        context.shared_state["pipeline_elapsed_s"] = context.shared_state.get("pipeline_elapsed_s", 0.0) + round_time
        context.logger.info("Round %s completed in %.2fs", round_number, round_time)
        print(f"{COLORS.OKCYAN}Round {round_number} completed in {round_time:.2f}s{COLORS.ENDC}")

        if _ckpt:
            clear_mid_round(context, "fedkdids")

        return {}

    def finalize(self, context: PipelineContext) -> None:
        from ..memory import clear_session
        clear_session()
        if os.path.isdir(FEDKDIDS_CACHE_DIR):
            for name in os.listdir(FEDKDIDS_CACHE_DIR):
                os.remove(os.path.join(FEDKDIDS_CACHE_DIR, name))
