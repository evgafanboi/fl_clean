from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List

import numpy as np
from ..backend import get_torch_loader_kwargs as _torch_loader_kwargs, use_tf as _use_tf
if _use_tf():
    import tensorflow as tf
    from models.dense_discri import create_discriminator
else:
    from models.pt_dense_discri import create_discriminator

from ..colors import COLORS
from ..data_utils import create_client_dataset
from ..memory import aggressive_memory_cleanup
from ..context import PipelineContext, ModelPool
from ..poison_utils import parse_poison_config, apply_gradient_scale_poison, poisonedfl_unified_weights
from .base import DistillationStrategy
from ._checkpoint import save_mid_round, load_mid_round, clear_mid_round
from .common import create_model, load_public_dataset_from_clients, lma_targets, numpy_from_dataset, poisonedfl_ghost_model_type

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
                      **_torch_loader_kwargs())


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
    dis_rng: np.random.RandomState,
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
    sure_known_feature = np.array(X_mmap, dtype=np.float32)
    del X_mmap

    if len(sure_unknown_feature) > len(sure_known_feature):
        idx = dis_rng.choice(len(sure_unknown_feature), len(sure_known_feature), replace=False)
        sure_unknown_feature = sure_unknown_feature[idx]

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
                             **_torch_loader_kwargs())
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

    logits = logits_model.predict(X_open, batch_size=batch_size, verbose=0)
    probs = _softmax_np(logits)
    del logits

    dis_pred = discri_model.predict(X_open, batch_size=batch_size, verbose=0).reshape(-1)
    probs[dis_pred > 0.5] = 1.0 / num_classes
    del dis_pred

    max_p = np.max(probs, axis=1)
    best = np.argmax(probs, axis=1).astype(np.int32)
    hard_labels = np.where(max_p > boundary, best, np.int32(num_classes)).astype(np.int32)
    del probs, max_p, best

    np.save(output_path, hard_labels)
    del hard_labels


def hard_label_vote(pred_files: List[str], num_classes: int, logger=None) -> np.ndarray:
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

    total = max(len(voted), 1)
    counts = np.bincount(voted, minlength=num_classes)[:num_classes]
    parts = [f"{i} ({100 * c / total:.1f} %)" for i, c in enumerate(counts)]
    lines = [" | ".join(parts[i:i + 10]) for i in range(0, len(parts), 10)]
    msg = "Vote: " + "\n      ".join(lines)
    print(msg)
    if logger is not None:
        logger.info("Vote: %s", " | ".join(parts))

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

        _mixed = getattr(config, 'mixed_models', False)
        if not _mixed:
            reusable = create_model(context.input_dim, context.num_classes,
                                    config.batch_size, model_type=config.model_type)
            context.shared_state["init_w"] = reusable.get_weights()
            self._model = reusable
        else:
            self._model = None

        disc_pool = ModelPool(
            pool_size=min(5, len(context.paths)),
            factory_fn=lambda: create_discriminator(context.input_dim),
        )
        context.shared_state["disc_pool"] = disc_pool
        context.shared_state["dis_rng"] = np.random.RandomState(42)

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
        dis_rng = context.shared_state["dis_rng"]
        _mixed = getattr(config, 'mixed_models', False)
        model = self._model
        _REFRESH_EVERY = getattr(config, "cleanup_interval", 25)
        attack_type, poison_value, _ = parse_poison_config(getattr(config, "poison", None))
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
            disc_pool.refresh()
        elif round_number > 1:
            from ..memory import clear_session
            clear_session()
            aggressive_memory_cleanup()
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

            if _mixed:
                if client_idx > 0 and client_idx % _REFRESH_EVERY == 0:
                    from ..memory import clear_session
                    clear_session()
                    aggressive_memory_cleanup()
                    disc_pool.refresh()
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
                    disc_pool.refresh()
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
            old_w = model.get_weights()
            model.fit(ds, epochs=config.train_rounds, verbose=0)
            del ds
            new_w = model.get_weights()

            if np.sum(state.data["class_counts"] > 0) <= 1:
                print("  Skipping discriminator (insufficient classes)")
                if cid in context.poisoned_clients:
                    context.logger.info("Round %s | Client %s [POISONED] SKIPPED (insufficient classes) — excluded from vote", round_number, cid)
                state.data["w"] = new_w
                if _mixed:
                    del model
                aggressive_memory_cleanup()
                continue

            disc = disc_pool.checkout(cid)
            trained = train_discriminator(
                model, disc, open_feature,
                config.dis_rounds, config.batch_size,
                state.paths["train_X"], dis_rng,
            )
            if not trained:
                print("  Discriminator training skipped (no uncertain samples)")
                if cid in context.poisoned_clients:
                    context.logger.info("Round %s | Client %s [POISONED] SKIPPED (discriminator not trained) — excluded from vote", round_number, cid)
                disc_pool.checkin(cid, disc)
                state.data["w"] = new_w
                if _mixed:
                    del model
                aggressive_memory_cleanup()
                continue

            poisoned_w = None
            if cid in context.poisoned_clients and attack_type == "gradient_scale":
                poisoned_w = apply_gradient_scale_poison(new_w, poison_value)
                model.set_weights(poisoned_w)
                print(f"{COLORS.WARNING}  [POISON] Client {cid}: gradient_scale applied (×{poison_value}){COLORS.ENDC}")
                context.logger.info("Round %s | Client %s [POISON] gradient_scale ×%s applied — WILL vote", round_number, cid, poison_value)

            pred_path = _pred_path(cid, round_number)
            predict_with_discriminator(
                model, disc, open_feature,
                context.num_classes, config.batch_size, pred_path,
            )
            disc_pool.checkin(cid, disc)
            pred_files.append(pred_path)

            _preds = np.load(pred_path, mmap_mode="r")
            _known_mask = _preds < context.num_classes
            _known = int(_known_mask.sum())
            _top = ""
            if _known > 0:
                _counts = np.bincount(_preds[_known_mask].astype(np.int32), minlength=context.num_classes)
                _top3 = np.argsort(_counts)[::-1][:3]
                _top = " | ".join(f"cls{c}:{100*_counts[c]/max(_known,1):.1f}%" for c in _top3 if _counts[c] > 0)
            _poison_tag = " [POISONED]" if cid in context.poisoned_clients else ""
            context.logger.info("Round %s | Client %s%s | known=%d/%d (%.1f%%) | top3: %s",
                round_number, cid, _poison_tag, _known, len(_preds), 100*_known/max(len(_preds),1), _top)
            del _preds

            state.data["w"] = poisoned_w if poisoned_w is not None else new_w

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
                    if pred_path not in pred_files:
                        pred_files.append(pred_path)
                    context.logger.info("Round %s | Client %s [LMA] | top3 pred-argmax: %s", round_number, cid, lma_top)
                del lma_hard, lma_counts

        if _pfl_stage1 is not None and context.poisoned_clients and context.shared_state.get("poisonedfl_ghost_w") is not None:
            ghost = create_model(context.input_dim, context.num_classes,
                                 config.batch_size, model_type=poisonedfl_ghost_model_type(config))
            ghost.set_weights(context.shared_state["poisonedfl_ghost_w"])
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
                pred_files.append(pred_path)
                context.logger.info("Round %s | Client %s [POISONED] | top3 pred-argmax: %s", round_number, cid, ghost_top)
            del ghost_hard, ghost_counts

        del open_feature
        aggressive_memory_cleanup()

        print(f"\n{COLORS.HEADER}Round {round_number} Stage II{COLORS.ENDC}")
        pseudo_y_path = os.path.join(SSFLIDS_CACHE_DIR, f"r{round_number}_pseudo_y.npy")
        if not skip_stage2_init and not os.path.exists(pseudo_y_path):
            global_labels_np = hard_label_vote(pred_files, context.num_classes, logger=context.logger)
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
            if _pfl_s2 is not None and state.client_id in context.poisoned_clients:
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

            model.set_weights(state.data["w"] if state.data["w"] is not None else context.shared_state["init_w"])
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

            if _mixed:
                del model

        # ---- PoisonedFL: ghost model trained on public_ds ----
        _pfl = getattr(context, 'poisoned_fl_state', None)
        if _pfl is not None and context.poisoned_clients:
            ghost = create_model(context.input_dim, context.num_classes,
                                 config.batch_size, model_type=poisonedfl_ghost_model_type(config))
            if context.shared_state.get("poisonedfl_ghost_w") is not None:
                ghost.set_weights(context.shared_state["poisonedfl_ghost_w"])
            ghost.fit(public_ds, epochs=config.dist_rounds, verbose=0)
            ghost_w = ghost.get_weights()
            del ghost
            poisoned_w = poisonedfl_unified_weights([ghost_w], _pfl)
            context.shared_state["poisonedfl_ghost_w"] = [w.copy() for w in poisoned_w]
            if not _mixed:
                for st in context.client_states:
                    if st.client_id in context.poisoned_clients:
                        st.data["w"] = poisoned_w
            context.logger.info("Round %s | PoisonedFL | Ghost trained on public_ds, injected into %d byzantine clients", round_number, len(context.poisoned_clients))
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
