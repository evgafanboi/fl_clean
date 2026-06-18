import gc
import numpy as np
from .ours2 import Ours2, LOGIT_CLIP

class Ours3(Ours2):
    def __init__(self, num_classes: int, memory: float = 1.0, kd_gamma: float = 1.0,
                 robust_threshold: float = 0.9, robust_workers: int = 8,
                 ekd_epochs: int = 1, ekd_lambda: float = 1.0):
        super().__init__(num_classes, memory, kd_gamma, robust_threshold, robust_workers, ekd_epochs, ekd_lambda)
        self.name = 'Ours3'
        self._last_replay_support = 0.0
        self._last_replay_kept = 0
        self._last_replay_total = 0
        self._scratch_pool = None

    def _consensus_logits(self, scratch_model, X, client_weights, client_ids, config):
        if not self._scratch_pool:
            return super()._consensus_logits(scratch_model, X, client_weights, client_ids, config)
        n_samples = len(X)
        n_clients = len(client_weights)
        chunk_size = max(config.batch_size * 8, 8192)
        X_chunks, consensus_chunks = [], []
        total_discard = np.zeros(n_clients, dtype=np.int64)
        total_eva = 0
        total_eva_cells = 0
        for start in range(0, n_samples, chunk_size):
            X_chunk = X[start:start + chunk_size]
            preds = []
            eva_keep = np.ones((len(X_chunk), n_clients), dtype=bool)
            for k, (cid, weights) in enumerate(zip(client_ids, client_weights)):
                model = self._scratch_pool[cid]
                model.set_weights(weights)
                pred = self._raw_logits(model, X_chunk, config.batch_size)
                pred = np.clip(pred, -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)
                preds.append(pred)
                if cid in self.eva_thresholds:
                    from .ours2 import _energy_from_logits
                    eva_keep[:, k] = _energy_from_logits(pred) <= self.eva_thresholds[cid]
            logits_stack = np.stack(preds, axis=1)
            discard_mask, _ = self.robust_filter.count_discards_mask(logits_stack)
            robust_keep = ~discard_mask
            keep_mask = eva_keep & robust_keep
            support = keep_mask.sum(axis=1)
            bad = support == 0
            if bad.any():
                eva_support = eva_keep[bad].sum(axis=1)
                fallback = np.where(eva_support[:, None] > 0, eva_keep[bad], robust_keep[bad])
                still_bad = fallback.sum(axis=1) == 0
                fallback[still_bad] = True
                keep_mask[bad] = fallback
                support[bad] = keep_mask[bad].sum(axis=1)
            consensus = (np.einsum('nk,nkd->nd', keep_mask.astype(np.float32), logits_stack) / support[:, None]).astype(np.float32)
            X_chunks.append(X_chunk.copy())
            consensus_chunks.append(consensus)
            total_discard += discard_mask.sum(axis=0)
            total_eva += int(eva_keep.sum())
            total_eva_cells += eva_keep.size
            del preds, logits_stack, discard_mask, robust_keep, keep_mask, eva_keep
            gc.collect()
        return X_chunks, consensus_chunks, total_discard, total_eva, total_eva_cells

    def distill_round(self, scratch_model, client_models, client_weights, config, task_id: int, active_n: int, num_tasks: int, client_ids=None):
        if client_ids is None:
            client_ids = list(range(len(client_weights)))
        if self.task_candidates is None or len(self.task_candidates) == 0:
            self._last_kd = 0.0
            self._last_survivor_clients = list(range(len(client_weights)))
            self._last_survivors = len(client_weights)
            return 0.0

        n_samples = len(self.task_candidates)
        n_clients = len(client_weights)
        cur_X_chunks, cur_consensus_chunks, total_discard, total_eva, total_eva_cells = (
            self._consensus_logits(scratch_model, self.task_candidates, client_weights, client_ids, config))

        discard_frac = total_discard.astype(np.float64) / max(n_samples, 1)
        survivor = discard_frac <= self.robust_filter.robust_threshold
        self._last_survivor_clients = [i for i in range(n_clients) if survivor[i]]
        self._last_survivors = int(survivor.sum())
        self._last_eva_support = float(total_eva / max(total_eva_cells, 1))

        n_classes = len(self.class_order)
        cur_X = np.concatenate(cur_X_chunks, axis=0)
        cur_logits = np.concatenate(cur_consensus_chunks, axis=0)
        if cur_logits.shape[1] < n_classes:
            pad = np.zeros((cur_logits.shape[0], n_classes - cur_logits.shape[1]), dtype=np.float32)
            cur_logits = np.concatenate([cur_logits, pad], axis=1)

        replay_X_parts, replay_logits_parts = [], []
        replay_kept, replay_total = 0, 0
        for chunk_file in self._mem_files:
            data = np.load(chunk_file)
            X_old = np.asarray(data['X'], dtype=np.float32)
            y_old = np.asarray(data['y'], dtype=np.int64)
            X_chunks_old, cons_chunks_old, _, _, _ = self._consensus_logits(
                scratch_model, X_old, client_weights, client_ids, config)
            cons_old = np.concatenate(cons_chunks_old, axis=0)
            X_old_all = np.concatenate(X_chunks_old, axis=0)
            if cons_old.shape[1] < n_classes:
                pad = np.zeros((cons_old.shape[0], n_classes - cons_old.shape[1]), dtype=np.float32)
                cons_old = np.concatenate([cons_old, pad], axis=1)
            keep = cons_old.argmax(axis=1) == y_old
            replay_total += len(y_old)
            replay_kept += int(keep.sum())
            if keep.any():
                replay_X_parts.append(X_old_all[keep])
                replay_logits_parts.append(cons_old[keep])
            del X_old, y_old, X_chunks_old, cons_chunks_old, cons_old, X_old_all, keep
            gc.collect()

        if replay_X_parts:
            X_all = np.concatenate([cur_X] + replay_X_parts, axis=0).astype(np.float32)
            L_all = np.concatenate([cur_logits] + replay_logits_parts, axis=0).astype(np.float32)
        else:
            X_all, L_all = cur_X.astype(np.float32), cur_logits.astype(np.float32)
        self._last_replay_kept = replay_kept
        self._last_replay_total = replay_total
        self._last_replay_support = float(replay_kept) / max(replay_total, 1)

        total_ekd, n_students = 0.0, 0
        for student in client_models.values():
            total_ekd += self._ekd_pt(student, X_all, L_all, config.batch_size)
            n_students += 1
        self._last_kd = total_ekd / max(n_students, 1)
        del cur_X_chunks, cur_consensus_chunks, cur_X, cur_logits, X_all, L_all, replay_X_parts, replay_logits_parts
        gc.collect()
        return self._last_kd
