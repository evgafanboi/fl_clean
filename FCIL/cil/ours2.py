import gc
import os
import numpy as np
from FL.strategy.robust_filter import RobustFilterV3
from .finetune import Finetune

LOGIT_CLIP = 30.0
EVA_QUANTILE = 0.95
EVA_EPS = 1e-6


def _energy_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    row_max = np.max(scaled, axis=1, keepdims=True)
    lse = row_max[:, 0] + np.log(np.exp(scaled - row_max).sum(axis=1) + EVA_EPS)
    return (-temperature * lse).astype(np.float32)


class Ours2(Finetune):
    def __init__(self, num_classes: int, memory: float = 1.0, kd_gamma: float = 1.0,
                 robust_threshold: float = 0.9, robust_workers: int = 8):
        super().__init__(num_classes=num_classes)
        self.name = 'Ours2'
        self.memory = memory
        self.kd_gamma = kd_gamma
        self.class_order = []
        self.label_map = {}
        self.robust_filter = RobustFilterV3(robust_threshold=robust_threshold, reference="Blom", workers=robust_workers)
        self.task_candidates = None
        self._mem_dir = None
        self._mem_files = []
        self.eva_thresholds = {}
        self._last_ce = 0.0
        self._last_kd = 0.0
        self._last_survivors = 0
        self._last_survivor_clients = []
        self._last_eva_support = 0.0

    def before_task(self, task_id: int, task_classes: list):
        super().before_task(task_id, task_classes)
        for cls in task_classes:
            if cls not in self.label_map:
                self.label_map[cls] = len(self.class_order)
                self.class_order.append(cls)

    def map_labels_np(self, y):
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        return np.vectorize(self.label_map.get, otypes=[int])(labels)

    def build_public_candidates(self, config, task_id: int, active_n: int, num_tasks: int):
        from ..pipeline import cil_partition_dir, _task_active_n
        base = cil_partition_dir(config.partition_root, config.partition_type, config.n_clients)
        xs = []
        for cid in range(_task_active_n(config.n_clients, num_tasks, task_id)):
            x_path = base / f"client_{cid}_task_{task_id}_X_public.npy"
            if x_path.exists():
                xs.append(np.asarray(np.load(str(x_path), mmap_mode='r'), dtype=np.float32))
        self.task_candidates = np.concatenate(xs, axis=0).astype(np.float32) if xs else np.empty((0, 0), dtype=np.float32)
        return len(self.task_candidates)

    def update_eva_threshold(self, client_id: int, model, X_path: str, batch_size: int):
        logits_model = model.get_logits_model() if hasattr(model, 'get_logits_model') else model
        X = np.load(X_path, mmap_mode='r')
        energies = []
        for start in range(0, len(X), 100_000):
            chunk = np.asarray(X[start:start + 100_000], dtype=np.float32)
            logits = logits_model.predict(chunk, batch_size=batch_size)
            logits = np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)
            energies.append(_energy_from_logits(logits))
        del X
        if energies:
            energy = np.concatenate(energies, axis=0)
            self.eva_thresholds[client_id] = float(np.quantile(energy, EVA_QUANTILE))
        gc.collect()

    def freeze_task_memory(self, scratch_model, client_weights, config, task_id: int, client_ids=None):
        if client_ids is None:
            client_ids = list(range(len(client_weights)))
        if self.task_candidates is None or len(self.task_candidates) == 0:
            self._last_survivor_clients = list(range(len(client_weights)))
            self._last_survivors = len(client_weights)
            self.eva_thresholds.clear()
            return

        n_samples = len(self.task_candidates)
        n_clients = len(client_weights)
        total_discard = np.zeros(n_clients, dtype=np.int64)
        total_eva = 0
        total_eva_cells = 0
        chunk_size = max(config.batch_size * 8, 8192)
        if self._mem_dir is None:
            self._mem_dir = os.path.join("temp_weights", f"ours2_memory_{config.partition_type}_{config.n_clients}client")
            os.makedirs(self._mem_dir, exist_ok=True)
            self._mem_files = []

        for start in range(0, n_samples, chunk_size):
            X_chunk = self.task_candidates[start:start + chunk_size]
            preds = []
            eva_keep = np.ones((len(X_chunk), n_clients), dtype=bool)
            for k, (cid, weights) in enumerate(zip(client_ids, client_weights)):
                scratch_model.set_weights(weights)
                pred = scratch_model.get_logits_model().predict(X_chunk, batch_size=config.batch_size)
                pred = np.clip(pred, -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)
                preds.append(pred)
                if cid in self.eva_thresholds:
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
            pseudo = consensus.argmax(axis=1).astype(np.int64)
            keep_idx = []
            for cls in range(consensus.shape[1]):
                idx = np.flatnonzero(pseudo == cls)
                if not len(idx):
                    continue
                n_keep = max(1, int(len(idx) * float(self.memory) / 100.0))
                if n_keep >= len(idx):
                    keep_idx.append(idx)
                    continue
                cls_logits = consensus[idx]
                centroid = cls_logits.mean(axis=0)
                dist = np.linalg.norm(cls_logits - centroid, axis=1)
                keep_idx.append(idx[np.argsort(dist)[:n_keep]])
            keep_idx = np.concatenate(keep_idx, axis=0) if keep_idx else np.array([], dtype=np.int64)
            if len(keep_idx):
                chunk_path = os.path.join(self._mem_dir, f"task_{task_id}_chunk_{len(self._mem_files)}.npz")
                np.savez_compressed(chunk_path, X=X_chunk[keep_idx].astype(np.float32), y=pseudo[keep_idx].astype(np.int64), logits=consensus[keep_idx].astype(np.float32))
                self._mem_files.append(chunk_path)
            total_discard += discard_mask.sum(axis=0)
            total_eva += int(eva_keep.sum())
            total_eva_cells += eva_keep.size
            del X_chunk, preds, logits_stack, discard_mask, robust_keep, keep_mask, consensus, pseudo, keep_idx, eva_keep
            gc.collect()

        discard_frac = total_discard.astype(np.float64) / max(n_samples, 1)
        survivor = discard_frac <= self.robust_filter.robust_threshold
        self._last_survivor_clients = [i for i in range(n_clients) if survivor[i]]
        self._last_survivors = int(survivor.sum())
        self._last_eva_support = float(total_eva / max(total_eva_cells, 1))
        print(f"  Ours2 freeze T{task_id}: Blom {self._last_survivors}/{n_clients} clients | EVA support={self._last_eva_support:.3f}")
        self.task_candidates = None
        self.eva_thresholds.clear()
        print(f"  Ours2 frozen memory: {len(self._mem_files)} disk chunks across {len(self.class_order)} classes")
        gc.collect()

    def build_dataset(self, X_path: str, y_path: str, batch_size: int):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        X_priv = np.load(X_path, mmap_mode='r')
        y_priv = np.load(y_path, mmap_mode='r')
        y_priv = y_priv if len(y_priv.shape) == 1 else np.argmax(y_priv, axis=1)
        X_parts = [np.asarray(X_priv, dtype=np.float32)]
        y_parts = [np.vectorize(self.label_map.get, otypes=[np.int64])(y_priv).astype(np.int64)]
        kd_parts = [np.zeros((len(y_priv), len(self.class_order)), dtype=np.float32)]

        if self._mem_files:
            for chunk_file in self._mem_files:
                data = np.load(chunk_file)
                X_parts.append(data['X'])
                y_parts.append(data['y'].astype(np.int64))
                logits = data['logits']
                if logits.shape[1] < len(self.class_order):
                    pad = np.zeros((logits.shape[0], len(self.class_order) - logits.shape[1]), dtype=np.float32)
                    logits = np.concatenate([logits, pad], axis=1)
                kd_parts.append(logits.astype(np.float32))

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        kd = np.concatenate(kd_parts, axis=0)

        if X.shape[0] == 0:
            dummy_X = torch.empty((1, 1), dtype=torch.float32)
            dummy_y = torch.empty((1, len(self.class_order)), dtype=torch.float32)
            dummy_kd = torch.empty((1, len(self.class_order)), dtype=torch.float32)
            loader = DataLoader(TensorDataset(dummy_X, dummy_y, dummy_kd), batch_size=batch_size, shuffle=True, drop_last=False)
            return loader, 0

        y_oh = np.zeros((len(y), len(self.class_order)), dtype=np.float32)
        y_oh[np.arange(len(y)), y] = 1.0
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y_oh), torch.from_numpy(kd)), batch_size=batch_size, shuffle=True, drop_last=False)
        return loader, len(X)

    def get_train_step(self, model, optimizer, loss_fn):
        import torch
        import torch.nn.functional as F
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.nn.to(dev)
        def step(X_b, y_b, kd_b):
            X_b = X_b.to(dev); y_b = y_b.to(dev); kd_b = kd_b.to(dev)
            y_cls = y_b.argmax(dim=1)
            mask = kd_b.abs().sum(dim=1) > 0
            model.nn.train()
            optimizer.zero_grad(set_to_none=True)
            logits = torch.clamp(model.nn(X_b, return_logits=True), -LOGIT_CLIP, LOGIT_CLIP)
            ce_loss = F.cross_entropy(logits, y_cls, label_smoothing=0.05)
            kd_loss = torch.tensor(0.0, device=dev)
            if mask.any():
                kd_loss = F.kl_div(F.log_softmax(logits[mask], dim=1), F.softmax(kd_b[mask], dim=1), reduction='batchmean')
            total = ce_loss + self.kd_gamma * kd_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 1.0)
            optimizer.step()
            self._last_ce = float(ce_loss.item())
            self._last_kd = float(kd_loss.item()) * self.kd_gamma
            return ce_loss.item(), kd_loss.item(), total.item()
        return step
