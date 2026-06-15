import gc
import numpy as np
from FL.strategy.robust_filter import RobustFilterV3
from .finetune import Finetune

LOGIT_CLIP = 30.0


class Ours2(Finetune):
    def __init__(self, num_classes: int, memory: float = 1.0, kd_gamma: float = 1.0,
                 robust_threshold: float = 0.9, robust_workers: int = 8, no_filter: bool = True):
        super().__init__(num_classes=num_classes)
        self.name = 'Ours2'
        self.memory = memory
        self.kd_gamma = kd_gamma
        self.no_filter = no_filter
        self.class_order = []
        self.label_map = {}
        self.robust_filter = RobustFilterV3(robust_threshold=robust_threshold, reference="Blom", workers=robust_workers)
        self.task_candidates = None
        self.frozen_X = None
        self.frozen_y = None
        self.frozen_logits = None
        self._last_ce = 0.0
        self._last_kd = 0.0
        self._last_survivors = 0
        self._last_survivor_clients = []

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

    def freeze_task_memory(self, scratch_model, client_weights, config, task_id: int):
        if self.task_candidates is None or len(self.task_candidates) == 0:
            self._last_survivor_clients = list(range(len(client_weights)))
            self._last_survivors = len(client_weights)
            return
        preds = []
        for weights in client_weights:
            scratch_model.set_weights(weights)
            pred = scratch_model.get_logits_model().predict(self.task_candidates, batch_size=config.batch_size)
            preds.append(np.clip(pred, -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32))
        logits_stack = np.stack(preds, axis=1)
        if self.no_filter:
            consensus = logits_stack.mean(axis=1).astype(np.float32)
            self._last_survivor_clients = list(range(len(client_weights)))
            self._last_survivors = len(client_weights)
            print(f"  Ours2 freeze T{task_id}: mean logits {len(client_weights)} clients")
        else:
            discard_mask, _ = self.robust_filter.count_discards_mask(logits_stack)
            keep_mask = ~discard_mask
            support = keep_mask.sum(axis=1)
            bad = support == 0
            if bad.any():
                keep_mask[bad] = True
                support[bad] = keep_mask.shape[1]
            consensus = (np.einsum('nk,nkd->nd', keep_mask.astype(np.float32), logits_stack) / support[:, None]).astype(np.float32)
            discard_frac = discard_mask.sum(axis=0).astype(np.float64) / max(len(self.task_candidates), 1)
            survivor = discard_frac <= self.robust_filter.robust_threshold
            self._last_survivor_clients = [i for i in range(len(client_weights)) if survivor[i]]
            self._last_survivors = int(survivor.sum())
            print(f"  Ours2 freeze T{task_id}: Blom {self._last_survivors}/{len(client_weights)} clients")
        pseudo = consensus.argmax(axis=1).astype(np.int64)
        keep_idx = []
        rng = np.random.default_rng(2026 + task_id)
        n_classes = consensus.shape[1]
        for cls in range(n_classes):
            idx = np.flatnonzero(pseudo == cls)
            if len(idx):
                n_keep = max(1, int(len(idx) * float(self.memory) / 100.0))
                keep_idx.append(rng.choice(idx, size=min(n_keep, len(idx)), replace=False))
        keep_idx = np.concatenate(keep_idx, axis=0) if keep_idx else np.array([], dtype=np.int64)
        rng.shuffle(keep_idx)
        new_X = self.task_candidates[keep_idx].astype(np.float32)
        new_y = pseudo[keep_idx].astype(np.int64)
        new_logits = consensus[keep_idx].astype(np.float32)
        if self.frozen_X is None:
            self.frozen_X = new_X
            self.frozen_y = new_y
            self.frozen_logits = new_logits
        else:
            old_w = self.frozen_logits.shape[1]
            new_w = new_logits.shape[1]
            if old_w < new_w:
                pad = np.zeros((len(self.frozen_logits), new_w - old_w), dtype=np.float32)
                self.frozen_logits = np.concatenate([self.frozen_logits, pad], axis=1)
            self.frozen_X = np.concatenate([self.frozen_X, new_X], axis=0)
            self.frozen_y = np.concatenate([self.frozen_y, new_y], axis=0)
            self.frozen_logits = np.concatenate([self.frozen_logits, new_logits], axis=0)
        self.task_candidates = None
        print(f"  Ours2 frozen memory: {len(self.frozen_X)} exemplars across {len(self.class_order)} classes")
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
        if self.frozen_X is not None and len(self.frozen_X):
            X_parts.append(self.frozen_X)
            y_parts.append(self.frozen_y.astype(np.int64))
            logits = self.frozen_logits
            w = logits.shape[1]
            target_w = len(self.class_order)
            if w < target_w:
                logits = np.concatenate([logits, np.zeros((len(logits), target_w - w), dtype=np.float32)], axis=1)
            kd_parts.append(logits.astype(np.float32))
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        kd = np.concatenate(kd_parts, axis=0)
        if X.shape[0] == 0:
            # Create a minimal dummy DataLoader so that torch's RandomSampler has a positive length.
            # The pipeline will skip this client because n_samples == 0, so the dummy data will never be used.
            dummy_X = torch.empty((1, 1), dtype=torch.float32)
            dummy_y = torch.empty((1, len(self.class_order)), dtype=torch.float32)
            dummy_kd = torch.empty((1, len(self.class_order)), dtype=torch.float32)
            loader = DataLoader(TensorDataset(dummy_X, dummy_y, dummy_kd),
                                batch_size=batch_size, shuffle=True, drop_last=False)
            return loader, 0
        y_oh = np.zeros((len(y), len(self.class_order)), dtype=np.float32)
        y_oh[np.arange(len(y)), y] = 1.0
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y_oh), torch.from_numpy(kd)),
                            batch_size=batch_size, shuffle=True, drop_last=False)
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
