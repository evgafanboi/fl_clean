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
        self.public_X = None
        self.public_y = None
        self.consensus_logits = None
        self.consensus_labels = None
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

    def build_public_memory(self, config, task_id: int, active_n: int, num_tasks: int):
        from ..pipeline import cil_partition_dir, _task_active_n
        base = cil_partition_dir(config.partition_root, config.partition_type, config.n_clients)
        known = set(self.class_order)
        xs, ys = [], []
        for task_idx in range(task_id + 1):
            for cid in range(_task_active_n(config.n_clients, num_tasks, task_idx)):
                x_path = base / f"client_{cid}_task_{task_idx}_X_public.npy"
                y_path = base / f"client_{cid}_task_{task_idx}_y_public.npy"
                if x_path.exists() and y_path.exists():
                    X = np.load(str(x_path), mmap_mode='r')
                    y = np.load(str(y_path), mmap_mode='r')
                    labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
                    mask = np.isin(labels, list(known))
                    if mask.any():
                        xs.append(np.asarray(X[mask], dtype=np.float32))
                        ys.append(np.asarray(labels[mask], dtype=np.int64))
        X_all = np.concatenate(xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        per_class = max(1, int(len(y_all) * float(self.memory) / max(len(known), 1)))
        keep = []
        rng = np.random.default_rng(2026 + task_id)
        for cls in sorted(known):
            idx = np.flatnonzero(y_all == cls)
            if len(idx):
                keep.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
        keep = np.concatenate(keep, axis=0)
        rng.shuffle(keep)
        self.public_X = X_all[keep].astype(np.float32)
        self.public_y = y_all[keep].astype(np.int64)
        self.consensus_logits = None
        return len(self.public_y), per_class

    def build_dataset(self, X_path: str, y_path: str, batch_size: int):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        X_priv = np.load(X_path, mmap_mode='r')
        y_priv = np.load(y_path, mmap_mode='r')
        y_priv = y_priv if len(y_priv.shape) == 1 else np.argmax(y_priv, axis=1)
        X_parts = [np.asarray(X_priv, dtype=np.float32)]
        y_parts = [np.vectorize(self.label_map.get, otypes=[np.int64])(y_priv).astype(np.int64)]
        kd_parts = [np.zeros((len(y_priv), len(self.class_order)), dtype=np.float32)]
        if self.public_X is not None and len(self.public_X):
            X_parts.append(self.public_X)
            if self.consensus_logits is None:
                y_parts.append(np.vectorize(self.label_map.get, otypes=[np.int64])(self.public_y).astype(np.int64))
                kd_parts.append(np.zeros((len(self.public_y), len(self.class_order)), dtype=np.float32))
            else:
                y_parts.append(self.consensus_labels.astype(np.int64))
                kd_parts.append(self.consensus_logits.astype(np.float32))
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        kd = np.concatenate(kd_parts, axis=0)
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

    def update_consensus(self, scratch_model, client_weights, config):
        preds = []
        for weights in client_weights:
            scratch_model.set_weights(weights)
            pred = scratch_model.get_logits_model().predict(self.public_X, batch_size=config.batch_size)
            preds.append(np.clip(pred, -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32))
        logits = np.stack(preds, axis=1)
        if self.no_filter:
            consensus = logits.mean(axis=1).astype(np.float32)
            self._last_survivor_clients = list(range(len(client_weights)))
            self._last_survivors = len(client_weights)
            print(f"  Ours2 mean logits: {len(client_weights)}/{len(client_weights)} clients")
        else:
            discard_mask, _ = self.robust_filter.count_discards_mask(logits)
            keep = ~discard_mask
            support = keep.sum(axis=1)
            bad = support == 0
            if bad.any():
                keep[bad] = True
                support[bad] = keep.shape[1]
            consensus = (np.einsum('nk,nkd->nd', keep.astype(np.float32), logits) / support[:, None]).astype(np.float32)
            discard_frac = discard_mask.sum(axis=0).astype(np.float64) / max(len(self.public_X), 1)
            survivor = discard_frac <= self.robust_filter.robust_threshold
            self._last_survivor_clients = [i for i in range(len(client_weights)) if survivor[i]]
            self._last_survivors = int(survivor.sum())
            print(f"  Ours2 Blom logits: {self._last_survivors}/{len(client_weights)} clients")
        self.consensus_logits = consensus
        self.consensus_labels = consensus.argmax(axis=1).astype(np.int64)
        gc.collect()
