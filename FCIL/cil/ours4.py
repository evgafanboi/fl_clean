import numpy as np
from .ours2 import LOGIT_CLIP
from .ours3 import Ours3


class Ours4(Ours3):
    def __init__(self, num_classes: int, memory: float = 1.0, kd_gamma: float = 1.0,
                 robust_threshold: float = 0.9, robust_workers: int = 8,
                 ekd_epochs: int = 1, ekd_lambda: float = 1.0, replay_cap: bool = True,
                 replay_min_per_class: int = 128, replay_balance: float = 1.0,
                 entropy_beta: float = 0.02):
        super().__init__(num_classes=num_classes,
                         memory=memory,
                         kd_gamma=kd_gamma,
                         robust_threshold=robust_threshold,
                         robust_workers=robust_workers,
                         ekd_epochs=ekd_epochs,
                         ekd_lambda=ekd_lambda,
                         replay_cap=replay_cap,
                         replay_min_per_class=replay_min_per_class)
        self.name = 'Ours4'
        self.replay_balance = float(replay_balance)
        self.entropy_beta = float(entropy_beta)
        self._last_replay_budget = 0
        self._last_replay_per_class = 0
        self._last_private_size = 0
        self._last_exemplar_total = 0
        self._last_old_classes = 0
        self._last_kept_count = 0

    def _compute_replay_budget(self, n_private: int, n_old_classes: int, exemplar_total: int):
        budget = int(np.floor(float(n_private) * self.replay_balance))
        budget = max(budget, n_old_classes * int(self.replay_min_per_class))
        budget = min(budget, exemplar_total)
        per_class = int(np.floor(budget / max(n_old_classes, 1))) if n_old_classes > 0 else 0
        return budget, per_class

    def get_train_step(self, model, optimizer, loss_fn):
        import torch
        import torch.nn.functional as F
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.nn.to(dev)
        beta = self.entropy_beta
        def step(X_b, y_b):
            X_b = X_b.to(dev); y_b = y_b.to(dev)
            y_cls = y_b.argmax(dim=1) if y_b.ndim > 1 else y_b.long()
            model.nn.train()
            optimizer.zero_grad(set_to_none=True)
            logits = torch.clamp(model.nn(X_b, return_logits=True), -LOGIT_CLIP, LOGIT_CLIP)
            ce_loss = F.cross_entropy(logits, y_cls, label_smoothing=0.05)
            if beta > 0:
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
                total = ce_loss - beta * entropy
            else:
                total = ce_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.nn.parameters(), 1.0)
            optimizer.step()
            self._last_ce = float(ce_loss.item())
            return ce_loss.item(), 0.0, total.item()
        return step

    def build_dataset(self, X_path: str, y_path: str, batch_size: int):
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        X_priv = np.load(X_path, mmap_mode='r')
        y_priv = np.load(y_path, mmap_mode='r')
        y_priv = y_priv if len(y_priv.shape) == 1 else np.argmax(y_priv, axis=1)
        y_priv_mapped = np.vectorize(self.label_map.get, otypes=[np.int64])(y_priv).astype(np.int64)

        X_parts = [np.asarray(X_priv, dtype=np.float32)]
        y_parts = [y_priv_mapped]

        replay_X, replay_y = [], []
        if self._mem_files:
            for chunk_file in self._mem_files:
                data = np.load(chunk_file)
                replay_X.append(np.asarray(data['X'], dtype=np.float32))
                replay_y.append(np.asarray(data['y'], dtype=np.int64))

        if replay_X:
            X_rep = np.concatenate(replay_X, axis=0)
            y_rep = np.concatenate(replay_y, axis=0)
            old_classes = sorted(np.unique(y_rep).tolist())
            exemplar_total = len(y_rep)
            n_private = len(X_parts[0])
            budget, per_class = self._compute_replay_budget(
                n_private=n_private,
                n_old_classes=len(old_classes),
                exemplar_total=exemplar_total,
            )

            self._last_replay_budget = budget
            self._last_replay_per_class = per_class
            self._last_private_size = n_private
            self._last_exemplar_total = exemplar_total
            self._last_old_classes = len(old_classes)

            if self.replay_cap and budget > 0 and len(X_rep) > budget:
                cls_to_idx = {}
                for idx, cls in enumerate(y_rep.tolist()):
                    cls_to_idx.setdefault(int(cls), []).append(idx)
                classes = sorted(cls_to_idx.keys())
                per_class = max(1, budget // len(classes))
                remainder = budget % len(classes)
                chosen = []
                for pos, cls in enumerate(classes):
                    idxs = np.asarray(cls_to_idx[cls], dtype=np.int64)
                    take = per_class + int(pos < remainder)
                    if take <= 0:
                        continue
                    chosen.append(idxs if len(idxs) <= take else np.random.choice(idxs, take, replace=False))
                chosen_idx = np.concatenate(chosen, axis=0) if chosen else np.array([], dtype=np.int64)
                X_rep = X_rep[chosen_idx]
                y_rep = y_rep[chosen_idx]
            elif self.replay_cap and budget <= 0:
                X_rep = X_rep[:0]
                y_rep = y_rep[:0]

            self._last_kept_count = len(X_rep)
            X_parts.append(X_rep)
            y_parts.append(y_rep)

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)

        if X.shape[0] == 0:
            dummy_X = torch.empty((1, 1), dtype=torch.float32)
            dummy_y = torch.empty((1, len(self.class_order)), dtype=torch.float32)
            loader = DataLoader(TensorDataset(dummy_X, dummy_y), batch_size=batch_size, shuffle=True, drop_last=False)
            return loader, 0

        y_oh = np.zeros((len(y), len(self.class_order)), dtype=np.float32)
        y_oh[np.arange(len(y)), y] = 1.0
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y_oh)), batch_size=batch_size, shuffle=True, drop_last=False)
        return loader, len(X)

