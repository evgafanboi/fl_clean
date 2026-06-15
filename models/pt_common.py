import contextlib
import math
import numpy as np
import torch
import torch.nn as nn


def _infer_ctx(dev):
    """Return an autocast context for GPU inference, or a no-op for CPU."""
    if dev.type == 'cuda':
        return torch.amp.autocast('cuda')
    return contextlib.nullcontext()


def _to_gpu_once(X, dev):
    """Pin a numpy array in host memory and transfer to *dev* in one DMA shot.

    Returns a CUDA tensor.  Much faster than per-batch .to(dev) because the
    PCIe DMA engine runs without CPU involvement when the source buffer is
    pinned, and only one round-trip is needed for the entire array.
    """
    if isinstance(X, torch.Tensor):
        return X if X.device.type == dev.type else X.to(dev, non_blocking=True)
    arr = X if X.dtype == np.float32 else X.astype(np.float32)
    t = torch.from_numpy(arr)
    if dev.type == 'cuda':
        t = t.pin_memory()
    return t.to(dev, non_blocking=True)


def _dev():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class _PTHistory:
    def __init__(self, d):
        self.history = d


class _PTFeatureWrapper:
    def __init__(self, wrapper):
        self._w = wrapper
        self._features = None
        self._hook = None
        for name, module in self._w.nn.named_modules():
            if name == 'logits':
                self._hook = module.register_forward_pre_hook(self._capture)
                self._feat_dim = module.in_features
                break

    @property
    def output_shape(self):
        return (None, self._feat_dim)

    def _capture(self, module, inp):
        self._features = inp[0]

    def predict(self, X, batch_size=None, verbose=None, **kwargs):
        dev = _dev()
        self._w.nn.to(dev)
        self._w.nn.eval()
        bs = batch_size or self._w.batch_size
        results = []
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.from_numpy(X.astype(np.float32))
                for i in range(0, len(X_t), bs):
                    self._w.nn(X_t[i:i + bs].to(dev))
                    results.append(self._features.cpu().numpy())
            else:
                for batch in X:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    self._w.nn(x.to(dev))
                    results.append(self._features.cpu().numpy())
        return np.concatenate(results)

    def __del__(self):
        if self._hook:
            self._hook.remove()


class _PTLogitsWrapper:
    def __init__(self, wrapper):
        self._w = wrapper

    def predict(self, X, batch_size=None, verbose=None, **kwargs):
        dev = _dev()
        self._w.nn.to(dev)
        self._w.nn.eval()
        bs = batch_size or self._w.batch_size
        parts = []
        with torch.no_grad(), _infer_ctx(dev):
            if isinstance(X, torch.Tensor) and X.device.type == dev.type:
                for i in range(0, len(X), bs):
                    parts.append(self._w.nn(X[i:i + bs], return_logits=True))
            elif isinstance(X, (np.ndarray, torch.Tensor)):
                arr = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
                arr = arr if arr.dtype == np.float32 else arr.astype(np.float32)
                for i in range(0, len(arr), bs):
                    parts.append(self._w.nn(torch.from_numpy(arr[i:i + bs]).to(dev, non_blocking=True), return_logits=True))
            else:
                for batch in X:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    parts.append(self._w.nn(x.to(dev, non_blocking=True), return_logits=True))
        return torch.cat(parts).float().cpu().numpy()

    def __call__(self, X, training=False):
        dev = _dev()
        self._w.nn.to(dev)
        X_t = torch.from_numpy(X.astype(np.float32)).to(dev)
        if training:
            return self._w.nn(X_t, return_logits=True)
        self._w.nn.eval()
        with torch.no_grad():
            return self._w.nn(X_t, return_logits=True)


class _PTModelWrapper:
    def __init__(self, nn_module, input_dim, num_classes, batch_size, lr,
                 l2_modules=None, optimizer_eps=1e-7):
        self.nn = nn_module
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = lr
        l2_ids = {id(m.weight) for m in (l2_modules or []) if hasattr(m, 'weight')}
        decay, no_decay = [], []
        for p in nn_module.parameters():
            (decay if id(p) in l2_ids else no_decay).append(p)
        groups = []
        if decay:
            groups.append({'params': decay, 'weight_decay': 1e-4})
        if no_decay:
            groups.append({'params': no_decay, 'weight_decay': 0.0})
        self.optimizer = torch.optim.Adam(groups, lr=lr, eps=optimizer_eps)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self._logits_model = None

    def _lr_for_epoch(self, epoch):
        if epoch < 5:
            return self.learning_rate * (epoch + 1) / 5
        decay_epochs = max(1, epoch - 5)
        return self.learning_rate * 0.5 * (1 + math.cos(math.pi * decay_epochs / 50))

    def fit(self, dataloader, epochs=5, **kwargs):
        dev = _dev()
        self.nn.to(dev)
        self.nn.train()
        history = {"loss": []}
        for epoch in range(epochs):
            lr = self._lr_for_epoch(epoch)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            total_loss, batches = 0.0, 0
            for X_b, y_b in dataloader:
                X_b = X_b.to(dev, non_blocking=True)
                y_b = y_b.to(dev, non_blocking=True)
                if y_b.ndim > 1:
                    y_b = y_b.argmax(dim=1)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.criterion(self.nn(X_b, return_logits=True), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
                batches += 1
            avg = total_loss / max(batches, 1)
            history["loss"].append(avg)
        return _PTHistory(history)

    def predict(self, X, batch_size=None, verbose=None, **kwargs):
        dev = _dev()
        self.nn.to(dev)
        self.nn.eval()
        bs = batch_size or self.batch_size
        parts = []
        with torch.no_grad(), _infer_ctx(dev):
            if isinstance(X, torch.Tensor) and X.device.type == dev.type:
                for i in range(0, len(X), bs):
                    parts.append(self.nn(X[i:i + bs]))
            elif isinstance(X, (np.ndarray, torch.Tensor)):
                arr = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
                arr = arr if arr.dtype == np.float32 else arr.astype(np.float32)
                for i in range(0, len(arr), bs):
                    parts.append(self.nn(torch.from_numpy(arr[i:i + bs]).to(dev, non_blocking=True)))
            else:
                for batch in X:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    parts.append(self.nn(x.to(dev, non_blocking=True)))
        return torch.cat(parts).cpu().numpy()

    def predict_proba(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def evaluate(self, dataset, verbose=0):
        dev = _dev()
        self.nn.to(dev)
        self.nn.eval()
        total_loss, total = 0.0, 0
        with torch.no_grad(), _infer_ctx(dev):
            for X_b, y_b in dataset:
                X_b = X_b.to(dev, non_blocking=True)
                y_cls = y_b.argmax(dim=1).to(dev, non_blocking=True) if y_b.ndim > 1 else y_b.long().to(dev, non_blocking=True)
                loss = self.criterion(self.nn(X_b, return_logits=True), y_cls)
                total_loss += loss.item() * len(X_b)
                total += len(X_b)
        return [total_loss / max(total, 1), 0.0]

    def get_weights(self):
        return [v.detach().cpu().float().numpy() for v in self.nn.state_dict().values()]

    def set_weights(self, weights):
        sd = self.nn.state_dict()
        if len(weights) != len(sd):
            raise ValueError(
                f"Weight count mismatch: model expects {len(sd)} arrays "
                f"but received {len(weights)}. This usually means the "
                f"checkpoint was saved with a different backend (TF vs PT) "
                f"or a different model architecture."
            )
        keys = list(sd.keys())
        weight_map = {k: np.array(w, dtype=np.float32) for k, w in zip(keys, weights)}
        target_classes = self.num_classes
        if 'logits.weight' in weight_map:
            target_classes = int(weight_map['logits.weight'].shape[0])
        if target_classes > self.num_classes:
            self.expand_classes(target_classes)
            sd = self.nn.state_dict()
            keys = list(sd.keys())
            weight_map = {k: np.array(w, dtype=np.float32) for k, w in zip(keys, weights)}
        new_sd = {}
        for k, v in sd.items():
            wt = torch.from_numpy(weight_map[k]).to(dtype=v.dtype)
            if wt.shape == v.shape:
                new_sd[k] = wt
            else:
                dst = v.clone()
                sl = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, wt.shape))
                dst[sl] = wt[sl]
                new_sd[k] = dst
        self.nn.load_state_dict(new_sd, strict=False)

    def get_logits_model(self):
        if self._logits_model is None:
            self._logits_model = _PTLogitsWrapper(self)
        return self._logits_model

    def get_feature_model(self):
        if not hasattr(self, '_feature_model') or self._feature_model is None:
            self._feature_model = _PTFeatureWrapper(self)
        return self._feature_model

    def expand_classes(self, new_num_classes):
        if new_num_classes <= self.num_classes:
            return
        old_nn = self.nn
        old_nc = self.num_classes
        new_nn = old_nn.__class__(self.input_dim, new_num_classes)
        old_sd = old_nn.state_dict()
        new_sd = new_nn.state_dict()
        for k in new_sd:
            if k not in old_sd:
                continue
            if new_sd[k].shape == old_sd[k].shape:
                new_sd[k].copy_(old_sd[k])
            else:
                slices = tuple(slice(0, s) for s in old_sd[k].shape)
                new_sd[k][slices] = old_sd[k]
        new_nn.load_state_dict(new_sd)
        dev = next(old_nn.parameters()).device
        new_nn.to(dev)
        self.nn = new_nn
        self.num_classes = new_num_classes
        self._logits_model = None
        self._feature_model = None
        lr = self.optimizer.param_groups[0]['lr']
        eps = self.optimizer.param_groups[0].get('eps', 1e-6)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr, eps=eps)
