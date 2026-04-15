import math
import numpy as np
import torch
import torch.nn as nn


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
        results = []
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.from_numpy(X.astype(np.float32))
                for i in range(0, len(X_t), bs):
                    results.append(self._w.nn(X_t[i:i + bs].to(dev), return_logits=True).cpu().numpy())
            else:
                for batch in X:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    results.append(self._w.nn(x.to(dev), return_logits=True).cpu().numpy())
        return np.concatenate(results)

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
            print(f"    epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
        return _PTHistory(history)

    def predict(self, X, batch_size=None, verbose=None, **kwargs):
        dev = _dev()
        self.nn.to(dev)
        self.nn.eval()
        bs = batch_size or self.batch_size
        results = []
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                for i in range(0, len(X), bs):
                    results.append(self.nn(X[i:i + bs]).cpu().numpy())
            elif isinstance(X, np.ndarray):
                X_t = torch.from_numpy(X.astype(np.float32))
                for i in range(0, len(X_t), bs):
                    results.append(self.nn(X_t[i:i + bs].to(dev)).cpu().numpy())
            else:
                for batch in X:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    results.append(self.nn(x.to(dev)).cpu().numpy())
        return np.concatenate(results)

    def predict_proba(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def evaluate(self, dataset, verbose=0):
        dev = _dev()
        self.nn.to(dev)
        self.nn.eval()
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in dataset:
                X_b = X_b.to(dev)
                y_cls = y_b.argmax(dim=1).to(dev) if y_b.ndim > 1 else y_b.long().to(dev)
                loss = self.criterion(self.nn(X_b, return_logits=True), y_cls)
                total_loss += loss.item() * len(X_b)
                total += len(X_b)
        return [total_loss / max(total, 1), 0.0]

    def get_weights(self):
        return [v.detach().cpu().float().numpy() for v in self.nn.state_dict().values()]

    def set_weights(self, weights):
        sd = self.nn.state_dict()
        new_sd = {
            k: torch.from_numpy(np.array(w, dtype=np.float32)).to(dtype=v.dtype)
            for (k, v), w in zip(sd.items(), weights)
        }
        self.nn.load_state_dict(new_sd)

    def get_logits_model(self):
        if self._logits_model is None:
            self._logits_model = _PTLogitsWrapper(self)
        return self._logits_model

    def get_feature_model(self):
        if not hasattr(self, '_feature_model') or self._feature_model is None:
            self._feature_model = _PTFeatureWrapper(self)
        return self._feature_model

    def expand_classes(self, new_num_classes):
        pass
