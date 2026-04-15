import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pt_common import _PTHistory, _PTLogitsWrapper, _dev


def _freeze(modules):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = False


# ── Dense FedMLB ─────────────────────────────────────────────────────

class _Block(nn.Module):
    def __init__(self, in_dim, out_dim, drop, residual=False):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(drop)
        self.residual = residual

    def forward(self, x):
        h = self.drop(self.ln(F.silu(self.dense(x))))
        return h + x if self.residual else h


class _FedMLBDenseNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.b1 = _Block(input_dim, 128, 0.15)
        self.b2 = _Block(128, 128, 0.20, residual=True)
        self.b3 = _Block(128, 64, 0.15)
        self.logits = nn.Linear(64, num_classes)
        self.b2g = _Block(128, 128, 0.20, residual=True)
        self.b3g = _Block(128, 64, 0.15)
        self.logits_g = nn.Linear(64, num_classes)
        _freeze([self.b2g, self.b3g, self.logits_g])

    def train(self, mode=True):
        super().train(mode)
        self.b2g.eval(); self.b3g.eval(); self.logits_g.eval()
        return self

    def forward(self, x, return_logits=False):
        _, _, lg, probs = self._local(x)
        return lg if return_logits else probs

    def _local(self, x):
        x = self.input_norm(x)
        z1 = self.b1(x)
        z2 = self.b2(z1)
        z3 = self.b3(z2)
        lg = self.logits(z3)
        return z1, z2, lg, F.softmax(lg, dim=-1)

    def _hybrid2(self, z1):
        h = self.b3g(self.b2g(z1))
        lg = self.logits_g(h)
        return lg, F.softmax(lg, dim=-1)

    def _hybrid3(self, z2):
        h = self.b3g(z2)
        lg = self.logits_g(h)
        return lg, F.softmax(lg, dim=-1)

    def sync_hybrid(self):
        self.b2g.load_state_dict(self.b2.state_dict())
        self.b3g.load_state_dict(self.b3.state_dict())
        self.logits_g.load_state_dict(self.logits.state_dict())
        _freeze([self.b2g, self.b3g, self.logits_g])


# ── GRU FedMLB ───────────────────────────────────────────────────────

class _GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, last_only=False):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(0.15)
        self.last_only = last_only

    def forward(self, x):
        out, _ = self.gru(x)
        if self.last_only:
            out = out[:, -1, :]
        out = self.drop(self.ln(out))
        return out


class _DenseHead(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.2):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.ln(F.relu(self.dense(x))))


class _FedMLBGRUNet(nn.Module):
    def __init__(self, input_dim, num_classes, gru_units=128):
        super().__init__()
        self.input_dim = input_dim
        self.gru1 = _GRUBlock(1, gru_units)
        self.gru2 = _GRUBlock(gru_units, gru_units)
        self.gru3 = _GRUBlock(gru_units, gru_units, last_only=True)
        self.head = _DenseHead(gru_units, 64)
        self.logits = nn.Linear(64, num_classes)
        self.gru2g = _GRUBlock(gru_units, gru_units)
        self.gru3g = _GRUBlock(gru_units, gru_units, last_only=True)
        self.headg = _DenseHead(gru_units, 64)
        self.logits_g = nn.Linear(64, num_classes)
        _freeze([self.gru2g, self.gru3g, self.headg, self.logits_g])

    def train(self, mode=True):
        super().train(mode)
        self.gru2g.eval(); self.gru3g.eval(); self.headg.eval(); self.logits_g.eval()
        return self

    def forward(self, x, return_logits=False):
        _, _, _, lg, probs = self._local(x)
        return lg if return_logits else probs

    def _local(self, x):
        x = x.unsqueeze(-1)
        z1 = self.gru1(x)
        z2 = self.gru2(z1)
        z3 = self.gru3(z2)
        z4 = self.head(z3)
        lg = self.logits(z4)
        return z1, z2, z3, lg, F.softmax(lg, dim=-1)

    def _hybrid_gru2(self, z1):
        h = self.gru3g(self.gru2g(z1))
        h = self.headg(h)
        lg = self.logits_g(h)
        return lg, F.softmax(lg, dim=-1)

    def _hybrid_gru3(self, z2):
        h = self.gru3g(z2)
        h = self.headg(h)
        lg = self.logits_g(h)
        return lg, F.softmax(lg, dim=-1)

    def _hybrid_head(self, z3):
        h = self.headg(z3)
        lg = self.logits_g(h)
        return lg, F.softmax(lg, dim=-1)

    def sync_hybrid(self):
        self.gru2g.load_state_dict(self.gru2.state_dict())
        self.gru3g.load_state_dict(self.gru3.state_dict())
        self.headg.load_state_dict(self.head.state_dict())
        self.logits_g.load_state_dict(self.logits.state_dict())
        _freeze([self.gru2g, self.gru3g, self.headg, self.logits_g])


# ── DCBLSTM FedMLB ───────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding='same')
        self.ln = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        return self.ln(x)


class _BiLSTMBlock(nn.Module):
    def __init__(self, input_size, units1, units2):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_size, units1, bidirectional=True, batch_first=True)
        self.ln = nn.LayerNorm(units1 * 2)
        self.bilstm2 = nn.LSTM(units1 * 2, units2, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x, _ = self.bilstm1(x)
        x = self.ln(x)
        x, _ = self.bilstm2(x)
        return self.drop(x[:, -1, :])


class _MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop, has_ln=False):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(drop)
        self.ln = nn.LayerNorm(out_dim) if has_ln else None

    def forward(self, x):
        x = self.drop(F.relu(self.dense(x)))
        return self.ln(x) if self.ln else x


class _FedMLBDCBLSTMNet(nn.Module):
    def __init__(self, input_dim, num_classes, conv_filters=64, lstm_units=64, lstm_units_2=128, dnn_sizes=(64, 32, 16)):
        super().__init__()
        self.input_dim = input_dim
        bi2_out = lstm_units_2 * 2
        self.conv = _ConvBlock(1, conv_filters, input_dim)
        self.bilstm = _BiLSTMBlock(conv_filters, lstm_units, lstm_units_2)
        self.mlp1 = _MLPBlock(bi2_out, dnn_sizes[0], 0.1)
        self.mlp2 = _MLPBlock(dnn_sizes[0], dnn_sizes[1], 0.1)
        self.mlp3 = _MLPBlock(dnn_sizes[1], dnn_sizes[2], 0.1, has_ln=True)
        self.logits = nn.Linear(dnn_sizes[2], num_classes)
        self.bilstm_g = _BiLSTMBlock(conv_filters, lstm_units, lstm_units_2)
        self.mlp1_g = _MLPBlock(bi2_out, dnn_sizes[0], 0.1)
        self.mlp2_g = _MLPBlock(dnn_sizes[0], dnn_sizes[1], 0.1)
        self.mlp3_g = _MLPBlock(dnn_sizes[1], dnn_sizes[2], 0.1, has_ln=True)
        self.logits_g = nn.Linear(dnn_sizes[2], num_classes)
        _freeze([self.bilstm_g, self.mlp1_g, self.mlp2_g, self.mlp3_g, self.logits_g])

    def train(self, mode=True):
        super().train(mode)
        for m in [self.bilstm_g, self.mlp1_g, self.mlp2_g, self.mlp3_g, self.logits_g]:
            m.eval()
        return self

    def forward(self, x, return_logits=False):
        _, _, _, _, lg, probs = self._local(x)
        return lg if return_logits else probs

    def _local(self, x):
        x = x.unsqueeze(-1)
        z1 = self.conv(x)
        z2 = self.bilstm(z1)
        z3 = self.mlp1(z2)
        z4 = self.mlp2(z3)
        z5 = self.mlp3(z4)
        lg = self.logits(z5)
        return z1, z2, z3, z4, lg, F.softmax(lg, dim=-1)

    def _hybrid_bilstm(self, z1):
        h = self.bilstm_g(z1)
        h = self.mlp1_g(h); h = self.mlp2_g(h); h = self.mlp3_g(h)
        lg = self.logits_g(h)
        return lg, F.softmax(lg, dim=-1)

    def _hybrid_mlp1(self, z2):
        h = self.mlp1_g(z2); h = self.mlp2_g(h); h = self.mlp3_g(h)
        lg = self.logits_g(h)
        return lg, F.softmax(lg, dim=-1)

    def _hybrid_mlp2(self, z3):
        h = self.mlp2_g(z3); h = self.mlp3_g(h)
        lg = self.logits_g(h)
        return lg, F.softmax(lg, dim=-1)

    def _hybrid_mlp3(self, z4):
        h = self.mlp3_g(z4)
        lg = self.logits_g(h)
        return lg, F.softmax(lg, dim=-1)

    def sync_hybrid(self):
        self.bilstm_g.load_state_dict(self.bilstm.state_dict())
        self.mlp1_g.load_state_dict(self.mlp1.state_dict())
        self.mlp2_g.load_state_dict(self.mlp2.state_dict())
        self.mlp3_g.load_state_dict(self.mlp3.state_dict())
        self.logits_g.load_state_dict(self.logits.state_dict())
        _freeze([self.bilstm_g, self.mlp1_g, self.mlp2_g, self.mlp3_g, self.logits_g])


# ── Generic FedMLB Wrapper ───────────────────────────────────────────

class _FedMLBWrapper:
    def __init__(self, nn_module, input_dim, num_classes, batch_size, lr,
                 lambda1, lambda2, temperature, n_hybrids, hybrid_fn,
                 l2_modules=None):
        self.nn = nn_module
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.temperature = temperature
        self._n_hybrids = n_hybrids
        self._hybrid_fn = hybrid_fn
        self.base_model = nn_module
        trainable = [p for p in nn_module.parameters() if p.requires_grad]
        l2_ids = {id(m.weight) for m in (l2_modules or []) if hasattr(m, 'weight')}
        decay, no_decay = [], []
        for p in trainable:
            (decay if id(p) in l2_ids else no_decay).append(p)
        groups = []
        if decay:
            groups.append({'params': decay, 'weight_decay': 1e-4})
        if no_decay:
            groups.append({'params': no_decay, 'weight_decay': 0.0})
        self.optimizer = torch.optim.Adam(groups, lr=lr, eps=1e-7)
        self._logits_model = None

    def fit(self, dataloader, epochs=5, **kwargs):
        dev = _dev()
        self.nn.to(dev)
        self.nn.train()
        T = self.temperature
        l1, l2 = self.lambda1, self.lambda2
        history = {"loss": []}
        for epoch in range(epochs):
            total_loss, batches = 0.0, 0
            for X_b, y_b in dataloader:
                X_b = X_b.to(dev, non_blocking=True)
                y_b = y_b.to(dev, non_blocking=True)
                if y_b.ndim > 1:
                    y_b = y_b.argmax(dim=1)
                intermediates = self.nn._local(X_b)
                logits_L = intermediates[-2]
                hybrid_logits = self._hybrid_fn(self.nn, intermediates)
                loss_main = F.cross_entropy(logits_L, y_b)
                loss_hce = sum(F.cross_entropy(hl, y_b) for hl in hybrid_logits) / len(hybrid_logits)
                q_L = F.softmax(logits_L / T, dim=-1)
                loss_hkl = sum(
                    F.kl_div(q_L.log(), F.softmax(hl / T, dim=-1), reduction='batchmean')
                    for hl in hybrid_logits
                ) / len(hybrid_logits)
                loss = loss_main + l1 * loss_hce + l2 * loss_hkl
                self.optimizer.zero_grad(set_to_none=True)
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
            if isinstance(X, np.ndarray):
                X_t = torch.from_numpy(X.astype(np.float32))
                for i in range(0, len(X_t), bs):
                    results.append(self.nn(X_t[i:i + bs].to(dev)).cpu().numpy())
            else:
                for batch in X:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    results.append(self.nn(x.to(dev)).cpu().numpy())
        return np.concatenate(results)

    def evaluate(self, dataset, verbose=0):
        dev = _dev()
        self.nn.to(dev)
        self.nn.eval()
        total_loss, total = 0.0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for X_b, y_b in dataset:
                X_b = X_b.to(dev)
                y_cls = y_b.argmax(dim=1).to(dev) if y_b.ndim > 1 else y_b.long().to(dev)
                loss = criterion(self.nn(X_b, return_logits=True), y_cls)
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

    def set_global_weights(self, _):
        self.nn.sync_hybrid()

    def get_logits_model(self):
        if self._logits_model is None:
            self._logits_model = _PTLogitsWrapper(self)
        return self._logits_model

    def get_feature_model(self):
        return self.get_logits_model()


# ── Factory helpers (per-architecture hybrid extraction) ──────────────

def _dense_hybrids(net, intermediates):
    z1, z2, lg, probs = intermediates
    l2, _ = net._hybrid2(z1)
    l3, _ = net._hybrid3(z2)
    return [l2, l3]

def _gru_hybrids(net, intermediates):
    z1, z2, z3, lg, probs = intermediates
    l2, _ = net._hybrid_gru2(z1)
    l3, _ = net._hybrid_gru3(z2)
    lh, _ = net._hybrid_head(z3)
    return [l2, l3, lh]

def _dcblstm_hybrids(net, intermediates):
    z1, z2, z3, z4, lg, probs = intermediates
    lb, _ = net._hybrid_bilstm(z1)
    l1, _ = net._hybrid_mlp1(z2)
    l2, _ = net._hybrid_mlp2(z3)
    l3, _ = net._hybrid_mlp3(z4)
    return [lb, l1, l2, l3]


# ── Public factory functions ─────────────────────────────────────────

def create_fedmlb_model(input_dim, num_classes, batch_size, lambda1=1.0, lambda2=1.0, temperature=1.0):
    lr = 0.001 * np.sqrt(batch_size / 1024)
    net = _FedMLBDenseNet(input_dim, num_classes)
    net.sync_hybrid()
    print(f"PT FedMLB Dense - lr={lr:.6f} bs={batch_size}")
    return _FedMLBWrapper(net, input_dim, num_classes, batch_size, lr, lambda1, lambda2, temperature, 2, _dense_hybrids,
                          l2_modules=[net.b1.dense, net.b2.dense, net.b3.dense])


def create_fedmlb_gru_model(input_dim, num_classes, batch_size, lambda1=1.0, lambda2=1.0, temperature=1.0, gru_units=128):
    lr = 0.001 * np.sqrt(batch_size / 1024)
    net = _FedMLBGRUNet(input_dim, num_classes, gru_units)
    net.sync_hybrid()
    print(f"PT FedMLB GRU - lr={lr:.6f} bs={batch_size}")
    return _FedMLBWrapper(net, input_dim, num_classes, batch_size, lr, lambda1, lambda2, temperature, 3, _gru_hybrids,
                          l2_modules=[net.head.dense])


def create_fedmlb_dcblstm_model(input_dim, num_classes, batch_size, lambda1=1.0, lambda2=1.0, temperature=1.0):
    lr = 0.001 * np.sqrt(batch_size / 1024)
    net = _FedMLBDCBLSTMNet(input_dim, num_classes)
    net.sync_hybrid()
    print(f"PT FedMLB DCBLSTM - lr={lr:.6f} bs={batch_size}")
    return _FedMLBWrapper(net, input_dim, num_classes, batch_size, lr, lambda1, lambda2, temperature, 4, _dcblstm_hybrids,
                          l2_modules=[])
