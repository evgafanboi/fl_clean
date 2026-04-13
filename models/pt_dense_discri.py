import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pt_common import _PTHistory, _dev


class _DiscriminatorNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.ln_input = nn.LayerNorm(input_dim)
        self.d1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.drop1 = nn.Dropout(0.15)
        self.d2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(0.15)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.ln_input(x)
        x = self.drop1(self.ln1(F.silu(self.d1(x))))
        x = self.drop2(self.ln2(F.silu(self.d2(x))))
        return torch.sigmoid(self.out(x))


class PTDiscriminator:
    def __init__(self, input_dim, learning_rate=0.0001):
        self.model = _DiscriminatorNet(input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

    def fit(self, dataset, epochs=1, verbose=0):
        dev = _dev()
        self.model.to(dev)
        self.model.train()
        for _ in range(epochs):
            for X_b, y_b in dataset:
                X_b = X_b.to(dev, non_blocking=True)
                y_b = y_b.to(dev, non_blocking=True).unsqueeze(-1) if y_b.ndim == 1 else y_b.to(dev)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.criterion(self.model(X_b), y_b)
                loss.backward()
                self.optimizer.step()
        return _PTHistory({"loss": [0.0]})

    def predict(self, X, batch_size=8192, verbose=0):
        dev = _dev()
        self.model.to(dev)
        self.model.eval()
        results = []
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.from_numpy(X.astype(np.float32))
                for i in range(0, len(X_t), batch_size):
                    results.append(self.model(X_t[i:i + batch_size].to(dev)).cpu().numpy())
            else:
                for batch in X:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    results.append(self.model(x.to(dev)).cpu().numpy())
        return np.concatenate(results)

    def get_weights(self):
        return [v.detach().cpu().float().numpy() for v in self.model.state_dict().values()]

    def set_weights(self, weights):
        sd = self.model.state_dict()
        new_sd = {
            k: torch.from_numpy(np.array(w, dtype=np.float32)).to(dtype=v.dtype)
            for (k, v), w in zip(sd.items(), weights)
        }
        self.model.load_state_dict(new_sd)


def create_discriminator(input_dim, learning_rate=0.0001):
    return PTDiscriminator(input_dim, learning_rate)
