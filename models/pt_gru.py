import torch.nn as nn
import torch.nn.functional as F
from .pt_common import _PTModelWrapper


class _GRUNet(nn.Module):
    def __init__(self, input_dim, num_classes, gru_units=128):
        super().__init__()
        self.gru1 = nn.GRU(1, gru_units, batch_first=True)
        self.ln1 = nn.LayerNorm(gru_units)
        self.drop1 = nn.Dropout(0.15)
        self.gru2 = nn.GRU(gru_units, gru_units, batch_first=True)
        self.ln2 = nn.LayerNorm(gru_units)
        self.drop2 = nn.Dropout(0.15)
        self.gru3 = nn.GRU(gru_units, gru_units, batch_first=True)
        self.ln3 = nn.LayerNorm(gru_units)
        self.drop3 = nn.Dropout(0.15)
        self.head = nn.Linear(gru_units, 64)
        self.ln_head = nn.LayerNorm(64)
        self.drop_head = nn.Dropout(0.2)
        self.logits = nn.Linear(64, num_classes)

    def forward(self, x, return_logits=False):
        x = x.unsqueeze(-1)
        x, _ = self.gru1(x)
        x = self.drop1(self.ln1(x))
        x, _ = self.gru2(x)
        x = self.drop2(self.ln2(x))
        x, _ = self.gru3(x)
        x = self.drop3(self.ln3(x[:, -1, :]))
        x = F.relu(self.head(x))
        x = self.drop_head(self.ln_head(x))
        lg = self.logits(x)
        return lg if return_logits else F.softmax(lg, dim=-1)


class PTGRUModel(_PTModelWrapper):
    def __init__(self, input_dim, num_classes, batch_size=4096, learning_rate=None, gru_units=128):
        lr = learning_rate or 0.001 * (batch_size / 1024) ** 0.5
        print(f"PT GRU Model - Using learning rate: {lr:.6f} for batch size {batch_size}")
        net = _GRUNet(input_dim, num_classes, gru_units)
        eps = 1e-6 if batch_size >= 2048 else 1e-7
        super().__init__(net, input_dim, num_classes, batch_size, lr,
                         l2_modules=[net.head], optimizer_eps=eps)


def create_gru_model(input_dim, num_classes, batch_size=4096, learning_rate=None, gru_units=128):
    return PTGRUModel(input_dim, num_classes, batch_size, learning_rate, gru_units)
