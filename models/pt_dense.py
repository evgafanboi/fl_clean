import torch.nn as nn
import torch.nn.functional as F
from .pt_common import _PTModelWrapper


class _DenseNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.ln_input = nn.LayerNorm(input_dim)
        self.d1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.drop1 = nn.Dropout(0.15)
        self.d2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(0.2)
        self.d3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.drop3 = nn.Dropout(0.15)
        self.logits = nn.Linear(64, num_classes)

    def forward(self, x, return_logits=False):
        x = self.ln_input(x)
        x = self.drop1(self.ln1(F.silu(self.d1(x))))
        res = x
        x = self.drop2(self.ln2(F.silu(self.d2(x))))
        x = x + res
        x = self.drop3(self.ln3(F.silu(self.d3(x))))
        lg = self.logits(x)
        return lg if return_logits else F.softmax(lg, dim=-1)


class PTDenseModel(_PTModelWrapper):
    def __init__(self, input_dim, num_classes, batch_size=4096, learning_rate=None):
        lr = learning_rate or 0.001 * (batch_size / 1024) ** 0.5
        print(f"PT Dense Model - lr={lr:.6f} bs={batch_size}")
        super().__init__(_DenseNet(input_dim, num_classes), input_dim, num_classes, batch_size, lr)


def create_dense_model(input_dim, num_classes, batch_size=4096, learning_rate=None):
    return PTDenseModel(input_dim, num_classes, batch_size, learning_rate)


create_enhanced_dense_model = create_dense_model
