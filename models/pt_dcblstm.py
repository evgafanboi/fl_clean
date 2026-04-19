import torch
import torch.nn as nn
import torch.nn.functional as F
from .pt_common import _PTModelWrapper


class _DCBLSTMNet(nn.Module):
    def __init__(self, input_dim, num_classes, conv_filters=64, lstm_units=64, lstm_units_2=128, dnn_sizes=(64, 32, 16)):
        super().__init__()
        self.conv = nn.Conv1d(1, conv_filters, kernel_size=input_dim, padding='same')
        self.ln_conv = nn.LayerNorm(conv_filters)
        self.bilstm1 = nn.LSTM(conv_filters, lstm_units, batch_first=True, bidirectional=True)
        self.ln_b1 = nn.LayerNorm(lstm_units * 2)
        self.bilstm2 = nn.LSTM(lstm_units * 2, lstm_units_2, batch_first=True, bidirectional=True)
        self.drop_b = nn.Dropout(0.1)
        dnn_in = lstm_units_2 * 2
        layers = []
        for s in dnn_sizes:
            layers += [nn.Linear(dnn_in, s), nn.ReLU(), nn.Dropout(0.1)]
            dnn_in = s
        self.dnn = nn.Sequential(*layers)
        self.ln_feat = nn.LayerNorm(dnn_in)
        self.logits = nn.Linear(dnn_in, num_classes)

    def forward(self, x, return_logits=False):
        x = self.ln_conv(F.relu(self.conv(x.unsqueeze(1))).transpose(1, 2))
        x, _ = self.bilstm1(x)
        x = self.ln_b1(x)
        x, (h_n, _) = self.bilstm2(x)
        x = self.drop_b(torch.cat([h_n[0], h_n[1]], dim=-1))
        x = self.ln_feat(self.dnn(x))
        lg = self.logits(x)
        return lg if return_logits else F.softmax(lg, dim=-1)


class PTDCBLSTMModel(_PTModelWrapper):
    def __init__(self, input_dim, num_classes, batch_size=8192, learning_rate=None):
        lr = learning_rate or 0.001 * (batch_size / 1024) ** 0.5
        print(f"PT DCBLSTM Model - Using learning rate: {lr:.6f} for batch size {batch_size}")
        net = _DCBLSTMNet(input_dim, num_classes)
        super().__init__(net, input_dim, num_classes, batch_size, lr,
                         l2_modules=[], optimizer_eps=1e-7)


def create_dcblstm_model(input_dim, num_classes, batch_size=8192, learning_rate=None):
    return PTDCBLSTMModel(input_dim, num_classes, batch_size, learning_rate)
