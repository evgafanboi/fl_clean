import torch
import torch.nn as nn
import torch.nn.functional as F
from .pt_common import _PTModelWrapper


def get_model_type_for_client(client_id, n_clients):
    quarter = n_clients // 4
    if client_id < quarter:
        return "gru"
    if client_id < 2 * quarter:
        return "mixed_dcblstm"
    if client_id < 3 * quarter:
        return "dense"
    return "cnn"


# ── CNN ──────────────────────────────────────────────────────────────────

class _CNNNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.15)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.15)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.15)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 64)
        self.ln_fc = nn.LayerNorm(64)
        self.drop_fc = nn.Dropout(0.2)
        self.logits = nn.Linear(64, num_classes)

    def forward(self, x, return_logits=False):
        x = x.unsqueeze(1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = F.max_pool1d(x, 2)
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = F.max_pool1d(x, 2)
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1)
        x = self.drop_fc(self.ln_fc(F.silu(self.fc(x))))
        lg = self.logits(x)
        return lg if return_logits else F.softmax(lg, dim=-1)


class PTCNNModel(_PTModelWrapper):
    def __init__(self, input_dim, num_classes, batch_size=4096, learning_rate=None):
        lr = learning_rate or 0.001 * (batch_size / 1024) ** 0.5
        print(f"PT CNN Model - Using learning rate: {lr:.6f} for batch size {batch_size}")
        net = _CNNNet(input_dim, num_classes)
        super().__init__(net, input_dim, num_classes, batch_size, lr,
                         l2_modules=[net.fc], optimizer_eps=1e-7)


def create_cnn_model(input_dim, num_classes, batch_size=4096, learning_rate=None):
    return PTCNNModel(input_dim, num_classes, batch_size, learning_rate)


# ── Altered DCBLSTM (near-last layer 64 instead of 16) ──────────────────

class _MixedDCBLSTMNet(nn.Module):
    def __init__(self, input_dim, num_classes, conv_filters=64, lstm_units=64, lstm_units_2=128, dnn_sizes=(64, 32, 64)):
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


class PTMixedDCBLSTMModel(_PTModelWrapper):
    def __init__(self, input_dim, num_classes, batch_size=8192, learning_rate=None):
        lr = learning_rate or 0.001 * (batch_size / 1024) ** 0.5
        print(f"PT Mixed DCBLSTM Model - Using learning rate: {lr:.6f} for batch size {batch_size}")
        net = _MixedDCBLSTMNet(input_dim, num_classes)
        super().__init__(net, input_dim, num_classes, batch_size, lr,
                         l2_modules=[], optimizer_eps=1e-7)


def create_mixed_dcblstm_model(input_dim, num_classes, batch_size=8192, learning_rate=None):
    return PTMixedDCBLSTMModel(input_dim, num_classes, batch_size, learning_rate)
