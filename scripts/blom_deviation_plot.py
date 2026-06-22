import argparse, glob, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtri
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

    def forward(self, x):
        x = x.unsqueeze(-1)
        x, _ = self.gru1(x)
        x = self.drop1(self.ln1(x))
        x, _ = self.gru2(x)
        x = self.drop2(self.ln2(x))
        x, _ = self.gru3(x)
        x = self.drop3(self.ln3(x[:, -1, :]))
        x = F.relu(self.head(x))
        x = self.drop_head(self.ln_head(x))
        return F.softmax(self.logits(x), dim=-1)


def _load_net(weights, input_dim, num_classes, gru_units):
    net = _GRUNet(input_dim, num_classes, gru_units)
    sd = net.state_dict()
    net.load_state_dict({k: torch.from_numpy(w) for k, w in zip(sd.keys(), weights)})
    return net


@torch.no_grad()
def _predict(net, X, dev, batch_size=4096):
    net.to(dev).eval()
    t = torch.from_numpy(X)
    return np.concatenate([
        net(t[i:i + batch_size].to(dev)).cpu().numpy()
        for i in range(0, len(t), batch_size)
    ])


def _compute_sorted_z(S):
    N, K, C = S.shape
    S = S.astype(np.float64)
    mu = S.mean(axis=1, keepdims=True)
    centered = S - mu
    Sigma = np.einsum('nkd,nke->nde', centered, centered) / max(K - 1, 1)
    v_star = np.linalg.eigh(Sigma)[1][:, :, -1]
    proj = np.einsum('nkd,nd->nk', centered, v_star)
    med = np.median(proj, axis=1, keepdims=True)
    abs_dev = np.abs(proj - med)
    sigma_r = np.maximum(np.median(abs_dev, axis=1, keepdims=True) * 1.4826, 1e-10)
    return np.sort(abs_dev / sigma_r, axis=1)


def _blom_ref(K):
    k = np.arange(1, K + 1, dtype=np.float64)
    return ndtri(0.5 + 0.5 * (k - 0.375) / (K + 0.25))


def _plot(sorted_z, ref, out_path):
    mean_z = sorted_z.mean(axis=0)
    std_z = sorted_z.std(axis=0)
    ranks = np.arange(1, len(ref) + 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ranks, ref, color='#444444', linewidth=1.5, linestyle='--', label='Expected Blom deviation')
    ax.fill_between(ranks, mean_z - std_z, mean_z + std_z, color='#1f77b4', alpha=0.15)
    ax.plot(ranks, mean_z, color='#1f77b4', linewidth=1.5, label='Actual deviation')
    ax.set_xlabel('Rank $r$')
    ax.set_ylabel('Deviation')
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight_record', default=(
        'temp_weights/Ours_100client_label_skew_0.1_gru_ekd_1.0_49_eva_v3_0.7_weight_record'
    ))
    ap.add_argument('--out_dir', default='scripts/results/blom')
    ap.add_argument('--max_samples', type=int, default=2000)
    ap.add_argument('--gru_units', type=int, default=128)
    args = ap.parse_args()

    X_all = np.load('data/X_test.npy', mmap_mode='r')
    y_all = np.load('data/y_test.npy', mmap_mode='r')
    input_dim = X_all.shape[1]
    num_classes = int(y_all.max()) + 1

    idx = np.random.default_rng(0).choice(len(X_all), size=min(args.max_samples, len(X_all)), replace=False)
    pub_X = np.array(X_all[idx], dtype=np.float32)
    del X_all, y_all

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    round_dirs = sorted(
        glob.glob(os.path.join(args.weight_record, 'round_*')),
        key=lambda p: int(os.path.basename(p).split('_')[1])
    )

    for rdir in round_dirs:
        r = int(os.path.basename(rdir).split('_')[1])
        if r < 3:
            continue

        bins = sorted(
            glob.glob(os.path.join(rdir, 'client_*_weight.bin')),
            key=lambda p: int(os.path.basename(p).split('_')[1])
        )
        K = len(bins)
        S = np.empty((len(pub_X), K, num_classes), dtype=np.float32)

        for k, fpath in enumerate(bins):
            with open(fpath, 'rb') as f:
                weights = pickle.load(f)
            net = _load_net(weights, input_dim, num_classes, args.gru_units)
            S[:, k, :] = _predict(net, pub_X, dev)
            del net

        sorted_z = _compute_sorted_z(S)
        ref = _blom_ref(K)
        out_path = os.path.join(args.out_dir, f'round_{r}.png')
        _plot(sorted_z, ref, out_path)
        print(f'round {r}: saved {out_path}')


if __name__ == '__main__':
    main()
