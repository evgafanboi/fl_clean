import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot client class distribution heatmap")
    parser.add_argument("--partition_type", type=str, required=True, help="Partition descriptor, e.g., label_skew_0.1-100")
    parser.add_argument("--n_clients", type=int, required=True, help="Number of clients to read")
    parser.add_argument("--out", type=str, default=None, help="Output path without extension")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution")
    return parser


def read_label_mapping(path=os.path.join(REPO_ROOT, "data", "label_mapping.txt")):
    rows = [line.strip().split(": ", 1) for line in open(path, encoding="utf-8") if line.strip()]
    names = {int(idx): name for idx, name in rows}
    return [names[i] for i in range(max(names) + 1)]


def partition_folder(partition_type, n_clients):
    suffix = f"-{n_clients}"
    return partition_type[:-len(suffix)] if partition_type.endswith(suffix) else partition_type


def load_client_labels(client_id, partition_type, n_clients):
    partition_dir = os.path.join(REPO_ROOT, "data", "partitions", f"{n_clients}_client", partition_folder(partition_type, n_clients))
    merged_y = os.path.join(partition_dir, f"client_{client_id}_y_merged.npy")
    train_y = os.path.join(partition_dir, f"client_{client_id}_y_train.npy")
    public_y = os.path.join(partition_dir, f"client_{client_id}_y_public.npy")
    if os.path.exists(merged_y):
        return np.load(merged_y, mmap_mode="r")
    y_train = np.load(train_y, mmap_mode="r")
    y_public = np.load(public_y, mmap_mode="r")
    return np.concatenate([np.asarray(y_train), np.asarray(y_public)])


def build_distribution(partition_type, n_clients, num_classes):
    counts = np.zeros((num_classes, n_clients), dtype=np.int64)
    for cid in range(n_clients):
        y = np.asarray(load_client_labels(cid, partition_type, n_clients), dtype=np.int64)
        counts[:, cid] = np.bincount(y, minlength=num_classes)[:num_classes]
    return counts


def plot_heatmap(counts, class_names, out_base, dpi):
    fig_w = max(18, counts.shape[1] * 0.24)
    fig_h = 12
    bounds = [0, 500, 10000, 50000, 100000, 300000, 550000, 800000]
    cmap = LinearSegmentedColormap.from_list("white_blue", ["#ffffff", "#08306b"], N=len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(counts, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    xticks = sorted(set([0] + [i - 1 for i in range(10, counts.shape[1] + 1, 10)] + [counts.shape[1] - 1]))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i + 1) for i in xticks], fontsize=18)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=16)
    ax.set_xlabel("Client ID", fontsize=22)
    ax.set_ylabel("Class", fontsize=22)
    ax.set_title("Client Class Distribution", fontsize=26, pad=16)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, ticks=bounds, fraction=0.018, pad=0.01)
    cbar.set_label("Samples", fontsize=20)
    cbar.ax.set_yticklabels([f"{v:,}" for v in bounds])
    cbar.ax.tick_params(labelsize=16)

    fig.savefig(f"{out_base}.pdf")
    fig.savefig(f"{out_base}.svg")
    fig.savefig(f"{out_base}.png", dpi=dpi)
    np.savetxt(f"{out_base}.csv", counts, fmt="%d", delimiter=",")
    plt.close(fig)


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    class_names = read_label_mapping()
    counts = build_distribution(args.partition_type, args.n_clients, len(class_names))
    out_base = args.out or os.path.join(REPO_ROOT, "results", f"partition_heatmap_{args.n_clients}client_{args.partition_type}")
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    plot_heatmap(counts, class_names, out_base, args.dpi)
    print(f"Saved {out_base}.pdf, {out_base}.svg, {out_base}.png, {out_base}.csv")


if __name__ == "__main__":
    main()
