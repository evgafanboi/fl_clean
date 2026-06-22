import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot CIL task-grouped client heatmap")
    parser.add_argument("--partition_type", type=str, required=True)
    parser.add_argument("--n_clients", type=int, required=True)
    parser.add_argument("--task_size", type=float, default=5.0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def read_label_mapping(path=os.path.join(REPO_ROOT, "data", "label_mapping.txt")):
    rows = [line.strip().split(": ", 1) for line in open(path, encoding="utf-8") if line.strip()]
    names = {int(idx): name for idx, name in rows}
    return [names[i] for i in range(max(names) + 1)]


def load_task_order(partition_type, n_clients, task_size):
    base = f"{partition_type}_{n_clients}_client_{task_size}_task"
    json_path = os.path.join(REPO_ROOT, "results", "incremental_order", f"{base}.json")
    with open(json_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def task_active_n(n_clients, num_tasks, task_id):
    if num_tasks != 7:
        return n_clients
    return max(1, int(n_clients * (0.4 + task_id * 0.1)))


def load_client_task_labels(partition_dir, client_id, task_id):
    y_train = os.path.join(partition_dir, f"client_{client_id}_task_{task_id}_y_train.npy")
    y_public = os.path.join(partition_dir, f"client_{client_id}_task_{task_id}_y_public.npy")
    arrays = []
    if os.path.exists(y_train):
        arrays.append(np.load(y_train, mmap_mode="r"))
    if os.path.exists(y_public):
        arrays.append(np.load(y_public, mmap_mode="r"))
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.asarray(a) for a in arrays])


def build_cil_distribution(partition_dir, n_clients, task_order, num_classes):
    num_tasks = len(task_order)
    counts = np.zeros((num_classes, n_clients), dtype=np.int64)
    for task_id, task_classes in task_order.items():
        active_n = task_active_n(n_clients, num_tasks, task_id)
        for cid in range(active_n):
            y = load_client_task_labels(partition_dir, cid, task_id)
            if len(y) == 0:
                continue
            bc = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes)
            counts[:, cid] += bc[:num_classes]
    return counts


def build_ordering(task_order, n_clients):
    num_tasks = len(task_order)
    task_class_order = []
    for tid in sorted(task_order.keys()):
        task_class_order.extend(sorted(task_order[tid]))

    client_groups = []
    prev_active = 0
    for tid in sorted(task_order.keys()):
        active_n = task_active_n(n_clients, num_tasks, tid)
        if active_n > prev_active:
            client_groups.append((tid, list(range(prev_active, active_n))))
        prev_active = active_n
    client_order = []
    for _, cids in client_groups:
        client_order.extend(cids)
    return task_class_order, client_groups, client_order


def plot_cil_heatmap(counts, class_names, task_order, client_groups, client_order,
                     task_class_order, out_base, dpi):
    n_classes = counts.shape[0]
    n_clients = counts.shape[1]

    reordered = counts[np.ix_(task_class_order, client_order)]

    fig_w = max(20, n_clients * 0.26)
    fig_h = max(10, n_classes * 0.45)
    bounds = [0, 100, 500, 1000, 10000, 50000, 100000, 500000, 1000000]
    colors = ["#ffffff", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
    cmap = LinearSegmentedColormap.from_list("white_blue", colors, N=len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(reordered, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    num_tasks = len(task_order)
    task_class_starts = []
    idx = 0
    for tid in sorted(task_order.keys()):
        task_class_starts.append((tid, idx))
        idx += len(task_order[tid])

    for tid, start in task_class_starts:
        if tid == 0:
            continue
        ax.axhline(y=start - 0.5, color="#000000", linewidth=1.2, linestyle="--")

    client_boundaries = []
    cumulative = 0
    for tid, cids in client_groups:
        cumulative += len(cids)
        client_boundaries.append(cumulative)

    for boundary in client_boundaries:
        if boundary < n_clients:
            ax.axvline(x=boundary - 0.5, color="#000000", linewidth=1.2, linestyle="--")

    task_y_labels = []
    for tid in sorted(task_order.keys()):
        classes = sorted(task_order[tid])
        mid = task_class_starts[tid][1] + len(classes) / 2 - 0.5
        task_y_labels.append((mid, f"Task {tid}"))

    ax.set_yticks([mid for mid, _ in task_y_labels])
    ax.set_yticklabels([label for _, label in task_y_labels], fontsize=14, fontweight="bold")

    class_y_positions = []
    for i, orig_cls_idx in enumerate(task_class_order):
        class_y_positions.append((i, class_names[orig_cls_idx]))

    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks([pos for pos, _ in class_y_positions])
    ax2.set_yticklabels([name for _, name in class_y_positions], fontsize=9)
    ax2.tick_params(length=0, pad=8)

    xtick_positions = []
    xtick_labels = []
    cumulative = 0
    for tid, cids in client_groups:
        group_size = len(cids)
        mid = cumulative + group_size / 2 - 0.5
        xtick_positions.append(mid)
        if (tid > 0):
            xtick_labels.append(f"Task {tid}\n(+ {group_size})")
        else:
            xtick_labels.append(f"Task {tid}\n({group_size})")
        cumulative += group_size

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=13, fontweight="bold")

    ax.set_xlabel("Client Groups (by task entry)", fontsize=18, labelpad=10)
    ax.set_ylabel("Task", fontsize=18, labelpad=10)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, ticks=bounds, fraction=0.015, pad=0.02)
    cbar.set_label("Samples", fontsize=16)
    cbar.ax.set_yticklabels([f"{v:,}" for v in bounds])
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    class_names = read_label_mapping()
    task_order = load_task_order(args.partition_type, args.n_clients, args.task_size)
    partition_dir = os.path.join(REPO_ROOT, "data", "partitions",
                                 f"{args.n_clients}_client", f"{args.partition_type}_cil")
    counts = build_cil_distribution(partition_dir, args.n_clients, task_order, len(class_names))
    task_class_order, client_groups, client_order = build_ordering(task_order, args.n_clients)
    out_base = args.out or os.path.join(
        REPO_ROOT, "results",
        f"cil_heatmap_{args.n_clients}client_{args.partition_type}_{args.task_size}task")
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    plot_cil_heatmap(counts, class_names, task_order, client_groups, client_order,
                     task_class_order, out_base, args.dpi)
    print(f"Saved {out_base}.pdf, {out_base}.svg, {out_base}.png")


if __name__ == "__main__":
    main()
