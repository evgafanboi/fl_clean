"""
Repartition CIC23 class data into per-task client splits for Federated CIL.

Generates fresh Dirichlet partitions from scratch for each task into
``data/partitions/<n_clients>_client/<partition_type>_cil/``.

When ``len(task_order) == 7`` the active client count grows from 40% at
task 0 to 100% at task 6 (``+10%`` per task). Inactive clients receive
empty task files so the FCIL pipeline can still load by ``client_id``.
Original ``partition_data`` files are untouched.
"""

import argparse
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def determine_num_classes(data_dir='data/CIC23'):
    data_path = Path(data_dir)
    if data_path.exists():
        class_files = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
        return len(class_files)
    raise FileNotFoundError("Cannot determine number of classes from data directory, run partition_data.py first")


def generate_task_order(num_classes, task_size, benign_class=0):
    if isinstance(task_size, float) and task_size < 1:
        classes_per_task = max(1, int(num_classes * task_size))
    else:
        classes_per_task = int(task_size)
    all_classes = list(range(num_classes))
    if benign_class in all_classes:
        all_classes.remove(benign_class)
    np.random.shuffle(all_classes)
    task_order = {}
    task_id = 0
    task_order[task_id] = [benign_class] + all_classes[:classes_per_task - 1]
    all_classes = all_classes[classes_per_task - 1:]
    task_id += 1
    while all_classes:
        task_order[task_id] = all_classes[:classes_per_task]
        all_classes = all_classes[classes_per_task:]
        task_id += 1
    return task_order


def save_task_order(task_order, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("# Task-Class Assignment for Continual Learning\n")
        f.write("# Format: task_id: [class_indices]\n\n")
        for task_id, classes in task_order.items():
            f.write(f"{task_id}: {classes}\n")
    json_path = save_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump({str(k): v for k, v in task_order.items()}, f, indent=2)
    print(f"Task order saved to:\n  {save_path}\n  {json_path}")


def load_task_order(load_path):
    json_path = load_path.replace('.txt', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            task_order_str = json.load(f)
            return {int(k): v for k, v in task_order_str.items()}
    task_order = {}
    with open(load_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            task_id, classes = line.split(':', 1)
            task_order[int(task_id)] = eval(classes.strip())
    return task_order


def sample_dirichlet_counts(n_samples, n_clients, alpha, rng):
    proportions = rng.dirichlet([alpha] * n_clients)
    proportions = proportions / proportions.sum()
    counts = np.floor(proportions * n_samples).astype(int)
    diff = n_samples - np.sum(counts)
    if diff != 0:
        remainders = proportions * n_samples - counts
        indices = np.argsort(remainders)[::-1]
        for i in range(abs(diff)):
            idx = indices[i % n_clients]
            counts[idx] += 1 if diff > 0 else -1
    return counts


def task_active_n(n_clients, num_tasks, task_id):
    if num_tasks != 7:
        return n_clients
    return max(1, int(n_clients * (0.4 + task_id * 0.1)))


def build_label_encoder(data_dir):
    class_pkls = sorted(glob.glob(os.path.join(data_dir, "*_label*.pkl")))
    if not class_pkls:
        raise FileNotFoundError(f"No class files found in {data_dir}")
    all_labels = []
    for class_pkl in class_pkls:
        with open(class_pkl, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
            all_labels.extend(data['label'].tolist())
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            _, y = data
            all_labels.extend(y.tolist() if hasattr(y, 'tolist') else list(y))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    return label_encoder


def load_class_data(class_pkl, label_encoder):
    with open(class_pkl, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (tuple, list)) and len(data) == 2:
        X, y = data
        feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df['label'] = y
    else:
        raise ValueError(f"Unsupported class file format: {class_pkl}")
    df['label_encoded'] = label_encoder.transform(df['label'])
    return df


def detect_input_dim(data_dir, label_encoder):
    class_pkls = sorted(glob.glob(os.path.join(data_dir, "*_label*.pkl")))
    sample_df = load_class_data(class_pkls[0], label_encoder)
    return sample_df.drop(['label', 'label_encoded'], axis=1).shape[1]


def save_client_task_files(partition_path, client_id, task_id, X_train, y_train, X_public, y_public):
    np.save(partition_path / f'client_{client_id}_task_{task_id}_X_train.npy', X_train)
    np.save(partition_path / f'client_{client_id}_task_{task_id}_y_train.npy', y_train)
    np.save(partition_path / f'client_{client_id}_task_{task_id}_X_public.npy', X_public)
    np.save(partition_path / f'client_{client_id}_task_{task_id}_y_public.npy', y_public)


def repartition_tasks_from_scratch(partition_dir, data_dir, n_clients, task_order, partition_type, alpha, seed):
    partition_path = Path(partition_dir)
    partition_path.mkdir(parents=True, exist_ok=True)
    label_encoder = build_label_encoder(data_dir)
    class_pkls = sorted(glob.glob(os.path.join(data_dir, "*_label*.pkl")))
    input_dim = detect_input_dim(data_dir, label_encoder)
    num_tasks = len(task_order)
    base_rng = np.random.default_rng(seed)
    empty_X = np.empty((0, input_dim), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.int64)
    print(f"\nRepartitioning CIL tasks into {partition_path}")
    print(f"Task order: {num_tasks} tasks")
    for task_id, task_classes in task_order.items():
        active_n = task_active_n(n_clients, num_tasks, task_id)
        task_seed = int(base_rng.integers(0, 2**32 - 1))
        task_rng = np.random.default_rng(task_seed)
        client_X = {cid: [] for cid in range(active_n)}
        client_y = {cid: [] for cid in range(active_n)}
        print(f"\nTask {task_id}: {len(task_classes)} classes -> {active_n}/{n_clients} active clients (seed={task_seed})")
        for class_pkl in tqdm(class_pkls, desc=f"Task {task_id}", leave=False):
            df = load_class_data(class_pkl, label_encoder)
            if int(df['label_encoded'].iloc[0]) not in task_classes:
                continue
            shuffled_df = df.sample(frac=1, random_state=int(task_rng.integers(0, 2**32 - 1))).reset_index(drop=True)
            n_samples = len(shuffled_df)
            if partition_type == 'iid':
                splits = np.array_split(np.arange(n_samples), active_n)
                counts = np.array([len(split) for split in splits], dtype=int)
            elif partition_type.startswith('label_skew_'):
                counts = sample_dirichlet_counts(n_samples, active_n, alpha, task_rng)
            else:
                raise ValueError(f"Unsupported partition_type for CIL repartition: {partition_type}")
            start_idx = 0
            for client_id, count in enumerate(counts):
                end_idx = start_idx + count
                if count > 0:
                    client_df = shuffled_df.iloc[start_idx:end_idx]
                    client_X[client_id].append(client_df.drop(['label', 'label_encoded'], axis=1).values.astype(np.float32))
                    client_y[client_id].append(client_df['label_encoded'].values.astype(np.int64))
                start_idx = end_idx
        for client_id in range(n_clients):
            if client_id < active_n and client_y[client_id]:
                X_all = np.concatenate(client_X[client_id], axis=0)
                y_all = np.concatenate(client_y[client_id], axis=0)
                perm = task_rng.permutation(len(y_all))
                X_all = X_all[perm]
                y_all = y_all[perm]
                public_ratio = task_rng.uniform(0.05, 0.20)
                n_public = int(np.floor(public_ratio * len(y_all)))
                X_public = X_all[:n_public]
                y_public = y_all[:n_public]
                X_train = X_all[n_public:]
                y_train = y_all[n_public:]
            else:
                X_train, y_train, X_public, y_public = empty_X, empty_y, empty_X, empty_y
            save_client_task_files(partition_path, client_id, task_id, X_train, y_train, X_public, y_public)
            if client_id < active_n:
                print(f"  client_{client_id}: total={len(y_train) + len(y_public):,} train={len(y_train):,} public={len(y_public):,}")
        if n_clients - active_n > 0:
            print(f"  inactive clients with empty task files: {n_clients - active_n}")
    print(f"\nCIL repartition created in {partition_path}")


def create_centralized_tasks(partition_dir, n_clients, task_order, file_types=['X_train', 'y_train']):
    partition_path = Path(partition_dir)
    centralized_dir = partition_path / 'centralized'
    centralized_dir.mkdir(exist_ok=True)
    print(f"\nCreating centralized task files in {centralized_dir}")
    for task_id in task_order.keys():
        for file_type in file_types:
            merged_data = []
            for client_id in range(n_clients):
                task_file = partition_path / f'client_{client_id}_task_{task_id}_{file_type}.npy'
                if not task_file.exists():
                    continue
                merged_data.append(np.load(task_file))
            if merged_data:
                merged = np.concatenate(merged_data, axis=0)
                output_file = centralized_dir / f'task_{task_id}_{file_type}.npy'
                np.save(output_file, merged)
                print(f"  Task {task_id} {file_type}: {merged.shape[0]:,} samples ({n_clients} clients merged) -> {output_file.name}")
    print(f"\n✓ Centralized task files created in {centralized_dir}")


def parse_partition_alpha(partition_type):
    if partition_type.startswith('label_skew_'):
        return float(partition_type.split('label_skew_', 1)[1])
    return None


def main():
    parser = argparse.ArgumentParser(description='Repartition CIL task files from scratch')
    parser.add_argument('--task_size', type=float, default=5.0,
                        help='Classes per task (int) or proportion of classes (float < 1). Default: 5')
    parser.add_argument('--benign_class', type=int, default=0,
                        help='Benign class to include in first task. Default: 0')
    parser.add_argument('--partition_type', type=str, required=True,
                        help='Partition type (e.g., label_skew_0.1, iid)')
    parser.add_argument('--n_clients', type=int, required=True,
                        help='Number of clients (e.g., 100)')
    parser.add_argument('--data_dir', type=str, default='data/CIC23',
                        help='Data directory to determine number of classes')
    parser.add_argument('--partition_root', type=str, default='data/partitions',
                        help='Root directory containing partitions')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for task order generation')
    parser.add_argument('--create_centralized', action='store_true',
                        help='Create centralized merged task files for centralized baseline')
    args = parser.parse_args()
    np.random.seed(args.seed)
    num_classes = determine_num_classes(args.data_dir)
    print(f"Detected {num_classes} classes from {args.data_dir}")
    task_order_file = f"results/incremental_order/{args.partition_type}_{args.n_clients}_client_{args.task_size}_task.txt"
    if os.path.exists(task_order_file.replace('.txt', '.json')):
        print(f"\nLoading existing task order from {task_order_file}")
        task_order = load_task_order(task_order_file)
    else:
        print(f"\nGenerating new task order (task_size={args.task_size}, benign={args.benign_class})")
        task_order = generate_task_order(num_classes, args.task_size, args.benign_class)
        save_task_order(task_order, task_order_file)
    print("\nTask assignment:")
    for task_id, classes in task_order.items():
        print(f"  Task {task_id}: {len(classes)} classes - {classes}")
    partition_dir = f"{args.partition_root}/{args.n_clients}_client/{args.partition_type}_cil"
    alpha = parse_partition_alpha(args.partition_type)
    repartition_tasks_from_scratch(
        partition_dir=partition_dir,
        data_dir=args.data_dir,
        n_clients=args.n_clients,
        task_order=task_order,
        partition_type=args.partition_type,
        alpha=alpha,
        seed=args.seed,
    )
    if args.create_centralized:
        create_centralized_tasks(
            partition_dir=partition_dir,
            n_clients=args.n_clients,
            task_order=task_order,
            file_types=['X_train', 'y_train']
        )
    print("\nTask partitioning completed!")
    print(f"\nTask order: {task_order_file}")
    print(f"CIL partitions: {partition_dir}")
    if args.create_centralized:
        print(f"Centralized: {partition_dir}/centralized/task_*_*.npy")


if __name__ == '__main__':
    main()
