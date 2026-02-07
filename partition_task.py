"""
Partition existing client data into task-specific files for Continual/Incremental Learning.

Reads existing partition files and splits them by class into task-ordered files.
Does NOT modify original partition files.
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
import pickle


def determine_num_classes(data_dir='data/CIC23'):
    data_path = Path(data_dir)
    
    if (data_path).exists():
        # list files and read len to determine number of classes
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


def split_partition_into_tasks(
    partition_dir,
    n_clients,
    task_order,
    file_types=['X_train', 'y_train', 'X_public', 'y_public']
):
    partition_path = Path(partition_dir)
    
    if not partition_path.exists():
        raise FileNotFoundError(f"Partition directory not found: {partition_dir}")
    
    print(f"\nProcessing {n_clients} clients in {partition_dir}")
    print(f"Task order: {len(task_order)} tasks")
    
    for client_idx in range(n_clients):
        print(f"\nClient {client_idx}:")
        
        for file_type in file_types:
            data_file = partition_path / f'client_{client_idx}_{file_type}.npy'
            
            if not data_file.exists():
                print(f"  Warning: {data_file.name} not found, skipping")
                continue
            
            data = np.load(data_file, mmap_mode='r')
            is_label = file_type.startswith('y_')
            
            for task_id, task_classes in task_order.items():
                if is_label:
                    if len(data.shape) == 1:
                        task_mask = np.isin(data, task_classes)
                    else:
                        task_mask = np.isin(np.argmax(data, axis=1), task_classes)
                    
                    task_data = np.array(data[task_mask])
                else:
                    label_file = data_file.parent / data_file.name.replace('X_', 'y_')
                    labels = np.load(label_file, mmap_mode='r')
                    
                    if len(labels.shape) == 1:
                        task_mask = np.isin(labels, task_classes)
                    else:
                        task_mask = np.isin(np.argmax(labels, axis=1), task_classes)
                    
                    task_data = np.array(data[task_mask])
                
                output_file = partition_path / f'client_{client_idx}_task_{task_id}_{file_type}.npy'
                np.save(output_file, task_data)
                
                if task_id == 0:
                    print(f"  {file_type}: {data.shape[0]:,} total samples -> {len(task_order)} tasks")
                
                print(f"    Task {task_id} ({len(task_classes)} classes): {task_data.shape[0]:,} samples -> {output_file.name}")
    
    print(f"\nIncremental partitions created")


def create_centralized_tasks(
    partition_dir,
    n_clients,
    task_order,
    file_types=['X_train', 'y_train']
):
    """
    Merge all client task files into centralized task files.
    Creates centralized/task_{task_id}_X_train.npy files.
    """
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
                    print(f"  Warning: {task_file.name} not found, skipping")
                    continue
                
                data = np.load(task_file)
                merged_data.append(data)
            
            if merged_data:
                merged = np.concatenate(merged_data, axis=0)
                output_file = centralized_dir / f'task_{task_id}_{file_type}.npy'
                np.save(output_file, merged)
                
                print(f"  Task {task_id} {file_type}: {merged.shape[0]:,} samples "
                      f"({n_clients} clients merged) -> {output_file.name}")
    
    print(f"\n✓ Centralized task files created in {centralized_dir}")


def main():
    parser = argparse.ArgumentParser(description='Split partitions into task-ordered files for CIL')
    parser.add_argument('--task_size', type=float, default=5,
                        help='Classes per task (int) or proportion of classes (float < 1). Default: 5')
    parser.add_argument('--benign_class', type=int, default=0,
                        help='Benign class to include in first task. Default: 0')
    parser.add_argument('--partition_type', type=str, required=True,
                        help='Partition type (e.g., label_skew, iid)')
    parser.add_argument('--n_clients', type=int, required=True,
                        help='Number of clients (e.g., 10)')
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
    
    partition_dir = f"{args.partition_root}/{args.n_clients}_client/{args.partition_type}"
    
    split_partition_into_tasks(
        partition_dir=partition_dir,
        n_clients=args.n_clients,
        task_order=task_order,
        file_types=['X_train', 'y_train', 'X_public', 'y_public']
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
    if args.create_centralized:
        print(f"Centralized: {partition_dir}/centralized/task_*_*.npy")


if __name__ == '__main__':
    main()
