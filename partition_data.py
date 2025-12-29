#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Partitioning for FL-IDS")
    parser.add_argument("--output_dir", type=str, default="data/partitions", help="Output directory for client partitions")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--partition_type", type=str, choices=["iid", "label_skew", "iid_poisoning"], 
                       default="iid", help="Partitioning strategy")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for label_skew")
    parser.add_argument("--poison_ratio", type=float, default=0.2, help="Ratio of clients to poison")
    parser.add_argument("--poison_intensity", type=float, default=0.5, help="Fraction of samples to poison per client")
    parser.add_argument("--public_ratio", type=float, default=0.285, help="Fraction of training data as public dataset")
    return parser.parse_args()

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

def create_iid_poisoning(all_data, n_clients, poison_ratio, poison_intensity, rng):
    n_poisoned = max(1, int(n_clients * poison_ratio))
    poisoned_clients = rng.choice(n_clients, size=n_poisoned, replace=False)
    
    shuffled_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
    n_samples = len(shuffled_data)
    
    splits = np.array_split(np.arange(n_samples), n_clients)
    client_data = {i: [] for i in range(n_clients)}
    
    unique_labels = sorted(all_data['label_encoded'].unique())
    num_classes = len(unique_labels)
    
    poison_mappings = {}
    
    if 0 in unique_labels and num_classes > 1:
        target_attack = rng.choice([l for l in unique_labels if l != 0])
        poison_mappings[0] = target_attack
    
    if num_classes > 2:
        attack_labels = [l for l in unique_labels if l != 0]
        if len(attack_labels) >= 2:
            attack_sample = rng.choice(attack_labels, size=min(len(attack_labels), 6), replace=False)
            for i in range(0, len(attack_sample) - 1, 2):
                if i + 1 < len(attack_sample):
                    poison_mappings[attack_sample[i]] = attack_sample[i + 1]
    
    if not poison_mappings:
        for i, label in enumerate(unique_labels):
            target_label = unique_labels[(i + 1) % num_classes]
            poison_mappings[label] = target_label
    
    poisoning_stats = {}
    
    for i in range(n_clients):
        if len(splits[i]) > 0:
            client_chunk = shuffled_data.iloc[splits[i]].copy()
            
            if i in poisoned_clients:
                original_labels = client_chunk['label_encoded'].copy()
                base_intensity = max(0.1, min(0.9, poison_intensity))
                intensity_variance = 0.2
                poison_fraction = base_intensity + (rng.random() - 0.5) * intensity_variance
                poison_fraction = max(0.1, min(0.9, poison_fraction))
                n_to_poison = int(len(client_chunk) * poison_fraction)
                poison_indices = rng.choice(len(client_chunk), size=n_to_poison, replace=False)
                
                poisoned_count = 0
                for idx in poison_indices:
                    original_label = client_chunk.iloc[idx]['label_encoded']
                    if original_label in poison_mappings:
                        client_chunk.iloc[idx, client_chunk.columns.get_loc('label_encoded')] = poison_mappings[original_label]
                        poisoned_count += 1
                
                final_labels = client_chunk['label_encoded']
                flipped_count = (original_labels != final_labels).sum()
                poisoning_stats[i] = {
                    'total_samples': len(client_chunk),
                    'poisoned_samples': poisoned_count,
                    'flipped_samples': flipped_count,
                    'poison_fraction': poison_fraction
                }
            
            client_data[i].append(client_chunk)
    
    return client_data, (poisoned_clients, poison_mappings, poisoning_stats)

def main():
    args = parse_arguments()
    n_clients = args.n_clients
    class_pkls = sorted(glob.glob(os.path.join("data/CIC23", "*_label*.pkl")))
    partition_type = args.partition_type
    
    output_dir = os.path.join(args.output_dir, f"{n_clients}_client", partition_type)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Step 1/4: Encoding labels from {len(class_pkls)} class files...")
    all_labels = []
    for class_pkl in tqdm(class_pkls, desc="Loading class files"):
        with open(class_pkl, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
            all_labels.extend(data['label'].tolist())
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            _, y = data
            all_labels.extend(y.tolist() if hasattr(y, 'tolist') else list(y))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    print(f"Found {len(label_encoder.classes_)} unique classes")
    np.save(os.path.join(output_dir, "label_classes.npy"), label_encoder.classes_)
    num_classes = len(label_encoder.classes_)

    for i in range(n_clients):
        client_pkl = os.path.join(output_dir, f"client_{i}.pkl")
        if os.path.exists(client_pkl):
            os.remove(client_pkl)

    client_data = {i: [] for i in range(n_clients)}
    
    if partition_type == "iid":
        print(f"Step 2/4: Partitioning data (IID) across {n_clients} clients...")
        for class_pkl in tqdm(class_pkls, desc="Processing classes"):
            with open(class_pkl, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, (tuple, list)) and len(data) == 2:
                X, y = data
                if hasattr(X, 'shape') and len(X.shape) == 2:
                    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
                    df = pd.DataFrame(X, columns=feature_cols)
                    df['label'] = y
                else:
                    continue
            else:
                continue
            df['label_encoded'] = label_encoder.transform(df['label'])
            for label_val, group_df in df.groupby('label_encoded'):
                n_samples = len(group_df)
                if n_samples == 0:
                    continue
                shuffled_df = group_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
                splits = np.array_split(np.arange(n_samples), n_clients)
                for i in range(n_clients):
                    if len(splits[i]) > 0:
                        client_data[i].append(shuffled_df.iloc[splits[i]])
            del df, data, shuffled_df
    
    elif partition_type == "label_skew":
        print(f"Step 2/4: Partitioning data (Label Skew, alpha={args.alpha}) across {n_clients} clients...")
        for class_pkl in tqdm(class_pkls, desc="Processing classes"):
            with open(class_pkl, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, (tuple, list)) and len(data) == 2:
                X, y = data
                if hasattr(X, 'shape') and len(X.shape) == 2:
                    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
                    df = pd.DataFrame(X, columns=feature_cols)
                    df['label'] = y
                else:
                    continue
            else:
                continue
            df['label_encoded'] = label_encoder.transform(df['label'])
            for label_val, group_df in df.groupby('label_encoded'):
                n_samples = len(group_df)
                if n_samples == 0:
                    continue
                shuffled_df = group_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
                counts = sample_dirichlet_counts(n_samples, n_clients, args.alpha, rng)
                start_idx = 0
                for i in range(n_clients):
                    end_idx = start_idx + counts[i]
                    if counts[i] > 0:
                        client_data[i].append(shuffled_df.iloc[start_idx:end_idx])
                    start_idx = end_idx
            del df, data
    
    elif partition_type == "iid_poisoning":
        print(f"Step 2/4: Partitioning data (IID + Poisoning) across {n_clients} clients...")
        all_dataframes = []
        for class_pkl in tqdm(class_pkls, desc="Loading classes"):
            with open(class_pkl, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, (tuple, list)) and len(data) == 2:
                X, y = data
                if hasattr(X, 'shape') and len(X.shape) == 2:
                    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
                    df = pd.DataFrame(X, columns=feature_cols)
                    df['label'] = y
                else:
                    continue
            else:
                continue
            all_dataframes.append(df)
        all_data = pd.concat(all_dataframes, ignore_index=True)
        all_data['label_encoded'] = label_encoder.transform(all_data['label'])
        client_data, poisoning_info = create_iid_poisoning(
            all_data, n_clients, args.poison_ratio, args.poison_intensity, rng
        )
        poisoned_clients, poison_mappings, poisoning_stats = poisoning_info
        with open(os.path.join(output_dir, "poisoned_clients.txt"), "w") as f:
            f.write("Poisoned Clients Information:\n")
            f.write("=" * 40 + "\n")
            f.write(f"Poisoned clients: {list(poisoned_clients)}\n")
            f.write(f"Total poisoned: {len(poisoned_clients)}/{n_clients}\n\n")
            
            f.write("Label Poisoning Mappings:\n")
            for orig, target in poison_mappings.items():
                f.write(f"  Class {orig} -> Class {target}\n")
            
            f.write("\nPer-Client Poisoning Statistics:\n")
            for client_id, stats in poisoning_stats.items():
                f.write(f"  Client {client_id}:\n")
                f.write(f"    Total samples: {stats['total_samples']}\n")
                f.write(f"    Poisoned samples: {stats['poisoned_samples']}\n") 
                f.write(f"    Poison fraction: {stats['poison_fraction']:.1%}\n")
                f.write(f"    Actually flipped: {stats['flipped_samples']}\n\n")
        del all_data, all_dataframes

    print(f"Step 3/4: Saving client data files...")
    for i in tqdm(range(n_clients), desc="Saving clients"):
        if not client_data[i]:
            continue
            
        client_df = pd.concat(client_data[i], ignore_index=True)
        client_pkl = os.path.join(output_dir, f"client_{i}.pkl")
        with open(client_pkl, 'wb') as f:
            pickle.dump(client_df, f)
        
        X = client_df.drop(['label', 'label_encoded'], axis=1).values.astype(np.float32)
        y = client_df['label_encoded'].values
        
        np.save(os.path.join(output_dir, f"client_{i}_X_train.npy"), X)
        np.save(os.path.join(output_dir, f"client_{i}_y_train.npy"), y)
        
        del X, y, client_df

    print(f"Step 4/4: Generating distribution tables...")
    generate_distribution_table(output_dir, num_classes)
    
    if args.public_ratio > 0:
        print(f"Carving public splits ({args.public_ratio:.1%} of data)...")
        carve_public_splits(output_dir, n_clients, num_classes, args.public_ratio, partition_type, rng)
        print(f"Regenerating distribution tables with public/private splits...")
        generate_distribution_table(output_dir, num_classes)
    
    print(f"Partitioning complete! Data saved to: {output_dir}")

def generate_distribution_table(output_dir, num_classes):
    import re
    
    def extract_client_num(filename):
        match = re.search(r'client_(\d+)(?:\.pkl|_y_train\.npy)', filename)
        return int(match.group(1)) if match else -1
    
    def detect_partition_format(data_dir):
        pkl_files = glob.glob(os.path.join(data_dir, 'client_*.pkl'))
        npy_y_train_files = glob.glob(os.path.join(data_dir, 'client_*_y_train.npy'))
        
        if pkl_files:
            return 'pkl', pkl_files
        elif npy_y_train_files:
            return 'npy_split', npy_y_train_files
        else:
            return None, []
    
    format_type, file_list = detect_partition_format(output_dir)
    if not file_list:
        print("No client files found for distribution table generation")
        return
    
    file_list = sorted(file_list, key=extract_client_num)
    
    all_labels = set()
    for file in file_list:
        if format_type == 'pkl':
            with open(file, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, pd.DataFrame):
                all_labels.update(data['label_encoded'].unique() if 'label_encoded' in data.columns else data['label'].unique())
            elif isinstance(data, (tuple, list)) and len(data) == 2:
                _, y = data
                if hasattr(y, 'unique'):
                    all_labels.update(y.unique())
                else:
                    all_labels.update(set(y))
        
        elif format_type == 'npy_split':
            y_train = np.load(file)
            all_labels.update(set(y_train.flatten()))
    
    all_labels = sorted(all_labels)
    num_client = len(file_list)
    
    table = []
    label_totals = [0] * len(all_labels)
    grand_total = 0
    for file in file_list:
        if format_type == 'pkl':
            with open(file, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, pd.DataFrame):
                label_col = 'label_encoded' if 'label_encoded' in data.columns else 'label'
                label_counts = data[label_col].value_counts().to_dict()
            elif isinstance(data, (tuple, list)) and len(data) == 2:
                _, y = data
                if hasattr(y, 'value_counts'):
                    label_counts = y.value_counts().to_dict()
                else:
                    label_counts = pd.Series(y).value_counts().to_dict()
        
        elif format_type == 'npy_split':
            y_train = np.load(file)
            label_counts = pd.Series(y_train.flatten()).value_counts().to_dict()
        
        row = [os.path.basename(file)]
        total = 0
        for i, label in enumerate(all_labels):
            count = label_counts.get(label, 0)
            row.append(count)
            label_totals[i] += count
            total += count
        row.append(total)
        grand_total += total
        table.append(row)
    
    total_row = ['Total']
    total_row.extend(label_totals)
    total_row.append(grand_total)
    table.append(total_row)
    
    header = ['Client'] + [str(label) for label in all_labels] + ['Total']
    md = '| ' + ' | '.join(header) + ' |\n'
    md += '| ' + ' | '.join(['---'] * len(header)) + ' |\n'
    for row in table:
        md += '| ' + ' | '.join(map(str, row)) + ' |\n'
    
    partition_name = os.path.basename(output_dir)
    base_name = f'{partition_name}_distribution'
    
    with open(os.path.join(output_dir, f'{base_name}.md'), 'w') as f:
        f.write(md)
    
    df_out = pd.DataFrame(table, columns=header)
    df_out.to_excel(os.path.join(output_dir, f'{base_name}.xlsx'), index=False)

def largest_remainder_method(target, shares):
    contributions = np.floor(target * shares).astype(int)
    remainder = int(target - contributions.sum())
    
    if remainder > 0:
        fractional_parts = target * shares - contributions
        top_indices = np.argsort(-fractional_parts)[:remainder]
        contributions[top_indices] += 1
    
    return contributions

def compute_client_class_distribution(output_dir, n_clients, num_classes):
    distribution = np.zeros((n_clients, num_classes), dtype=np.int64)
    
    for client_id in range(n_clients):
        y_path = os.path.join(output_dir, f"client_{client_id}_y_train.npy")
        y = np.load(y_path, mmap_mode='r')
        
        for class_id in range(num_classes):
            distribution[client_id, class_id] = np.sum(y == class_id)
    
    return distribution

def carve_public_splits(output_dir, n_clients, num_classes, public_ratio, partition_type, rng):
    if partition_type == "iid":
        total_public = 0
        total_private = 0
        
        for client_id in tqdm(range(n_clients), desc="Carving public/private"):
            X = np.load(os.path.join(output_dir, f"client_{client_id}_X_train.npy"))
            y = np.load(os.path.join(output_dir, f"client_{client_id}_y_train.npy"))
            
            n_samples = len(y)
            n_public = int(np.floor(public_ratio * n_samples))
            
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            
            public_indices = indices[:n_public]
            private_indices = indices[n_public:]
            
            X_public = X[public_indices]
            y_public = y[public_indices]
            X_private = X[private_indices]
            y_private = y[private_indices]
            
            np.save(os.path.join(output_dir, f"client_{client_id}_X_public.npy"), X_public)
            np.save(os.path.join(output_dir, f"client_{client_id}_y_public.npy"), y_public)
            np.save(os.path.join(output_dir, f"client_{client_id}_X_train.npy"), X_private)
            np.save(os.path.join(output_dir, f"client_{client_id}_y_train.npy"), y_private)
            
            total_public += len(X_public)
            total_private += len(X_private)
    
    else:
        distribution = compute_client_class_distribution(output_dir, n_clients, num_classes)
        class_totals = distribution.sum(axis=0)
        
        client_contributions = {i: {} for i in range(n_clients)}
        
        for class_id in range(num_classes):
            total_class = class_totals[class_id]
            target_public = int(np.floor(public_ratio * total_class))
            
            if total_class == 0:
                continue
            
            client_shares = distribution[:, class_id] / total_class
            contributions = largest_remainder_method(target_public, client_shares)
            
            for client_id in range(n_clients):
                if contributions[client_id] > 0:
                    client_contributions[client_id][class_id] = int(contributions[client_id])
        
        for client_id in tqdm(range(n_clients), desc="Carving public/private"):
            X = np.load(os.path.join(output_dir, f"client_{client_id}_X_train.npy"))
            y = np.load(os.path.join(output_dir, f"client_{client_id}_y_train.npy"))
            
            public_indices = []
            
            for class_id, target_count in client_contributions[client_id].items():
                class_indices = np.where(y == class_id)[0]
                rng.shuffle(class_indices)
                selected = class_indices[:target_count]
                public_indices.extend(selected)
            
            public_indices = np.array(public_indices, dtype=np.int64)
            private_indices = np.setdiff1d(np.arange(len(y)), public_indices)
            
            X_public = X[public_indices]
            y_public = y[public_indices]
            X_private = X[private_indices]
            y_private = y[private_indices]
            
            np.save(os.path.join(output_dir, f"client_{client_id}_X_public.npy"), X_public)
            np.save(os.path.join(output_dir, f"client_{client_id}_y_public.npy"), y_public)
            np.save(os.path.join(output_dir, f"client_{client_id}_X_train.npy"), X_private)
            np.save(os.path.join(output_dir, f"client_{client_id}_y_train.npy"), y_private)

if __name__ == "__main__":
    main()