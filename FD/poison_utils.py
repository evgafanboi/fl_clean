import os
import numpy as np


def parse_poison_config(poison_arg):
    if not poison_arg:
        return None, None
    
    parts = poison_arg.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid poison format: {poison_arg}. Expected format: <attack_type>-<ratio>")
    
    attack_type = parts[0]
    try:
        ratio = float(parts[1])
        if not 0 < ratio <= 1:
            raise ValueError("Poison ratio must be between 0 and 1")
    except ValueError as e:
        raise ValueError(f"Invalid poison ratio: {parts[1]}. {e}")
    
    return attack_type, ratio


def get_or_create_poisoned_clients(partition_type, attack_type, ratio, n_clients, seed=42):
    history_dir = os.path.join("results", "poison_history")
    os.makedirs(history_dir, exist_ok=True)
    
    filename = f"{partition_type}_{attack_type}_{ratio}.txt"
    filepath = os.path.join(history_dir, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            poisoned_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        print(f"Loaded existing poisoned clients from {filepath}: {poisoned_ids}")
        return poisoned_ids
    
    n_poisoned = max(1, int(n_clients * ratio))
    rng = np.random.default_rng(seed)
    poisoned_ids = sorted(rng.choice(n_clients, size=n_poisoned, replace=False).tolist())
    
    with open(filepath, 'w') as f:
        f.write(f"# Poisoned clients for {partition_type} with {attack_type} attack (ratio={ratio})\n")
        f.write(f"# Total clients: {n_clients}, Poisoned: {n_poisoned} ({ratio*100:.1f}%)\n")
        f.write(f"# Attack type: {attack_type}\n")
        f.write(f"# Seed: {seed}\n")
        f.write("\n")
        for client_id in poisoned_ids:
            f.write(f"{client_id}\n")
    
    print(f"Created new poisoned clients list at {filepath}: {poisoned_ids}")
    return poisoned_ids


def apply_label_flip_poison(y, num_classes):
    y_poisoned = num_classes - y - 1
    return y_poisoned


class PoisonedDataLoader:
    
    def __init__(self, attack_type, num_classes):
        self.attack_type = attack_type
        self.num_classes = num_classes
    
    def poison_labels(self, y):
        """
        Apply poisoning to labels based on attack type.
        
        Args:
            y: original labels
            
        Returns:
            poisoned labels
        """
        if self.attack_type == "label_flip":
            return apply_label_flip_poison(y, self.num_classes)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
