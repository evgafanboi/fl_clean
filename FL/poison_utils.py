import os
import numpy as np


def parse_poison_config(poison_arg):
    if not poison_arg:
        return None, None, None
    parts = poison_arg.split()
    attack_type = parts[0]
    value = float(parts[1])
    ratio = float(parts[2])
    return attack_type, value, ratio


def get_or_create_poisoned_clients(partition_type, attack_type, value, ratio, n_clients, seed=42):
    history_dir = os.path.join("results", "poison_history")
    os.makedirs(history_dir, exist_ok=True)
    
    filename = f"{partition_type}_{attack_type}_{value}_{ratio}.txt"
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
        f.write(f"# Poisoned clients for {partition_type} with {attack_type} attack (value={value}, ratio={ratio})\n")
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


def apply_gaussian_noise_scale(model, scale):
    """Gaussian noise model poisoning for federated distillation (no global model).

    Computes the variance *v* of the entire flattened weight vector, then
    for every weight tensor *w*:
        w' = w + N(0,1) * v          (noisy weights)
        w_fin = scale * (w' - w) + w  (scaled update, same as standard FL)
             = w + scale * N(0,1) * v
    """
    keras_model = model.model if hasattr(model, "model") else model
    weights = keras_model.get_weights()

    all_params = np.concatenate([w.flatten() for w in weights])
    v = float(np.var(all_params))

    poisoned_weights = []
    for w in weights:
        noise = np.random.normal(0.0, 1.0, size=w.shape).astype(w.dtype)
        w_fin = w + scale * noise * v
        poisoned_weights.append(w_fin)

    keras_model.set_weights(poisoned_weights)


class PoisonedDataLoader:
    
    def __init__(self, attack_type, num_classes):
        self.attack_type = attack_type
        self.num_classes = num_classes
    
    def poison_labels(self, y):
        if self.attack_type == "label_flip":
            return apply_label_flip_poison(y, self.num_classes)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
