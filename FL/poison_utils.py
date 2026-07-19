import os
import numpy as np


def parse_poison_config(poison_arg):
    if not poison_arg:
        return None, None, None
    parts = poison_arg.split()
    attack_type = parts[0]
    assert attack_type in ("gradient_scale", "label_flip", "targeted_flip", "poisonedfl", "ipoisonedfl", "lma", "ilma"), f"Unknown attack: {attack_type}"
    no_value = attack_type in ("label_flip", "lma", "ilma")
    value = None if no_value else float(parts[1])
    ratio = float(parts[-1])
    return attack_type, value, ratio


class PoisonedFLState:
    def __init__(self, c0: float):
        self.initial_scaling_factor = c0
        self.fixed_rand = None
        self.scaling_factor = c0
        self.has_applied_poison = False
        self.prev_global = None
        self.last_grad = None
        self.last_poisoned = None
        self.cached_update = None
        self.last_hypothesis_success = None
        self.last_aligned = None
        self.last_dimension = None

    def compute_update(self, current_flat: np.ndarray, allow_decay: bool = True):
        fmax = float(np.finfo(np.float32).max)
        current_flat = np.nan_to_num(current_flat.astype(np.float64, copy=False), nan=0.0, posinf=fmax, neginf=-fmax)
        d = current_flat.size
        if self.fixed_rand is None:
            rng = np.random.default_rng(42)
            self.fixed_rand = np.where(rng.random(d) < 0.5, 1.0, -1.0).astype(np.float32)
        if self.prev_global is None:
            self.last_aligned = None
            self.last_dimension = None
            self.prev_global = current_flat.copy()
            return None
        prev_global = np.nan_to_num(np.asarray(self.prev_global, dtype=np.float64), nan=0.0, posinf=fmax, neginf=-fmax)
        last_grad = current_flat - prev_global
        last_grad_norm = float(np.linalg.norm(last_grad))
        sf = self.scaling_factor
        if self.last_grad is None:
            deviation = self.fixed_rand / np.sqrt(float(d))
            m_update = np.nan_to_num(sf * last_grad_norm * deviation, nan=0.0, posinf=fmax, neginf=-fmax)
            m_update = np.clip(m_update, -fmax, fmax).astype(np.float32)
            self.last_grad = np.clip(last_grad, -fmax, fmax).astype(np.float32)
            self.last_poisoned = m_update.copy()
            self.prev_global = current_flat.copy()
            self.last_aligned = None
            self.last_dimension = None
            self.last_hypothesis_success = None
            print(f"  [PoisonedFL] armed c={sf:.4f}, 1-gradient attack, mal_norm={float(np.linalg.norm(m_update)):.4e}")
            return m_update
        history_vec = np.nan_to_num(np.asarray(self.last_grad, dtype=np.float64), nan=0.0, posinf=fmax, neginf=-fmax)
        history = history_vec.reshape(d, 1)
        history_norm = float(np.linalg.norm(history))
        residual = history - last_grad.reshape(d, 1) * history_norm / (last_grad_norm + 1e-9)
        scale = np.nan_to_num(np.linalg.norm(residual, axis=1), nan=0.0, posinf=fmax, neginf=0.0)
        deviation = np.nan_to_num(scale * self.fixed_rand / (float(np.linalg.norm(scale)) + 1e-9), nan=0.0, posinf=1.0, neginf=-1.0)
        if self.has_applied_poison:
            aligned = int(np.sum(np.sign(last_grad) == self.fixed_rand))
            k_99 = int(d / 2 + 2.326 * np.sqrt(d) / 2)
            hypothesis_success = aligned >= k_99
            if allow_decay and not hypothesis_success and sf * 0.7 >= 0.5:
                sf *= 0.7
            self.last_aligned = aligned
            self.last_dimension = d
        else:
            aligned = None
            k_99 = None
            hypothesis_success = None
            self.last_aligned = None
            self.last_dimension = None
        self.scaling_factor = sf
        self.last_hypothesis_success = hypothesis_success
        mal_update = np.nan_to_num(sf * history_norm * deviation, nan=0.0, posinf=fmax, neginf=-fmax)
        mal_update = np.clip(mal_update, -fmax, fmax).astype(np.float32)
        self.last_grad = np.clip(last_grad, -fmax, fmax).astype(np.float32)
        self.last_poisoned = mal_update.copy()
        self.prev_global = current_flat.copy()
        if hypothesis_success is None:
            print(f"  [PoisonedFL] armed c={sf:.4f}, hypothesis deferred, mal_norm={float(np.linalg.norm(mal_update)):.4e}")
        else:
            hypothesis = "H1" if hypothesis_success else "H0"
            print(f"  [PoisonedFL] {hypothesis} c={sf:.4f}, aligned={aligned}/{d} (k99={k_99}), mal_norm={float(np.linalg.norm(mal_update)):.4e}")
        return mal_update


def poisonedfl_unified_weights(byz_weights_list, state: PoisonedFLState, allow_decay: bool = True):
    consensus = [np.mean([w[i] for w in byz_weights_list], axis=0) for i in range(len(byz_weights_list[0]))]
    fmax = float(np.finfo(np.float32).max)
    current_flat = np.nan_to_num(np.concatenate([w.ravel() for w in consensus]).astype(np.float64), nan=0.0, posinf=fmax, neginf=-fmax)
    mal_update = state.compute_update(current_flat, allow_decay=allow_decay)
    state.cached_update = mal_update
    if mal_update is None:
        return consensus
    poisoned_flat = np.nan_to_num(current_flat + mal_update.astype(np.float64), nan=0.0, posinf=fmax, neginf=-fmax)
    poisoned_flat = np.clip(poisoned_flat, -fmax, fmax)
    offset, poisoned = 0, []
    for w in consensus:
        n = w.size
        poisoned.append(poisoned_flat[offset:offset + n].reshape(w.shape).astype(w.dtype))
        offset += n
    return poisoned


def ipoisonedfl_client_weights(weights, client_id: int):
    eps = np.finfo(np.float16).eps
    low, high = float(eps), float(0.1 * eps) # too large?
    rng = np.random.default_rng(int(client_id))
    out = []
    for w in weights:
        modifier = rng.uniform(low, high, size=w.shape).astype(w.dtype, copy=False)
        out.append((w + w * modifier).astype(w.dtype, copy=False))
    return out


def ipoisonedfl_client_weight_map(weights, client_ids):
    return {int(cid): ipoisonedfl_client_weights(weights, int(cid)) for cid in client_ids}


def poisonedfl_broadcast_weights(current_weights, state: PoisonedFLState, allow_decay: bool = True):
    fmax = float(np.finfo(np.float32).max)
    current_flat = np.nan_to_num(np.concatenate([w.ravel() for w in current_weights]).astype(np.float64), nan=0.0, posinf=fmax, neginf=-fmax)
    d = current_flat.size
    if state.fixed_rand is None:
        rng = np.random.default_rng(42)
        state.fixed_rand = np.where(rng.random(d) < 0.5, 1.0, -1.0).astype(np.float32)
    if state.prev_global is None:
        state.cached_update = None
        state.last_hypothesis_success = None
        state.last_aligned = None
        state.last_dimension = None
        state.prev_global = current_flat.copy()
        return [w.copy() for w in current_weights]
    prev_global = np.nan_to_num(np.asarray(state.prev_global, dtype=np.float64), nan=0.0, posinf=fmax, neginf=-fmax)
    last_grad = current_flat - prev_global
    last_grad_norm = float(np.linalg.norm(last_grad))
    if state.last_grad is None:
        scale = np.abs(last_grad)
    else:
        history_vec = np.nan_to_num(np.asarray(state.last_grad, dtype=np.float64), nan=0.0, posinf=fmax, neginf=-fmax)
        history_norm = float(np.linalg.norm(history_vec))
        scale = np.abs(history_vec - last_grad * history_norm / (last_grad_norm + 1e-9))
    scale = np.nan_to_num(scale, nan=0.0, posinf=fmax, neginf=0.0)
    deviation = np.nan_to_num(scale * state.fixed_rand / (float(np.linalg.norm(scale)) + 1e-9), nan=0.0, posinf=1.0, neginf=-1.0)
    sf = state.scaling_factor
    if state.has_applied_poison:
        aligned = int(np.sum(np.sign(last_grad) == state.fixed_rand))
        k_99 = int(d / 2 + 2.326 * np.sqrt(d) / 2)
        hypothesis_success = aligned >= k_99
        if allow_decay and not hypothesis_success and sf * 0.7 >= 0.5:
            sf *= 0.7
        state.last_aligned = aligned
        state.last_dimension = d
    else:
        aligned = None
        k_99 = None
        hypothesis_success = None
        state.last_aligned = None
        state.last_dimension = None
    state.scaling_factor = sf
    state.has_applied_poison = True
    state.last_hypothesis_success = hypothesis_success
    mal_update = np.nan_to_num(sf * last_grad_norm * deviation, nan=0.0, posinf=fmax, neginf=-fmax)
    mal_update = np.clip(mal_update, -fmax, fmax).astype(np.float32)
    state.last_grad = np.clip(last_grad, -fmax, fmax).astype(np.float32)
    state.last_poisoned = mal_update.copy()
    state.prev_global = current_flat.copy()
    state.cached_update = mal_update
    if hypothesis_success is None:
        print(f"  [PoisonedFL] armed c={sf:.4f}, hypothesis deferred, mal_norm={float(np.linalg.norm(mal_update)):.4e}")
    else:
        hypothesis = "H1" if hypothesis_success else "H0"
        print(f"  [PoisonedFL] {hypothesis} c={sf:.4f}, aligned={aligned}/{d} (k99={k_99}), mal_norm={float(np.linalg.norm(mal_update)):.4e}")
    poisoned_flat = np.nan_to_num(current_flat + mal_update.astype(np.float64), nan=0.0, posinf=fmax, neginf=-fmax)
    poisoned_flat = np.clip(poisoned_flat, -fmax, fmax)
    offset, poisoned = 0, []
    for w in current_weights:
        n = w.size
        poisoned.append(poisoned_flat[offset:offset + n].reshape(w.shape).astype(w.dtype))
        offset += n
    return poisoned


def poisonedfl_warmstart_weights(shared_state, state: PoisonedFLState, fallback=None, prefer_poisoned: bool = False):
    proxy_w = shared_state.get("poisonedfl_proxy_w")
    poisoned_w = shared_state.get("poisonedfl_ghost_w")
    if prefer_poisoned:
        selected = poisoned_w or proxy_w or fallback
    else:
        selected = proxy_w or poisoned_w or fallback
    if selected is None:
        return None
    return [w.copy() for w in selected]


def poisonedfl_store_round_weights(shared_state, proxy_w, poisoned_w, state: PoisonedFLState = None):
    shared_state["poisonedfl_proxy_w"] = [w.copy() for w in proxy_w]
    shared_state["poisonedfl_ghost_w"] = [w.copy() for w in poisoned_w]
    if state is not None and state.cached_update is not None:
        state.has_applied_poison = True


def poisonedfl_distill_loss_text(distill_loss):
    loss_value = None
    if distill_loss is not None:
        if hasattr(distill_loss, "history"):
            loss_hist = distill_loss.history.get("loss") if isinstance(distill_loss.history, dict) else None
            if loss_hist:
                loss_value = float(loss_hist[-1])
        else:
            loss_value = float(distill_loss)
    return "n/a" if loss_value is None else f"{loss_value:.4f}"


def poisonedfl_log_values(state: PoisonedFLState, distill_loss=None):
    distill_text = poisonedfl_distill_loss_text(distill_loss)
    alignment_text = "n/a"
    if state.last_aligned is not None and state.last_dimension is not None:
        alignment_text = f"{state.last_aligned}/{state.last_dimension}"
    mal_norm = float(np.linalg.norm(state.cached_update)) if state.cached_update is not None else 0.0
    return distill_text, float(state.initial_scaling_factor), float(state.scaling_factor), mal_norm, alignment_text


def get_or_create_poisoned_clients(partition_type, attack_type, value, ratio, n_clients, seed=42):
    history_dir = os.path.join("results", "poison_history")
    os.makedirs(history_dir, exist_ok=True)

    filename = f"{partition_type}_{attack_type}_{value}_{ratio}_{n_clients}.txt"
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


def apply_gradient_scale_poison(weights, scale):
    """Random-sign multiplicative poisoning: w_byz = w_trained * scale * S, S ∈ {-1, +1}^shape."""
    return [
        w * scale * np.where(np.random.rand(*w.shape) < 0.5, 1.0, -1.0).astype(w.dtype)
        for w in weights
    ]

def apply_targeted_flip_poison(y, target_label):
    return target_label


def apply_targeted_flip_poison_batch(y, target_label):
    y_out = y.copy()
    y_out[:] = target_label
    return y_out


def get_client_poison_loader(attack_type, client_id, poisoned_clients, num_classes, y_path, global_loader=None):
    if not attack_type or client_id not in poisoned_clients:
        return None
    if attack_type == "label_flip":
        return global_loader
    return None


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

    def __init__(self, attack_type, num_classes, dominant_class=None, target_label=None):
        self.attack_type = attack_type
        self.num_classes = num_classes
        self.dominant_class = dominant_class
        self.target_label = int(target_label) if target_label is not None else None

    def poison_labels(self, y):
        if self.attack_type == "label_flip":
            return apply_label_flip_poison(y, self.num_classes)
        if self.attack_type == "targeted_flip":
            mask = y == self.dominant_class
            y_out = y.copy()
            y_out[mask] = apply_targeted_flip_poison_batch(y[mask], self.target_label)
            return y_out
        raise ValueError(f"Unknown attack type: {self.attack_type}")
