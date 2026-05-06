import os


_use_tf = True


def use_tf() -> bool:
    return _use_tf


def set_backend(tf_mode: bool) -> None:
    global _use_tf
    _use_tf = tf_mode


def get_torch_device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_torch_loader_kwargs(workers: int | None = None, *, pin_memory: bool = False):
    n_workers = int(workers) if workers is not None else min(4, max(1, (os.cpu_count() or 1) // 8))
    kwargs = {
        'pin_memory': bool(pin_memory),
        'num_workers': max(0, n_workers),
    }
    if kwargs['num_workers'] > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4
    return kwargs
