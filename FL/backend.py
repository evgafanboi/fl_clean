_use_tf = True


def use_tf() -> bool:
    return _use_tf


def set_backend(tf_mode: bool) -> None:
    global _use_tf
    _use_tf = tf_mode


def get_torch_device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
