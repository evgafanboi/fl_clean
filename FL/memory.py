import gc
import time

from .backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf


def aggressive_memory_cleanup(sleep_seconds: float = 0.0) -> None:
    gc.collect()
    gc.collect()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def clear_tf_session_and_gc() -> None:
    """Full TF reset — only call when truly switching model architecture."""
    if _use_tf():
        tf.keras.backend.clear_session()
    gc.collect()
    gc.collect()


def clear_session() -> None:
    from .backend import use_tf
    if use_tf():
        tf.keras.backend.clear_session()
    else:
        import torch
        torch.cuda.empty_cache()
    gc.collect()
    gc.collect()
