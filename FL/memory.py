import gc
import time

import tensorflow as tf


def aggressive_memory_cleanup(sleep_seconds: float = 0.0) -> None:
    gc.collect()
    gc.collect()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def clear_tf_session_and_gc() -> None:
    """Full TF reset — only call when truly switching model architecture."""
    tf.keras.backend.clear_session()
    gc.collect()
    gc.collect()
