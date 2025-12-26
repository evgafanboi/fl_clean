import gc
import time

import tensorflow as tf


def aggressive_memory_cleanup(sleep_seconds: float = 0.1) -> None:
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    gc.collect()
    gc.collect()
    time.sleep(sleep_seconds)
