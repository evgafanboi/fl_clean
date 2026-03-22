import os
import tensorflow as tf

from .colors import COLORS

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)


def configure_gpu_memory() -> None:
    """Configure TensorFlow GPU memory growth if GPUs are available."""
    import warnings
    warnings.filterwarnings('ignore')
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return

    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"{COLORS.OKCYAN}GPU memory growth enabled + mixed precision enabled{COLORS.ENDC}")
    except Exception as exc:
        print(f"GPU configuration error: {exc}")
