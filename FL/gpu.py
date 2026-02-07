import os
import tensorflow as tf

from .colors import COLORS

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
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
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.9,
            allow_growth=True,
            polling_active_delay=10,
            allocator_type='BFC'
        )
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        print(f"{COLORS.OKCYAN}GPU memory growth enabled{COLORS.ENDC}")
    except Exception as exc:
        print(f"GPU configuration error: {exc}")
