import os
import tensorflow as tf

from .colors import COLORS

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Use CUDA's async memory allocator (CUDA 11.2+) which uses virtual memory
# management.  Unlike TF's default BFC allocator with memory_growth=True,
# it can defragment and never fails to find contiguous workspace for CuDNN.
os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
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
        allocator = os.environ.get('TF_GPU_ALLOCATOR', 'default')
        print(f"{COLORS.OKCYAN}GPU memory growth enabled + mixed precision enabled (allocator={allocator}){COLORS.ENDC}")
    except Exception as exc:
        print(f"GPU configuration error: {exc}")
