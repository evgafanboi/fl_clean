import os

from .backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf

from .colors import COLORS

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if _use_tf():
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)


def configure_gpu_memory() -> None:
    import warnings
    warnings.filterwarnings('ignore')
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return

    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        with tf.device('/GPU:0'):
            _ = tf.zeros([1])
        print(f"{COLORS.OKCYAN}GPU memory growth enabled + mixed precision enabled{COLORS.ENDC}")
    except Exception as exc:
        print(f"GPU configuration error: {exc}")


def configure_pytorch_gpu() -> None:
    import torch
    if not torch.cuda.is_available():
        print(f"{COLORS.WARNING}No CUDA GPU found for PyTorch — running on CPU{COLORS.ENDC}")
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    with torch.cuda.device(0):
        _ = torch.zeros(1, device='cuda')
    print(f"{COLORS.OKCYAN}PyTorch GPU ready: {torch.cuda.get_device_name(0)}{COLORS.ENDC}")


def configure_gpu() -> None:
    from .backend import use_tf
    if use_tf():
        configure_gpu_memory()
    else:
        configure_pytorch_gpu()
