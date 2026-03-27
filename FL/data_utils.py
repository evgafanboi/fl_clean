import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .colors import COLORS

_TestDatasetCache = Optional[Tuple[np.ndarray, np.ndarray]]
_test_dataset_cache: _TestDatasetCache = None
_test_tf_dataset_cache = None
_test_tf_dataset_batch_size = None


def parse_partition_type(partition_type: str) -> Tuple[str, int]:
    if '-' in partition_type:
        parts = partition_type.split('-')
        if len(parts) == 2:
            return parts[0], int(parts[1])
    raise ValueError(f"Invalid partition type '{partition_type}'. Use format 'iid-10'")


def setup_paths(client_id: str, partition_type: str, client_count: int) -> Dict[str, Any]:
    partitions_root = os.path.join("data", "partitions")
    client_folder = f"{client_count}_client"
    partition_dir = os.path.join(partitions_root, client_folder, partition_type)

    paths = {
        'train_X': os.path.join(partition_dir, f"client_{client_id}_X_train.npy"),
        'train_y': os.path.join(partition_dir, f"client_{client_id}_y_train.npy"),
        'public_X': os.path.join(partition_dir, f"client_{client_id}_X_public.npy"),
        'public_y': os.path.join(partition_dir, f"client_{client_id}_y_public.npy"),
        'test_X': os.path.join("data", "X_test.npy"),
        'test_y': os.path.join("data", "y_test.npy")
    }

    test_X_path = os.path.join(partition_dir, f"client_{client_id}_X_test.npy")
    test_y_path = os.path.join(partition_dir, f"client_{client_id}_y_test.npy")

    if os.path.exists(test_X_path) and os.path.exists(test_y_path):
        paths['local_test_X'] = test_X_path
        paths['local_test_y'] = test_y_path

    return paths


def restore_full_partition(paths: Dict[str, Any]) -> bool:
    """Mark partition for lazy in-memory merge (no disk I/O).
    
    Instead of loading and writing merged files to disk (slow on Colab),
    we just flag that public data exists. The actual merge happens
    on-demand in create_client_dataset when the data is accessed.
    
    This approach avoids disk writes which can be slow on cloud environments
    like Google Colab, while still allowing full partition access.
    """
    if not os.path.exists(paths['public_X']):
        return False
    
    # Just mark that this partition needs merging - actual merge happens lazily
    paths['_needs_merge'] = True
    return True


def create_client_dataset(
    X_path: str,
    y_path: str,
    input_dim: int,
    num_classes: int,
    batch_size: int,
    poison_loader=None,
    cache: bool = True,
    paths: Optional[Dict[str, Any]] = None,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset for client training.
    
    Args:
        X_path: Path to features file (used if no lazy merge needed)
        y_path: Path to labels file (used if no lazy merge needed)
        input_dim: Input feature dimension
        num_classes: Number of output classes
        batch_size: Batch size for training
        poison_loader: Optional poison loader for label flipping attacks
        cache: Whether to cache dataset (unused, kept for compatibility)
        paths: Optional paths dict with '_needs_merge' flag for lazy merging
    """
    # Check if we need to do lazy in-memory merge
    needs_merge = paths is not None and paths.get('_needs_merge', False)
    
    def generator():
        # Load data - merge public+private if needed
        if needs_merge:
            # Lazy merge: load both partitions and concatenate in memory
            X_train = np.load(paths['train_X'], mmap_mode='r')
            y_train = np.load(paths['train_y'], mmap_mode='r')
            X_public = np.load(paths['public_X'], mmap_mode='r')
            y_public = np.load(paths['public_y'], mmap_mode='r')
            
            # Concatenate (converts mmap to arrays)
            X_data = np.concatenate([X_train, X_public], axis=0)
            y_data = np.concatenate([y_train, y_public], axis=0)
        else:
            X_data = np.load(X_path, mmap_mode='r')
            y_data = np.load(y_path, mmap_mode='r')
        
        total_samples = X_data.shape[0]
        indices = np.random.permutation(total_samples)

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            idx = indices[start_idx:end_idx]
            X_chunk = np.array(X_data[idx], dtype=np.float32)
            y_chunk = np.array(y_data[idx], dtype=np.int32)

            if poison_loader is not None:
                y_chunk = poison_loader.poison_labels(y_chunk)

            if len(y_chunk.shape) == 1 or y_chunk.shape[1] == 1:
                y_chunk = tf.keras.utils.to_categorical(
                    y_chunk.astype(np.int32),
                    num_classes=num_classes
                ).astype(np.float32)
            yield X_chunk, y_chunk

    output_signature = (
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_client_sample_size(paths: Dict[str, Any]) -> int:
    """Get the total sample size for a client, accounting for lazy merge.
    
    This avoids loading the full data just to get the sample count.
    """
    if paths.get('_needs_merge', False):
        # Count samples from both partitions
        X_train = np.load(paths['train_X'], mmap_mode='r')
        X_public = np.load(paths['public_X'], mmap_mode='r')
        total = X_train.shape[0] + X_public.shape[0]
        del X_train, X_public
        return total
    else:
        X_train = np.load(paths['train_X'], mmap_mode='r')
        total = X_train.shape[0]
        del X_train
        return total


def load_test_dataset(batch_size: int, num_classes: int) -> tf.data.Dataset:
    global _test_dataset_cache, _test_tf_dataset_cache, _test_tf_dataset_batch_size

    if _test_dataset_cache is None:
        X_test = np.load("data/X_test.npy")
        y_test = np.load("data/y_test.npy")
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if len(y_test.shape) == 1:
            y_test = tf.keras.utils.to_categorical(y_test.astype(np.int32), num_classes).astype(np.float32)
        _test_dataset_cache = (X_test, y_test)
        print(f"{COLORS.OKGREEN}Test dataset cached ({X_test.shape[0]} samples){COLORS.ENDC}")

    if _test_tf_dataset_cache is None or _test_tf_dataset_batch_size != batch_size:
        X_test, y_test = _test_dataset_cache
        dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        _test_tf_dataset_cache = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        _test_tf_dataset_batch_size = batch_size

    return _test_tf_dataset_cache
