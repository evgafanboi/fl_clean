import os
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .colors import COLORS

_TestDatasetCache = Optional[Tuple[np.ndarray, np.ndarray]]
_test_dataset_cache: _TestDatasetCache = None


def parse_partition_type(partition_type: str) -> Tuple[str, int]:
    if '-' in partition_type:
        parts = partition_type.split('-')
        if len(parts) == 2:
            return parts[0], int(parts[1])
    raise ValueError(f"Invalid partition type '{partition_type}'. Use format 'iid-10'")


def setup_paths(client_id: str, partition_type: str, client_count: int) -> Dict[str, str]:
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


def restore_full_partition(paths: Dict[str, str]) -> bool:
    X_train = np.load(paths['train_X'], mmap_mode='r')
    y_train = np.load(paths['train_y'], mmap_mode='r')

    if os.path.exists(paths['public_X']):
        X_public = np.load(paths['public_X'])
        y_public = np.load(paths['public_y'])

        X_full = np.concatenate([np.array(X_train), X_public], axis=0)
        y_full = np.concatenate([np.array(y_train), y_public], axis=0)

        np.save(paths['train_X'], X_full)
        np.save(paths['train_y'], y_full)
        return True

    return False


def create_client_dataset(
    X_path: str,
    y_path: str,
    input_dim: int,
    num_classes: int,
    batch_size: int,
    chunk_size: int = 10000,
    poison_loader=None
) -> tf.data.Dataset:
    def generator():
        X_mmap = np.load(X_path, mmap_mode='r')
        y_mmap = np.load(y_path, mmap_mode='r')
        total_samples = X_mmap.shape[0]

        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
            y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.float32)
            
            if poison_loader is not None:
                y_chunk = poison_loader.poison_labels(y_chunk.astype(np.int32))
            
            if len(y_chunk.shape) == 1 or y_chunk.shape[1] == 1:
                y_chunk = tf.keras.utils.to_categorical(
                    y_chunk.astype(np.int32),
                    num_classes=num_classes
                ).astype(np.float32)
            yield X_chunk, y_chunk
            del X_chunk, y_chunk

    output_signature = (
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_test_dataset(batch_size: int, num_classes: int) -> tf.data.Dataset:
    global _test_dataset_cache

    if _test_dataset_cache is None:
        X_test = np.load("data/X_test.npy")
        y_test = np.load("data/y_test.npy")
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if len(y_test.shape) == 1:
            y_test = tf.keras.utils.to_categorical(y_test.astype(np.int32), num_classes).astype(np.float32)
        _test_dataset_cache = (X_test, y_test)
        print(f"{COLORS.OKGREEN}Test dataset cached ({X_test.shape[0]} samples){COLORS.ENDC}")

    X_test, y_test = _test_dataset_cache
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
