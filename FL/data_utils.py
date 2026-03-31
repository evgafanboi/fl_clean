import os
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .colors import COLORS

_test_tf_dataset_cache = None
_test_tf_dataset_batch_size = None
_test_tf_num_classes = None

_client_data_cache: dict = {}


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
    """Point paths to merged partition files, creating them if needed."""
    if not os.path.exists(paths['public_X']):
        return False

    merged_X_path = paths['train_X'].replace('_train.npy', '_merged.npy')
    merged_y_path = paths['train_y'].replace('_train.npy', '_merged.npy')

    # Skip the expensive concat+save if pre-built merged files exist on disk
    # (created by restore_partition.py)
    if not (os.path.exists(merged_X_path) and os.path.exists(merged_y_path)):
        X_train = np.load(paths['train_X'], mmap_mode='r')
        y_train = np.load(paths['train_y'], mmap_mode='r')
        X_public = np.load(paths['public_X'], mmap_mode='r')
        y_public = np.load(paths['public_y'], mmap_mode='r')

        np.save(merged_X_path, np.concatenate([np.array(X_train), np.array(X_public)], axis=0))
        np.save(merged_y_path, np.concatenate([np.array(y_train), np.array(y_public)], axis=0))

    paths['train_X'] = merged_X_path
    paths['train_y'] = merged_y_path

    return True


def create_client_dataset(
    X_path: str,
    y_path: str,
    input_dim: int,
    num_classes: int,
    batch_size: int,
    poison_loader=None,
    cache: bool = True,
) -> tf.data.Dataset:
    cache_key = (X_path, y_path, num_classes, id(poison_loader))
    if cache_key in _client_data_cache:
        X, y = _client_data_cache[cache_key]
    else:
        X = np.load(X_path, mmap_mode='r').astype(np.float32, copy=False)
        y = np.load(y_path, mmap_mode='r').astype(np.int32, copy=False)
        if poison_loader is not None:
            y = poison_loader.poison_labels(y)
        if len(y.shape) == 1 or y.shape[1] == 1:
            y = tf.keras.utils.to_categorical(
                y.astype(np.int32), num_classes=num_classes
            ).astype(np.float32)
        _client_data_cache[cache_key] = (X, y)

    idx = np.random.permutation(len(X))
    dataset = tf.data.Dataset.from_tensor_slices((X[idx], y[idx]))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_test_dataset(batch_size: int, num_classes: int) -> tf.data.Dataset:
    global _test_tf_dataset_cache, _test_tf_dataset_batch_size, _test_tf_num_classes

    if (_test_tf_dataset_cache is None
            or _test_tf_dataset_batch_size != batch_size
            or _test_tf_num_classes != num_classes):
        X_test = np.load("data/X_test.npy").astype(np.float32, copy=False)
        y_test = np.load("data/y_test.npy")
        if y_test.ndim == 1 or y_test.shape[1] == 1:
            y_test = tf.keras.utils.to_categorical(y_test.astype(np.int32), num_classes).astype(np.float32)
        else:
            y_test = y_test.astype(np.float32, copy=False)
        _test_tf_dataset_cache = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
                                  .batch(batch_size).prefetch(tf.data.AUTOTUNE))
        _test_tf_dataset_batch_size = batch_size
        _test_tf_num_classes = num_classes
        print(f"{COLORS.OKGREEN}Test dataset cached ({X_test.shape[0]} samples){COLORS.ENDC}")

    return _test_tf_dataset_cache
