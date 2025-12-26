import os
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


def parse_partition_type(partition_type: str) -> Tuple[str, int]:
    if '-' in partition_type:
        parts = partition_type.split('-')
        if len(parts) == 2:
            return parts[0], int(parts[1])
    raise ValueError(f"Invalid partition type '{partition_type}'. Use format 'label_skew-10'")


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

    local_test_X = os.path.join(partition_dir, f"client_{client_id}_X_test.npy")
    local_test_y = os.path.join(partition_dir, f"client_{client_id}_y_test.npy")
    if os.path.exists(local_test_X) and os.path.exists(local_test_y):
        paths['local_test_X'] = local_test_X
        paths['local_test_y'] = local_test_y

    return paths


def load_test_dataset(batch_size: int, num_classes: int) -> tf.data.Dataset:
    X_test = np.load("data/X_test.npy", mmap_mode='r')
    y_test = np.load("data/y_test.npy", mmap_mode='r')

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    if len(y_test.shape) == 1:
        y_test = tf.keras.utils.to_categorical(y_test.astype(np.int32), num_classes).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
