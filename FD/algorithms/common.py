from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf

from models import dense

try:  # Optional GRU dependency used by select algorithms
    from models import gru  # type: ignore
except Exception:  # pragma: no cover - gracefully handle absence
    gru = None  # type: ignore


def create_model(input_dim: int, num_classes: int, batch_size: int, model_type: str = "dense"):
    model_type_normalized = model_type.lower()
    if model_type_normalized == "dense":
        return dense.create_enhanced_dense_model(input_dim, num_classes, batch_size)
    if model_type_normalized == "gru":
        if gru is None:
            raise ValueError("GRU model requested but models.gru is unavailable")
        return gru.create_enhanced_gru_model((1, input_dim), num_classes, batch_size)
    raise ValueError(f"Unknown model type: {model_type}")


def create_private_dataset(
    X_path: str,
    y_path: str,
    input_dim: int,
    num_classes: int,
    batch_size: int,
    *,
    chunk_size: int = 100000,
    to_categorical: bool = True,
    is_sequence: bool = False,
) -> tf.data.Dataset:
    def generator() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        X_mmap = np.load(X_path, mmap_mode="r")
        y_mmap = np.load(y_path, mmap_mode="r")
        total_samples = X_mmap.shape[0]

        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
            y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.float32)

            if is_sequence:
                X_chunk = X_chunk.reshape(-1, 1, input_dim)

            if to_categorical and num_classes and (y_chunk.ndim == 1 or y_chunk.shape[1] == 1):
                y_chunk = tf.keras.utils.to_categorical(y_chunk.astype(np.int32), num_classes)

            yield X_chunk, y_chunk
            del X_chunk, y_chunk

    output_signature = (
        tf.TensorSpec(
            shape=(None, 1, input_dim),
            dtype=tf.float32,
        )
        if is_sequence
        else tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_public_dataset_from_clients(
    paths_iterable: Iterable[Dict[str, str]],
    *,
    batch_size: int,
    num_classes: Optional[int] = None,
    shuffle: bool = False,
    return_labels: bool = False,
    is_sequence: bool = False,
) -> Tuple[tf.data.Dataset, int]:
    X_collections = []
    y_collections = []

    for paths in paths_iterable:
        public_X = paths.get("public_X")
        public_y = paths.get("public_y")
        if public_X and tf.io.gfile.exists(public_X):
            X_data = np.load(public_X)
            X_collections.append(X_data.astype(np.float32))
            if return_labels:
                if not public_y or not tf.io.gfile.exists(public_y):
                    raise ValueError("Public labels requested but missing")
                y_data = np.load(public_y)
                y_collections.append(y_data.astype(np.float32))

    if not X_collections:
        raise ValueError("No public data found in client partitions")

    X_public = np.concatenate(X_collections, axis=0)
    total_samples = X_public.shape[0]

    if shuffle:
        shuffle_idx = np.random.permutation(total_samples)
        X_public = X_public[shuffle_idx]
        if return_labels:
            y_public = np.concatenate(y_collections, axis=0)
            y_public = y_public.astype(np.float32)[shuffle_idx]
        else:
            y_public = None
    else:
        y_public = np.concatenate(y_collections, axis=0).astype(np.float32) if return_labels else None

    if is_sequence:
        X_public = X_public.reshape(-1, 1, X_public.shape[-1])

    if return_labels:
        if num_classes is None:
            raise ValueError("num_classes required when return_labels is True")
        if y_public.ndim == 1 or y_public.shape[1] == 1:
            y_public = tf.keras.utils.to_categorical(y_public.astype(np.int32), num_classes).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((X_public, y_public))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(X_public)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), total_samples


def numpy_from_dataset(dataset: tf.data.Dataset) -> np.ndarray:
    arrays = []
    for batch in dataset:
        if isinstance(batch, tuple):
            X_batch = batch[0]
        else:
            X_batch = batch
        arrays.append(np.array(X_batch))
    if not arrays:
        return np.empty((0,))
    return np.concatenate(arrays, axis=0)
