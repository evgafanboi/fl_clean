from __future__ import annotations

import os
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
from ..backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf
    from models import dense

    try:
        from models import gru  # type: ignore
    except Exception:
        gru = None  # type: ignore

    try:
        from models import dcblstm  # type: ignore
    except Exception:
        dcblstm = None  # type: ignore
else:
    dense = gru = dcblstm = None  # type: ignore


def create_model(input_dim: int, num_classes: int, batch_size: int, model_type: str = "dense"):
    from ..backend import use_tf
    if not use_tf():
        return _create_pt_model(input_dim, num_classes, batch_size, model_type)
    model_type_normalized = model_type.lower()
    if model_type_normalized == "dense":
        return dense.create_enhanced_dense_model(input_dim, num_classes, batch_size)
    if model_type_normalized == "gru":
        if gru is None:
            raise ValueError("GRU model requested but models.gru is unavailable")
        return gru.create_gru_model(input_dim, num_classes, batch_size)
    if model_type_normalized == "dcblstm":
        if dcblstm is None:
            raise ValueError("DCBLSTM model requested but models.dcblstm is unavailable")
        return dcblstm.create_dcblstm_model(input_dim, num_classes, batch_size)
    raise ValueError(f"Unknown model type: {model_type}")


def _create_pt_model(input_dim: int, num_classes: int, batch_size: int, model_type: str = "dense"):
    mt = model_type.lower()
    if mt == "dense":
        from models import pt_dense
        return pt_dense.create_dense_model(input_dim, num_classes, batch_size)
    if mt == "gru":
        from models import pt_gru
        return pt_gru.create_gru_model(input_dim, num_classes, batch_size)
    if mt == "dcblstm":
        from models import pt_dcblstm
        return pt_dcblstm.create_dcblstm_model(input_dim, num_classes, batch_size)
    if mt == "cnn":
        from models.mixed_models import create_cnn_model
        return create_cnn_model(input_dim, num_classes, batch_size)
    if mt == "mixed_dcblstm":
        from models.mixed_models import create_mixed_dcblstm_model
        return create_mixed_dcblstm_model(input_dim, num_classes, batch_size)
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
    poison_loader=None,
    cache: bool = True,
):
    from ..backend import use_tf
    if not use_tf():
        return _create_private_dataloader(X_path, y_path, num_classes, batch_size, poison_loader)
    X = np.load(X_path, mmap_mode="r").astype(np.float32, copy=False)
    y = np.load(y_path, mmap_mode="r").astype(np.int32, copy=False)

    if poison_loader is not None:
        y = poison_loader.poison_labels(y)

    if is_sequence:
        X = X.reshape(-1, 1, input_dim)

    if to_categorical and num_classes and (y.ndim == 1 or y.shape[1] == 1):
        y = tf.keras.utils.to_categorical(y.astype(np.int32), num_classes).astype(np.float32)

    idx = np.random.permutation(len(X))
    return (tf.data.Dataset.from_tensor_slices((X[idx], y[idx]))
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))


def _create_private_dataloader(X_path, y_path, num_classes, batch_size, poison_loader=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    X = np.load(X_path, mmap_mode='r').astype(np.float32, copy=False)
    y = np.load(y_path, mmap_mode='r').astype(np.int32, copy=False)
    if poison_loader is not None:
        y = poison_loader.poison_labels(y)
    y_oh = np.zeros((len(y), num_classes), dtype=np.float32)
    y_oh[np.arange(len(y)), y] = 1.0
    idx = np.random.permutation(len(X))
    ds = TensorDataset(torch.from_numpy(X[idx].copy()), torch.from_numpy(y_oh[idx]))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      pin_memory=False, num_workers=0)


def load_public_dataset_from_clients(
    paths_iterable: Iterable[Dict[str, str]],
    *,
    batch_size: int,
    num_classes: Optional[int] = None,
    shuffle: bool = False,
    return_labels: bool = False,
    is_sequence: bool = False,
):
    from ..backend import use_tf
    if not use_tf():
        return _load_public_dataloader_from_clients(
            paths_iterable, batch_size=batch_size, num_classes=num_classes,
            shuffle=shuffle, return_labels=return_labels)
    chunk_size = 10000
    public_files = []
    feature_dim = None
    
    for paths in paths_iterable:
        public_X = paths.get("public_X")
        public_y = paths.get("public_y")
        if public_X and tf.io.gfile.exists(public_X):
            X_mmap = np.load(public_X, mmap_mode='r')
            if feature_dim is None:
                feature_dim = int(X_mmap.shape[1]) if X_mmap.ndim > 1 else int(X_mmap.shape[0])
            if return_labels:
                if not public_y or not tf.io.gfile.exists(public_y):
                    raise ValueError("Public labels requested but missing")
                public_files.append((public_X, public_y, X_mmap.shape[0]))
            else:
                public_files.append((public_X, None, X_mmap.shape[0]))
            del X_mmap
    
    if not public_files:
        raise ValueError("No public data found in client partitions")
    
    total_samples = sum([p[2] for p in public_files])
    
    def generator():
        for x_path, y_path, total in public_files:
            X_mmap = np.load(x_path, mmap_mode='r')
            
            if return_labels:
                y_mmap = np.load(y_path, mmap_mode='r')
            
            for start_idx in range(0, total, chunk_size):
                end_idx = min(start_idx + chunk_size, total)
                X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
                
                if is_sequence:
                    X_chunk = X_chunk.reshape(-1, 1, X_chunk.shape[-1])
                
                if return_labels:
                    y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.float32)
                    
                    if num_classes and (len(y_chunk.shape) == 1 or y_chunk.shape[1] == 1):
                        y_chunk = tf.keras.utils.to_categorical(y_chunk.astype(np.int32), num_classes).astype(np.float32)
                    
                    yield X_chunk, y_chunk
                    del y_chunk
                else:
                    yield X_chunk
                
                del X_chunk
            
            del X_mmap
            if return_labels:
                del y_mmap
    
    if return_labels:
        if num_classes is None:
            raise ValueError("num_classes required when return_labels is True")
        
        if is_sequence:
            output_signature = (
                tf.TensorSpec(shape=(None, 1, feature_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
            )
        else:
            output_signature = (
                tf.TensorSpec(shape=(None, feature_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
            )
        
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    else:
        if is_sequence:
            output_signature = tf.TensorSpec(shape=(None, 1, feature_dim), dtype=tf.float32)
        else:
            output_signature = tf.TensorSpec(shape=(None, feature_dim), dtype=tf.float32)
        
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, total_samples))
    
    return dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE), total_samples


def numpy_from_dataset(dataset) -> np.ndarray:
    arrays = []
    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            X_batch = batch[0]
        else:
            X_batch = batch
        if isinstance(X_batch, np.ndarray):
            arrays.append(X_batch)
        else:
            arrays.append(X_batch.detach().cpu().numpy())
    if not arrays:
        return np.empty((0,))
    return np.concatenate(arrays, axis=0)


def _load_public_dataloader_from_clients(
    paths_iterable,
    *,
    batch_size: int,
    num_classes: Optional[int] = None,
    shuffle: bool = False,
    return_labels: bool = False,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    X_parts, y_parts = [], []
    total_samples = 0
    for paths in paths_iterable:
        public_X = paths.get("public_X")
        public_y = paths.get("public_y")
        if not public_X or not os.path.exists(public_X):
            continue
        X_mmap = np.load(public_X, mmap_mode='r').astype(np.float32)
        X_parts.append(np.array(X_mmap))
        total_samples += len(X_mmap)
        if return_labels and public_y and os.path.exists(public_y):
            y_raw = np.load(public_y, mmap_mode='r').astype(np.int32)
            y_oh = np.zeros((len(y_raw), num_classes), dtype=np.float32)
            y_oh[np.arange(len(y_raw)), y_raw] = 1.0
            y_parts.append(y_oh)
    X_all = np.concatenate(X_parts, axis=0)
    if shuffle:
        idx = np.random.permutation(len(X_all))
        X_all = X_all[idx]
    if return_labels:
        y_all = np.concatenate(y_parts, axis=0)
        if shuffle:
            y_all = y_all[idx]
        ds = TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all))
    else:
        ds = TensorDataset(torch.from_numpy(X_all))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        pin_memory=False, num_workers=0)
    return loader, total_samples
