"""FCIL Pipeline - Main execution logic for continual learning"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import gc

from .config import FCILConfig
from .colors import COLORS


def load_task_order(task_order_file: str) -> dict:
    """Load task order from JSON file"""
    with open(task_order_file, 'r') as f:
        task_order_str = json.load(f)
        return {int(k): v for k, v in task_order_str.items()}


def get_task_files(partition_root: str, partition_type: str, n_clients: int,
                    client_id: int, task_id: int, strategy: str):
    """Get file paths for a specific task"""
    if strategy == "Centralized":
        base_dir = Path(partition_root) / f"{n_clients}_client" / partition_type / "centralized"
        return {
            'X': str(base_dir / f'task_{task_id}_X_train.npy'),
            'y': str(base_dir / f'task_{task_id}_y_train.npy'),
        }
    else:
        base_dir = Path(partition_root) / f"{n_clients}_client" / partition_type
        return {
            'X': str(base_dir / f'client_{client_id}_task_{task_id}_X_train.npy'),
            'y': str(base_dir / f'client_{client_id}_task_{task_id}_y_train.npy'),
        }


def create_dataset(X_path: str, y_path: str, batch_size: int, num_classes: int):
    """Load data and create TensorFlow dataset"""
    X = np.load(X_path)
    y = np.load(y_path)
    
    if len(y.shape) == 1:
        y = tf.keras.utils.to_categorical(y, num_classes)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, X.shape[0]


def create_model(num_classes: int, input_dim: int, batch_size: int):
    """Create model (uses dense architecture with built-in compilation)"""
    from models.dense import DenseModel
    model = DenseModel(num_classes=num_classes, input_dim=input_dim, batch_size=batch_size)
    return model


def train_client(model, dataset, cil_method, epochs: int):
    """Train client model with CIL method"""
    keras_model = model.model if hasattr(model, 'model') else model
    optimizer = keras_model.optimizer
    loss_fn = keras_model.loss
    
    custom_train_step = cil_method.get_train_step(model, optimizer, loss_fn)
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch_X, batch_y in dataset:
            if custom_train_step:
                ce_loss, ewc_loss, total = custom_train_step(batch_X, batch_y)
                total_loss += total.numpy()
            else:
                with tf.GradientTape() as tape:
                    predictions = keras_model(batch_X, training=True)
                    loss = loss_fn(batch_y, predictions)
                
                gradients = tape.gradient(loss, keras_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
                total_loss += loss.numpy()
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, batch_size: int, num_classes: int):
    """Evaluate model on test set"""
    keras_model = model.model if hasattr(model, 'model') else model
    
    if len(y_test.shape) == 1:
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.map(lambda x, y: (x, tf.one_hot(y, num_classes)))
    else:
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size)
    
    results = keras_model.evaluate(test_dataset, verbose=0)
    return {'loss': results[0], 'acc': results[1]}


def run_fcil_pipeline(config: FCILConfig):
    """Main FCIL pipeline execution"""
    
    # Load task order
    task_order = load_task_order(config.task_order_file)
    num_tasks = len(task_order)
    all_classes = sorted(set(c for classes in task_order.values() for c in classes))
    num_classes = len(all_classes)
    
    print(f"\n{COLORS.OKCYAN}Task Order: {num_tasks} tasks, {num_classes} total classes{COLORS.ENDC}")
    for task_id, classes in task_order.items():
        print(f"  Task {task_id}: {len(classes)} classes - {classes}")
    
    def load_test_data():
        import pickle
        test_data = pickle.load(open('data/CIC23_test.pkl', 'rb'))
        if isinstance(test_data, tuple):
            X_full, y_full = test_data
        else:
            X_full = test_data.iloc[:, :-1].values
            y_full = test_data.iloc[:, -1].values
        return X_full, y_full
    
    X_test, y_test = load_test_data()
    input_dim = X_test.shape[1]
    print(f"{COLORS.OKCYAN}Test set: {X_test.shape[0]:,} samples, {input_dim} features{COLORS.ENDC}")
    del X_test, y_test
    gc.collect()
    
    # Import strategy and CIL method
    if config.strategy == "FedAvg":
        from .strategy.FedAvg import FedAvg
        aggregator = FedAvg()
    elif config.strategy == "Centralized":
        from .strategy.Centralized import Centralized
        aggregator = Centralized()
    elif config.strategy == "FedCoMed":
        from .strategy.FedCoMed import FedCoMed
        aggregator = FedCoMed()
    
    if config.cil_method == "finetune":
        from .cil.finetune import Finetune
        cil_method = Finetune(num_classes=num_classes)
    elif config.cil_method == "ewc":
        from .cil.ewc import EWC
        cil_method = EWC(num_classes=num_classes, ewc_lambda=config.ewc_lambda)
    
    # Initialize global model
    global_model = create_model(num_classes, input_dim, config.batch_size)
    
    # Logging
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(config.log_file, 'w')
        print(f"{COLORS.OKCYAN}Logging to {config.log_file}{COLORS.ENDC}")
    else:
        log_file = None
    
    def log(msg, file_msg=None):
        print(msg)
        if log_file:
            log_file.write((file_msg if file_msg is not None else msg) + '\n')
            log_file.flush()
    
    # Main loop: iterate through tasks
    for task_id in range(num_tasks):
        task_classes = task_order[task_id]
        
        log(
            f"\n{COLORS.HEADER}TASK {task_id}: {len(task_classes)} classes - {task_classes}{COLORS.ENDC}",
            f"TASK {task_id}: {len(task_classes)} classes - {task_classes}"
        )
        
        cil_method.before_task(task_id, task_classes)
        
        # Run rounds for this task
        for round_num in range(config.rounds_per_task):
            
            client_weights = []
            sample_sizes = []
            
            n_clients = 1 if config.strategy == "Centralized" else config.n_clients
            
            for client_id in range(n_clients):
                # Load task data
                paths = get_task_files(
                    config.partition_root, config.partition_type,
                    config.n_clients, client_id, task_id, config.strategy
                )
                
                dataset, n_samples = create_dataset(
                    paths['X'], paths['y'], config.batch_size, num_classes
                )
                
                # Create client model and set global weights
                client_model = create_model(num_classes, input_dim, config.batch_size)
                client_model.set_weights(global_model.get_weights())
                
                # Train
                loss = train_client(client_model, dataset, cil_method, config.epochs_per_round)
                
                client_weights.append(client_model.get_weights())
                sample_sizes.append(n_samples)
                
                # Clean up client model and dataset
                del client_model, dataset
                
                if config.strategy != "Centralized":
                    log(
                        f"{COLORS.WARNING}Client {client_id}: Loss={loss:.4f}{COLORS.ENDC}",
                        f"Client {client_id}: Loss={loss:.4f}"
                    )
            
            # Aggregate
            global_weights = aggregator.aggregate(client_weights, sample_sizes)
            global_model.set_weights(global_weights)
            del global_weights
            
            # Clean up client data
            del client_weights, sample_sizes
            gc.collect()
            
            # Evaluate per-task classes (separately) using full test data
            X_test, y_test = load_test_data()
            for eval_task_id in range(task_id + 1):
                eval_classes = task_order[eval_task_id]
                eval_mask = np.isin(
                    y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1),
                    eval_classes
                )
                eval_results = evaluate_model(
                    global_model,
                    X_test[eval_mask],
                    y_test[eval_mask],
                    config.batch_size,
                    num_classes
                )
                log(
                    f"{COLORS.OKGREEN}Round {round_num + 1} | Task {eval_task_id} | Acc: {eval_results['acc']:.4f}, Loss: {eval_results['loss']:.4f}{COLORS.ENDC}",
                    f"Round {round_num + 1} | Task {eval_task_id} | Acc: {eval_results['acc']:.4f} | Loss: {eval_results['loss']:.4f}"
                )
                del eval_mask
            
            del X_test, y_test
            gc.collect()
        
        # After task: compute Fisher (for EWC) or other post-task operations
        if task_id < num_tasks - 1:
            paths = get_task_files(
                config.partition_root, config.partition_type,
                config.n_clients, 0, task_id, config.strategy
            )
            task_dataset, _ = create_dataset(paths['X'], paths['y'], config.batch_size, num_classes)
            cil_method.after_task(global_model, task_dataset)
            
            # Clean up task dataset after Fisher computation
            del task_dataset
            gc.collect()
    
    log(f"{COLORS.OKGREEN}Pipeline completed!{COLORS.ENDC}", "Pipeline completed!")
    
    # Final evaluation on all classes
    X_test, y_test = load_test_data()
    final_results = evaluate_model(global_model, X_test, y_test, config.batch_size, num_classes)
    del X_test, y_test
    gc.collect()
    log(
        f"{COLORS.OKCYAN}Final test (all {num_classes} classes): Acc={final_results['acc']:.4f}, Loss={final_results['loss']:.4f}{COLORS.ENDC}",
        f"Final test (all {num_classes} classes): Acc={final_results['acc']:.4f} | Loss={final_results['loss']:.4f}"
    )
    
    if log_file:
        log_file.close()
