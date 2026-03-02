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


def create_dataset(X_path: str, y_path: str, batch_size: int, num_classes: int, label_map: dict | None = None):
    """Load data and create TensorFlow dataset"""
    X = np.load(X_path)
    y = np.load(y_path)
    
    if label_map:
        labels = y if len(y.shape) == 1 else np.argmax(y, axis=1)
        mapped = np.vectorize(label_map.get)(labels)
        y = mapped
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
        for batch in dataset:
            if custom_train_step:
                if len(batch) == 3:
                    batch_X, batch_y, batch_old = batch
                    ce_loss, ewc_loss, total = custom_train_step(batch_X, batch_y, batch_old)
                else:
                    batch_X, batch_y = batch
                    ce_loss, ewc_loss, total = custom_train_step(batch_X, batch_y)
                total_loss += total.numpy()
            else:
                batch_X, batch_y = batch
                with tf.GradientTape() as tape:
                    predictions = keras_model(batch_X, training=True)
                    loss = loss_fn(batch_y, predictions)
                
                gradients = tape.gradient(loss, keras_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
                total_loss += loss.numpy()
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_client_fast(model, dataset, train_step, epochs: int):
    keras_model = model.model if hasattr(model, 'model') else model
    total_loss = 0.0
    num_batches = 0
    for epoch in range(epochs):
        for batch in dataset:
            if train_step:
                if len(batch) == 3:
                    _, _, total = train_step(batch[0], batch[1], batch[2])
                else:
                    _, _, total = train_step(batch[0], batch[1])
                total_loss += total.numpy()
            else:
                with tf.GradientTape() as tape:
                    predictions = keras_model(batch[0], training=True)
                    loss = keras_model.loss(batch[1], predictions)
                gradients = tape.gradient(loss, keras_model.trainable_variables)
                keras_model.optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
                total_loss += loss.numpy()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


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
        with open('data/CIC23_test.pkl', 'rb') as fh:
            test_data = pickle.load(fh)
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
    if config.strategy in ("FedAvg", "FedSSD"):
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
    elif config.cil_method == "lwf":
        from .cil.lwf import LwF
        cil_method = LwF(num_classes=num_classes, alpha=config.lwf_alpha, temperature=config.lwf_temperature)
    elif config.cil_method == "icarl":
        from .cil.icarl import ICaRL
        cil_method = ICaRL(num_classes=num_classes, memory=config.icarl_memory, use_bce=config.icarl_bce)
    elif config.cil_method == "bic":
        from .cil.bic import BiC
        cil_method = BiC(num_classes=num_classes, memory=config.icarl_memory,
                         alpha=config.lwf_alpha, temperature=config.lwf_temperature,
                         val_split=config.bic_val_split)
    elif config.cil_method == "foster":
        from .cil.foster import FOSTER
        cil_method = FOSTER(num_classes=num_classes, memory=config.icarl_memory,
                            beta1=config.foster_beta1, beta2=config.foster_beta2,
                            lambda_okd=config.foster_lambda_okd,
                            temperature=config.lwf_temperature,
                            compression_epochs=config.foster_compression_epochs)
    elif config.cil_method == "glfc":
        from .cil.glfc import GLFC
        cil_method = GLFC(num_classes=num_classes, memory=config.icarl_memory,
                          encoder_epochs=config.glfc_encoder_epochs,
                          model_selection=config.glfc_model_selection)
    
    # Initialize global model
    initial_classes = len(task_order[0]) if config.cil_method in ["lwf", "icarl", "bic", "foster", "glfc"] else num_classes
    global_model = create_model(initial_classes, input_dim, config.batch_size)
    
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
    
    label_map = None

    # Main loop: iterate through tasks
    for task_id in range(num_tasks):
        task_classes = task_order[task_id]
        
        log(
            f"\n{COLORS.HEADER}TASK {task_id}: {len(task_classes)} classes - {task_classes}{COLORS.ENDC}",
            f"TASK {task_id}: {len(task_classes)} classes - {task_classes}"
        )
        
        cil_method.before_task(task_id, task_classes)
        if hasattr(cil_method, "set_old_model") and task_id > 0:
            cil_method.set_old_model(global_model)
        if hasattr(global_model, "expand_classes") and hasattr(cil_method, "class_order"):
            global_model.expand_classes(len(cil_method.class_order))
        
        # FOSTER: zero old-class logits so boosted residual starts at 0
        if config.cil_method == "foster" and task_id > 0:
            cil_method.reinit_logits(global_model)
        
        # Create one reusable client model per task
        n_clients = 1 if config.strategy == "Centralized" else config.n_clients
        current_classes = len(cil_method.class_order) if hasattr(cil_method, "class_order") else num_classes
        label_map = cil_method.label_map if hasattr(cil_method, "label_map") else None
        client_model = create_model(current_classes, input_dim, config.batch_size)
        cached_train_step = cil_method.get_train_step(client_model, client_model.model.optimizer, client_model.model.loss)
        
        # SSD: merge public data to temp file once per task
        ssd_pub_X_path, ssd_pub_y_path = None, None
        if config.strategy == "FedSSD" and hasattr(cil_method, "get_ssd_train_step"):
            ssd_dir = Path("temp_weights/ssd_public")
            ssd_dir.mkdir(parents=True, exist_ok=True)
            ssd_pub_X_path = str(ssd_dir / f"task_{task_id}_X_public.npy")
            ssd_pub_y_path = str(ssd_dir / f"task_{task_id}_y_public.npy")
            base_dir = Path(config.partition_root) / f"{config.n_clients}_client" / config.partition_type
            pub_X_parts, pub_y_parts = [], []
            for t in range(task_id + 1):
                for c in range(n_clients):
                    px = base_dir / f"client_{c}_task_{t}_X_public.npy"
                    py = base_dir / f"client_{c}_task_{t}_y_public.npy"
                    if px.exists():
                        pub_X_parts.append(np.load(str(px)))
                        pub_y_parts.append(np.load(str(py)))
            np.save(ssd_pub_X_path, np.concatenate(pub_X_parts))
            np.save(ssd_pub_y_path, np.concatenate(pub_y_parts))
            del pub_X_parts, pub_y_parts
            gc.collect()

        # Run federated rounds for this task
        for round_num in range(config.rounds_per_task):
            
            # SSD: compute M_class each round from merged public dataset
            ssd_train_step = None
            if ssd_pub_X_path is not None:
                from .strategy.FedSSD import compute_class_metrics, build_ssd_components
                X_pub = np.load(ssd_pub_X_path, mmap_mode='r')
                y_pub = np.load(ssd_pub_y_path, mmap_mode='r')
                M_class = compute_class_metrics(
                    global_model, X_pub, y_pub,
                    current_classes, label_map)
                log(
                    f"{COLORS.OKCYAN}SSD M_class: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}{COLORS.ENDC}",
                    f"SSD M_class: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}")
                global_logits_model, M_class_tf, m_max_tf = build_ssd_components(
                    client_model, global_model, M_class, config.m_max)
                local_logits_model = client_model.get_logits_model()
                ssd_train_step = cil_method.get_ssd_train_step(
                    client_model, client_model.model.optimizer, client_model.model.loss,
                    global_logits_model, local_logits_model,
                    M_class_tf, m_max_tf)
                del X_pub, y_pub
                gc.collect()

            active_train_step = ssd_train_step if ssd_train_step is not None else cached_train_step
            
            client_weights = []
            sample_sizes = []
            
            for client_id in range(n_clients):
                if hasattr(cil_method, "set_client"):
                    cil_method.set_client(client_id)
                # Load task data
                paths = get_task_files(
                    config.partition_root, config.partition_type,
                    config.n_clients, client_id, task_id, config.strategy
                )
                
                if hasattr(cil_method, "build_dataset"):
                    dataset, n_samples = cil_method.build_dataset(
                        paths['X'], paths['y'], global_model, config.batch_size
                    )
                else:
                    dataset, n_samples = create_dataset(
                        paths['X'], paths['y'], config.batch_size, current_classes, label_map
                    )
                
                # Reuse client model, just reset weights
                client_model.set_weights(global_model.get_weights())
                
                # Train using active train step (SSD or cached CIL)
                loss = train_client_fast(client_model, dataset, active_train_step, config.epochs_per_round)
                
                client_weights.append(client_model.get_weights())
                sample_sizes.append(n_samples)

                # GLFC: collect proto gradients after client trains
                if hasattr(cil_method, "collect_proto_gradients"):
                    cil_method.collect_proto_gradients(
                        client_model, client_id, paths['X'], paths['y'])
                
                del dataset
                gc.collect()
                
                if config.strategy != "Centralized":
                    log(
                        f"{COLORS.WARNING}Client {client_id}: Loss={loss:.4f}{COLORS.ENDC}",
                        f"Client {client_id}: Loss={loss:.4f}"
                    )
            
            # Aggregate
            global_weights = aggregator.aggregate(client_weights, sample_sizes)
            global_model.set_weights(global_weights)
            del global_weights

            # GLFC: proxy server update with collected proto gradients
            if hasattr(cil_method, "proxy_update"):
                proxy_perf = cil_method.proxy_update(global_model)
                if proxy_perf is not None:
                    log(
                        f"{COLORS.OKCYAN}GLFC Proxy: accuracy={proxy_perf:.4f}{COLORS.ENDC}",
                        f"GLFC Proxy: accuracy={proxy_perf:.4f}")

            if hasattr(cil_method, "update_exemplars"):
                reduce_old = round_num == 0
                for client_id in range(n_clients):
                    if hasattr(cil_method, "set_client"):
                        cil_method.set_client(client_id)
                    paths = get_task_files(
                        config.partition_root, config.partition_type,
                        config.n_clients, client_id, task_id, config.strategy
                    )
                    cil_method.update_exemplars(global_model, paths['X'], paths['y'], task_classes, reduce_old=reduce_old)
                cil_method.rebuild_global_exemplars(global_model)
            
            # Clean up client data
            del client_weights
            gc.collect()

            # BiC: per-client bias correction → aggregate coefficients
            if config.cil_method == "bic" and task_id > 0:
                corrector_weights = []
                for cid in range(n_clients):
                    a, b = cil_method.run_client_bias_correction(global_model, cid)
                    corrector_weights.append([np.array([a, b])])
                agg_ab = aggregator.aggregate(corrector_weights, sample_sizes)
                cil_method.bias_alpha = float(agg_ab[0][0])
                cil_method.bias_beta = float(agg_ab[0][1])
                cil_method.apply_bias_correction(global_model)
                log(
                    f"{COLORS.OKCYAN}BiC correction: alpha={cil_method.bias_alpha:.4f}, beta={cil_method.bias_beta:.4f}{COLORS.ENDC}",
                    f"BiC correction: alpha={cil_method.bias_alpha:.4f}, beta={cil_method.bias_beta:.4f}")
                del corrector_weights

            del sample_sizes
            gc.collect()

            # Evaluate per-task classes (separately) using full test data
            X_test, y_test = load_test_data()
            for eval_task_id in range(task_id + 1):
                eval_classes = task_order[eval_task_id]
                eval_mask = np.isin(
                    y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1),
                    eval_classes
                )
                eval_y = y_test[eval_mask]
                if label_map and not hasattr(cil_method, "evaluate"):
                    eval_y = cil_method.map_labels_np(eval_y)
                if hasattr(cil_method, "evaluate"):
                    eval_results = cil_method.evaluate(global_model, X_test[eval_mask], eval_y, current_classes, logger=log)
                else:
                    eval_results = evaluate_model(
                        global_model,
                        X_test[eval_mask],
                        eval_y,
                        config.batch_size,
                        current_classes
                    )
                log(
                    f"{COLORS.OKGREEN}Round {round_num + 1} | Task {eval_task_id} | Acc: {eval_results['acc']:.4f}, Loss: {eval_results['loss']:.4f}{COLORS.ENDC}",
                    f"Round {round_num + 1} | Task {eval_task_id} | Acc: {eval_results['acc']:.4f} | Loss: {eval_results['loss']:.4f}"
                )
                del eval_mask
            
            del X_test, y_test
            gc.collect()

            # BiC: undo correction so next round trains on uncorrected model
            if config.cil_method == "bic" and task_id > 0:
                cil_method.undo_bias_correction(global_model)

        # FOSTER Stage 2: feature compression after last round of each task
        if config.cil_method == "foster" and task_id > 0:
            log(f"{COLORS.OKCYAN}FOSTER Stage 2: Feature Compression{COLORS.ENDC}",
                "FOSTER Stage 2: Feature Compression")
            task_paths = [
                get_task_files(config.partition_root, config.partition_type,
                               config.n_clients, cid, task_id, config.strategy)
                for cid in range(n_clients)
            ]
            cil_method.run_compression(global_model, config, task_paths)
            log(f"{COLORS.OKCYAN}FOSTER compression complete{COLORS.ENDC}",
                "FOSTER compression complete")

            # Post-compression eval so log reflects actual student quality
            X_test_pc, y_test_pc = load_test_data()
            for eval_task_id in range(task_id + 1):
                eval_classes = task_order[eval_task_id]
                eval_mask = np.isin(
                    y_test_pc if len(y_test_pc.shape) == 1 else np.argmax(y_test_pc, axis=1),
                    eval_classes)
                eval_results = cil_method.evaluate(
                    global_model, X_test_pc[eval_mask], y_test_pc[eval_mask],
                    current_classes, logger=log)
                log(
                    f"{COLORS.OKCYAN}Post-compress | Task {eval_task_id} | Acc: {eval_results['acc']:.4f}, Loss: {eval_results['loss']:.4f}{COLORS.ENDC}",
                    f"Post-compress | Task {eval_task_id} | Acc: {eval_results['acc']:.4f} | Loss: {eval_results['loss']:.4f}")
                del eval_mask
            del X_test_pc, y_test_pc
            gc.collect()

        # Clean up reusable client model for this task
        del client_model, cached_train_step
        gc.collect()
        
        # After task: compute Fisher (for EWC)
        if task_id < num_tasks - 1 and config.cil_method == "ewc":
            paths = get_task_files(
                config.partition_root, config.partition_type,
                config.n_clients, 0, task_id, config.strategy
            )
            task_dataset, _ = create_dataset(paths['X'], paths['y'], config.batch_size, num_classes)
            cil_method.after_task(global_model, task_dataset)
            del task_dataset
            gc.collect()
    
    log(f"{COLORS.OKGREEN}Pipeline completed!{COLORS.ENDC}", "Pipeline completed!")
    
    # Final evaluation on all classes
    X_test, y_test = load_test_data()
    final_classes = len(cil_method.class_order) if hasattr(cil_method, "class_order") else num_classes
    if label_map and not hasattr(cil_method, "evaluate"):
        y_test = cil_method.map_labels_np(y_test)
    if hasattr(cil_method, "evaluate"):
        final_results = cil_method.evaluate(global_model, X_test, y_test, final_classes, logger=log)
    else:
        final_results = evaluate_model(global_model, X_test, y_test, config.batch_size, final_classes)
    del X_test, y_test
    gc.collect()
    log(
        f"{COLORS.OKCYAN}Final test (all {num_classes} classes): Acc={final_results['acc']:.4f}, Loss={final_results['loss']:.4f}{COLORS.ENDC}",
        f"Final test (all {num_classes} classes): Acc={final_results['acc']:.4f} | Loss={final_results['loss']:.4f}"
    )
    
    if log_file:
        log_file.close()

    if hasattr(cil_method, "cleanup_temp"):
        cil_method.cleanup_temp()
