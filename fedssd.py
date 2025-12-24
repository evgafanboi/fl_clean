import os
import argparse
import numpy as np
import tensorflow as tf
import gc
import logging
import pandas as pd
import time
import pickle
from datetime import datetime
from strategy.FedAvg import FedAvg
from models import dense, gru
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    except Exception as e:
        print(f"GPU error: {e}")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    PURPLE = '\033[95m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log_timestamp(logger, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {message}")
    print(f"{bcolors.OKCYAN}[{timestamp}] {message}{bcolors.ENDC}")

def setup_logger(n_clients, partition_type):
    log_filename = f"results/FedSSD_{n_clients}client_{partition_type}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(levelname)s: %(message)s', filemode='w')
    print(f"{bcolors.OKCYAN}Logging to {log_filename}{bcolors.ENDC}")
    
    detailed_log_filename = log_filename.replace('.log', '_detailed_class_metrics.log')
    detailed_logger = logging.getLogger('detailed_metrics')
    detailed_logger.setLevel(logging.INFO)
    detailed_handler = logging.FileHandler(detailed_log_filename, mode='w')
    detailed_handler.setFormatter(logging.Formatter('%(message)s'))
    detailed_logger.addHandler(detailed_handler)
    detailed_logger.propagate = False
    
    return logging.getLogger(), log_filename, detailed_logger

def parse_partition_type(partition_type):
    if '-' in partition_type:
        parts = partition_type.split('-')
        if len(parts) == 2:
            return parts[0], int(parts[1])
    raise ValueError(f"Invalid partition type '{partition_type}'. Use format 'label_skew-10'")

def setup_paths(client_id, partition_type, client_count):
    partitions_root = os.path.join("data", "partitions")
    client_folder = f"{client_count}_client"
    partition_dir = os.path.join(partitions_root, client_folder, partition_type)
    
    return {
        'train_X': os.path.join(partition_dir, f"client_{client_id}_X_train.npy"),
        'train_y': os.path.join(partition_dir, f"client_{client_id}_y_train.npy"),
        'public_X': os.path.join(partition_dir, f"client_{client_id}_X_public.npy"),
        'public_y': os.path.join(partition_dir, f"client_{client_id}_y_public.npy"),
    }

def load_public_dataset_from_clients(paths_list, input_dim, num_classes, batch_size, is_gru=False):
    chunk_size = 10000
    public_files = []
    for paths in paths_list:
        if os.path.exists(paths['public_X']):
            X_mmap = np.load(paths['public_X'], mmap_mode='r')
            public_files.append((paths['public_X'], paths['public_y'], X_mmap.shape[0]))
            del X_mmap
    
    def generator():
        for x_path, y_path, total in public_files:
            X_mmap = np.load(x_path, mmap_mode='r')
            y_mmap = np.load(y_path, mmap_mode='r')
            for start_idx in range(0, total, chunk_size):
                end_idx = min(start_idx + chunk_size, total)
                X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
                y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.float32)
                if is_gru:
                    X_chunk = X_chunk.reshape(-1, 1, input_dim)
                if len(y_chunk.shape) == 1 or y_chunk.shape[1] == 1:
                    y_chunk = tf.keras.utils.to_categorical(y_chunk.astype(np.int32), num_classes=num_classes)
                yield X_chunk, y_chunk
                del X_chunk, y_chunk
            del X_mmap, y_mmap
    
    output_signature = (
        tf.TensorSpec(shape=(None, 1, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    ) if is_gru else (
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE), sum([p[2] for p in public_files])

def restore_full_partition(paths):
    """Restore original partition by concatenating public and private splits"""
    X_train = np.load(paths['train_X'])
    y_train = np.load(paths['train_y'])
    
    if os.path.exists(paths['public_X']):
        X_public = np.load(paths['public_X'])
        y_public = np.load(paths['public_y'])
        
        X_full = np.concatenate([X_train, X_public], axis=0)
        y_full = np.concatenate([y_train, y_public], axis=0)
        
        return X_full, y_full
    
    return X_train, y_train

def create_model(input_dim, num_classes, model_type, batch_size):
    if model_type.lower() == "dense":
        return dense.create_dense_model(input_dim, num_classes, batch_size)
    elif model_type.lower() == "gru":
        return gru.create_enhanced_gru_model((1, input_dim), num_classes, batch_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_dataset(X_path, y_path, input_dim, num_classes, batch_size, is_gru=False):
    def generator():
        X_mmap = np.load(X_path, mmap_mode='r')
        y_mmap = np.load(y_path, mmap_mode='r')
        total_samples = X_mmap.shape[0]
        chunk_size = 10000
        
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
            y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.float32)
            
            if is_gru:
                X_chunk = X_chunk.reshape(-1, 1, input_dim)
            
            if len(y_chunk.shape) == 1:
                y_chunk = tf.keras.utils.to_categorical(y_chunk.astype(np.int32), num_classes).astype(np.float32)
            
            yield X_chunk, y_chunk
            del X_chunk, y_chunk
    
    if is_gru:
        output_signature = (
            tf.TensorSpec(shape=(None, 1, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_test_dataset(input_dim, num_classes, batch_size, is_gru=False):
    X_test_path = os.path.join("data", "X_test.npy")
    y_test_path = os.path.join("data", "y_test.npy")
    
    def generator():
        X_mmap = np.load(X_test_path, mmap_mode='r')
        y_mmap = np.load(y_test_path, mmap_mode='r')
        total_samples = X_mmap.shape[0]
        chunk_size = 10000
        
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
            y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.float32)
            
            if is_gru:
                X_chunk = X_chunk.reshape(-1, 1, input_dim)
            
            if len(y_chunk.shape) == 1:
                y_chunk = tf.keras.utils.to_categorical(y_chunk.astype(np.int32), num_classes).astype(np.float32)
            
            yield X_chunk, y_chunk
            del X_chunk, y_chunk
    
    if is_gru:
        output_signature = (
            tf.TensorSpec(shape=(None, 1, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)

def compute_class_metrics(model, aux_dataset, num_classes):
    keras_model = model.model if hasattr(model, 'model') else model
    
    all_preds = []
    all_labels = []
    
    for batch_X, batch_y in aux_dataset:
        preds = keras_model(batch_X, training=False)
        pred_classes = tf.argmax(preds, axis=1).numpy()
        true_classes = tf.argmax(batch_y, axis=1).numpy()
        
        all_preds.extend(pred_classes)
        all_labels.extend(true_classes)
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    M_class = np.zeros(num_classes, dtype=np.float32)
    for k in range(num_classes):
        total_k = cm[:, k].sum()
        if total_k > 0:
            A_k_k = cm[k, k] / cm[k, :].sum() if cm[k, :].sum() > 0 else 0
            confusion_rates = []
            for j in range(num_classes):
                if j != k and cm[j, :].sum() > 0:
                    A_j_k = cm[j, k] / cm[j, :].sum()
                    confusion_rates.append(A_j_k)
            
            max_confusion = max(confusion_rates) if confusion_rates else 0
            M_class[k] = A_k_k * (1 - max_confusion)
    
    return M_class, cm

def train_with_ssd_loss(model, private_dataset, global_model, M_class, m_max, num_classes, 
                        epochs, is_gru=False):
    keras_model = model.model if hasattr(model, 'model') else model
    global_keras = global_model.model if hasattr(global_model, 'model') else global_model
    
    optimizer = keras_model.optimizer
    ce_loss_fn = keras_model.loss
    
    logits_layer = keras_model.get_layer('logits')
    logits_model = tf.keras.Model(inputs=keras_model.input, outputs=[logits_layer.output, keras_model.output])
    
    global_logits_layer = global_keras.get_layer('logits')
    global_logits_model = tf.keras.Model(inputs=global_keras.input, outputs=global_logits_layer.output)
    
    M_class_expanded = tf.constant(tf.expand_dims(M_class, axis=0), dtype=tf.float32)
    
    @tf.function
    def train_step(batch_X, batch_y):
        with tf.GradientTape() as tape:
            local_logits, predictions = logits_model(batch_X, training=True)
            ce_loss = ce_loss_fn(batch_y, predictions)
            
            global_logits = global_logits_model(batch_X, training=False)
            
            global_probs = tf.nn.softmax(global_logits)
            true_labels = tf.cast(tf.argmax(batch_y, axis=1), tf.int32)
            batch_size = tf.shape(batch_y)[0]
            
            indices = tf.stack([tf.range(batch_size), true_labels], axis=1)
            p_g_k2 = tf.gather_nd(global_probs, indices)
            M_sample_expanded = tf.expand_dims(1.0 - tf.sqrt(1.0 - p_g_k2), axis=1)
            
            M = tf.nn.relu(m_max * M_class_expanded * M_sample_expanded - 0.1)
            
            logit_diff = M * (global_logits - local_logits)
            ssd_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logit_diff), axis=1))
            
            total_loss = ce_loss + ssd_loss
        
        gradients = tape.gradient(total_loss, keras_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
        
        return ce_loss, ssd_loss, total_loss
    
    print(f"  Training with SSD loss (m_max={m_max:.2f}) for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_losses = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        num_batches = 0
        
        for batch_X, batch_y in private_dataset:
            ce_loss, ssd_loss, total_loss = train_step(batch_X, batch_y)
            epoch_losses = epoch_losses + tf.stack([ce_loss, ssd_loss, total_loss])
            num_batches += 1
        
        if num_batches > 0:
            avg_losses = epoch_losses / num_batches
            print(f"    Epoch {epoch+1}/{epochs} - CE: {avg_losses[0]:.4f}, "
                  f"SSD: {avg_losses[1]:.4f}, Total: {avg_losses[2]:.4f}")

def evaluate_model(model, test_dataset, num_classes):
    keras_model = model.model if hasattr(model, 'model') else model
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch_X, batch_y in test_dataset:
        preds = keras_model(batch_X, training=False)
        pred_classes = tf.argmax(preds, axis=1).numpy()
        true_classes = tf.argmax(batch_y, axis=1).numpy()
        
        loss = keras_model.loss(batch_y, preds).numpy()
        total_loss += loss
        num_batches += 1
        
        all_preds.extend(pred_classes)
        all_labels.extend(true_classes)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    
    return avg_loss, accuracy, f1, precision, recall, (f1_per_class, precision_per_class, recall_per_class)

def main():
    parser = argparse.ArgumentParser(description="FedSSD: Federated Selective Soft Distillation")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--partition_type", type=str, default="label_skew-10")
    parser.add_argument("--model_type", type=str, default="dense")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--m_max", type=float, default=1.0, help="Maximum value for mask M")
    
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    
    partition_type, client_count = parse_partition_type(args.partition_type)
    n_clients = min(args.n_clients, client_count)
    
    logger, log_filename, detailed_logger = setup_logger(n_clients, partition_type)
    excel_filename = log_filename.replace('.log', '.xlsx')
    
    sample_X = np.load(os.path.join("data", "X_test.npy"), mmap_mode='r')
    y_test = np.load(os.path.join("data", "y_test.npy"), mmap_mode='r')
    input_dim = sample_X.shape[1]
    num_classes = len(np.unique(y_test))
    del sample_X, y_test
    
    paths_list = [setup_paths(str(i), partition_type, client_count) for i in range(n_clients)]
    
    is_gru = args.model_type.lower() == "gru"
    
    pipeline_start = time.time()
    log_timestamp(logger, "FedSSD PIPELINE STARTED")
    log_timestamp(logger, f"Clients: {n_clients}, Rounds: {args.rounds}, Epochs: {args.epochs}")
    log_timestamp(logger, f"m_max: {args.m_max}")
    
    aux_dataset, public_len = load_public_dataset_from_clients(paths_list, input_dim, num_classes, args.batch_size, is_gru)
    test_dataset = load_test_dataset(input_dim, num_classes, args.batch_size, is_gru)
    
    log_timestamp(logger, f"Auxiliary dataset (from client public splits): {public_len} samples")
    
    print(f"\n{bcolors.HEADER}ROUND 0: WARMUP{bcolors.ENDC}")
    log_timestamp(logger, "Round 0: Warmup")
    
    client_weights = []
    sample_sizes = []
    
    for i in range(n_clients):
        print(f"\n{bcolors.BOLD}Client {i}{bcolors.ENDC}")
        
        model = create_model(input_dim, num_classes, args.model_type, args.batch_size)
        train_dataset = create_dataset(
            paths_list[i]['train_X'],
            paths_list[i]['train_y'],
            input_dim, num_classes, args.batch_size, is_gru
        )
        
        model.fit(train_dataset, epochs=args.epochs, verbose=1)

        y_mmap = np.load(paths_list[i]['train_y'], mmap_mode='r')
        sample_sizes.append(y_mmap.shape[0])
        client_weights.append(model.get_weights())
        
        del train_dataset, model
        gc.collect()
    
    log_timestamp(logger, "All clients trained (warmup)")
    
    strategy = FedAvg()
    aggregated_weights = strategy.aggregate(client_weights, sample_sizes)
    
    global_model = create_model(input_dim, num_classes, args.model_type, args.batch_size)
    global_model.set_weights(aggregated_weights)
    
    loss, acc, f1, prec, rec, _ = evaluate_model(global_model, test_dataset, num_classes)
    logger.info(f"Round 0 - Global: Acc={acc:.4f}, Loss={loss:.4f}, F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
    print(f"{bcolors.OKGREEN}Round 0 - Global: Acc={acc:.4f}, F1={f1:.4f}{bcolors.ENDC}")
    
    results_data = [{
        'Round': 0,
        'Accuracy': acc,
        'Loss': loss,
        'F1_Score': f1,
        'Precision': prec,
        'Recall': rec
    }]
    
    for round_num in range(1, args.rounds + 1):
        round_start = time.time()
        
        print(f"\n{bcolors.HEADER}ROUND {round_num}/{args.rounds}{bcolors.ENDC}")
        log_timestamp(logger, f"Round {round_num} started")
        
        print(f"\n{bcolors.PURPLE}[SERVER] Computing class metrics on auxiliary dataset{bcolors.ENDC}")
        M_class, cm = compute_class_metrics(global_model, aux_dataset, num_classes)
        print(f"  M_class: min={M_class.min():.4f}, max={M_class.max():.4f}, mean={M_class.mean():.4f}")
        
        print(f"\n{bcolors.PURPLE}[CLIENTS] Local training with SSD loss{bcolors.ENDC}")
        
        round_personalized_metrics = []
        new_client_weights = []
        
        for i in range(n_clients):
            print(f"\n{bcolors.BOLD}Client {i}{bcolors.ENDC}")
            
            model = create_model(input_dim, num_classes, args.model_type, args.batch_size)
            model.set_weights(global_model.get_weights())
            
            train_dataset = create_dataset(
                paths_list[i]['train_X'],
                paths_list[i]['train_y'],
                input_dim, num_classes, args.batch_size, is_gru
            )
            
            train_with_ssd_loss(
                model, train_dataset, global_model,
                M_class, args.m_max, num_classes, args.epochs, is_gru
            )
            
            new_client_weights.append(model.get_weights())
            del train_dataset, model
            gc.collect()
        
        log_timestamp(logger, "All clients completed local training")
        
        client_weights = new_client_weights
        aggregated_weights = strategy.aggregate(client_weights, sample_sizes)
        global_model.set_weights(aggregated_weights)
        
        loss, acc, f1, prec, rec, per_class = evaluate_model(global_model, test_dataset, num_classes)
        logger.info(f"Round {round_num} - Global: Acc={acc:.4f}, Loss={loss:.4f}, F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
        print(f"{bcolors.OKGREEN}Round {round_num} - Global: Acc={acc:.4f}, F1={f1:.4f}{bcolors.ENDC}")
        
        results_data.append({
            'Round': round_num,
            'Accuracy': acc,
            'Loss': loss,
            'F1_Score': f1,
            'Precision': prec,
            'Recall': rec
        })
        
        round_time = time.time() - round_start
        log_timestamp(logger, f"Round {round_num} completed in {round_time:.2f}s")
    
    results_df = pd.DataFrame(results_data)
    results_df.to_excel(excel_filename, index=False, sheet_name='Overall_Metrics')
    
    pipeline_time = time.time() - pipeline_start
    log_timestamp(logger, f"Pipeline completed in {pipeline_time:.2f}s ({pipeline_time/60:.2f}m)")
    
    print(f"\n{bcolors.OKGREEN}Results saved to {excel_filename}{bcolors.ENDC}")

if __name__ == "__main__":
    main()
