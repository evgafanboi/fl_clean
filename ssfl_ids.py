#!/usr/bin/env python3

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
from models import dense, dense_discri
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

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
    log_filename = f"results/SSFL_IDS_{n_clients}client_{partition_type}.log"
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
    
    paths = {
        'train_X': os.path.join(partition_dir, f"client_{client_id}_X_train.npy"),
        'train_y': os.path.join(partition_dir, f"client_{client_id}_y_train.npy"),
        'test_X': os.path.join("data", "X_test.npy"),
        'test_y': os.path.join("data", "y_test.npy")
    }
    
    return paths

def create_private_dataset(X_path, y_path, input_dim, num_classes, batch_size):
    def generator():
        X_mmap = np.load(X_path, mmap_mode='r')
        y_mmap = np.load(y_path, mmap_mode='r')
        total_samples = X_mmap.shape[0]
        chunk_size = 100000
        
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
            y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.int64)
            
            if len(y_chunk.shape) == 1:
                y_chunk = tf.keras.utils.to_categorical(y_chunk, num_classes)
            
            yield X_chunk, y_chunk
            del X_chunk, y_chunk
    
    output_signature = (
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)

_public_dataset_cache = None

def load_public_dataset_from_clients(paths_list):
    """Load and concatenate public splits from all clients"""
    global _public_dataset_cache
    
    if _public_dataset_cache is None:
        X_public_list = []
        
        for paths in paths_list:
            if os.path.exists(paths.get('public_X', '')):
                X_pub = np.load(paths['public_X'])
                X_public_list.append(X_pub)
        
        if not X_public_list:
            raise ValueError("No public splits found in client partitions")
        
        _public_dataset_cache = np.concatenate(X_public_list, axis=0).astype(np.float32)
        print(f"{bcolors.OKGREEN}Public dataset cached from client splits ({_public_dataset_cache.shape[0]} samples){bcolors.ENDC}")
    
    X_pub = _public_dataset_cache
    indices = np.random.permutation(len(X_pub))
    X_pub = X_pub[indices]
    
    return X_pub

def load_public_dataset(batch_size):
    global _public_dataset_cache
    
    if _public_dataset_cache is None:
        with open("data/X_public.pkl", "rb") as f:
            X_pub_raw = pickle.load(f)
        _public_dataset_cache = np.asarray(X_pub_raw, dtype=np.float32)
        print(f"{bcolors.OKGREEN}Public dataset cached ({_public_dataset_cache.shape[0]} samples){bcolors.ENDC}")
    
    X_pub = _public_dataset_cache
    indices = np.random.permutation(len(X_pub))
    X_pub = X_pub[indices]
    
    return X_pub

def load_test_dataset(batch_size, num_classes):
    X_test = np.load("data/X_test.npy", mmap_mode='r')
    y_test = np.load("data/y_test.npy", mmap_mode='r')
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    if len(y_test.shape) == 1:
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def predict_with_discriminator(classify_model, discri_model, X_open, num_classes, batch_size=4096):
    classify_logits_model = classify_model.get_logits_model()
    
    all_logits = []
    for start_idx in range(0, len(X_open), batch_size):
        end_idx = min(start_idx + batch_size, len(X_open))
        X_batch = X_open[start_idx:end_idx]
        
        logits_batch = classify_logits_model(X_batch, training=False).numpy()
        logits_batch = tf.nn.softmax(logits_batch).numpy()
        
        dis_pred = discri_model.model(X_batch, training=False).numpy()
        
        for i in range(len(dis_pred)):
            if dis_pred[i] > 0.5:
                logits_batch[i] = np.ones(num_classes, dtype=np.float32) / num_classes
        
        all_logits.append(logits_batch)
        del logits_batch, dis_pred, X_batch
    
    result = np.vstack(all_logits)
    del all_logits
    return result

def hard_label_from_soft(soft_label, num_classes):
    boundary = 1.0 / num_classes
    hard_labels = []
    for i in range(len(soft_label)):
        max_prob = np.max(soft_label[i])
        if max_prob > boundary:
            hard_labels.append(np.argmax(soft_label[i]))
        else:
            hard_labels.append(num_classes)
    return hard_labels

def hard_label_vote(all_client_hard_labels, num_classes):
    client_cnt = len(all_client_hard_labels)
    sample_cnt = len(all_client_hard_labels[0])
    
    pred_labels = []
    for i in range(sample_cnt):
        label_votes = [0] * num_classes
        for j in range(client_cnt):
            pred_label = all_client_hard_labels[j][i]
            if pred_label != num_classes:
                label_votes[pred_label] += 1
        
        max_vote_idx = label_votes.index(max(label_votes))
        pred_labels.append(max_vote_idx)
    
    return pred_labels

def distill_knowledge(model_wrapper, target_logits, X_data, epochs, batch_size):
    keras_model = model_wrapper.model
    logits_model = model_wrapper.get_logits_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    
    @tf.function
    def distillation_step(X_batch, target_batch):
        with tf.GradientTape() as tape:
            student_logits = logits_model(X_batch, training=True)
            loss = mae_loss(target_batch, student_logits)
        gradients = tape.gradient(loss, keras_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
        return loss
    
    dataset = tf.data.Dataset.from_tensor_slices((X_data, target_logits)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    for epoch in range(epochs):
        for X_batch, target_batch in dataset:
            distillation_step(X_batch, target_batch)

def train_discriminator(client, open_feature, theta, dis_rounds, batch_size, num_classes):
    classify_logits_model = client['classify_model'].get_logits_model()
    
    # Batch prediction to avoid OOM with large open_feature
    all_logits = []
    prediction_batch_size = 10000
    for start_idx in range(0, len(open_feature), prediction_batch_size):
        end_idx = min(start_idx + prediction_batch_size, len(open_feature))
        batch_logits = classify_logits_model(open_feature[start_idx:end_idx], training=False)
        batch_logits = tf.nn.softmax(batch_logits).numpy()
        all_logits.append(batch_logits)
    
    dis_logits = np.vstack(all_logits)
    del all_logits
    
    max_probs = np.max(dis_logits, axis=1)
    if theta < 0:
        theta = np.median(max_probs)
    
    sure_unknown_mask = max_probs < theta
    sure_unknown_feature = open_feature[sure_unknown_mask]
    
    if len(sure_unknown_feature) == 0:
        return False
    
    private_X_path = client['paths']['train_X']
    X_mmap = np.load(private_X_path, mmap_mode='r')
    sure_known_feature = np.array(X_mmap[:len(sure_unknown_feature)], dtype=np.float32)
    del X_mmap
    
    dis_X = np.vstack([sure_known_feature, sure_unknown_feature])
    dis_y = np.concatenate([np.zeros(len(sure_known_feature)), np.ones(len(sure_unknown_feature))])
    
    indices = np.random.permutation(len(dis_X))
    dis_X = dis_X[indices]
    dis_y = dis_y[indices]
    
    dis_dataset = tf.data.Dataset.from_tensor_slices((dis_X, dis_y))
    dis_dataset = dis_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    for r in range(dis_rounds):
        client['discri_model'].fit(dis_dataset, epochs=1, verbose=0)
    
    return True

def aggressive_memory_cleanup():
    gc.collect()
    tf.keras.backend.clear_session()
    if gpus:
        try:
            tf.config.experimental.reset_memory_stats(gpus[0])
        except:
            pass
    time.sleep(0.5)

def main():
    parser = argparse.ArgumentParser(description="SSFL-IDS")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--partition_type", type=str, default="label_skew-10", help="Partition type")
    parser.add_argument("--rounds", type=int, default=10, help="Communication rounds")
    parser.add_argument("--train_rounds", type=int, default=3, help="Training rounds per communication round")
    parser.add_argument("--dis_rounds", type=int, default=3, help="Discriminator training rounds")
    parser.add_argument("--dist_rounds", type=int, default=2, help="Distillation rounds")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size")
    parser.add_argument("--theta", type=float, default=-1, help="Confidence threshold (auto if -1)")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    
    partition_type, client_count = parse_partition_type(args.partition_type)
    n_clients = min(args.n_clients, client_count)
    
    logger, log_filename, detailed_logger = setup_logger(n_clients, partition_type)
    excel_filename = log_filename.replace('.log', '.xlsx')
    
    partition_dir = os.path.join("data", "partitions", f"{client_count}_client", partition_type)
    
    y_test = np.load(os.path.join("data", "y_test.npy"), mmap_mode='r')
    num_classes = len(np.unique(y_test))
    del y_test
    
    sample_X = np.load(os.path.join("data", "X_test.npy"), mmap_mode='r')
    input_dim = int(sample_X.shape[1])
    del sample_X
    
    paths_list = []
    for i in range(n_clients):
        paths = setup_paths(str(i), partition_type, client_count)
        paths_list.append(paths)
    
    # Load full public dataset (unlabeled)
    X_public_full = load_public_dataset(args.batch_size)
    print(f"{bcolors.OKGREEN}Using full public dataset ({X_public_full.shape[0]} samples){bcolors.ENDC}")
    
    test_dataset = load_test_dataset(args.batch_size, num_classes)
    
    pipeline_start = time.time()
    log_timestamp(logger, "SIMULATION STARTED")
    log_timestamp(logger, f"Clients: {n_clients}, Rounds: {args.rounds}, Partition: {partition_type}")
    
    clients = []
    for i in range(n_clients):
        classify_model = dense.create_enhanced_dense_model(input_dim, num_classes, args.batch_size)
        discri_model = dense_discri.create_discriminator(input_dim)
        
        private_dataset = create_private_dataset(
            paths_list[i]['train_X'],
            paths_list[i]['train_y'],
            input_dim,
            num_classes,
            args.batch_size
        )
        
        y_mmap = np.load(paths_list[i]['train_y'], mmap_mode='r')
        class_counts = np.bincount(y_mmap, minlength=num_classes)
        del y_mmap
        
        clients.append({
            'classify_model': classify_model,
            'discri_model': discri_model,
            'private_dataset': private_dataset,
            'class_counts': class_counts,
            'paths': paths_list[i]
        })
    
    server_model = dense.create_enhanced_dense_model(input_dim, num_classes, args.batch_size)
    
    global_results = []
    
    for round_num in range(1, args.rounds + 1):
        round_start = time.time()
        
        print(f"\n{bcolors.HEADER}Round {round_num}/{args.rounds} - Stage I{bcolors.ENDC}")
        log_timestamp(logger, f"Round {round_num} Stage I started")
        
        # Use full public dataset for this round
        open_feature = X_public_full
        
        all_client_hard_labels = []
        sure_unknown_none = set()
        
        for i in range(n_clients):
            print(f"\n{bcolors.BOLD}Client {i} Training...{bcolors.ENDC}")
            
            client = clients[i]
            
            for _ in range(args.train_rounds):
                client['classify_model'].fit(client['private_dataset'], epochs=1, verbose=0)
            
            if np.sum(client['class_counts'] > 0) <= 1:
                continue
            
            succ = train_discriminator(client, open_feature, args.theta, args.dis_rounds, args.batch_size, num_classes)
            
            if not succ:
                sure_unknown_none.add(i)
            
            if i not in sure_unknown_none:
                local_logits = predict_with_discriminator(
                    client['classify_model'],
                    client['discri_model'],
                    open_feature,
                    num_classes,
                    args.batch_size
                )
                
                hard_labels = hard_label_from_soft(local_logits, num_classes)
                all_client_hard_labels.append(hard_labels)
        
        print(f"\n{bcolors.PURPLE}[VOTING] Aggregating predictions{bcolors.ENDC}")
        global_labels = hard_label_vote(all_client_hard_labels, num_classes)
        global_labels_onehot = tf.keras.utils.to_categorical(global_labels, num_classes)
        
        del all_client_hard_labels
        aggressive_memory_cleanup()
        
        print(f"\n{bcolors.HEADER}Round {round_num}/{args.rounds} - Stage II{bcolors.ENDC}")
        log_timestamp(logger, f"Round {round_num} Stage II started")
        
        for i in range(n_clients):
            print(f"Client {i} Distillation Training...")
            
            client = clients[i]
            distill_knowledge(client['classify_model'], global_labels_onehot, open_feature, args.dist_rounds, args.batch_size)
        
        print(f"\n{bcolors.PURPLE}Server Training...{bcolors.ENDC}")
        distill_knowledge(server_model, global_labels_onehot, open_feature, args.dist_rounds, args.batch_size)
        
        del global_labels_onehot, open_feature
        aggressive_memory_cleanup()
        
        print(f"\n{bcolors.PURPLE}[EVALUATION]{bcolors.ENDC}")
        
        test_results = server_model.evaluate(test_dataset, verbose=0)
        test_loss, accuracy = test_results[0], test_results[1]
        
        y_pred = []
        y_true = []
        for batch_X, batch_y in test_dataset:
            batch_pred = server_model.model(batch_X, training=False)
            y_pred.extend(np.argmax(batch_pred.numpy(), axis=1))
            y_true.extend(np.argmax(batch_y.numpy(), axis=1))
            del batch_pred
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        min_size = min(len(y_true), len(y_pred))
        y_true = y_true[:min_size]
        y_pred = y_pred[:min_size]
        
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        detailed_logger.info(f"Round {round_num} - Per-Class Metrics")
        for cls in range(num_classes):
            support = np.sum(y_true == cls)
            detailed_logger.info(f"Class {cls}: F1={f1_per_class[cls]:.4f}, Precision={precision_per_class[cls]:.4f}, Recall={recall_per_class[cls]:.4f}, Support={support}")
        
        detailed_logger.info(f"\nRound {round_num} Confusion Matrix")
        cm_df = pd.DataFrame(cm)
        for line in cm_df.to_string().splitlines():
            detailed_logger.info(line)
        
        logger.info(f"Round {round_num} - Server Model - Acc: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"{bcolors.OKGREEN}Round {round_num}: Acc={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}{bcolors.ENDC}")
        
        global_results.append({
            'Round': round_num,
            'Loss': test_loss,
            'Accuracy': accuracy,
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        })
        
        aggressive_memory_cleanup()
        
        round_time = time.time() - round_start
        log_timestamp(logger, f"Round {round_num} completed in {round_time:.2f}s")
    
    results_df = pd.DataFrame(global_results)
    results_df.to_excel(excel_filename, index=False)
    
    total_time = time.time() - pipeline_start
    log_timestamp(logger, f"Pipeline completed in {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"\n{bcolors.OKGREEN}Results saved to {excel_filename}{bcolors.ENDC}")

if __name__ == "__main__":
    main()
