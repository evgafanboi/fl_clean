import os
import argparse
import numpy as np
import tensorflow as tf
import gc
import logging
import pandas as pd
import time
from datetime import datetime
from models import dense
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

def setup_logger(n_clients, partition_type, gamma=1.0):
    log_filename = f"results/FedProto_{n_clients}client_{partition_type}_gamma{gamma}.log"
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

def aggressive_memory_cleanup():
    gc.collect()
    time.sleep(0.5)

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

def create_model(input_dim, num_classes, batch_size):
    return dense.create_enhanced_dense_model(input_dim, num_classes, batch_size)

def create_client_dataset(X_path, y_path, input_dim, num_classes, batch_size):
    def generator():
        X_mmap = np.load(X_path, mmap_mode='r')
        y_mmap = np.load(y_path, mmap_mode='r')
        total_samples = X_mmap.shape[0]
        chunk_size = 100000
        
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
            y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.float32)
            
            if num_classes and (len(y_chunk.shape) == 1 or y_chunk.shape[1] == 1):
                y_chunk = tf.keras.utils.to_categorical(y_chunk, num_classes)
            
            yield (X_chunk, y_chunk)
            del X_chunk, y_chunk
    
    output_signature = (
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_test_dataset(batch_size, num_classes):
    X_test = np.load("data/X_test.npy", mmap_mode='r')
    y_test = np.load("data/y_test.npy", mmap_mode='r')
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    if len(y_test.shape) == 1:
        y_test = tf.keras.utils.to_categorical(y_test.astype(np.int32), num_classes).astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def extract_features(model, X_path, y_path, num_classes):
    keras_model = model.model if hasattr(model, 'model') else model
    
    feature_extractor = tf.keras.Model(
        inputs=keras_model.input,
        outputs=keras_model.layers[-2].output
    )
    
    X_mmap = np.load(X_path, mmap_mode='r')
    y_mmap = np.load(y_path, mmap_mode='r')
    total_samples = X_mmap.shape[0]
    
    feature_dim = feature_extractor.output.shape[-1]
    class_features_sum = {c: np.zeros(feature_dim, dtype=np.float32) for c in range(num_classes)}
    class_counts = {c: 0 for c in range(num_classes)}
    
    chunk_size = 10000
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
        y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.int32)
        
        features = feature_extractor(X_chunk, training=False).numpy()
        
        for i, label in enumerate(y_chunk):
            class_features_sum[label] += features[i]
            class_counts[label] += 1
        
        del X_chunk, y_chunk, features
    
    prototypes = {}
    supports = {}
    
    for class_id in range(num_classes):
        if class_counts[class_id] > 0:
            prototypes[class_id] = class_features_sum[class_id] / class_counts[class_id]
            supports[class_id] = class_counts[class_id]
        else:
            prototypes[class_id] = None
            supports[class_id] = 0
    
    return prototypes, supports

def local_training_with_prototypes(model, private_dataset, global_prototypes, num_classes, 
                                   epochs, gamma, batch_size):
    keras_model = model.model if hasattr(model, 'model') else model
    
    print(f"  [LOCAL TRAINING] Training with prototype alignment for {epochs} epochs...")
    print(f"    Global prototypes available for {len(global_prototypes)}/{num_classes} classes")
    
    if len(global_prototypes) == 0:
        print(f"    No global prototypes - using standard training")
        history = model.fit(private_dataset, epochs=epochs, verbose=0)
        if 'loss' in history.history:
            final_loss = history.history['loss'][-1]
            print(f"    Loss: {final_loss:.4f}")
        return model
    
    feature_extractor = tf.keras.Model(
        inputs=keras_model.input,
        outputs=keras_model.layers[-2].output
    )
    
    feature_dim = feature_extractor.output.shape[-1]
    
    global_proto_array = np.zeros((num_classes, feature_dim), dtype=np.float32)
    has_global_proto = np.zeros(num_classes, dtype=bool)
    
    for class_id, proto in global_prototypes.items():
        if proto is not None:
            global_proto_array[class_id] = proto
            has_global_proto[class_id] = True
    
    global_proto_tensor = tf.constant(global_proto_array, dtype=tf.float32)
    has_global_proto_tensor = tf.constant(has_global_proto, dtype=tf.bool)
    
    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    
    if hasattr(keras_model, 'optimizer') and keras_model.optimizer is not None:
        optimizer = keras_model.optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        keras_model.optimizer = optimizer
    
    @tf.function
    def train_step(X_batch, y_batch):
        with tf.GradientTape() as tape:
            features = feature_extractor(X_batch, training=True)
            predictions = keras_model(X_batch, training=True)
            
            ce = ce_loss_fn(y_batch, predictions)
            
            y_labels = tf.argmax(y_batch, axis=1)
            batch_has_global = tf.gather(has_global_proto_tensor, y_labels)
            
            if tf.reduce_any(batch_has_global):
                valid_features = tf.boolean_mask(features, batch_has_global)
                valid_labels = tf.boolean_mask(y_labels, batch_has_global)
                
                batch_protos = tf.gather(global_proto_tensor, valid_labels)
                
                # Use L2 distance (not squared) to prevent explosion in high dimensions
                distances = tf.sqrt(tf.reduce_sum(tf.square(valid_features - batch_protos), axis=1) + 1e-8)
                
                proto_loss = gamma * tf.reduce_mean(distances)
            else:
                proto_loss = 0.0
            
            loss = ce + proto_loss
        
        gradients = tape.gradient(loss, keras_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
        return loss, ce, proto_loss
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_proto_loss = 0
        batches = 0
        
        for X_batch, y_batch in private_dataset:
            loss, ce, proto_loss = train_step(X_batch, y_batch)
            
            epoch_loss += loss.numpy()
            epoch_ce_loss += ce.numpy()
            epoch_proto_loss += proto_loss.numpy() if isinstance(proto_loss, tf.Tensor) else proto_loss
            batches += 1
        
        avg_loss = epoch_loss / batches if batches > 0 else 0
        avg_ce = epoch_ce_loss / batches if batches > 0 else 0
        avg_proto = epoch_proto_loss / batches if batches > 0 else 0
        epoch_time = time.time() - epoch_start
        
        print(f"    Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Proto: {avg_proto:.4f})")
    
    print(f"    Training completed")
    return model

def aggregate_prototypes(all_client_prototypes, num_classes):
    print(f"\n{bcolors.OKCYAN}[SERVER AGGREGATION] Weighted prototype aggregation (by dataset size){bcolors.ENDC}")
    
    global_prototypes = {}
    
    for class_id in range(num_classes):
        prototypes_weighted = []
        total_support = 0
        contributing_clients = []
        
        for client_id, protos in all_client_prototypes.items():
            if class_id in protos and protos[class_id]['prototype'] is not None:
                proto = protos[class_id]['prototype']
                support = protos[class_id]['support']
                
                prototypes_weighted.append(proto * support)
                total_support += support
                contributing_clients.append(client_id)
        
        if prototypes_weighted and total_support > 0:
            global_prototypes[class_id] = np.sum(prototypes_weighted, axis=0) / total_support
            print(f"  Class {class_id}: {len(contributing_clients)} clients contributed (total samples: {total_support})")
        else:
            global_prototypes[class_id] = None
            print(f"  Class {class_id}: No contributions")
    
    return global_prototypes

def main():
    parser = argparse.ArgumentParser(description="FedProto - Prototype-based Federated Learning")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--partition_type", type=str, default="label_skew-10", help="Partition type")
    parser.add_argument("--rounds", type=int, default=10, help="Communication rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size")
    parser.add_argument("--gamma", type=float, default=1.0, help="Prototype alignment loss weight")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    
    partition_type, client_count = parse_partition_type(args.partition_type)
    n_clients = min(args.n_clients, client_count)
    
    logger, log_filename, detailed_logger = setup_logger(n_clients, partition_type, args.gamma)
    excel_filename = log_filename.replace('.log', '.xlsx')
    
    results_dict = {f'Client_{i}': {} for i in range(n_clients)}
    
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
    
    # Load test dataset
    test_dataset = load_test_dataset(args.batch_size, num_classes)
    print(f"{bcolors.OKGREEN}Using FULL test set{bcolors.ENDC}")
    
    pipeline_start = time.time()
    log_timestamp(logger, "SIMULATION STARTED")
    log_timestamp(logger, f"Clients: {n_clients}, Rounds: {args.rounds}, Partition: {partition_type}")
    log_timestamp(logger, f"Gamma: {args.gamma}")
    
    client_models = []
    for i in range(n_clients):
        model = create_model(input_dim, num_classes, args.batch_size)
        client_models.append(model)
    
    for round_num in range(1, args.rounds + 1):
        round_start = time.time()
        
        logger.info(f"Round {round_num}/{args.rounds}")
        print(f"\n{bcolors.HEADER}Round {round_num}/{args.rounds}{bcolors.ENDC}")
        log_timestamp(logger, f"Round {round_num} started")
        
        print(f"\n{bcolors.OKCYAN}[STEP 1/4] Local training{bcolors.ENDC}")
        
        for i in range(n_clients):
            print(f"\n{bcolors.BOLD}Client {i}{bcolors.ENDC}")
            paths = paths_list[i]
            
            train_dataset = create_client_dataset(
                paths['train_X'], paths['train_y'],
                input_dim, num_classes, args.batch_size
            )
            
            history = client_models[i].fit(train_dataset, epochs=args.epochs, verbose=0)
            
            final_loss = history.history['loss'][-1] if 'loss' in history.history else 0
            print(f"  Training loss: {final_loss:.4f}")
            
            del train_dataset
            aggressive_memory_cleanup()
        
        print(f"\n{bcolors.OKCYAN}[STEP 2/4] Computing prototypes{bcolors.ENDC}")
        
        all_client_prototypes = {}
        
        for i in range(n_clients):
            print(f"\n{bcolors.BOLD}Client {i}{bcolors.ENDC}")
            paths = paths_list[i]
            
            prototypes, supports = extract_features(
                client_models[i], paths['train_X'], paths['train_y'], num_classes
            )
            
            client_proto_dict = {}
            sent_classes = []
            
            for class_id in range(num_classes):
                if prototypes[class_id] is not None:
                    client_proto_dict[class_id] = {
                        'prototype': prototypes[class_id],
                        'support': supports[class_id]
                    }
                    sent_classes.append(class_id)
            
            all_client_prototypes[i] = client_proto_dict
            
            print(f"  Sent {len(sent_classes)} class prototypes")
            if sent_classes:
                print(f"  Classes: {sent_classes[:10]}{'...' if len(sent_classes) > 10 else ''}")
            
            aggressive_memory_cleanup()
        
        print(f"\n{bcolors.OKCYAN}[STEP 3/4] Server aggregation{bcolors.ENDC}")
        
        global_prototypes = aggregate_prototypes(
            all_client_prototypes, num_classes
        )
        
        print(f"\n{bcolors.OKCYAN}[STEP 4/4] Local training with global prototypes{bcolors.ENDC}")
        
        for i in range(n_clients):
            print(f"\n{bcolors.BOLD}Client {i}{bcolors.ENDC}")
            paths = paths_list[i]
            
            train_dataset = create_client_dataset(
                paths['train_X'], paths['train_y'],
                input_dim, num_classes, args.batch_size
            )
            
            client_models[i] = local_training_with_prototypes(
                client_models[i],
                train_dataset,
                global_prototypes,
                num_classes,
                args.epochs,
                args.gamma,
                args.batch_size
            )
            
            del train_dataset
            aggressive_memory_cleanup()
        
        print(f"\n{bcolors.PURPLE}[EVALUATION] Testing all clients on global test set{bcolors.ENDC}")
        
        for i in range(n_clients):
            preds = client_models[i].predict(test_dataset, verbose=0)
            y_pred = np.argmax(preds, axis=1)
            del preds
            gc.collect()
            
            y_true = []
            for batch_x, batch_y in test_dataset:
                if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                    y_true.extend(np.argmax(batch_y.numpy(), axis=1))
                else:
                    y_true.extend(batch_y.numpy())
            y_true = np.array(y_true)
            
            min_size = min(len(y_true), len(y_pred))
            y_true = y_true[:min_size]
            y_pred = y_pred[:min_size]
            
            accuracy = np.mean(y_true == y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            logger.info(f"Round {round_num} | Client {i} | Acc: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            print(f"{bcolors.OKGREEN}Client {i}: Acc={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}{bcolors.ENDC}")
            
            results_dict[f'Client_{i}'][f'Round_{round_num}_Acc'] = accuracy
            results_dict[f'Client_{i}'][f'Round_{round_num}_F1'] = f1
            results_dict[f'Client_{i}'][f'Round_{round_num}_Precision'] = precision
            results_dict[f'Client_{i}'][f'Round_{round_num}_Recall'] = recall
            
            del y_pred, y_true
        
        round_time = time.time() - round_start
        log_timestamp(logger, f"Round {round_num} completed in {round_time:.2f}s")
        
        results_df = pd.DataFrame(results_dict).T
        results_df.to_excel(excel_filename)
    
    total_time = time.time() - pipeline_start
    log_timestamp(logger, f"Pipeline completed in {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"\n{bcolors.OKGREEN}Results saved to {excel_filename}{bcolors.ENDC}")

if __name__ == "__main__":
    main()
