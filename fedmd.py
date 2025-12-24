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
from models import dense
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Control GPU memory usage
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
    log_filename = f"results/FedMD_{n_clients}client_{partition_type}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(levelname)s: %(message)s', filemode='w')
    print(f"{bcolors.OKCYAN}Logging to {log_filename}{bcolors.ENDC}")
    
    # Setup class metrics logger
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
        'public_X': os.path.join(partition_dir, f"client_{client_id}_X_public.npy"),
        'public_y': os.path.join(partition_dir, f"client_{client_id}_y_public.npy"),
        'test_X': os.path.join("data", "X_test.npy"),
        'test_y': os.path.join("data", "y_test.npy")
    }
    
    return paths

def create_model(input_dim, num_classes, batch_size):
    return dense.create_enhanced_dense_model(input_dim, num_classes, batch_size)

# Cache public dataset to avoid repeated pickle loading
_public_dataset_cache = None
_public_labels_cache = None
def load_public_dataset_from_clients(paths_list, batch_size, num_classes, shuffle=True, return_labels=False):
    """Load and concatenate public splits from all clients"""
    global _public_dataset_cache, _public_labels_cache
    
    if _public_dataset_cache is None:
        X_public_list = []
        y_public_list = []
        
        for paths in paths_list:
            if os.path.exists(paths.get('public_X', '')):
                X_pub = np.load(paths['public_X'])
                y_pub = np.load(paths['public_y'])
                X_public_list.append(X_pub)
                y_public_list.append(y_pub)
        
        if not X_public_list:
            raise ValueError("No public splits found in client partitions")
        
        _public_dataset_cache = np.concatenate(X_public_list, axis=0).astype(np.float32)
        _public_labels_cache = np.concatenate(y_public_list, axis=0).astype(np.float32)
        print(f"{bcolors.OKGREEN}Public dataset cached from client splits ({_public_dataset_cache.shape[0]} samples){bcolors.ENDC}")
    
    X_pub = _public_dataset_cache
    y_pub = _public_labels_cache
    
    if shuffle:
        indices = np.random.permutation(len(X_pub))
        X_pub = X_pub[indices]
        y_pub = y_pub[indices]
    
    if return_labels:
        if len(y_pub.shape) == 1:
            y_pub_cat = tf.keras.utils.to_categorical(y_pub.astype(np.int32), num_classes).astype(np.float32)
        else:
            y_pub_cat = y_pub.astype(np.float32)
        
        dataset = tf.data.Dataset.from_tensor_slices((X_pub, y_pub_cat))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(X_pub)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
def load_public_dataset(batch_size, num_classes, shuffle=True, return_labels=False):
    global _public_dataset_cache, _public_labels_cache
    
    # Load once and cache
    if _public_dataset_cache is None:
        import pickle
        with open("data/X_public.pkl", "rb") as f:
            X_pub_raw = pickle.load(f)
        with open("data/y_public.pkl", "rb") as f:
            y_pub_raw = pickle.load(f)
        # Ensure float32 dtype to match model expectations
        _public_dataset_cache = np.asarray(X_pub_raw, dtype=np.float32)
        _public_labels_cache = np.asarray(y_pub_raw, dtype=np.float32)
        print(f"{bcolors.OKGREEN}Public dataset cached in memory ({_public_dataset_cache.shape[0]} samples){bcolors.ENDC}")
    
    X_pub = _public_dataset_cache
    y_pub = _public_labels_cache
    
    if shuffle:
        indices = np.random.permutation(len(X_pub))
        X_pub = X_pub[indices]
        y_pub = y_pub[indices]
    
    if return_labels:
        # Convert labels to categorical for training
        if len(y_pub.shape) == 1:
            y_pub_cat = tf.keras.utils.to_categorical(y_pub.astype(np.int32), num_classes).astype(np.float32)
        else:
            y_pub_cat = y_pub.astype(np.float32)
        
        dataset = tf.data.Dataset.from_tensor_slices((X_pub, y_pub_cat))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        # For logits generation, no labels needed
        dataset = tf.data.Dataset.from_tensor_slices(X_pub)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Cache public test dataset
_public_test_dataset_cache = None

def load_public_test_dataset(batch_size):
    global _public_test_dataset_cache
    
    if _public_test_dataset_cache is None:
        import pickle
        with open("data/X_public_test.pkl", "rb") as f:
            X_pub_test_raw = pickle.load(f)
        with open("data/y_public_test.pkl", "rb") as f:
            y_pub_test_raw = pickle.load(f)
        
        # Ensure float32 dtype
        X_pub_test = np.asarray(X_pub_test_raw, dtype=np.float32)
        y_pub_test = np.asarray(y_pub_test_raw, dtype=np.float32)
        
        num_classes = len(np.unique(y_pub_test))
        if len(y_pub_test.shape) == 1:
            y_pub_test = tf.keras.utils.to_categorical(y_pub_test.astype(np.int32), num_classes).astype(np.float32)
        
        _public_test_dataset_cache = (X_pub_test, y_pub_test)
        print(f"{bcolors.OKGREEN}Public test dataset cached in memory ({X_pub_test.shape[0]} samples){bcolors.ENDC}")
    
    X_pub_test, y_pub_test = _public_test_dataset_cache
    dataset = tf.data.Dataset.from_tensor_slices((X_pub_test, y_pub_test))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_private_dataset(X_path, y_path, input_dim, num_classes, batch_size):
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

def generate_logits(model_wrapper, public_dataset):
    # Get logits
    logits_model = model_wrapper.get_logits_model()
    
    all_logits = []
    for batch in public_dataset:
        X_batch = batch[0] if isinstance(batch, tuple) else batch
        logits = logits_model(X_batch, training=False)
        all_logits.append(logits.numpy())
    return np.vstack(all_logits)

def transfer_learning_on_public(model_wrapper, public_dataset_labeled, epochs):
    """Transfer learning on public dataset with labels"""
    print(f"  Transfer learning on public dataset for {epochs} epochs...")
    
    history = model_wrapper.fit(
        public_dataset_labeled,
        epochs=epochs,
        verbose=0
    )
    
    if 'loss' in history.history:
        final_loss = history.history['loss'][-1]
        print(f"    Transfer learning loss: {final_loss:.4f}")
    
    return model_wrapper

def digest_phase(model_wrapper, consensus_logits, public_dataset, epochs=1, batch_size=8192):
    keras_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
    
    # Get logits
    student_logits_model = model_wrapper.get_logits_model()
    
    print(f"  [DIGEST] Matching consensus logits for {epochs} epochs...")
    
    # Prepare public dataset features
    public_X = []
    for batch in public_dataset:
        X_batch = batch[0] if isinstance(batch, tuple) else batch
        public_X.append(X_batch.numpy().astype(np.float32))
    public_X = np.vstack(public_X)
    
    # Manual training loop with MAE on logits
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    
    public_tf_dataset = tf.data.Dataset.from_tensor_slices(public_X)
    public_tf_dataset = public_tf_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    for epoch in range(epochs):
        epoch_loss = 0
        batches = 0
        
        batch_idx = 0
        for X_batch in public_tf_dataset:
            current_batch_size = X_batch.shape[0]
            consensus_batch = consensus_logits[batch_idx:batch_idx+current_batch_size]
            batch_idx += current_batch_size
            
            with tf.GradientTape() as tape:
                # Use logits_model
                student_logits = student_logits_model(X_batch, training=True)
                loss = mae_loss(consensus_batch, student_logits)
            
            gradients = tape.gradient(loss, keras_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            
            epoch_loss += loss.numpy()
            batches += 1
        
        avg_loss = epoch_loss / batches if batches > 0 else 0
    
    print(f"    DIGEST loss (MAE): {avg_loss:.4f}")
    return model_wrapper

def revisit_phase(model_wrapper, private_dataset, epochs):
    """REVISIT phase: Train on private data"""
    print(f"  [REVISIT] Training on private data for {epochs} epochs...")
    
    history = model_wrapper.fit(
        private_dataset,
        epochs=epochs,
        verbose=0
    )
    
    if 'loss' in history.history:
        final_loss = history.history['loss'][-1]
        print(f"    REVISIT loss: {final_loss:.4f}")
    
    return model_wrapper

def main():
    parser = argparse.ArgumentParser(description="FedMD - Centralized Knowledge Distillation (Paper Implementation)")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--partition_type", type=str, default="label_skew-10", help="Partition type (e.g., 'label_skew-10')")
    parser.add_argument("--rounds", type=int, default=10, help="Communication rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs for REVISIT phase")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    
    partition_type, client_count = parse_partition_type(args.partition_type)
    n_clients = min(args.n_clients, client_count)
    
    logger, log_filename, detailed_logger = setup_logger(n_clients, partition_type)
    excel_filename = log_filename.replace('.log', '.xlsx')
    
    results_dict = {f'Client_{i}': {} for i in range(n_clients)}
    
    partition_dir = os.path.join("data", "partitions", f"{client_count}_client", partition_type)
    
    y_test = np.load(os.path.join("data", "y_test.npy"), mmap_mode='r')
    num_classes = len(np.unique(y_test))
    del y_test
    
    # Determine input_dim
    sample_X = np.load(os.path.join("data", "X_test.npy"), mmap_mode='r')
    input_dim = int(sample_X.shape[1])
    del sample_X
    
    # Setup paths for all clients
    paths_list = []
    for i in range(n_clients):
        paths = setup_paths(str(i), partition_type, client_count)
        paths_list.append(paths)
    
    # Load public datasets from client public slices (paper approach)
    public_dataset = load_public_dataset_from_clients(paths_list, args.batch_size, num_classes, shuffle=False, return_labels=False)
    public_dataset_labeled = load_public_dataset_from_clients(paths_list, args.batch_size, num_classes, shuffle=True, return_labels=True)
    
    # Load full test dataset
    X_test = np.load("data/X_test.npy", mmap_mode='r')
    y_test = np.load("data/y_test.npy", mmap_mode='r')
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    if len(y_test.shape) == 1:
        y_test = tf.keras.utils.to_categorical(y_test.astype(np.int32), num_classes).astype(np.float32)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    print(f"{bcolors.OKGREEN}Using FULL test set ({X_test.shape[0]} samples){bcolors.ENDC}")
    
    pipeline_start = time.time()
    log_timestamp(logger, "FedMD PIPELINE STARTED (Paper Implementation)")
    log_timestamp(logger, f"Clients: {n_clients}, Rounds: {args.rounds}, Partition: {partition_type}")
    log_timestamp(logger, f"DIGEST: 1 epoch, REVISIT: {args.epochs} epochs")
    
    # Transfer Learning: Each party trains on public D0 then on private Dk
    print(f"\n{bcolors.HEADER}TRANSFER LEARNING{bcolors.ENDC}")
    log_timestamp(logger, "Transfer learning started")
    
    client_models = []
    for i in range(n_clients):
        print(f"\n{bcolors.BOLD}Client {i}{bcolors.ENDC}")
        model = create_model(input_dim, num_classes, args.batch_size)
        
        # Step 1: Train on public dataset
        print(f"  [1/2] Training on public dataset ({args.epochs} epochs)...")
        model = transfer_learning_on_public(model, public_dataset_labeled, args.epochs)
        
        # Step 2: Train on private dataset
        print(f"  [2/2] Training on private dataset ({args.epochs} epochs)...")
        paths = paths_list[i]
        private_dataset = create_private_dataset(
            paths['train_X'], paths['train_y'],
            input_dim, num_classes, args.batch_size
        )
        model = revisit_phase(model, private_dataset, args.epochs)
        
        client_models.append(model)
        del private_dataset
        aggressive_memory_cleanup()
    
    log_timestamp(logger, "Transfer learning completed")
    
    for round_num in range(1, args.rounds + 1):
        round_start = time.time()
        
        logger.info(f"Round {round_num}/{args.rounds}")
        print(f"\n{bcolors.HEADER}Round {round_num}/{args.rounds}{bcolors.ENDC}")
        log_timestamp(logger, f"Round {round_num} started")
        
        # STEP 1: Generate logits from all clients
        print(f"\n{bcolors.OKCYAN}[STEP 1/3] Generating logits from all clients{bcolors.ENDC}")
        all_client_logits = []
        
        for i in range(n_clients):
            print(f"  Client {i}: Generating logits on public dataset...")
            logits = generate_logits(client_models[i], public_dataset)
            all_client_logits.append(logits)
            print(f"    Logits shape: {logits.shape}")
        
        # STEP 2: Compute mean logits
        print(f"\n{bcolors.OKCYAN}[STEP 2/3] Averaging logits{bcolors.ENDC}")
        consensus_logits = np.mean(all_client_logits, axis=0)
        
        print(f"  Consensus logits shape: {consensus_logits.shape}")
        
        # STEP 3: Each client enters DIGEST + REVISIT
        print(f"\n{bcolors.OKCYAN}[STEP 3/3] Client training (DIGEST + REVISIT){bcolors.ENDC}")
        
        for i in range(n_clients):
            print(f"\n{bcolors.BOLD}Client {i}{bcolors.ENDC}")
            
            # DIGEST phase: Match consensus logits (2 epochs hard-coded)
            client_models[i] = digest_phase(
                client_models[i],
                consensus_logits,
                public_dataset,
                epochs=1,
                batch_size=args.batch_size
            )
            
            # REVISIT phase: Train on private data
            paths = paths_list[i]
            private_dataset = create_private_dataset(
                paths['train_X'], paths['train_y'],
                input_dim, num_classes, int(args.batch_size * 0.25)  # Smaller batch for private data
            )
            
            client_models[i] = revisit_phase(
                client_models[i],
                private_dataset,
                int(args.epochs)
            )
            
            del private_dataset
            aggressive_memory_cleanup()
        
        # STEP 4: Personalized evaluation on full test set
        print(f"\n{bcolors.PURPLE}[EVALUATION] Personalized evaluation (each client on full test set){bcolors.ENDC}")
        
        for i in range(n_clients):
            print(f"\n{bcolors.BOLD}Client {i}{bcolors.ENDC}")
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
        
        # Save results after each round
        results_df = pd.DataFrame(results_dict).T
        results_df.to_excel(excel_filename)
    
    total_time = time.time() - pipeline_start
    log_timestamp(logger, f"Pipeline completed in {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"\n{bcolors.OKGREEN}Results saved to {excel_filename}{bcolors.ENDC}")

if __name__ == "__main__":
    main()
