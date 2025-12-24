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

def setup_logger(n_clients, partition_type, gamma=1.0):
    log_filename = f"results/FD_{n_clients}client_{partition_type}_gamma{gamma}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(levelname)s: %(message)s', filemode='w')
    print(f"{bcolors.OKCYAN}Logging to {log_filename}{bcolors.ENDC}")
    
    # Setup detailed class metrics logger
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
    }
    
    return paths

def create_model(input_dim, num_classes, batch_size):
    return dense.create_enhanced_dense_model(input_dim, num_classes, batch_size)

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

def load_test_dataset(batch_size, num_classes):
    X_test = np.load("data/X_test.npy", mmap_mode='r')
    y_test = np.load("data/y_test.npy", mmap_mode='r')
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    if len(y_test.shape) == 1:
        y_test = tf.keras.utils.to_categorical(y_test.astype(np.int32), num_classes).astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def generate_per_class_logits(model_wrapper, X_path, y_path, num_classes):
    # Extract accumulated logits from training (paper algorithm line 7-8)
    if hasattr(model_wrapper, '_accumulated_logits') and hasattr(model_wrapper, '_accumulated_counts'):
        class_logits_sum = model_wrapper._accumulated_logits
        class_counts = model_wrapper._accumulated_counts
        
        # Average the accumulated logits (paper line 8)
        per_class_logits = {}
        for class_id in range(num_classes):
            count = class_counts[class_id]
            if count > 0:
                per_class_logits[class_id] = class_logits_sum[class_id] / count
        
        # Convert class_counts to dict
        class_counts_dict = {i: int(class_counts[i]) for i in range(num_classes)}
        
        # Clean up accumulated data
        delattr(model_wrapper, '_accumulated_logits')
        delattr(model_wrapper, '_accumulated_counts')
        
        return per_class_logits, class_counts_dict
    
    # Fallback: if no accumulated logits, generate from final model (old behavior)
    print("  WARNING: No accumulated logits found, using final model state")
    if hasattr(model_wrapper, 'get_logits_model'):
        logits_model = model_wrapper.get_logits_model()
    else:
        logits_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
    
    # Load data with memory mapping
    X_mmap = np.load(X_path, mmap_mode='r')
    y_mmap = np.load(y_path, mmap_mode='r')
    total_samples = X_mmap.shape[0]
    
    # Initialize accumulators for each class
    class_logits_sum = np.zeros((num_classes, num_classes), dtype=np.float32)
    class_counts = np.zeros(num_classes, dtype=np.int32)
    
    # Process in chunks
    chunk_size = 10000
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        X_chunk = np.array(X_mmap[start_idx:end_idx], dtype=np.float32)
        y_chunk = np.array(y_mmap[start_idx:end_idx], dtype=np.int32)
        
        # Get logits for chunk (training=False for stable outputs)
        logits_chunk = logits_model(X_chunk, training=False).numpy()
        
        # Vectorized accumulation using np.add.at
        np.add.at(class_logits_sum, y_chunk, logits_chunk)
        np.add.at(class_counts, y_chunk, 1)
        
        del X_chunk, y_chunk, logits_chunk
    
    # Average logits
    per_class_logits = {}
    for class_id in range(num_classes):
        count = class_counts[class_id]
        if count > 0:
            per_class_logits[class_id] = class_logits_sum[class_id] / count
    
    # Convert class_counts to dict for consistent return type
    class_counts_dict = {i: int(class_counts[i]) for i in range(num_classes)}
    
    return per_class_logits, class_counts_dict

def local_training_with_distillation(model_wrapper, private_dataset, global_logits, num_classes, 
                                     epochs, gamma, batch_size, client_class_counts=None):
    setup_start = time.time()
    keras_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
    
    # Get logits model for distillation
    if hasattr(model_wrapper, 'get_logits_model'):
        logits_model = model_wrapper.get_logits_model()
    else:
        logits_model = keras_model
    
    print(f"  [LOCAL TRAINING] Training with distillation for {epochs} epochs...")
    print(f"    Global logits available for {len(global_logits)}/{num_classes} classes")
    print(f"    Setup time: {time.time() - setup_start:.3f}s")
    
    # Initialize accumulators for logits (paper line 6: accumulate during training)
    class_logits_sum = np.zeros((num_classes, num_classes), dtype=np.float32)
    class_counts = np.zeros(num_classes, dtype=np.int32)
    
    # If no global logits available, fall back to standard CE training
    if len(global_logits) == 0:
        print(f"    No global logits available - using standard CE training")
        history = model_wrapper.fit(
            private_dataset,
            epochs=epochs,
            verbose=1
        )
        if 'loss' in history.history:
            final_loss = history.history['loss'][-1]
            print(f"    Loss: {final_loss:.4f}")
        
        # Still need to accumulate logits for sending to server
        for X_batch, y_batch in private_dataset:
            batch_logits = logits_model(X_batch, training=False).numpy()
            y_labels = np.argmax(y_batch.numpy(), axis=1)
            np.add.at(class_logits_sum, y_labels, batch_logits)
            np.add.at(class_counts, y_labels, 1)
        
        model_wrapper._accumulated_logits = class_logits_sum
        model_wrapper._accumulated_counts = class_counts
        return model_wrapper
    
    # Prepare client-specific metadata
    tensor_start = time.time()
    client_owned_classes = set()
    
    if client_class_counts is not None:
        client_owned_classes = {c for c, count in client_class_counts.items() if count > 0}
        print(f"    Client owns {len(client_owned_classes)} classes")
    
    # Convert global_logits dict to tensor
    global_logits_tensor = np.zeros((num_classes, num_classes), dtype=np.float32)
    has_global_logits = np.zeros(num_classes, dtype=bool)
    
    for class_id, logits in global_logits.items():
        global_logits_tensor[class_id] = logits
        has_global_logits[class_id] = True
    
    global_logits_tensor = tf.constant(global_logits_tensor, dtype=tf.float32)
    has_global_logits_tensor = tf.constant(has_global_logits, dtype=tf.bool)
    
    print(f"    Tensor conversion time: {time.time() - tensor_start:.3f}s")
    
    # Create a model that outputs both logits and predictions in a single forward pass
    model_start = time.time()
    logits_layer = keras_model.get_layer('logits')
    dual_output_model = tf.keras.Model(
        inputs=keras_model.input,
        outputs=[logits_layer.output, keras_model.output]
    )
    print(f"    Dual model creation time: {time.time() - model_start:.3f}s")
    
    # Manual training loop with GradientTape
    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    
    # Use the model's existing optimizer if available, otherwise create new one
    if hasattr(keras_model, 'optimizer') and keras_model.optimizer is not None:
        optimizer = keras_model.optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        keras_model.optimizer = optimizer
    
    # Compile training step for faster execution - accumulate logits during training
    @tf.function
    def train_step(X_batch, y_batch):
        with tf.GradientTape() as tape:
            student_logits, predictions = dual_output_model(X_batch, training=True)
            y_labels = tf.argmax(y_batch, axis=1)
            
            # CE loss for all samples
            ce = ce_loss_fn(y_batch, predictions)
            
            # Distillation for samples with matching global logits
            batch_global_logits = tf.gather(global_logits_tensor, y_labels)
            batch_has_global = tf.gather(has_global_logits_tensor, y_labels)
            
            valid_student = tf.boolean_mask(student_logits, batch_has_global)
            valid_global = tf.boolean_mask(batch_global_logits, batch_has_global)
            
            if tf.shape(valid_student)[0] > 0:
                distill = gamma * mae_loss(valid_global, valid_student)
            else:
                distill = 0.0
            
            loss = ce + distill
        
        gradients = tape.gradient(loss, keras_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
        return loss, ce, distill, student_logits, y_labels
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_distill_loss = 0
        batches = 0
        
        forward_time = 0
        backward_time = 0
        loss_compute_time = 0
        
        print(f"    Epoch {epoch+1}/{epochs}", end='', flush=True)
        
        for X_batch, y_batch in private_dataset:
            step_start = time.time()
            loss, ce, distill, batch_logits, batch_labels = train_step(X_batch, y_batch)
            step_time = time.time() - step_start
            
            # Accumulate logits DURING training (paper algorithm line 6)
            batch_logits_np = batch_logits.numpy()
            batch_labels_np = batch_labels.numpy()
            np.add.at(class_logits_sum, batch_labels_np, batch_logits_np)
            np.add.at(class_counts, batch_labels_np, 1)
            
            epoch_loss += loss.numpy()
            epoch_ce_loss += ce.numpy()
            epoch_distill_loss += distill.numpy() if isinstance(distill, tf.Tensor) else distill
            forward_time += step_time  # Compiled function includes all steps
            batches += 1
        
        avg_loss = epoch_loss / batches if batches > 0 else 0
        avg_ce = epoch_ce_loss / batches if batches > 0 else 0
        avg_distill = epoch_distill_loss / batches if batches > 0 else 0
        epoch_time = time.time() - epoch_start
        print(f" - {epoch_time:.2f}s - loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Distill: {avg_distill:.4f})")
        print(f"      [Step time: {forward_time:.2f}s, {batches} batches, {forward_time/batches:.3f}s/batch]")
    
    print(f"    Training completed")
    
    # Store accumulated logits in model wrapper for later extraction
    model_wrapper._accumulated_logits = class_logits_sum
    model_wrapper._accumulated_counts = class_counts
    
    return model_wrapper

def aggregate_logits_server(all_client_logits, num_classes):
    """Server aggregation of per-class logits using simple average."""
    print(f"\n{bcolors.OKCYAN}[SERVER AGGREGATION]{bcolors.ENDC}")
    
    # Initialize global logits for this round
    global_logits_sum = {}
    class_contributors = {}  # {class_id: [client_ids]}
    
    # Collect contributions
    for client_id, client_logits in all_client_logits.items():
        for class_id, logits_vector in client_logits.items():
            if class_id not in global_logits_sum:
                global_logits_sum[class_id] = np.zeros(num_classes, dtype=np.float32)
                class_contributors[class_id] = []
            
            global_logits_sum[class_id] += logits_vector
            class_contributors[class_id].append(client_id)
    
    print(f"  Classes with contributions: {len(global_logits_sum)}/{num_classes}")
    for class_id, contributors in class_contributors.items():
        print(f"    Class {class_id}: {len(contributors)} clients contributed")
    
    # Broadcast to CONTRIBUTORS (remove their contribution)
    global_logits_to_broadcast = {}
    
    for client_id in all_client_logits.keys():
        client_global_logits = {}
        
        for class_id in global_logits_sum.keys():
            s_l = len(class_contributors[class_id])
            contributed = client_id in class_contributors[class_id]
            
            if contributed:
                # Client contributed: remove its contribution and average
                numerator = global_logits_sum[class_id] - all_client_logits[client_id][class_id]
                client_global_logits[class_id] = numerator / (s_l - 1) if s_l > 1 else np.zeros(num_classes, dtype=np.float32)
        
        global_logits_to_broadcast[client_id] = client_global_logits
    
    # Self-average for non-contributors
    global_logits_balanced = {}
    for class_id in global_logits_sum.keys():
        s_l = len(class_contributors[class_id])
        global_logits_balanced[class_id] = global_logits_sum[class_id] / s_l
    
    # Broadcast balanced average to NON-CONTRIBUTORS
    for client_id in all_client_logits.keys():
        for class_id in global_logits_balanced.keys():
            if class_id not in global_logits_to_broadcast[client_id]:
                global_logits_to_broadcast[client_id][class_id] = global_logits_balanced[class_id]
    
    return global_logits_to_broadcast

def main():
    parser = argparse.ArgumentParser(description="Federated Distillation (FD) - No Public Dataset")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--partition_type", type=str, default="label_skew-10", help="Partition type (e.g., 'label_skew-10')")
    parser.add_argument("--rounds", type=int, default=10, help="Communication rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size")
    parser.add_argument("--gamma", type=float, default=1.0, help="Distillation loss weight")
    parser.add_argument("--data_calc", action="store_true", help="Recalculate data composition every round (default: calculate once)")
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
    
    # Determine input_dim
    sample_X = np.load(os.path.join("data", "X_test.npy"), mmap_mode='r')
    input_dim = int(sample_X.shape[1])
    del sample_X
    
    # Setup paths for all clients
    paths_list = []
    for i in range(n_clients):
        paths = setup_paths(str(i), partition_type, client_count)
        paths_list.append(paths)
    
    # Load full test dataset
    test_dataset = load_test_dataset(args.batch_size, num_classes)
    print(f"{bcolors.OKGREEN}Using FULL test set{bcolors.ENDC}")
    
    pipeline_start = time.time()
    log_timestamp(logger, "SIMULATION STARTED")
    log_timestamp(logger, f"Clients: {n_clients}, Rounds: {args.rounds}, Partition: {partition_type}")
    log_timestamp(logger, f"Gamma: {args.gamma}")
    
    # Initialize client models
    client_models = []
    for i in range(n_clients):
        model = create_model(input_dim, num_classes, args.batch_size)
        client_models.append(model)
    
    # Initialize global logits for each client (from previous round)
    global_logits_per_client = {i: {} for i in range(n_clients)}
    
    # Pre-calculate data composition if not recalculating every round
    all_client_counts_cache = None
    if not args.data_calc:
        print(f"\n{bcolors.OKCYAN}[PRE-CALCULATION] Calculating data composition once for all rounds{bcolors.ENDC}")
        all_client_counts_cache = {}
        for i in range(n_clients):
            paths = paths_list[i]
            y_mmap = np.load(paths['train_y'], mmap_mode='r')
            y_array = np.array(y_mmap, dtype=np.int32)
            class_counts = np.zeros(num_classes, dtype=np.int32)
            np.add.at(class_counts, y_array, 1)
            all_client_counts_cache[i] = {j: int(class_counts[j]) for j in range(num_classes)}
            del y_mmap, y_array
        print(f"  Data composition cached for {n_clients} clients")
    
    # MAIN ROUNDS
    for round_num in range(1, args.rounds + 1):
        round_start = time.time()
        
        logger.info(f"Round {round_num}/{args.rounds}")
        print(f"\n{bcolors.HEADER}Round {round_num}/{args.rounds}{bcolors.ENDC}")
        log_timestamp(logger, f"Round {round_num} started")
        
        # STEP 1: Local training with distillation (using global logits from PREVIOUS round)
        print(f"\n{bcolors.OKCYAN}[STEP 1/3] Local training with distillation{bcolors.ENDC}")
        all_client_counts = {}
        
        for i in range(n_clients):
            print(f"\n{bcolors.BOLD}Client {i}{bcolors.ENDC}")
            paths = paths_list[i]
            
            # Get cached or fresh class counts
            if all_client_counts_cache is not None:
                client_counts = all_client_counts_cache[i]
            else:
                y_mmap = np.load(paths['train_y'], mmap_mode='r')
                y_array = np.array(y_mmap, dtype=np.int32)
                class_counts_arr = np.zeros(num_classes, dtype=np.int32)
                np.add.at(class_counts_arr, y_array, 1)
                client_counts = {j: int(class_counts_arr[j]) for j in range(num_classes)}
                del y_mmap, y_array
            all_client_counts[i] = client_counts
            
            private_dataset = create_private_dataset(
                paths['train_X'], paths['train_y'],
                input_dim, num_classes, args.batch_size
            )
            
            # Get global logits from PREVIOUS round (empty for round 1)
            global_logits = global_logits_per_client[i]
            
            client_models[i] = local_training_with_distillation(
                client_models[i],
                private_dataset,
                global_logits,
                num_classes,
                args.epochs,
                args.gamma,
                args.batch_size,
                client_class_counts=client_counts
            )
            
            del private_dataset
            aggressive_memory_cleanup()
        
        # STEP 2: Generate per-class logits from TRAINED models
        print(f"\n{bcolors.OKCYAN}[STEP 2/3] Generating per-class logits from trained models{bcolors.ENDC}")
        all_client_logits = {}
        
        for i in range(n_clients):
            print(f"\n{bcolors.BOLD}Client {i}{bcolors.ENDC}")
            paths = paths_list[i]
            
            per_class_logits, class_counts = generate_per_class_logits(
                client_models[i], 
                paths['train_X'], 
                paths['train_y'],
                num_classes
            )
            
            all_client_logits[i] = per_class_logits
            
            print(f"  Generated logits for {len(per_class_logits)}/{num_classes} classes")
            for class_id, count in class_counts.items():
                if class_id in per_class_logits and count > 0:
                    print(f"    Class {class_id}: {count} samples")
            
            aggressive_memory_cleanup()
        
        # STEP 3: Server aggregates and broadcasts for NEXT round
        global_logits_to_broadcast = aggregate_logits_server(
            all_client_logits,
            num_classes
        )
        
        # Store global logits for next round
        global_logits_per_client = global_logits_to_broadcast
        
        # STEP 4: Evaluate all clients on test set
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
        
        # Save results after each round
        results_df = pd.DataFrame(results_dict).T
        results_df.to_excel(excel_filename)
    
    total_time = time.time() - pipeline_start
    log_timestamp(logger, f"Pipeline completed in {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"\n{bcolors.OKGREEN}Results saved to {excel_filename}{bcolors.ENDC}")

if __name__ == "__main__":
    main()
