import os
import gc
import time
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from models import dense, dense_og
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

def setup_logger(n_clients, partition_type, og=False):
    suffix = "_og" if og else ""
    log_filename = f"results/NoFed_{n_clients}_client_{partition_type}{suffix}.log"
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

def create_model(input_dim, num_classes, batch_size, og=False):
    return dense.create_enhanced_dense_model(input_dim, num_classes, batch_size)

# Cache public test dataset to avoid repeated pickle loading
_public_test_dataset_cache = None

def load_public_test_dataset(batch_size, og=False):
    """Load public test dataset (default small fraction) respecting OG postfix."""
    global _public_test_dataset_cache
    
    if _public_test_dataset_cache is None:
        import pickle
        postfix = "_og" if og else ""
        with open(f"data/X_public_test{postfix}.pkl", "rb") as f:
            X_pub_test = pickle.load(f)
        with open(f"data/y_public_test{postfix}.pkl", "rb") as f:
            y_pub_test = pickle.load(f)
        
        # Convert labels to categorical
        num_classes = len(np.unique(y_pub_test))
        if len(y_pub_test.shape) == 1:
            y_pub_test = tf.keras.utils.to_categorical(y_pub_test, num_classes)
        
        _public_test_dataset_cache = (X_pub_test, y_pub_test)
        print(f"{bcolors.OKGREEN}Public test dataset cached ({X_pub_test.shape[0]} samples){bcolors.ENDC}")
    
    X_pub_test, y_pub_test = _public_test_dataset_cache
    dataset = tf.data.Dataset.from_tensor_slices((X_pub_test, y_pub_test))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Cache global test dataset
_global_test_dataset_cache = None

def load_global_test_dataset(batch_size, num_classes, og=False):
    """Load full global test dataset respecting OG postfix."""
    global _global_test_dataset_cache
    
    if _global_test_dataset_cache is None:
        postfix = "_og" if og else ""
        base_X = f"data/X_test{postfix}.npy"
        base_y = f"data/y_test{postfix}.npy"
        if not os.path.exists(base_X) or not os.path.exists(base_y):
            # Fallback to non-postfix if OG specific files absent
            base_X = "data/X_test.npy"
            base_y = "data/y_test.npy"
        X_test = np.load(base_X, mmap_mode='r')
        y_test = np.load(base_y, mmap_mode='r')
        
        # Load into memory
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
        
        # Convert labels to categorical
        if len(y_test.shape) == 1:
            y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        
        _global_test_dataset_cache = (X_test, y_test)
        print(f"{bcolors.OKGREEN}Global test dataset cached ({X_test.shape[0]} samples){bcolors.ENDC}")
    
    X_test, y_test = _global_test_dataset_cache
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_private_dataset(X_path, y_path, input_dim, num_classes, batch_size):
    """Create dataset from client's private data"""
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

def evaluate_on_test(model_wrapper, test_dataset, dataset_name="test", num_classes=None, class_names=None):
    """Evaluate model on test dataset and return metrics"""
    preds = model_wrapper.predict(test_dataset, verbose=0)
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
    
    # Per-class metrics - compute for ALL classes in dataset (not just ones in test set)
    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            num_classes = int(np.max(np.concatenate([y_true, y_pred])) + 1)
    labels = np.arange(num_classes)
    
    per_class_metrics = {}
    for cls in labels:
        cls_mask = (y_true == cls)
        num_samples = np.sum(cls_mask)
        
        if num_samples > 0:
            cls_y_true = y_true[cls_mask]
            cls_y_pred = y_pred[cls_mask]
            
            # Per-class accuracy: what % of true class samples were correctly predicted
            cls_acc = np.mean(cls_y_pred == cls)
            
            # Precision: of all predictions for this class, how many were correct
            cls_pred_mask = (y_pred == cls)
            if np.sum(cls_pred_mask) > 0:
                cls_precision = np.mean(y_true[cls_pred_mask] == cls)
            else:
                cls_precision = 0.0
            
            # Recall: of all true samples of this class, how many were predicted correctly
            cls_recall = cls_acc
            
            # F1 score
            if cls_precision + cls_recall > 0:
                cls_f1 = 2 * (cls_precision * cls_recall) / (cls_precision + cls_recall)
            else:
                cls_f1 = 0.0
            
            per_class_metrics[int(cls)] = {
                'accuracy': cls_acc,
                'f1': cls_f1,
                'precision': cls_precision,
                'recall': cls_recall,
                'support': int(num_samples)
            }
        else:
            per_class_metrics[int(cls)] = {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'support': 0
            }
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    del y_pred, y_true
    gc.collect()
    
    return accuracy, f1, precision, recall, per_class_metrics, conf_matrix

def main():
    parser = argparse.ArgumentParser(description="No Federation Baseline - Isolated Client Training")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--partition_type", type=str, default="label_skew", help="Partition type")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per client")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size")
    parser.add_argument("--full_test", action="store_true", help="Use full global test set instead of public test set")
    parser.add_argument("--og", action="store_true", help="Use OG 45-feature dataset partitions (adds _og suffix)")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    logger, log_filename, detailed_logger = setup_logger(args.n_clients, args.partition_type, og=args.og)
    excel_filename = log_filename.replace('.log', '.xlsx')
    
    results_dict = {f'Client_{i}': {} for i in range(args.n_clients)}
    
    partition_dir = os.path.join("data", "partitions", f"{args.n_clients}_client", f"{args.partition_type}{'_og' if args.og else ''}")
    
    y_test = np.load(os.path.join("data", "y_test.npy"), mmap_mode='r')
    num_classes = len(np.unique(y_test))
    del y_test
    
    client_paths = []
    for i in range(args.n_clients):
        paths = {
            'train_X': os.path.join(partition_dir, f"client_{i}_X_train.npy"),
            'train_y': os.path.join(partition_dir, f"client_{i}_y_train.npy")
        }
        client_paths.append(paths)

    # Determine input_dim from client training data to avoid mismatch with global test set
    input_dim = None
    for paths in client_paths:
        if os.path.exists(paths['train_X']):
            sample_X = np.load(paths['train_X'], mmap_mode='r')
            input_dim = int(sample_X.shape[1])
            del sample_X
            break
    if input_dim is None:
        # Fallback to test set as last resort
        test_X_path = os.path.join("data", f"X_test{'_og' if args.og else ''}.npy")
        if not os.path.exists(test_X_path):
            test_X_path = os.path.join("data", "X_test.npy")
        sample_X = np.load(test_X_path, mmap_mode='r')
        input_dim = int(sample_X.shape[1])
        del sample_X
    
    # Load test dataset based on flag
    if args.full_test:
        test_dataset = load_global_test_dataset(args.batch_size, num_classes, og=args.og)
        test_name = "Global Test Set (100%)"
    else:
        test_dataset = load_public_test_dataset(args.batch_size, og=args.og)
        test_name = "Public Test Set (20%)"

    class_names = None
    label_classes_path = os.path.join(partition_dir, "label_classes.npy")
    if os.path.exists(label_classes_path):
        try:
            class_names = np.load(label_classes_path, allow_pickle=True)
        except Exception as err:
            print(f"{bcolors.WARNING}Warning: failed to load class names ({err}){bcolors.ENDC}")
    
    pipeline_start = time.time()
    log_timestamp(logger, "NO FEDERATION BASELINE STARTED")
    log_timestamp(logger, f"Clients: {args.n_clients}, Epochs: {args.epochs}, Partition: {args.partition_type}")
    log_timestamp(logger, f"Test Set: {test_name}")
    log_timestamp(logger, "Each client trains independently on private data only (no communication)")
    
    print(f"\n{bcolors.HEADER}No Federation Baseline{bcolors.ENDC}")
    print(f"{bcolors.WARNING}Clients train independently for {args.epochs} epochs{bcolors.ENDC}")
    print(f"{bcolors.WARNING}Test set: {test_name}{bcolors.ENDC}\n")
    
    # Train each client independently
    for client_id in range(args.n_clients):
        client_start = time.time()
        
        print(f"\n{bcolors.BOLD}Training Client {client_id}{bcolors.ENDC}")
        log_timestamp(logger, f"Client {client_id} training started")
        
        # Create model
        model = create_model(input_dim, num_classes, args.batch_size, og=args.og)
        
        # Load private dataset
        paths = client_paths[client_id]
        private_dataset = create_private_dataset(
            paths['train_X'], paths['train_y'],
            input_dim, num_classes, args.batch_size
        )
        
        # Compute class weights from client's training labels
        y_train_raw = np.load(paths['train_y'], mmap_mode='r')
        y_train_raw = np.array(y_train_raw, dtype=np.int32)
        from collections import Counter
        class_counts = Counter(y_train_raw.tolist())
        total_samples = len(y_train_raw)
        
        # Inverse frequency weighting with smoothing
        class_weight = {}
        for cls in range(num_classes):
            count = class_counts.get(cls, 0)
            if count > 0:
                # Inverse frequency: total / (num_classes * count)
                class_weight[cls] = total_samples / (num_classes * count)
            else:
                # Assign small weight to unseen classes (won't affect training but keeps dict complete)
                class_weight[cls] = 0.0
        
        print(f"  Class distribution for client {client_id}:")
        print(f"    Present classes: {len([c for c in class_counts.values() if c > 0])}/{num_classes}")
        print(f"    Sample counts (non-zero): {dict(sorted(class_counts.items())[:10])}")
        print(f"    Weight range: min={min([w for w in class_weight.values() if w > 0], default=0):.4f}, max={max(class_weight.values()):.4f}")
        
        del y_train_raw
        gc.collect()
        
        # Train on private data
        print(f"  Training for {args.epochs} epochs on private data with class weighting...")
        history = model.fit(
            private_dataset,
            epochs=args.epochs,
            class_weight=class_weight,
            verbose=1
        )
        
        if 'loss' in history.history:
            final_loss = history.history['loss'][-1]
            print(f"  Final Training Loss: {final_loss:.4f}")
        
        # Evaluate on test set
        print(f"  Evaluating on {test_name}...")
        accuracy, f1, precision, recall, per_class_metrics, conf_matrix = evaluate_on_test(
            model,
            test_dataset,
            test_name,
            num_classes=num_classes,
            class_names=class_names
        )
        
        client_time = time.time() - client_start
        
        logger.info(f"Client {client_id} | Acc: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Time: {client_time:.2f}s")
        print(f"{bcolors.OKGREEN}Client {client_id} Results:{bcolors.ENDC}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Training Time: {client_time:.2f}s")
        
        # Log detailed per-class metrics
        detailed_logger.info(f"Client {client_id}:")
        class_metrics_str = ""
        for cls in sorted(per_class_metrics.keys()):
            metrics = per_class_metrics[cls]
            class_metrics_str += f" [Class {cls}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, Support={metrics['support']}]"
        detailed_logger.info(class_metrics_str)

        # Log confusion matrix for deeper diagnostics (convert to DataFrame for readability)
        if class_names is not None and len(class_names) == conf_matrix.shape[0]:
            cm_columns = [str(cls) for cls in class_names]
            cm_index = [str(cls) for cls in class_names]
        else:
            cm_columns = [str(i) for i in range(conf_matrix.shape[0])]
            cm_index = cm_columns
        cm_df = pd.DataFrame(conf_matrix, index=cm_index, columns=cm_columns)
        detailed_logger.info("Confusion Matrix (rows=true, cols=pred):")
        for line in cm_df.to_string().splitlines():
            detailed_logger.info(line)
        
        # Store results
        results_dict[f'Client_{client_id}']['Accuracy'] = accuracy
        results_dict[f'Client_{client_id}']['F1_Score'] = f1
        results_dict[f'Client_{client_id}']['Precision'] = precision
        results_dict[f'Client_{client_id}']['Recall'] = recall
        results_dict[f'Client_{client_id}']['Training_Time_s'] = client_time
        
        # Cleanup
        del model
        aggressive_memory_cleanup()
    
    # Save results to Excel
    results_df = pd.DataFrame(results_dict).T
    results_df.to_excel(excel_filename)
    
    # Calculate and print summary statistics
    accuracies = [results_dict[f'Client_{i}']['Accuracy'] for i in range(args.n_clients)]
    f1_scores = [results_dict[f'Client_{i}']['F1_Score'] for i in range(args.n_clients)]
    
    total_time = time.time() - pipeline_start
    
    print(f"\n{bcolors.HEADER}SUMMARY{bcolors.ENDC}")
    print(f"Accuracy - Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}, Min: {np.min(accuracies):.4f}, Max: {np.max(accuracies):.4f}")
    print(f"F1 Score - Mean: {np.mean(f1_scores):.4f}, Std: {np.std(f1_scores):.4f}, Min: {np.min(f1_scores):.4f}, Max: {np.max(f1_scores):.4f}")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f}m)")
    
    log_timestamp(logger, "SUMMARY")
    logger.info(f"Accuracy - Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}, Min: {np.min(accuracies):.4f}, Max: {np.max(accuracies):.4f}")
    logger.info(f"F1 Score - Mean: {np.mean(f1_scores):.4f}, Std: {np.std(f1_scores):.4f}, Min: {np.min(f1_scores):.4f}, Max: {np.max(f1_scores):.4f}")
    logger.info(f"Total Time: {total_time:.2f}s ({total_time/60:.2f}m)")
    
    print(f"\n{bcolors.OKGREEN}Results saved to {excel_filename}{bcolors.ENDC}")

if __name__ == "__main__":
    main()
