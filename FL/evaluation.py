import gc
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from .backend import get_torch_loader_kwargs as _torch_loader_kwargs, use_tf as _use_tf
if _use_tf():
    import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PerClassMetrics = Tuple[np.ndarray, np.ndarray, np.ndarray]


def plot_f1_threshold_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    plot_path: str,
    label: str = "model",
    n_steps: int = 20,
    logger=None,
) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    thresholds = np.linspace(0.0, 0.95, n_steps)
    max_conf = y_pred_proba.max(axis=1)
    labels = np.arange(y_pred_proba.shape[1])
    f1_vals, effective_f1_vals, coverage_vals, acc_vals, precision_vals, recall_vals = [], [], [], [], [], []
    missing_classes, missing_counts = [], []
    for theta in thresholds:
        mask = max_conf >= theta
        coverage = float(mask.mean())
        if mask.sum() == 0:
            missing = labels.tolist()
            f1_vals.append(0.0)
            effective_f1_vals.append(0.0)
            coverage_vals.append(0.0)
            acc_vals.append(0.0)
            precision_vals.append(0.0)
            recall_vals.append(0.0)
            missing_classes.append(missing)
            missing_counts.append(len(missing))
            continue
        preds = y_pred_proba[mask].argmax(axis=1)
        true = y_true[mask]
        present = np.unique(true)
        missing = np.setdiff1d(labels, present, assume_unique=True).astype(int).tolist()
        f1 = float(f1_score(true, preds, labels=labels, average="macro", zero_division=0))
        effective_f1 = float(f1_score(true, preds, labels=present, average="macro", zero_division=0)) if len(present) else 0.0
        f1_vals.append(f1)
        effective_f1_vals.append(effective_f1)
        coverage_vals.append(coverage)
        acc_vals.append(float(np.mean(preds == true)))
        precision_vals.append(float(precision_score(true, preds, labels=present, average="macro", zero_division=0)))
        recall_vals.append(float(recall_score(true, preds, labels=present, average="macro", zero_division=0)))
        missing_classes.append(missing)
        missing_counts.append(len(missing))

    best_idx = int(np.argmax(effective_f1_vals))
    best_theta = float(thresholds[best_idx])
    best_f1 = float(f1_vals[best_idx])
    best_effective_f1 = float(effective_f1_vals[best_idx])
    min_cov_idx = int(np.argmin(coverage_vals))
    min_cov_theta = float(thresholds[min_cov_idx])
    min_cov = float(coverage_vals[min_cov_idx])

    best_point = {
        "theta": best_theta, "Acc": acc_vals[best_idx], "F1": best_f1,
        "Effective_F1": best_effective_f1, "Missing_Count": missing_counts[best_idx],
        "Missing_Classes": missing_classes[best_idx], "Precision": precision_vals[best_idx],
        "Recall": recall_vals[best_idx], "Coverage": coverage_vals[best_idx],
    }
    min_cov_point = {
        "theta": min_cov_theta, "Acc": acc_vals[min_cov_idx], "F1": f1_vals[min_cov_idx],
        "Effective_F1": effective_f1_vals[min_cov_idx], "Missing_Count": missing_counts[min_cov_idx],
        "Missing_Classes": missing_classes[min_cov_idx], "Precision": precision_vals[min_cov_idx],
        "Recall": recall_vals[min_cov_idx], "Coverage": min_cov,
    }

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thresholds, effective_f1_vals, "b-o", markersize=4, label=f"{label} effective F1")
    ax1.axvline(best_theta, color="b", linestyle=":", alpha=0.6, label=f"best θ={best_theta:.2f} effF1={best_effective_f1:.4f}")
    ax1.set_xlabel("Confidence Threshold θ")
    ax1.set_ylabel("Macro F1", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, coverage_vals, "r--s", markersize=4, label="Coverage")
    ax2.bar(thresholds, missing_counts, width=0.95 / max(n_steps, 1) * 0.65, alpha=0.22, color="gray", label="Missing classes")
    ax2.set_ylabel("Coverage / missing class count", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(0, max(1.0, max(missing_counts) * 1.15))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    missing_detail = "; ".join(
        f"θ={theta:.2f}: missing={missing}"
        for theta, missing in zip(thresholds, missing_classes)
        if missing
    ) or "none"
    summary = (
        f"F1-threshold curve saved → {plot_path} | "
        f"Best θ={best_theta:.2f}: Acc={best_point['Acc']:.4f} F1={best_f1:.4f} EffF1={best_effective_f1:.4f} "
        f"P={best_point['Precision']:.4f} R={best_point['Recall']:.4f} Cov={best_point['Coverage']:.4f} "
        f"Missing={best_point['Missing_Classes']} | "
        f"MinCov θ={min_cov_theta:.2f}: Acc={min_cov_point['Acc']:.4f} F1={min_cov_point['F1']:.4f} "
        f"EffF1={min_cov_point['Effective_F1']:.4f} P={min_cov_point['Precision']:.4f} "
        f"R={min_cov_point['Recall']:.4f} Cov={min_cov:.4f} Missing={min_cov_point['Missing_Classes']} | "
        f"Missing detail: {missing_detail}"
    )
    print(summary)
    if logger is not None:
        logger.info("F1_CURVE | %s | %s", label, summary)
    return best_point, min_cov_point


def summarize_prob_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    num_classes: int,
    ece_bins: int = 15,
    compute_auprc: bool = True,
) -> Tuple[np.ndarray, float, float, float]:
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64 if compute_auprc else np.float32)
    pred_labels = np.argmax(y_pred_proba, axis=1).astype(np.int32)
    row_ids = np.arange(len(y_true))

    loss = float(-np.mean(np.log(np.clip(y_pred_proba[row_ids, y_true], 1e-7, 1.0))))

    if compute_auprc:
        y_true_oh = np.zeros((len(y_true), num_classes), dtype=np.float32)
        y_true_oh[row_ids, y_true] = 1.0
        present_classes = np.unique(y_true)
        if len(present_classes) > 0:
            auprc = float(np.mean([
                average_precision_score(y_true_oh[:, class_idx], y_pred_proba[:, class_idx])
                for class_idx in present_classes
            ]))
        else:
            auprc = 0.0
        del y_true_oh
    else:
        auprc = 0.0

    confidences = y_pred_proba[row_ids, pred_labels]
    correctness = (pred_labels == y_true).astype(np.float64)
    bin_ids = np.digitize(confidences, np.linspace(0.0, 1.0, ece_bins + 1)[1:-1], right=True)
    ece = 0.0
    total = max(len(confidences), 1)
    for bin_idx in range(ece_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        ece += abs(correctness[mask].mean() - confidences[mask].mean()) * (mask.sum() / total)

    del row_ids, confidences, correctness, bin_ids
    return pred_labels, loss, auprc, float(ece)


def evaluate_model_streaming(
    model,
    X_path: str,
    y_path: str,
    num_classes: int,
    batch_size: int,
    is_gru: bool = False,
    chunk_size: int = 2_000_000
):
    X_test = np.load(X_path, mmap_mode='r')
    y_test = np.load(y_path, mmap_mode='r')
    total_samples = len(X_test)

    all_y_true = []
    all_y_pred = []
    total_loss = 0.0
    total_observations = 0

    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        X_chunk = np.array(X_test[chunk_start:chunk_end], dtype=np.float32)
        y_chunk = np.array(y_test[chunk_start:chunk_end])

        if is_gru and len(X_chunk.shape) == 2:
            X_chunk = np.expand_dims(X_chunk, axis=1)

        if len(y_chunk.shape) == 1 or y_chunk.shape[1] == 1:
            y_chunk_cat = np.zeros((len(y_chunk), num_classes), dtype=np.float32)
            y_chunk_cat[np.arange(len(y_chunk)), y_chunk.astype(int).ravel()] = 1.0
        else:
            y_chunk_cat = y_chunk.astype(np.float32)

        if _use_tf():
            chunk_dataset = tf.data.Dataset.from_tensor_slices((X_chunk, y_chunk_cat)).batch(batch_size)
        else:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            chunk_dataset = DataLoader(
                TensorDataset(torch.from_numpy(X_chunk), torch.from_numpy(y_chunk_cat)),
                batch_size=batch_size, shuffle=False, **_torch_loader_kwargs())
        chunk_loss = model.evaluate(chunk_dataset, verbose=0)[0]
        total_loss += chunk_loss * (chunk_end - chunk_start)
        total_observations += (chunk_end - chunk_start)

        chunk_preds = model.predict(chunk_dataset, verbose=0)
        chunk_y_pred = np.argmax(chunk_preds, axis=1)

        if len(y_chunk.shape) == 1:
            chunk_y_true = y_chunk
        else:
            chunk_y_true = np.argmax(y_chunk, axis=1)

        all_y_true.extend(chunk_y_true)
        all_y_pred.extend(chunk_y_pred)

        del X_chunk, y_chunk, y_chunk_cat, chunk_dataset, chunk_preds, chunk_y_pred, chunk_y_true
        gc.collect()

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    avg_loss = total_loss / total_observations if total_observations > 0 else 0.0
    accuracy = np.mean(y_true == y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    del X_test, y_test, all_y_true, all_y_pred
    gc.collect()

    return avg_loss, accuracy, f1_macro, precision_macro, recall_macro, (f1_per_class, precision_per_class, recall_per_class), cm, y_true, y_pred


def compute_aurc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_steps: int = 20,
) -> float:
    thresholds = np.linspace(0.0, 0.95, n_steps)
    max_conf = y_pred_proba.max(axis=1)
    labels = np.arange(y_pred_proba.shape[1])
    risks, coverages = [0.0], [0.0]
    for theta in thresholds:
        mask = max_conf >= theta
        if mask.sum() == 0:
            continue
        preds = y_pred_proba[mask].argmax(axis=1)
        risks.append(1.0 - float(f1_score(y_true[mask], preds, labels=labels, average="macro", zero_division=0)))
        coverages.append(float(mask.mean()))
    pairs = sorted(zip(coverages, risks))
    covs, rks = zip(*pairs)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(rks, covs))


def evaluate_model_with_metrics(
    model,
    test_dataset: Iterable,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
    round_num: Optional[int] = None,
    strategy_name: Optional[str] = None,
    partition_type: Optional[str] = None,
    collect_details: bool = True,
    y_true_cache: Optional[np.ndarray] = None,
    compute_aurc_curve: bool = False,
):
    base_model = model
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'predict'):
        base_model = model.base_model
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'predict'):
        base_model = base_model.model

    y_pred_proba = base_model.predict(test_dataset, verbose=0)

    if y_true_cache is not None:
        y_true = y_true_cache
    else:
        y_true_parts = []
        for _, batch_y in test_dataset:
            b = batch_y.numpy() if hasattr(batch_y, 'numpy') else np.asarray(batch_y)
            if len(b.shape) > 1 and b.shape[1] > 1:
                y_true_parts.append(np.argmax(b, axis=1))
            else:
                y_true_parts.append(b.astype(int))
        y_true = np.concatenate(y_true_parts)
        del y_true_parts

    y_pred, test_loss, auprc, ece = summarize_prob_predictions(y_true, y_pred_proba, num_classes)
    aurc_score = compute_aurc(y_true, y_pred_proba) if compute_aurc_curve else None

    accuracy = float(np.mean(y_true == y_pred))

    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    is_attack_true = y_true != 0
    is_attack_pred = y_pred != 0
    n_attack = int(is_attack_true.sum())
    n_benign = int((~is_attack_true).sum())
    tpr = float((is_attack_true & is_attack_pred).sum() / n_attack) if n_attack > 0 else 0.0
    fpr = float(((~is_attack_true) & is_attack_pred).sum() / n_benign) if n_benign > 0 else 0.0

    per_class_metrics = None
    cm = None
    class_report = ""
    if collect_details:
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        per_class_metrics = (f1_per_class, precision_per_class, recall_per_class)

        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]

        class_report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )

    del y_pred, y_pred_proba
    if y_true_cache is None:
        del y_true
    gc.collect()

    return (
        test_loss,
        accuracy,
        f1_macro,
        precision_macro,
        recall_macro,
        per_class_metrics,
        cm,
        class_report,
        auprc,
        ece,
        aurc_score,
        tpr,
        fpr,
    )


def create_enhanced_excel_report(
    excel_filename: str,
    main_results_df: pd.DataFrame,
    per_class_metrics: Optional[PerClassMetrics],
    class_names: Sequence[str],
    current_round: int,
    conf_matrix: Optional[np.ndarray]
) -> None:
    import openpyxl
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils.dataframe import dataframe_to_rows

    try:
        wb = openpyxl.load_workbook(excel_filename)
    except FileNotFoundError:
        wb = openpyxl.Workbook()
        if 'Sheet' in wb.sheetnames:
            wb.remove(wb['Sheet'])

    if 'Overall_Metrics' in wb.sheetnames:
        wb.remove(wb['Overall_Metrics'])
    ws_main = wb.create_sheet('Overall_Metrics')

    for row in dataframe_to_rows(main_results_df, index=False, header=True):
        ws_main.append(row)

    for cell in ws_main[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")

    if per_class_metrics is not None and isinstance(per_class_metrics, tuple) and len(per_class_metrics) == 3:
        sheet_name = f'Round_{current_round}_PerClass'
        if sheet_name in wb.sheetnames:
            wb.remove(wb[sheet_name])
        ws_per_class = wb.create_sheet(sheet_name)

        f1_per_class, precision_per_class, recall_per_class = per_class_metrics
        per_class_df = pd.DataFrame({
            'Class': class_names,
            'F1_Score': f1_per_class,
            'Precision': precision_per_class,
            'Recall': recall_per_class
        })

        for row in dataframe_to_rows(per_class_df, index=False, header=True):
            ws_per_class.append(row)

    wb.save(excel_filename)
