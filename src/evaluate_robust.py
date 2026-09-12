#!/usr/bin/env python3
"""
AdvRoNIDS: Robust Classifier (Model B) Vulnerability Benchmark & Clean vs. Robust Comparison
Proposal References: Section 6.4.4 (Min-Max Adversarial Training),
                     Section 6.5 (Step 3-4), Section 8 (Evaluation Plan)

This script performs:
1. Ingestion of the held-out test split (stratified N=50,000 sample, matching Phase 3).
2. Clean baseline evaluation of Robust Model B (Accuracy, Macro-F1, Weighted-F1, Classification Report, Confusion Matrix).
3. FGSM evaluation (epsilon=0.10) under constrained (Pi_S) and unconstrained settings.
4. PGD-7 robustness sweep across epsilon in {0.01, 0.05, 0.10, 0.20} under both constrained (Pi_S) and unconstrained settings.
5. Generation of classification report on constrained-PGD-attacked test data at epsilon=0.10.
6. Export of ./results/robust_model_test_metrics.json and ./results/robust_model_classification_report.txt.
7. Generation of the central comparison table ./results/clean_vs_robust_comparison.csv.
8. Generation of the central 4-line robustness curves figure ./results/clean_vs_robust_robustness_curves.png.
"""

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Ensure project modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import AdvRoNIDS_CNN
from attacks import fgsm_attack, pgd_attack, project_feasible

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_memory_mb():
    """Returns current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class DetailedLogger:
    """Logs pipeline execution to stdout and appends detailed notes to project_documentation.txt."""

    def __init__(self, doc_path="project_documentation.txt"):
        self.doc_path = Path(doc_path)

    def log(self, message, header=False, section=False):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        mem_info = f"[RAM: {get_memory_mb():.1f} MB]"

        if section:
            formatted = f"\n{'='*80}\n{timestamp} {mem_info} {message}\n{'='*80}\n"
        elif header:
            formatted = f"\n{timestamp} {mem_info} --- {message} ---\n"
        else:
            formatted = f"{timestamp} {mem_info} {message}\n"

        try:
            sys.stdout.write(formatted)
            sys.stdout.flush()
        except Exception:
            sys.stdout.buffer.write(formatted.encode("utf-8", errors="replace"))
            sys.stdout.flush()

        with open(self.doc_path, "a", encoding="utf-8") as f:
            f.write(formatted)


def stratified_sample(X, y, sample_size, random_state=42):
    """Draws a stratified sample guaranteeing that every class present is represented."""
    rng = np.random.RandomState(random_state)
    unique_classes = np.unique(y)
    selected_indices = []

    # First ensure at least 1 sample from each class
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        selected_indices.append(rng.choice(cls_indices, size=1, replace=False)[0])

    remaining_needed = sample_size - len(selected_indices)
    if remaining_needed > 0:
        all_indices = np.arange(len(y))
        remaining_pool = np.setdiff1d(all_indices, selected_indices)
        pool_y = y[remaining_pool]

        # Allocate proportional quotas
        for cls in unique_classes:
            cls_pool = remaining_pool[pool_y == cls]
            quota = int(np.round((len(cls_pool) / len(remaining_pool)) * remaining_needed))
            if quota > 0 and len(cls_pool) > 0:
                chosen = rng.choice(cls_pool, size=min(quota, len(cls_pool)), replace=False)
                selected_indices.extend(chosen)

        # Pad or trim if rounding created slight difference
        selected_indices = list(set(selected_indices))
        if len(selected_indices) < sample_size:
            diff = sample_size - len(selected_indices)
            remaining_pool = np.setdiff1d(all_indices, selected_indices)
            chosen = rng.choice(remaining_pool, size=diff, replace=False)
            selected_indices.extend(chosen)
        elif len(selected_indices) > sample_size:
            selected_indices = selected_indices[:sample_size]

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)
    return X[selected_indices], y[selected_indices]


def load_test_sample(data_dir, sample_size=50000, random_state=42, logger=None):
    """Loads test data, applies stratified sampling, and converts to tensors."""
    data_dir = Path(data_dir)
    logger.log(f"Loading test dataset from '{data_dir}'...", header=True)

    X_test_df = pd.read_parquet(data_dir / "X_test.parquet")
    y_test_df = pd.read_parquet(data_dir / "y_test.parquet")

    with open(data_dir / "label_encoding.json", "r", encoding="utf-8") as f:
        encoding_data = json.load(f)
    class_to_idx = encoding_data["class_to_idx"]
    unique_classes = encoding_data["classes"]

    y_test_encoded = y_test_df["Label"].map(class_to_idx).to_numpy(dtype=np.int64)
    X_test_np = np.ascontiguousarray(X_test_df.to_numpy(dtype=np.float32))

    total_test = len(y_test_encoded)
    if sample_size and sample_size < total_test:
        logger.log(f"Drawing stratified sample of {sample_size:,} flows from {total_test:,} total test flows...")
        X_sample, y_sample = stratified_sample(
            X_test_np, y_test_encoded,
            sample_size=sample_size,
            random_state=random_state
        )
    else:
        logger.log(f"Using full test set of {total_test:,} flows...")
        X_sample, y_sample = X_test_np, y_test_encoded

    del X_test_df, y_test_df, X_test_np, y_test_encoded
    gc.collect()

    X_tensor = torch.from_numpy(X_sample)
    y_tensor = torch.from_numpy(y_sample)

    logger.log(f"Evaluation sample tensor shape: X = {X_tensor.shape}, y = {y_tensor.shape}")
    return X_tensor, y_tensor, unique_classes, class_to_idx


def load_masks_and_bounds(mask_path, device):
    """Loads feasibility mask vectors and bounds tensors onto compute device."""
    with open(mask_path, "r", encoding="utf-8") as f:
        mask_data = json.load(f)

    mask_t = torch.tensor(mask_data["mask_vector"], dtype=torch.float32, device=device)
    cum_mask_t = torch.tensor(mask_data["cumulative_mask_vector"], dtype=torch.float32, device=device)
    min_t = torch.tensor(mask_data["std_emp_min_vector"], dtype=torch.float32, device=device)
    max_t = torch.tensor(mask_data["std_emp_max_vector"], dtype=torch.float32, device=device)

    return mask_t, cum_mask_t, min_t, max_t, mask_data


def evaluate_clean(model, dataloader, device, unique_classes, results_dir, logger):
    """Evaluates clean baseline of Model B and saves classification report and confusion matrix."""
    logger.log("1. Clean Baseline Evaluation on Held-Out Test Sample", header=True)
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_b, y_b in dataloader:
            X_b = X_b.to(device)
            logits = model(X_b)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_b.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    logger.log(f"  Clean Accuracy:    {acc*100:.3f}%")
    logger.log(f"  Clean Macro-F1:    {macro_f1:.4f}")
    logger.log(f"  Clean Weighted-F1: {weighted_f1:.4f}")

    # Full classification report
    report_dict = classification_report(
        all_targets, all_preds,
        target_names=unique_classes,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        all_targets, all_preds,
        target_names=unique_classes,
        digits=4,
        zero_division=0
    )
    logger.log("Clean Classification Report:\n" + report_text)

    # Save classification report text
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "robust_model_classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AdvRoNIDS Model B (Robust 1D-CNN) - Clean Test Set Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report_text)
        f.write(f"\nClean Test Accuracy:    {acc*100:.3f}%\n")
        f.write(f"Clean Test Macro-F1:    {macro_f1:.4f}\n")
        f.write(f"Clean Test Weighted-F1: {weighted_f1:.4f}\n")
    logger.log(f"Saved clean classification report to '{report_file}'.")

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    cm_df = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
    cm_csv_path = results_dir / "robust_model_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    cm_png_path = results_dir / "robust_model_confusion_matrix.png"
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=unique_classes,
        yticklabels=unique_classes,
        cbar_kws={"label": "Normalized Recall Rate"}
    )
    plt.title("AdvRoNIDS Model B Robust 1D-CNN: Normalized Clean Test Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    logger.log(f"Saved clean confusion matrix heatmap to '{cm_png_path}'.")

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report_dict
    }


def evaluate_fgsm(model, dataloader, epsilon, mask_t, cum_mask_t, min_t, max_t, device, logger):
    """Evaluates Robust Model B against FGSM (constrained vs unconstrained) at epsilon=0.10."""
    logger.log(f"2. FGSM Vulnerability Evaluation (epsilon = {epsilon})", header=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    results = {}
    for constrained in [False, True]:
        tag = "Constrained (Pi_S)" if constrained else "Unconstrained (Naive)"
        all_preds, all_targets = [], []
        start_time = time.time()

        for X_b, y_b in dataloader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)

            X_adv = fgsm_attack(
                model=model,
                x=X_b,
                y=y_b,
                epsilon=epsilon,
                criterion=criterion,
                constrained=constrained,
                mask=mask_t,
                cumulative_mask=cum_mask_t,
                emp_min=min_t,
                emp_max=max_t
            )

            with torch.no_grad():
                logits = model(X_adv)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_b.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
        elapsed = time.time() - start_time

        logger.log(f"  FGSM [{tag:<22}] | Accuracy: {acc*100:6.3f}% | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f} | Time: {elapsed:.1f}s")
        key = "constrained" if constrained else "unconstrained"
        results[key] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "time_seconds": float(elapsed)
        }

    return results


def evaluate_pgd_sweep(model, dataloader, epsilons, num_steps, mask_t, cum_mask_t, min_t, max_t,
                       unique_classes, results_dir, device, logger):
    """
    Evaluates Robust Model B against PGD-7 sweep across epsilons for both constrained and unconstrained attacks.
    Generates adversarial classification report and confusion matrix at epsilon=0.10 constrained.
    """
    logger.log(f"3. Projected Gradient Descent (PGD-{num_steps}) Robustness Sweep across epsilons", header=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    results_dir = Path(results_dir)

    sweep_results = {
        "epsilons": epsilons,
        "num_steps": num_steps,
        "constrained": [],
        "unconstrained": []
    }

    for eps in epsilons:
        alpha = eps / 4.0
        logger.log(f"\n--- Testing Robust Model B under PGD-{num_steps} with epsilon = {eps:.2f} (alpha = {alpha:.4f}) ---")

        # 1. Unconstrained PGD
        start_uncon = time.time()
        all_preds_uncon, all_targets_uncon = [], []

        for X_b, y_b in dataloader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)

            X_adv_uncon = pgd_attack(
                model=model,
                x=X_b,
                y=y_b,
                epsilon=eps,
                alpha=alpha,
                num_steps=num_steps,
                criterion=criterion,
                constrained=False,
                random_start=True
            )

            with torch.no_grad():
                logits = model(X_adv_uncon)
                preds = torch.argmax(logits, dim=1)
                all_preds_uncon.append(preds.cpu().numpy())
                all_targets_uncon.append(y_b.cpu().numpy())

        all_preds_uncon = np.concatenate(all_preds_uncon)
        all_targets_uncon = np.concatenate(all_targets_uncon)

        acc_uncon = accuracy_score(all_targets_uncon, all_preds_uncon)
        f1_uncon = f1_score(all_targets_uncon, all_preds_uncon, average="macro", zero_division=0)
        wf1_uncon = f1_score(all_targets_uncon, all_preds_uncon, average="weighted", zero_division=0)
        time_uncon = time.time() - start_uncon

        logger.log(
            f"  PGD-{num_steps} [Unconstrained] | eps={eps:.2f} | "
            f"Accuracy: {acc_uncon*100:6.3f}% | Macro-F1: {f1_uncon:.4f} | Weighted-F1: {wf1_uncon:.4f} | Time: {time_uncon:.1f}s"
        )
        sweep_results["unconstrained"].append({
            "epsilon": float(eps),
            "accuracy": float(acc_uncon),
            "macro_f1": float(f1_uncon),
            "weighted_f1": float(wf1_uncon),
            "time_seconds": float(time_uncon)
        })

        # 2. Constrained PGD (Pi_S)
        start_con = time.time()
        all_preds_con, all_targets_con = [], []

        for X_b, y_b in dataloader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)

            X_adv_con = pgd_attack(
                model=model,
                x=X_b,
                y=y_b,
                epsilon=eps,
                alpha=alpha,
                num_steps=num_steps,
                criterion=criterion,
                constrained=True,
                mask=mask_t,
                cumulative_mask=cum_mask_t,
                emp_min=min_t,
                emp_max=max_t,
                random_start=True
            )

            with torch.no_grad():
                logits = model(X_adv_con)
                preds = torch.argmax(logits, dim=1)
                all_preds_con.append(preds.cpu().numpy())
                all_targets_con.append(y_b.cpu().numpy())

        all_preds_con = np.concatenate(all_preds_con)
        all_targets_con = np.concatenate(all_targets_con)

        acc_con = accuracy_score(all_targets_con, all_preds_con)
        f1_con = f1_score(all_targets_con, all_preds_con, average="macro", zero_division=0)
        wf1_con = f1_score(all_targets_con, all_preds_con, average="weighted", zero_division=0)
        time_con = time.time() - start_con

        logger.log(
            f"  PGD-{num_steps} [Constrained Pi_S] | eps={eps:.2f} | "
            f"Accuracy: {acc_con*100:6.3f}% | Macro-F1: {f1_con:.4f} | Weighted-F1: {wf1_con:.4f} | Time: {time_con:.1f}s"
        )
        sweep_results["constrained"].append({
            "epsilon": float(eps),
            "accuracy": float(acc_con),
            "macro_f1": float(f1_con),
            "weighted_f1": float(wf1_con),
            "time_seconds": float(time_con)
        })

        # If eps == 0.10, generate adversarial classification report & confusion matrix
        if abs(eps - 0.10) < 1e-4:
            logger.log("Generating Full Classification Report & Confusion Matrix for Constrained PGD (eps=0.10)...", header=True)
            adv_report_text = classification_report(
                all_targets_con, all_preds_con,
                target_names=unique_classes,
                digits=4,
                zero_division=0
            )
            adv_report_file = results_dir / "robust_model_adv_classification_report.txt"
            with open(adv_report_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("AdvRoNIDS Model B (Robust 1D-CNN) - Constrained PGD-7 (eps=0.10) Classification Report\n")
                f.write("=" * 80 + "\n\n")
                f.write(adv_report_text)
                f.write(f"\nAdversarial Test Accuracy:    {acc_con*100:.3f}%\n")
                f.write(f"Adversarial Test Macro-F1:    {f1_con:.4f}\n")
                f.write(f"Adversarial Test Weighted-F1: {wf1_con:.4f}\n")
            logger.log(f"Saved adversarial classification report to '{adv_report_file}'.")

            # Adversarial Confusion Matrix
            cm_adv = confusion_matrix(all_targets_con, all_preds_con)
            cm_adv_df = pd.DataFrame(cm_adv, index=unique_classes, columns=unique_classes)
            cm_adv_df.to_csv(results_dir / "robust_model_adv_confusion_matrix.csv")

            cm_adv_png = results_dir / "robust_model_adv_confusion_matrix.png"
            plt.figure(figsize=(12, 10))
            cm_adv_norm = cm_adv.astype("float") / (cm_adv.sum(axis=1)[:, np.newaxis] + 1e-9)
            sns.heatmap(
                cm_adv_norm,
                annot=True,
                fmt=".2f",
                cmap="Oranges",
                xticklabels=unique_classes,
                yticklabels=unique_classes,
                cbar_kws={"label": "Normalized Recall Rate"}
            )
            plt.title("AdvRoNIDS Model B: Normalized Confusion Matrix under Constrained PGD (ε=0.10)", fontsize=13, fontweight="bold", pad=15)
            plt.xlabel("Predicted Class", fontsize=11)
            plt.ylabel("True Class", fontsize=11)
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            plt.savefig(cm_adv_png, dpi=300)
            plt.close()
            logger.log(f"Saved adversarial confusion matrix heatmap to '{cm_adv_png}'.")

    return sweep_results


def build_comparison_table_and_plots(clean_report_path, robust_clean, robust_fgsm, robust_pgd, results_dir, logger):
    """
    Builds the central clean vs. robust comparative table (CSV) and 4-line robustness curves figure.
    """
    logger.log("4. Generating Central Clean Model A vs Robust Model B Comparison", section=True)
    results_dir = Path(results_dir)

    # Load Clean Model A benchmark report
    with open(clean_report_path, "r", encoding="utf-8") as f:
        clean_data = json.load(f)

    clean_base = clean_data["clean_baseline"]
    clean_fgsm = clean_data["fgsm_epsilon_0_1"]
    clean_pgd = clean_data["pgd_sweep"]

    # Construct comparison records
    records = []

    # 1. Clean Baseline (eps=0.0)
    acc_clean_model_clean = clean_base["accuracy"] * 100.0
    acc_rob_model_clean = robust_clean["accuracy"] * 100.0
    records.append({
        "Model": "Clean Model A",
        "Attack": "Clean Baseline",
        "Epsilon": 0.0,
        "Constraint": "N/A",
        "Accuracy_Pct": round(acc_clean_model_clean, 3),
        "Macro_F1": round(clean_base["macro_f1"], 4),
        "Weighted_F1": round(clean_base["weighted_f1"], 4),
        "Delta_Accuracy_Pct": 0.0
    })
    records.append({
        "Model": "Robust Model B",
        "Attack": "Clean Baseline",
        "Epsilon": 0.0,
        "Constraint": "N/A",
        "Accuracy_Pct": round(acc_rob_model_clean, 3),
        "Macro_F1": round(robust_clean["macro_f1"], 4),
        "Weighted_F1": round(robust_clean["weighted_f1"], 4),
        "Delta_Accuracy_Pct": round(acc_rob_model_clean - acc_clean_model_clean, 3)
    })

    # 2. FGSM (eps=0.10)
    for con_tag, con_name in [("unconstrained", "Unconstrained"), ("constrained", "Constrained (Pi_S)")]:
        c_acc = clean_fgsm[con_tag]["accuracy"] * 100.0
        r_acc = robust_fgsm[con_tag]["accuracy"] * 100.0
        records.append({
            "Model": "Clean Model A",
            "Attack": "FGSM",
            "Epsilon": 0.10,
            "Constraint": con_name,
            "Accuracy_Pct": round(c_acc, 3),
            "Macro_F1": round(clean_fgsm[con_tag]["macro_f1"], 4),
            "Weighted_F1": round(clean_fgsm[con_tag]["weighted_f1"], 4),
            "Delta_Accuracy_Pct": 0.0
        })
        records.append({
            "Model": "Robust Model B",
            "Attack": "FGSM",
            "Epsilon": 0.10,
            "Constraint": con_name,
            "Accuracy_Pct": round(r_acc, 3),
            "Macro_F1": round(robust_fgsm[con_tag]["macro_f1"], 4),
            "Weighted_F1": round(robust_fgsm[con_tag]["weighted_f1"], 4),
            "Delta_Accuracy_Pct": round(r_acc - c_acc, 3)
        })

    # 3. PGD Sweep across epsilons
    eps_list = clean_pgd["epsilons"]
    for i, eps in enumerate(eps_list):
        # Constrained
        c_p_con = clean_pgd["constrained"][i]
        r_p_con = robust_pgd["constrained"][i]
        c_con_acc = c_p_con["accuracy"] * 100.0
        r_con_acc = r_p_con["accuracy"] * 100.0

        records.append({
            "Model": "Clean Model A",
            "Attack": "PGD-7",
            "Epsilon": eps,
            "Constraint": "Constrained (Pi_S)",
            "Accuracy_Pct": round(c_con_acc, 3),
            "Macro_F1": round(c_p_con["macro_f1"], 4),
            "Weighted_F1": round(c_p_con["weighted_f1"], 4),
            "Delta_Accuracy_Pct": 0.0
        })
        records.append({
            "Model": "Robust Model B",
            "Attack": "PGD-7",
            "Epsilon": eps,
            "Constraint": "Constrained (Pi_S)",
            "Accuracy_Pct": round(r_con_acc, 3),
            "Macro_F1": round(r_p_con["macro_f1"], 4),
            "Weighted_F1": round(r_p_con["weighted_f1"], 4),
            "Delta_Accuracy_Pct": round(r_con_acc - c_con_acc, 3)
        })

        # Unconstrained
        c_p_unc = clean_pgd["unconstrained"][i]
        r_p_unc = robust_pgd["unconstrained"][i]
        c_unc_acc = c_p_unc["accuracy"] * 100.0
        r_unc_acc = r_p_unc["accuracy"] * 100.0

        records.append({
            "Model": "Clean Model A",
            "Attack": "PGD-7",
            "Epsilon": eps,
            "Constraint": "Unconstrained",
            "Accuracy_Pct": round(c_unc_acc, 3),
            "Macro_F1": round(c_p_unc["macro_f1"], 4),
            "Weighted_F1": round(c_p_unc["weighted_f1"], 4),
            "Delta_Accuracy_Pct": 0.0
        })
        records.append({
            "Model": "Robust Model B",
            "Attack": "PGD-7",
            "Epsilon": eps,
            "Constraint": "Unconstrained",
            "Accuracy_Pct": round(r_unc_acc, 3),
            "Macro_F1": round(r_p_unc["macro_f1"], 4),
            "Weighted_F1": round(r_p_unc["weighted_f1"], 4),
            "Delta_Accuracy_Pct": round(r_unc_acc - c_unc_acc, 3)
        })

    df_comp = pd.DataFrame(records)
    csv_path = results_dir / "clean_vs_robust_comparison.csv"
    df_comp.to_csv(csv_path, index=False)
    logger.log(f"Saved comparative metrics CSV to '{csv_path}'.")

    # Display comparison table
    logger.log("\n" + "=" * 100)
    logger.log("CENTRAL BENCHMARK: CLEAN MODEL A vs. ROBUST MODEL B (AdvRoNIDS Section 8)")
    logger.log("=" * 100)
    logger.log(f"{'Attack':<16} | {'Eps':<5} | {'Constraint':<20} | {'Clean Model A':<14} | {'Robust Model B':<14} | {'Robust Improvement (Δ)':<22}")
    logger.log("-" * 100)

    # Clean
    logger.log(f"{'Clean Baseline':<16} | {'0.00':<5} | {'N/A':<20} | {acc_clean_model_clean:6.2f}% (F1:{clean_base['macro_f1']:.3f}) | {acc_rob_model_clean:6.2f}% (F1:{robust_clean['macro_f1']:.3f}) | {acc_rob_model_clean - acc_clean_model_clean:+6.2f}% pts (Trade-off)")

    # FGSM
    for tag, name in [("constrained", "Constrained (Pi_S)"), ("unconstrained", "Unconstrained")]:
        c_acc = clean_fgsm[tag]["accuracy"] * 100.0
        r_acc = robust_fgsm[tag]["accuracy"] * 100.0
        logger.log(f"{'FGSM':<16} | {'0.10':<5} | {name:<20} | {c_acc:6.2f}% (F1:{clean_fgsm[tag]['macro_f1']:.3f}) | {r_acc:6.2f}% (F1:{robust_fgsm[tag]['macro_f1']:.3f}) | {r_acc - c_acc:+6.2f}% pts")

    # PGD
    for i, eps in enumerate(eps_list):
        c_c = clean_pgd["constrained"][i]["accuracy"] * 100.0
        r_c = robust_pgd["constrained"][i]["accuracy"] * 100.0
        c_u = clean_pgd["unconstrained"][i]["accuracy"] * 100.0
        r_u = robust_pgd["unconstrained"][i]["accuracy"] * 100.0

        logger.log(f"{'PGD-7':<16} | {eps:<5.2f} | {'Constrained (Pi_S)':<20} | {c_c:6.2f}% (F1:{clean_pgd['constrained'][i]['macro_f1']:.3f}) | {r_c:6.2f}% (F1:{robust_pgd['constrained'][i]['macro_f1']:.3f}) | {r_c - c_c:+6.2f}% pts [PRIMARY CLAIM]")
        logger.log(f"{'PGD-7':<16} | {eps:<5.2f} | {'Unconstrained':<20} | {c_u:6.2f}% (F1:{clean_pgd['unconstrained'][i]['macro_f1']:.3f}) | {r_u:6.2f}% (F1:{robust_pgd['unconstrained'][i]['macro_f1']:.3f}) | {r_u - c_u:+6.2f}% pts")

    logger.log("=" * 100)

    # --- 4-Line Robustness Curves Plot (Central Paper Figure) ---
    plot_path = results_dir / "clean_vs_robust_robustness_curves.png"
    all_eps = [0.0] + eps_list

    # Accuracy lines
    acc_clean_con = [clean_base["accuracy"] * 100.0] + [res["accuracy"] * 100.0 for res in clean_pgd["constrained"]]
    acc_clean_unc = [clean_base["accuracy"] * 100.0] + [res["accuracy"] * 100.0 for res in clean_pgd["unconstrained"]]
    acc_rob_con = [robust_clean["accuracy"] * 100.0] + [res["accuracy"] * 100.0 for res in robust_pgd["constrained"]]
    acc_rob_unc = [robust_clean["accuracy"] * 100.0] + [res["accuracy"] * 100.0 for res in robust_pgd["unconstrained"]]

    # Macro-F1 lines
    f1_clean_con = [clean_base["macro_f1"]] + [res["macro_f1"] for res in clean_pgd["constrained"]]
    f1_clean_unc = [clean_base["macro_f1"]] + [res["macro_f1"] for res in clean_pgd["unconstrained"]]
    f1_rob_con = [robust_clean["macro_f1"]] + [res["macro_f1"] for res in robust_pgd["constrained"]]
    f1_rob_unc = [robust_clean["macro_f1"]] + [res["macro_f1"] for res in robust_pgd["unconstrained"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Classification Accuracy (%)
    ax1.plot(all_eps, acc_rob_con, "o-", color="#1b9e77", linewidth=2.8, markersize=8, label="Model B (Robust) - Constrained PGD (Π_S Real Domain)")
    ax1.plot(all_eps, acc_rob_unc, "s--", color="#7570b3", linewidth=2.2, markersize=7, label="Model B (Robust) - Unconstrained PGD (Naive L∞)")
    ax1.plot(all_eps, acc_clean_con, "^-.", color="#d95f02", linewidth=2.2, markersize=7, label="Model A (Clean) - Constrained PGD (Π_S Real Domain)")
    ax1.plot(all_eps, acc_clean_unc, "x:", color="#e7298a", linewidth=2.2, markersize=7, label="Model A (Clean) - Unconstrained PGD (Naive L∞)")

    # Highlight training budget epsilon = 0.10
    ax1.axvline(x=0.10, color="gray", linestyle=":", alpha=0.7, label="Training Budget (ε = 0.10)")

    ax1.set_title("AdvRoNIDS Central Benchmark: Accuracy vs Perturbation Budget ε", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Perturbation Budget (ε in Standardized L∞ Space)", fontsize=11)
    ax1.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax1.set_xlim(-0.01, 0.21)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="lower left", framealpha=0.95)

    # Subplot 2: Macro-F1 Score
    ax2.plot(all_eps, f1_rob_con, "o-", color="#1b9e77", linewidth=2.8, markersize=8, label="Model B (Robust) - Constrained PGD (Π_S Real Domain)")
    ax2.plot(all_eps, f1_rob_unc, "s--", color="#7570b3", linewidth=2.2, markersize=7, label="Model B (Robust) - Unconstrained PGD (Naive L∞)")
    ax2.plot(all_eps, f1_clean_con, "^-.", color="#d95f02", linewidth=2.2, markersize=7, label="Model A (Clean) - Constrained PGD (Π_S Real Domain)")
    ax2.plot(all_eps, f1_clean_unc, "x:", color="#e7298a", linewidth=2.2, markersize=7, label="Model A (Clean) - Unconstrained PGD (Naive L∞)")

    ax2.axvline(x=0.10, color="gray", linestyle=":", alpha=0.7, label="Training Budget (ε = 0.10)")

    ax2.set_title("AdvRoNIDS Central Benchmark: Macro-F1 Degradation vs Budget ε", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Perturbation Budget (ε in Standardized L∞ Space)", fontsize=11)
    ax2.set_ylabel("Macro-F1 Score", fontsize=11)
    ax2.set_xlim(-0.01, 0.21)
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9, loc="lower left", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.log(f"Saved central 4-line robustness curves plot to '{plot_path}'.")

    return df_comp


def main():
    parser = argparse.ArgumentParser(description="Evaluate AdvRoNIDS Robust Model B & Benchmark Clean vs Robust")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Path to processed parquet data")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory to save evaluation results")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/robust_model_best.pt", help="Robust model checkpoint")
    parser.add_argument("--mask-path", type=str, default="./data/processed/feasibility_mask.json", help="Feasibility mask JSON path")
    parser.add_argument("--clean-report-path", type=str, default="./results/attack_vulnerability_report.json", help="Clean model vulnerability report JSON")
    parser.add_argument("--doc-path", type=str, default="project_documentation.txt", help="Documentation file path")
    parser.add_argument("--sample-size", type=int, default=50000, help="Stratified test sample size (matches Phase 3)")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument("--num-steps", type=int, default=7, help="PGD iteration steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        cpu_cores = os.cpu_count() or 4
        torch.set_num_threads(cpu_cores)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = DetailedLogger(doc_path=args.doc_path)

    logger.log("Phase 4: Robust Model B Post-Training Benchmark & Clean vs Robust Comparison", section=True)
    logger.log(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else f'Host CPU with {torch.get_num_threads()} threads'})")

    # 1. Ingest Held-Out Test Split
    X_test_tensor, y_test_tensor, unique_classes, class_to_idx = load_test_sample(
        args.data_dir, sample_size=args.sample_size, random_state=args.seed, logger=logger
    )
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Feasibility Masks
    mask_t, cum_mask_t, min_t, max_t, mask_metadata = load_masks_and_bounds(args.mask_path, device)

    # 3. Load Trained Robust Model B
    logger.log(f"Loading trained Robust Model B checkpoint from '{args.checkpoint_path}'...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = AdvRoNIDS_CNN(num_classes=len(unique_classes), in_channels=1, input_features=71, dropout_rate=0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.log(f"Loaded Model B checkpoint from Epoch {checkpoint.get('epoch', 'N/A')} (Val Robust Macro-F1: {checkpoint.get('val_robust_macro_f1', 0.0):.4f}).")

    # 4. Clean Baseline
    clean_res = evaluate_clean(model, test_loader, device, unique_classes, args.results_dir, logger)

    # 5. FGSM Evaluation (epsilon = 0.10)
    fgsm_res = evaluate_fgsm(model, test_loader, epsilon=0.10, mask_t=mask_t, cum_mask_t=cum_mask_t,
                             min_t=min_t, max_t=max_t, device=device, logger=logger)

    # 6. PGD Sweep (epsilons = [0.01, 0.05, 0.10, 0.20])
    epsilons = [0.01, 0.05, 0.10, 0.20]
    pgd_res = evaluate_pgd_sweep(
        model=model,
        dataloader=test_loader,
        epsilons=epsilons,
        num_steps=args.num_steps,
        mask_t=mask_t,
        cum_mask_t=cum_mask_t,
        min_t=min_t,
        max_t=max_t,
        unique_classes=unique_classes,
        results_dir=args.results_dir,
        device=device,
        logger=logger
    )

    # 7. Save Comprehensive Robust Model Metrics JSON
    robust_metrics = {
        "model_name": "AdvRoNIDS_ModelB_Robust_1D_CNN",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_samples": len(X_test_tensor),
        "clean_baseline": clean_res,
        "fgsm_epsilon_0_1": fgsm_res,
        "pgd_sweep": pgd_res
    }
    metrics_path = Path(args.results_dir) / "robust_model_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(robust_metrics, f, indent=2)
    logger.log(f"Saved robust model test metrics to '{metrics_path}'.")

    # 8. Build Central Comparison Table & 4-Line Robustness Plot
    build_comparison_table_and_plots(
        clean_report_path=args.clean_report_path,
        robust_clean=clean_res,
        robust_fgsm=fgsm_res,
        robust_pgd=pgd_res,
        results_dir=args.results_dir,
        logger=logger
    )

    logger.log("Robust Model B Benchmark & Clean vs. Robust Comparison Complete!", section=True)


if __name__ == "__main__":
    main()
