#!/usr/bin/env python3
"""
AdvRoNIDS: Min-Max Adversarial Training for Robust Classifier (Model B)
Proposal References: Section 6.4.4 (Min-Max Adversarial Training Formulation),
                     Section 6.5 (Phase 1 Step 3), Section 6.2.4 (Hyperparameters)

Mathematical Formulation:
    min_theta E_{(x,y)~D} [ max_{delta in S} L(f_theta(x + delta), y) ]

Where:
    - Inner maximization: Solved dynamically per batch using 7-step domain-constrained
      PGD (project_feasible Pi_S operator enforcing frozen feature freezing, non-negative
      cumulative counter monotonicity, and standardized empirical bounds).
    - Outer minimization: Stochastic optimization of classifier parameters theta using
      Adam (lr=1e-3, batch size=128) over adversarial examples x_adv with class-balanced
      CrossEntropyLoss.
    - Early stopping & Checkpoint criterion: Monitored on Validation Robust Macro-F1
      under constrained PGD at epsilon=0.10 (patience=5).
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

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

# Ensure project modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import AdvRoNIDS_CNN
from attacks import pgd_attack, project_feasible

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


def prevent_sleep():
    """Prevents Windows system sleep/hibernation during long training jobs."""
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception:
        pass



class RobustLogger:
    """Logs pipeline execution to stdout and appends detailed logs to project_documentation.txt."""

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


def load_datasets(data_dir, val_sample_size=50000, seed=42, logger=None):
    """Loads training and validation splits from parquet files and constructs PyTorch tensors."""
    logger.log("Phase 4 - Step 1: Ingesting Data & Preparing Adversarial Training Tensors", section=True)
    data_dir = Path(data_dir)

    logger.log(f"Reading training and validation parquet splits from '{data_dir}'...")
    X_train_df = pd.read_parquet(data_dir / "X_train.parquet")
    y_train_df = pd.read_parquet(data_dir / "y_train.parquet")
    X_val_df = pd.read_parquet(data_dir / "X_val.parquet")
    y_val_df = pd.read_parquet(data_dir / "y_val.parquet")

    # Load canonical label encoding
    encoding_path = data_dir / "label_encoding.json"
    with open(encoding_path, "r", encoding="utf-8") as f:
        encoding_data = json.load(f)
    class_to_idx = encoding_data["class_to_idx"]
    unique_classes = encoding_data["classes"]
    num_classes = len(unique_classes)

    logger.log(f"Loaded label encoding: {num_classes} classes.")
    logger.log(f"  Train: X shape = {X_train_df.shape}, y shape = {y_train_df.shape}")
    logger.log(f"  Val:   X shape = {X_val_df.shape}, y shape = {y_val_df.shape}")

    # Encode labels
    y_train_encoded = y_train_df["Label"].map(class_to_idx).to_numpy(dtype=np.int64)
    y_val_encoded = y_val_df["Label"].map(class_to_idx).to_numpy(dtype=np.int64)

    X_train_np = np.ascontiguousarray(X_train_df.to_numpy(dtype=np.float32))
    X_val_np = np.ascontiguousarray(X_val_df.to_numpy(dtype=np.float32))

    del X_train_df, y_train_df, X_val_df, y_val_df
    gc.collect()

    # Create stratified validation sample for fast per-epoch robust evaluation
    total_val = len(y_val_encoded)
    if val_sample_size and val_sample_size < total_val:
        logger.log(f"Drawing stratified validation sample of {val_sample_size:,} flows for robust evaluation (from {total_val:,} total val flows)...")
        X_val_eval_np, y_val_eval_np = stratified_sample(
            X_val_np, y_val_encoded,
            sample_size=val_sample_size,
            random_state=seed
        )
    else:
        logger.log(f"Using full validation set of {total_val:,} flows for evaluation...")
        X_val_eval_np, y_val_eval_np = X_val_np, y_val_encoded

    # Convert to Tensors
    X_train_tensor = torch.from_numpy(X_train_np)
    y_train_tensor = torch.from_numpy(y_train_encoded)
    X_val_tensor = torch.from_numpy(X_val_np)
    y_val_tensor = torch.from_numpy(y_val_encoded)
    X_val_eval_tensor = torch.from_numpy(X_val_eval_np)
    y_val_eval_tensor = torch.from_numpy(y_val_eval_np)

    logger.log(f"Allocated Tensors:")
    logger.log(f"  X_train:     {X_train_tensor.shape} ({X_train_tensor.element_size() * X_train_tensor.nelement() / (1024**2):.1f} MB)")
    logger.log(f"  X_val:       {X_val_tensor.shape} ({X_val_tensor.element_size() * X_val_tensor.nelement() / (1024**2):.1f} MB)")
    logger.log(f"  X_val_eval:  {X_val_eval_tensor.shape} ({X_val_eval_tensor.element_size() * X_val_eval_tensor.nelement() / (1024**2):.1f} MB)")

    return (X_train_tensor, y_train_tensor,
            X_val_tensor, y_val_tensor,
            X_val_eval_tensor, y_val_eval_tensor,
            unique_classes, class_to_idx)


def load_feasibility_masks(mask_path, device, logger):
    """Loads feasibility mask vectors and standardized empirical bounds."""
    logger.log(f"Loading feasibility masks & empirical bounds from '{mask_path}'...", header=True)
    with open(mask_path, "r", encoding="utf-8") as f:
        mask_data = json.load(f)

    mask_t = torch.tensor(mask_data["mask_vector"], dtype=torch.float32, device=device)
    cum_mask_t = torch.tensor(mask_data["cumulative_mask_vector"], dtype=torch.float32, device=device)
    min_t = torch.tensor(mask_data["std_emp_min_vector"], dtype=torch.float32, device=device)
    max_t = torch.tensor(mask_data["std_emp_max_vector"], dtype=torch.float32, device=device)

    logger.log(f"Feasibility Configuration: {mask_data['frozen_count']} Frozen features, "
               f"{mask_data['perturbable_count']} Perturbable features, "
               f"{mask_data['cumulative_counter_count']} Cumulative counters.")
    return mask_t, cum_mask_t, min_t, max_t


def compute_balanced_weights(y_train_tensor, num_classes, unique_classes, logger):
    """Computes exact inverse class frequency weights matching Phase 2."""
    logger.log("Phase 4 - Step 2: Computing Inverse Class Frequency Weights for Loss", section=True)
    y_train_np = y_train_tensor.numpy()
    classes_arr = np.arange(num_classes)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_arr,
        y=y_train_np
    )

    logger.log("Computed class weights (weight_c = N / (C * count_c)):")
    for idx, (cls_name, weight) in enumerate(zip(unique_classes, class_weights)):
        cnt = (y_train_np == idx).sum()
        logger.log(f"  [{idx:2d}] {cls_name:<26} | Count: {cnt:9,d} | Weight: {weight:12.4f}")

    return torch.tensor(class_weights, dtype=torch.float32)


def evaluate_clean_validation(model, dataloader, criterion, device):
    """Evaluates loss, accuracy, and macro-F1 on clean validation data."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, macro_f1, weighted_f1


def evaluate_robust_validation(model, dataloader, criterion, epsilon, alpha, num_steps,
                                mask_t, cum_mask_t, min_t, max_t, device):
    """
    Evaluates robust accuracy and robust macro-F1 on validation data under constrained PGD.
    This provides the primary early-stopping and checkpointing signal (Section 6.2.4 & 6.4.4).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Generate constrained adversarial validation examples
        X_adv = pgd_attack(
            model=model,
            x=X_batch,
            y=y_batch,
            epsilon=epsilon,
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
            logits = model(X_adv)
            loss = criterion(logits, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, macro_f1, weighted_f1


def train_robust_model(model, train_loader, val_clean_loader, val_robust_loader,
                       criterion, optimizer, device, train_eps=0.10, train_alpha=0.025,
                       train_steps=7, mask_t=None, cum_mask_t=None, min_t=None, max_t=None,
                       num_epochs=30, patience=5, checkpoint_path="./checkpoints/robust_model_best.pt",
                       results_dir="./results", resume=False, logger=None):
    """
    Executes min-max adversarial training loop for AdvRoNIDS Model B.
    Inner Maximization: 7-step domain-constrained PGD (Pi_S) generated dynamically per batch.
    Outer Minimization: CrossEntropy loss on adversarial batch with inverse class weighting.
    Early stopping and best checkpoint tracked strictly on Validation Robust Macro-F1.
    """
    logger.log("Phase 4 - Step 3: Min-Max Adversarial Training Loop (Model B)", section=True)
    logger.log(
        f"Training configuration:\n"
        f"  Epochs: {num_epochs} | Early Stopping Patience: {patience} (on Val Robust Macro-F1)\n"
        f"  Batch Size: {train_loader.batch_size} | Optimizer: Adam (lr={optimizer.param_groups[0]['lr']})\n"
        f"  Inner PGD (Dynamic per Batch): steps={train_steps}, epsilon={train_eps}, alpha={train_alpha}, constrained=True (Pi_S)\n"
        f"  Checkpoint Path: '{checkpoint_path}'"
    )

    prevent_sleep()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = checkpoint_path.parent
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoints_dir / "robust_model_last.pt"
    jsonl_history_path = results_dir / "robust_model_training_history.jsonl"
    json_history_path = results_dir / "robust_model_training_history.json"

    history = {
        "epoch": [],
        "train_adv_loss": [],
        "val_clean_loss": [],
        "val_clean_acc": [],
        "val_clean_macro_f1": [],
        "val_robust_loss": [],
        "val_robust_acc": [],
        "val_robust_macro_f1": [],
        "epoch_time": []
    }

    start_epoch = 1
    best_val_robust_macro_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    # Resume capability if requested
    if resume:
        target_ckpt = None
        if last_checkpoint_path.exists():
            target_ckpt = last_checkpoint_path
        elif checkpoint_path.exists():
            target_ckpt = checkpoint_path

        if target_ckpt:
            logger.log(f"Resuming training from checkpoint '{target_ckpt}'...")
            ckpt = torch.load(target_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass
            start_epoch = ckpt["epoch"] + 1
            best_val_robust_macro_f1 = ckpt.get("best_val_robust_macro_f1", ckpt.get("val_robust_macro_f1", -1.0))
            best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", -1))
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)

            # Load existing history if available
            if json_history_path.exists():
                try:
                    with open(json_history_path, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except Exception:
                    pass
            logger.log(f"Resumed at Epoch {start_epoch} (Previous Best Epoch: {best_epoch}, Best Val Robust Macro-F1: {best_val_robust_macro_f1:.4f}, Patience: {epochs_no_improve}/{patience}).")

    total_train_start = time.time()
    num_batches = len(train_loader)

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        running_adv_loss = 0.0
        train_samples = 0

        logger.log(f"\n--- Epoch [{epoch:02d}/{num_epochs:02d}] Starting Training Loop ({num_batches:,} batches) ---")
        batch_log_interval = max(1, num_batches // 5)  # Log ~5 times per epoch

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader, 1):
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # --- 1. Inner Maximization: Dynamic PGD adversarial example generation ---
            model.eval()
            X_adv = pgd_attack(
                model=model,
                x=X_batch,
                y=y_batch,
                epsilon=train_eps,
                alpha=train_alpha,
                num_steps=train_steps,
                criterion=criterion,
                constrained=True,
                mask=mask_t,
                cumulative_mask=cum_mask_t,
                emp_min=min_t,
                emp_max=max_t,
                random_start=True
            )

            # --- 2. Outer Minimization: Forward/backward on adversarial batch ---
            model.train()
            optimizer.zero_grad()
            logits = model(X_adv)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            running_adv_loss += loss.item() * batch_size
            train_samples += batch_size

            if batch_idx % batch_log_interval == 0 or batch_idx == num_batches:
                cur_elapsed = time.time() - epoch_start
                cur_loss = running_adv_loss / train_samples
                est_epoch_total = (cur_elapsed / batch_idx) * num_batches
                logger.log(f"  Epoch [{epoch:02d}/{num_epochs:02d}] Batch [{batch_idx:5d}/{num_batches:5d}] | Adv Train Loss: {cur_loss:.4f} | Elapsed: {cur_elapsed:.1f}s (Est Epoch: {est_epoch_total:.1f}s)")

        train_adv_loss = running_adv_loss / train_samples

        # --- Validation: Clean and Robust Evaluation ---
        logger.log(f"Evaluating Epoch {epoch} on Clean & Robust Validation Sets...")
        val_clean_loss, val_clean_acc, val_clean_macro_f1, _ = evaluate_clean_validation(
            model, val_clean_loader, criterion, device
        )
        val_rob_loss, val_rob_acc, val_rob_macro_f1, _ = evaluate_robust_validation(
            model, val_robust_loader, criterion,
            epsilon=train_eps, alpha=train_alpha, num_steps=train_steps,
            mask_t=mask_t, cum_mask_t=cum_mask_t, min_t=min_t, max_t=max_t,
            device=device
        )
        epoch_elapsed = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_adv_loss"].append(train_adv_loss)
        history["val_clean_loss"].append(val_clean_loss)
        history["val_clean_acc"].append(val_clean_acc)
        history["val_clean_macro_f1"].append(val_clean_macro_f1)
        history["val_robust_loss"].append(val_rob_loss)
        history["val_robust_acc"].append(val_rob_acc)
        history["val_robust_macro_f1"].append(val_rob_macro_f1)
        history["epoch_time"].append(epoch_elapsed)

        # 1. Immediate per-epoch JSONL streaming append with disk flush
        epoch_record = {
            "epoch": epoch,
            "train_adv_loss": float(train_adv_loss),
            "val_clean_loss": float(val_clean_loss),
            "val_clean_acc": float(val_clean_acc),
            "val_clean_macro_f1": float(val_clean_macro_f1),
            "val_robust_loss": float(val_rob_loss),
            "val_robust_acc": float(val_rob_acc),
            "val_robust_macro_f1": float(val_rob_macro_f1),
            "epoch_time_seconds": float(epoch_elapsed),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(jsonl_history_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(epoch_record) + "\n")
            f_jsonl.flush()

        # 2. Immediate per-epoch JSON update & plot refresh
        with open(json_history_path, "w", encoding="utf-8") as f_json:
            json.dump(history, f_json, indent=2)
            f_json.flush()
        
        try:
            plot_training_curves(history, results_dir, logger, verbose=False)
        except Exception:
            pass

        # Early stopping and checkpointing on Validation Robust Macro-F1
        is_best = val_rob_macro_f1 > best_val_robust_macro_f1
        if is_best:
            best_val_robust_macro_f1 = val_rob_macro_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_robust_macro_f1": val_rob_macro_f1,
                "val_robust_acc": val_rob_acc,
                "val_clean_macro_f1": val_clean_macro_f1,
                "val_clean_acc": val_clean_acc,
                "val_clean_loss": val_clean_loss,
                "train_adv_loss": train_adv_loss
            }, checkpoint_path)
            save_tag = "[BEST ROBUST CHECKPOINT SAVED]"
        else:
            epochs_no_improve += 1
            save_tag = f"[No Improvement ({epochs_no_improve}/{patience})]"

        # Always save latest state for resume guarantee
        torch.save({
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_robust_macro_f1": best_val_robust_macro_f1,
            "epochs_no_improve": epochs_no_improve,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_robust_macro_f1": val_rob_macro_f1,
            "val_robust_acc": val_rob_acc,
            "val_clean_macro_f1": val_clean_macro_f1,
            "val_clean_acc": val_clean_acc,
            "val_clean_loss": val_clean_loss,
            "train_adv_loss": train_adv_loss
        }, last_checkpoint_path)

        logger.log(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] Summary:\n"
            f"  Adv Train Loss:       {train_adv_loss:.4f}\n"
            f"  Val Clean Acc:        {val_clean_acc*100:.2f}% | Val Clean Macro-F1:  {val_clean_macro_f1:.4f} | Val Clean Loss: {val_clean_loss:.4f}\n"
            f"  Val Robust Acc (ε=0.1):{val_rob_acc*100:.2f}% | Val Robust Macro-F1: {val_rob_macro_f1:.4f} | Val Robust Loss: {val_rob_loss:.4f}\n"
            f"  Epoch Duration:       {epoch_elapsed:.1f}s | {save_tag}"
        )

        if epochs_no_improve >= patience:
            logger.log(f"\nEarly stopping triggered after {epoch} epochs (no robust validation macro-F1 improvement for {patience} consecutive epochs).")
            break

    total_train_time = time.time() - total_train_start
    logger.log(
        f"\nMin-Max Adversarial Training Complete!\n"
        f"  Total Duration: {total_train_time/60:.2f} minutes ({total_train_time:.1f}s)\n"
        f"  Best Epoch: {best_epoch} with Val Robust Macro-F1 = {best_val_robust_macro_f1:.4f}\n"
        f"  Saved Checkpoint: '{checkpoint_path}'",
        section=True
    )
    return history, best_epoch


def plot_training_curves(history, results_dir, logger, verbose=True):
    """Plots clean and robust validation metrics over training epochs."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "robust_model_training_curves.png"

    epochs = history["epoch"]
    if len(epochs) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Subplot 1: Loss curves
    ax1.plot(epochs, history["train_adv_loss"], "o-", label="Train Adv Loss (PGD-7)", color="#d62728", linewidth=2)
    ax1.plot(epochs, history["val_clean_loss"], "s--", label="Val Clean Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_robust_loss"], "^-.", label="Val Robust Loss (ε=0.1)", color="#ff7f0e", linewidth=2)
    ax1.set_title("AdvRoNIDS Model B: Adversarial & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Subplot 2: Accuracy & Macro-F1 curves
    ax2.plot(epochs, [a * 100 for a in history["val_clean_acc"]], "s--", label="Val Clean Accuracy (%)", color="#1f77b4", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_robust_acc"]], "o-", label="Val Robust Accuracy (%) [ε=0.1]", color="#2ca02c", linewidth=2.2)
    ax2.plot(epochs, [f * 100 for f in history["val_clean_macro_f1"]], "d:", label="Val Clean Macro-F1 (x100)", color="#9467bd", linewidth=1.8)
    ax2.plot(epochs, [f * 100 for f in history["val_robust_macro_f1"]], "^-", label="Val Robust Macro-F1 (x100) [Criterion]", color="#d62728", linewidth=2.2)
    ax2.set_title("AdvRoNIDS Model B: Clean vs Robust Validation Performance", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score / Percentage (%)", fontsize=11)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    if verbose and logger:
        logger.log(f"Saved robust model training curves plot to '{plot_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Min-Max Adversarial Training for AdvRoNIDS Robust Classifier (Model B)")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Path to processed parquet data")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory to save evaluation results")
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--mask-path", type=str, default="./data/processed/feasibility_mask.json", help="Feasibility mask JSON path")
    parser.add_argument("--doc-path", type=str, default="project_documentation.txt", help="Documentation file path")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for adversarial training (Section 6.2.4)")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs on robust val macro-F1)")
    parser.add_argument("--train-eps", type=float, default=0.10, help="Training perturbation budget epsilon (Section 6.2.4)")
    parser.add_argument("--train-steps", type=int, default=7, help="Training PGD inner-loop steps")
    parser.add_argument("--val-sample-size", type=int, default=50000, help="Stratified sample size for fast per-epoch robust validation")
    parser.add_argument("--resume", action="store_true", help="Resume from last saved checkpoint if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Reproducibility seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set PyTorch thread count for optimal CPU performance
    if not torch.cuda.is_available():
        cpu_cores = os.cpu_count() or 4
        torch.set_num_threads(cpu_cores)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = RobustLogger(doc_path=args.doc_path)

    logger.log("AdvRoNIDS Min-Max Adversarial Training (Model B - Section 6.4.4)", section=True)
    logger.log(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else f'Host CPU with {torch.get_num_threads()} threads'})")

    # 1. Ingest Data & Construct Tensors
    (X_train, y_train,
     X_val, y_val,
     X_val_eval, y_val_eval,
     unique_classes, class_to_idx) = load_datasets(
        args.data_dir, val_sample_size=args.val_sample_size, seed=args.seed, logger=logger
    )
    num_classes = len(unique_classes)

    # 2. DataLoaders
    use_pin = torch.cuda.is_available()
    train_dataset = TensorDataset(X_train, y_train)
    val_clean_dataset = TensorDataset(X_val, y_val)
    val_robust_dataset = TensorDataset(X_val_eval, y_val_eval)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=use_pin)
    val_clean_loader = DataLoader(val_clean_dataset, batch_size=256, shuffle=False, pin_memory=use_pin)
    val_robust_loader = DataLoader(val_robust_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=use_pin)

    # 3. Feasibility Mask & Domain Constraints
    mask_t, cum_mask_t, min_t, max_t = load_feasibility_masks(args.mask_path, device, logger)

    # 4. Balanced Class Weights & Loss
    class_weights = compute_balanced_weights(y_train, num_classes, unique_classes, logger)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 5. Model Architecture & Optimizer
    model = AdvRoNIDS_CNN(num_classes=num_classes, in_channels=1, input_features=71, dropout_rate=0.3)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"AdvRoNIDS_CNN Model B Initialized ({total_params:,} trainable parameters).")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6. Train Model B
    checkpoint_file = Path(args.checkpoints_dir) / "robust_model_best.pt"
    train_alpha = args.train_eps / 4.0

    history, best_epoch = train_robust_model(
        model=model,
        train_loader=train_loader,
        val_clean_loader=val_clean_loader,
        val_robust_loader=val_robust_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_eps=args.train_eps,
        train_alpha=train_alpha,
        train_steps=args.train_steps,
        mask_t=mask_t,
        cum_mask_t=cum_mask_t,
        min_t=min_t,
        max_t=max_t,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint_file,
        results_dir=args.results_dir,
        resume=args.resume,
        logger=logger
    )

    logger.log(f"Robust Model B Training Finished. Best Model Checkpoint at '{checkpoint_file}'.")


if __name__ == "__main__":
    main()

