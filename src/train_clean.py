#!/usr/bin/env python3
"""
AdvRoNIDS: Training the Clean Classifier (Model A)
Proposal References: Section 6.2 (Clean Classifier Architecture & Training),
                     Section 6.5 (Phase 1 Steps 1-2), Section 8 (Evaluation Plan)

This script performs:
1. Ingestion of processed train/val/test splits from ./data/processed/.
2. Consistent label encoding across all splits and export of label_encoding.json.
3. Computation of inverse-class-frequency balanced loss weights.
4. Instantiation of AdvRoNIDS_CNN (1D-CNN) matching Section 6.2.2.
5. Training loop with Adam (lr=1e-3), batch size 256, max 30 epochs,
   early stopping (patience=5 on val loss), and checkpointing best model by val macro-F1.
6. Evaluation on the held-out test split:
   - Accuracy, Macro-F1, Weighted-F1
   - Full per-class classification report
   - Confusion matrix (CSV + PNG heatmap)
   - Training curves plot (Loss & Macro-F1)
   - Metrics export (clean_model_test_metrics.json, clean_model_classification_report.txt)
7. Comprehensive execution logging to project_documentation.txt.
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix

from model import AdvRoNIDS_CNN

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
    """Logs pipeline execution to stdout and appends in-depth notes to project_documentation.txt."""

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


def load_datasets(data_dir, logger):
    """Loads processed splits from parquet files and encodes labels."""
    logger.log("Phase 2 - Step 1: Loading Processed Datasets & Constructing Tensors", section=True)
    data_dir = Path(data_dir)
    
    logger.log(f"Reading parquet splits from '{data_dir}'...")
    X_train_df = pd.read_parquet(data_dir / "X_train.parquet")
    y_train_df = pd.read_parquet(data_dir / "y_train.parquet")
    X_val_df = pd.read_parquet(data_dir / "X_val.parquet")
    y_val_df = pd.read_parquet(data_dir / "y_val.parquet")
    X_test_df = pd.read_parquet(data_dir / "X_test.parquet")
    y_test_df = pd.read_parquet(data_dir / "y_test.parquet")

    logger.log(f"  Train: X shape = {X_train_df.shape}, y shape = {y_train_df.shape}")
    logger.log(f"  Val:   X shape = {X_val_df.shape}, y shape = {y_val_df.shape}")
    logger.log(f"  Test:  X shape = {X_test_df.shape}, y shape = {y_test_df.shape}")

    # Establish canonical sorted class ordering
    unique_classes = sorted(y_train_df["Label"].unique().tolist())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    idx_to_class = {idx: cls_name for idx, cls_name in enumerate(unique_classes)}
    num_classes = len(unique_classes)

    logger.log(f"Canonical Class Encoding ({num_classes} classes):", header=True)
    for cls_name, idx in class_to_idx.items():
        logger.log(f"  Index {idx:2d} -> '{cls_name}'")

    # Save label encoding mapping
    encoding_path = data_dir / "label_encoding.json"
    with open(encoding_path, "w", encoding="utf-8") as f:
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "classes": unique_classes
        }, f, indent=2)
    logger.log(f"Exported label encoding mapping to '{encoding_path}'.")

    # Map labels to integer vectors
    y_train_encoded = y_train_df["Label"].map(class_to_idx).to_numpy(dtype=np.int64)
    y_val_encoded = y_val_df["Label"].map(class_to_idx).to_numpy(dtype=np.int64)
    y_test_encoded = y_test_df["Label"].map(class_to_idx).to_numpy(dtype=np.int64)

    # Convert features to contiguous float32 numpy arrays
    X_train_np = np.ascontiguousarray(X_train_df.to_numpy(dtype=np.float32))
    X_val_np = np.ascontiguousarray(X_val_df.to_numpy(dtype=np.float32))
    X_test_np = np.ascontiguousarray(X_test_df.to_numpy(dtype=np.float32))

    # Free dataframes
    del X_train_df, y_train_df, X_val_df, y_val_df, X_test_df, y_test_df
    gc.collect()

    # Convert to PyTorch Tensors
    X_train_tensor = torch.from_numpy(X_train_np)
    y_train_tensor = torch.from_numpy(y_train_encoded)
    X_val_tensor = torch.from_numpy(X_val_np)
    y_val_tensor = torch.from_numpy(y_val_encoded)
    X_test_tensor = torch.from_numpy(X_test_np)
    y_test_tensor = torch.from_numpy(y_test_encoded)

    logger.log(f"Tensors successfully allocated in memory:")
    logger.log(f"  X_train: {X_train_tensor.shape}, {X_train_tensor.element_size() * X_train_tensor.nelement() / (1024**2):.1f} MB")
    logger.log(f"  X_val:   {X_val_tensor.shape}, {X_val_tensor.element_size() * X_val_tensor.nelement() / (1024**2):.1f} MB")
    logger.log(f"  X_test:  {X_test_tensor.shape}, {X_test_tensor.element_size() * X_test_tensor.nelement() / (1024**2):.1f} MB")

    return (X_train_tensor, y_train_tensor,
            X_val_tensor, y_val_tensor,
            X_test_tensor, y_test_tensor,
            unique_classes, class_to_idx, idx_to_class)


def compute_balanced_weights(y_train_tensor, num_classes, unique_classes, logger):
    """Computes inverse class frequency weights using sklearn."""
    logger.log("Phase 2 - Step 2: Computing Inverse Class Frequency Weights for Loss", section=True)
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


def evaluate_model(model, dataloader, criterion, device):
    """Evaluates model on a dataloader, returns loss, accuracy, macro_f1, weighted_f1, all_preds, all_targets."""
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

    return avg_loss, acc, macro_f1, weighted_f1, all_preds, all_targets


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs, patience, checkpoint_path, logger):
    """Executes the clean classifier training loop with early stopping and best macro-F1 checkpointing."""
    logger.log("Phase 2 - Step 3: Model A Clean Training Loop", section=True)
    logger.log(f"Training configuration: Epochs={num_epochs}, Patience={patience}, Batch Size={train_loader.batch_size}, LR={optimizer.param_groups[0]['lr']}")
    
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "epoch_time": []
    }

    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    total_train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            running_loss += loss.item() * batch_size
            train_samples += batch_size

        train_loss = running_loss / train_samples
        val_loss, val_acc, val_macro_f1, val_weighted_f1, _, _ = evaluate_model(
            model, val_loader, criterion, device
        )
        epoch_elapsed = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_weighted_f1"].append(val_weighted_f1)
        history["epoch_time"].append(epoch_elapsed)

        is_best_f1 = val_macro_f1 > best_val_macro_f1
        if is_best_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            # Save checkpoint by best validation Macro-F1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, checkpoint_path)
            save_tag = "[BEST CHECKPOINT SAVED]"
        else:
            save_tag = ""

        logger.log(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_macro_f1:.4f} | "
            f"Time: {epoch_elapsed:.1f}s {save_tag}"
        )

        # Early stopping tracking based on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.log(f"Early stopping triggered after {epoch} epochs (no validation loss improvement for {patience} epochs).")
                break

    total_train_time = time.time() - total_train_start
    logger.log(f"Training completed in {total_train_time:.2f} seconds. Best checkpoint from Epoch {best_epoch} with Val Macro-F1 = {best_val_macro_f1:.4f}.")
    return history, best_epoch


def plot_training_curves(history, results_dir, logger):
    """Plots training/validation loss and validation macro-F1 over epochs."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "clean_model_training_curves.png"

    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title("Training & Validation Loss (CrossEntropy)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11)

    # Macro-F1 & Accuracy Curve
    ax2.plot(epochs, history["val_macro_f1"], "o-", label="Val Macro-F1", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "^-.", label="Val Accuracy", color="#d62728", linewidth=2)
    ax2.set_title("Validation Macro-F1 & Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.log(f"Saved training curves plot to '{plot_path}'.")


def evaluate_and_report_test(model, test_loader, criterion, device, unique_classes,
                             results_dir, checkpoint_path, logger):
    """Loads best checkpoint and performs thorough evaluation on the held-out test split."""
    logger.log("Phase 2 - Step 4: Comprehensive Held-Out Test Evaluation", section=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load best checkpoint
    logger.log(f"Loading best checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.log(f"Loaded model checkpoint from Epoch {checkpoint['epoch']} (Val Macro-F1: {checkpoint['val_macro_f1']:.4f})")

    test_loss, test_acc, test_macro_f1, test_weighted_f1, test_preds, test_targets = evaluate_model(
        model, test_loader, criterion, device
    )

    logger.log(f"Test Set Evaluation Results:", header=True)
    logger.log(f"  Test Loss:        {test_loss:.4f}")
    logger.log(f"  Test Accuracy:    {test_acc*100:.3f}%")
    logger.log(f"  Test Macro-F1:    {test_macro_f1:.4f}")
    logger.log(f"  Test Weighted-F1: {test_weighted_f1:.4f}")

    # Generate full classification report
    report_dict = classification_report(
        test_targets, test_preds,
        target_names=unique_classes,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        test_targets, test_preds,
        target_names=unique_classes,
        digits=4,
        zero_division=0
    )

    logger.log("Classification Report (Precision / Recall / F1-Score per Class):", header=True)
    logger.log("\n" + report_text)

    # Save classification report text
    report_file = results_dir / "clean_model_classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AdvRoNIDS Model A (Clean 1D-CNN) - Test Set Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report_text)
        f.write(f"\nTest Accuracy:    {test_acc*100:.3f}%\n")
        f.write(f"Test Macro-F1:    {test_macro_f1:.4f}\n")
        f.write(f"Test Weighted-F1: {test_weighted_f1:.4f}\n")
    logger.log(f"Saved classification report to '{report_file}'.")

    # Compute and save Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds)
    cm_df = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
    cm_csv_path = results_dir / "clean_model_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    logger.log(f"Saved confusion matrix CSV to '{cm_csv_path}'.")

    # Plot Confusion Matrix Heatmap
    cm_png_path = results_dir / "clean_model_confusion_matrix.png"
    plt.figure(figsize=(12, 10))
    # Normalize by row (true class) for clear visual interpretation
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=unique_classes,
        yticklabels=unique_classes,
        cbar_kws={"label": "Normalized Recall Rate"}
    )
    plt.title("AdvRoNIDS Model A Clean 1D-CNN: Normalized Test Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    logger.log(f"Saved confusion matrix heatmap to '{cm_png_path}'.")

    # Save test metrics JSON
    metrics_json_path = results_dir / "clean_model_test_metrics.json"
    metrics_summary = {
        "model_name": "AdvRoNIDS_ModelA_Clean_1D_CNN",
        "best_epoch": int(checkpoint["epoch"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "classification_report": report_dict
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.log(f"Saved test metrics JSON to '{metrics_json_path}'.")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Train Clean 1D-CNN Classifier (AdvRoNIDS Model A)")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Path to processed parquet data")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory to save evaluation results")
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--doc-path", type=str, default="project_documentation.txt", help="Documentation file path")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = DetailedLogger(doc_path=args.doc_path)

    logger.log("AdvRoNIDS Clean Classifier Training (Model A - Section 6.2)", section=True)
    logger.log(f"Using Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Host CPU'})")

    # 1. Load Data
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     unique_classes, class_to_idx, idx_to_class) = load_datasets(args.data_dir, logger)

    num_classes = len(unique_classes)

    # 2. Build Datasets and DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # DataLoader with pinned memory if CUDA is available
    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=use_pin)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=use_pin)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=use_pin)

    # 3. Balanced Class Weights
    class_weights = compute_balanced_weights(y_train, num_classes, unique_classes, logger)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 4. Initialize Model
    model = AdvRoNIDS_CNN(num_classes=num_classes, in_channels=1, input_features=71, dropout_rate=0.3)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"AdvRoNIDS_CNN Model Initialized with {total_params:,} trainable parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. Train Model
    checkpoint_file = Path(args.checkpoints_dir) / "clean_model_best.pt"
    history, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint_file,
        logger=logger
    )

    # 6. Plot Training Curves
    plot_training_curves(history, args.results_dir, logger)

    # 7. Evaluate on Held-Out Test Set
    metrics_summary = evaluate_and_report_test(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        unique_classes=unique_classes,
        results_dir=args.results_dir,
        checkpoint_path=checkpoint_file,
        logger=logger
    )

    logger.log("Phase 2 (Model A Training & Evaluation) Complete!", section=True)


if __name__ == "__main__":
    main()
