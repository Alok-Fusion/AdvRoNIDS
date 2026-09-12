#!/usr/bin/env python3
"""
AdvRoNIDS: Attack Vulnerability Benchmark & Feasibility Evaluation
Proposal References: Section 6.3 (Domain Feasibility Constraints),
                     Section 6.4 (Threat Model & Attack Algorithms),
                     Section 8 (Evaluation Plan - Clean Baseline Vulnerability Table)

Evaluates Clean Classifier (Model A) against:
1. Clean baseline.
2. FGSM (epsilon = 0.1) — Constrained (Pi_S) vs. Unconstrained.
3. PGD (epsilon in {0.01, 0.05, 0.1, 0.2}, 7 steps, alpha=epsilon/4) —
   Constrained (Pi_S) vs. Unconstrained.
4. Feasibility violation rate on unconstrained PGD perturbations (Section 4.3 reproduction).
5. Exports results JSON, robustness curves plot, and appends documentation to project_documentation.txt.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

from model import AdvRoNIDS_CNN
from attacks import project_feasible, fgsm_attack, pgd_attack, compute_feasibility_violations

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


class AttackLogger:
    """Logs execution to stdout and appends detailed notes to project_documentation.txt."""

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
        X_sample, _, y_sample, _ = train_test_split(
            X_test_np, y_test_encoded,
            train_size=sample_size,
            random_state=random_state,
            stratify=y_test_encoded
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
    """Loads feasibility mask vectors and bounds tensors onto target compute device."""
    with open(mask_path, "r", encoding="utf-8") as f:
        mask_data = json.load(f)

    mask_t = torch.tensor(mask_data["mask_vector"], dtype=torch.float32, device=device)
    cum_mask_t = torch.tensor(mask_data["cumulative_mask_vector"], dtype=torch.float32, device=device)
    min_t = torch.tensor(mask_data["std_emp_min_vector"], dtype=torch.float32, device=device)
    max_t = torch.tensor(mask_data["std_emp_max_vector"], dtype=torch.float32, device=device)

    return mask_t, cum_mask_t, min_t, max_t, mask_data


def evaluate_clean_baseline(model, dataloader, device, unique_classes, logger):
    """Evaluates clean accuracy and Macro-F1 on evaluation sample."""
    logger.log("1. Clean Baseline Evaluation on Evaluation Sample", header=True)
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

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }


def evaluate_fgsm(model, dataloader, epsilon, mask_t, cum_mask_t, min_t, max_t,
                  device, logger):
    """Evaluates both unconstrained and constrained FGSM attacks at given epsilon."""
    logger.log(f"2. Fast Gradient Sign Method (FGSM) Evaluation (epsilon = {epsilon})", header=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()

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

        if not constrained:
            uncon_res = {"accuracy": float(acc), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1)}
        else:
            con_res = {"accuracy": float(acc), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1)}

    return {"unconstrained": uncon_res, "constrained": con_res}


def evaluate_pgd_sweep(model, dataloader, epsilons, num_steps, mask_t, cum_mask_t,
                       min_t, max_t, device, logger):
    """Evaluates PGD across epsilon sweep for both constrained and unconstrained attacks."""
    logger.log(f"3. Projected Gradient Descent (PGD-{num_steps}) Robustness Sweep across epsilons", header=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    sweep_results = {
        "epsilons": epsilons,
        "num_steps": num_steps,
        "constrained": [],
        "unconstrained": [],
        "feasibility_violations": []
    }

    for eps in epsilons:
        alpha = eps / 4.0
        logger.log(f"\n--- Testing PGD-{num_steps} with epsilon = {eps:.2f} (alpha = {alpha:.4f}) ---")

        # 1. Unconstrained PGD
        start_uncon = time.time()
        all_preds_uncon, all_targets_uncon = [], []
        total_viol_dict = {"overall": 0, "frozen": 0, "mono": 0, "bound": 0, "count": 0}

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

            # Measure domain feasibility violations on unconstrained perturbations
            v_stats = compute_feasibility_violations(
                X_adv_uncon, X_b, mask_t, cum_mask_t, min_t, max_t
            )
            n_b = v_stats["total_samples"]
            total_viol_dict["overall"] += (v_stats["overall_violation_rate"] / 100.0) * n_b
            total_viol_dict["frozen"] += (v_stats["frozen_violation_rate"] / 100.0) * n_b
            total_viol_dict["mono"] += (v_stats["monotonicity_violation_rate"] / 100.0) * n_b
            total_viol_dict["bound"] += (v_stats["boundary_violation_rate"] / 100.0) * n_b
            total_viol_dict["count"] += n_b

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

        n_tot = total_viol_dict["count"]
        viol_summary = {
            "epsilon": float(eps),
            "overall_violation_rate": float((total_viol_dict["overall"] / n_tot) * 100.0),
            "frozen_violation_rate": float((total_viol_dict["frozen"] / n_tot) * 100.0),
            "monotonicity_violation_rate": float((total_viol_dict["mono"] / n_tot) * 100.0),
            "boundary_violation_rate": float((total_viol_dict["bound"] / n_tot) * 100.0)
        }
        sweep_results["feasibility_violations"].append(viol_summary)

        logger.log(
            f"  PGD-{num_steps} [Unconstrained] | eps={eps:.2f} | "
            f"Accuracy: {acc_uncon*100:6.3f}% | Macro-F1: {f1_uncon:.4f} | "
            f"Violation Rate: {viol_summary['overall_violation_rate']:5.1f}% | Time: {time_uncon:.1f}s"
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
            f"Accuracy: {acc_con*100:6.3f}% | Macro-F1: {f1_con:.4f} | "
            f"Realistic Domain Evading | Time: {time_con:.1f}s"
        )
        sweep_results["constrained"].append({
            "epsilon": float(eps),
            "accuracy": float(acc_con),
            "macro_f1": float(f1_con),
            "weighted_f1": float(wf1_con),
            "time_seconds": float(time_con)
        })

    return sweep_results


def plot_robustness_curves(clean_acc, clean_f1, fgsm_results, pgd_results, results_dir, logger):
    """Plots robustness degradation curves across epsilons."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "robustness_curve_clean_model.png"

    epsilons = [0.0] + pgd_results["epsilons"]
    
    # Accuracies
    acc_con = [clean_acc * 100.0] + [res["accuracy"] * 100.0 for res in pgd_results["constrained"]]
    acc_uncon = [clean_acc * 100.0] + [res["accuracy"] * 100.0 for res in pgd_results["unconstrained"]]

    # Macro-F1s
    f1_con = [clean_f1] + [res["macro_f1"] for res in pgd_results["constrained"]]
    f1_uncon = [clean_f1] + [res["macro_f1"] for res in pgd_results["unconstrained"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epsilons, acc_con, "o-", color="#2ca02c", linewidth=2.2, label="Constrained PGD (Π_S Real Domain)")
    ax1.plot(epsilons, acc_uncon, "s--", color="#d62728", linewidth=2.2, label="Unconstrained PGD (Naive L∞)")
    ax1.scatter([0.1], [fgsm_results["constrained"]["accuracy"] * 100.0], color="#1f77b4", marker="^", s=100, label="Constrained FGSM (ε=0.1)", zorder=5)
    ax1.scatter([0.1], [fgsm_results["unconstrained"]["accuracy"] * 100.0], color="#ff7f0e", marker="v", s=100, label="Unconstrained FGSM (ε=0.1)", zorder=5)

    ax1.set_title("Clean Model A: Robustness Curve (Accuracy vs Perturbation Budget ε)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Perturbation Budget (ε in Standardized L∞ Space)", fontsize=10)
    ax1.set_ylabel("Classification Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9.5, loc="lower left")

    # Macro-F1 Plot
    ax2.plot(epsilons, f1_con, "o-", color="#2ca02c", linewidth=2.2, label="Constrained PGD (Π_S Real Domain)")
    ax2.plot(epsilons, f1_uncon, "s--", color="#d62728", linewidth=2.2, label="Unconstrained PGD (Naive L∞)")
    ax2.scatter([0.1], [fgsm_results["constrained"]["macro_f1"]], color="#1f77b4", marker="^", s=100, label="Constrained FGSM (ε=0.1)", zorder=5)
    ax2.scatter([0.1], [fgsm_results["unconstrained"]["macro_f1"]], color="#ff7f0e", marker="v", s=100, label="Unconstrained FGSM (ε=0.1)", zorder=5)

    ax2.set_title("Clean Model A: Macro-F1 Degradation vs Perturbation Budget ε", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Perturbation Budget (ε in Standardized L∞ Space)", fontsize=10)
    ax2.set_ylabel("Macro-F1 Score", fontsize=10)
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9.5, loc="lower left")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.log(f"Saved robustness degradation plot to '{plot_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Clean Model A Vulnerability against FGSM and Constrained/Unconstrained PGD")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Path to processed parquet data")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory to save evaluation results")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/clean_model_best.pt", help="Clean model checkpoint")
    parser.add_argument("--mask-path", type=str, default="./data/processed/feasibility_mask.json", help="Feasibility mask JSON path")
    parser.add_argument("--doc-path", type=str, default="project_documentation.txt", help="Documentation file path")
    parser.add_argument("--sample-size", type=int, default=50000, help="Stratified test sample size (use 0 for full test set)")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument("--num-steps", type=int, default=7, help="PGD iteration steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = AttackLogger(doc_path=args.doc_path)

    logger.log("Phase 3: Attack Vulnerability & Feasibility Benchmark (Section 6.3 - 6.4)", section=True)
    logger.log(f"Using Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Host CPU'})")

    # 1. Load Data
    X_test_tensor, y_test_tensor, unique_classes, class_to_idx = load_test_sample(
        args.data_dir, sample_size=args.sample_size, random_state=args.seed, logger=logger
    )
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Load Feasibility Masks and Bounds
    mask_t, cum_mask_t, min_t, max_t, mask_metadata = load_masks_and_bounds(args.mask_path, device)
    logger.log(f"Feasibility Configuration: {mask_metadata['frozen_count']} Frozen features, {mask_metadata['perturbable_count']} Perturbable features, {mask_metadata['cumulative_counter_count']} Cumulative counters.")

    # 3. Load Trained Clean Model A
    logger.log(f"Loading trained Model A checkpoint from '{args.checkpoint_path}'...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = AdvRoNIDS_CNN(num_classes=len(unique_classes), in_channels=1, input_features=71, dropout_rate=0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 4. Clean Baseline
    clean_res = evaluate_clean_baseline(model, test_loader, device, unique_classes, logger)

    # 5. FGSM Evaluation (epsilon = 0.1)
    fgsm_res = evaluate_fgsm(model, test_loader, epsilon=0.1, mask_t=mask_t,
                             cum_mask_t=cum_mask_t, min_t=min_t, max_t=max_t,
                             device=device, logger=logger)

    # 6. PGD Robustness Sweep across epsilons
    epsilons = [0.01, 0.05, 0.1, 0.2]
    pgd_res = evaluate_pgd_sweep(
        model=model,
        dataloader=test_loader,
        epsilons=epsilons,
        num_steps=args.num_steps,
        mask_t=mask_t,
        cum_mask_t=cum_mask_t,
        min_t=min_t,
        max_t=max_t,
        device=device,
        logger=logger
    )

    # 7. Plot Robustness Curves
    plot_robustness_curves(
        clean_acc=clean_res["accuracy"],
        clean_f1=clean_res["macro_f1"],
        fgsm_results=fgsm_res,
        pgd_results=pgd_res,
        results_dir=args.results_dir,
        logger=logger
    )

    # 8. Save Full Vulnerability Report JSON
    report = {
        "benchmark": "AdvRoNIDS Clean Model A Vulnerability & Feasibility Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_samples": len(X_test_tensor),
        "clean_baseline": clean_res,
        "fgsm_epsilon_0_1": fgsm_res,
        "pgd_sweep": pgd_res,
        "feasibility_mask_summary": {
            "frozen_features_count": mask_metadata["frozen_count"],
            "perturbable_features_count": mask_metadata["perturbable_count"],
            "cumulative_counters_count": mask_metadata["cumulative_counter_count"]
        }
    }

    report_path = Path(args.results_dir) / "attack_vulnerability_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.log(f"Full vulnerability report successfully saved to '{report_path}'.")

    # Print Final Vulnerability Summary Table
    logger.log("Summary Table: Clean Model A Vulnerability (Section 8 Evaluation Baseline)", section=True)
    logger.log(f"{'Attack Configuration':<35} | {'Epsilon':<8} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Feasibility Violation':<22}")
    logger.log("-" * 95)
    logger.log(f"{'Clean Baseline (No Attack)':<35} | {'0.00':<8} | {clean_res['accuracy']*100:6.3f}%   | {clean_res['macro_f1']:6.4f}     | {'0.0% (N/A)':<22}")
    logger.log(f"{'FGSM [Unconstrained Naive]':<35} | {'0.10':<8} | {fgsm_res['unconstrained']['accuracy']*100:6.3f}%   | {fgsm_res['unconstrained']['macro_f1']:6.4f}     | {'N/A':<22}")
    logger.log(f"{'FGSM [Constrained Pi_S]':<35} | {'0.10':<8} | {fgsm_res['constrained']['accuracy']*100:6.3f}%   | {fgsm_res['constrained']['macro_f1']:6.4f}     | {'0.0% (Enforced)':<22}")
    
    for c_res, u_res, v_res in zip(pgd_res["constrained"], pgd_res["unconstrained"], pgd_res["feasibility_violations"]):
        eps_val = c_res["epsilon"]
        logger.log(f"{'PGD-7 [Unconstrained Naive]':<35} | {eps_val:<8.2f} | {u_res['accuracy']*100:6.3f}%   | {u_res['macro_f1']:6.4f}     | {v_res['overall_violation_rate']:5.1f}% (Broken Rules)")
        logger.log(f"{'PGD-7 [Constrained Pi_S]':<35} | {eps_val:<8.2f} | {c_res['accuracy']*100:6.3f}%   | {c_res['macro_f1']:6.4f}     | {'0.0% (Enforced)':<22}")

    logger.log("-" * 95)
    logger.log("Phase 3 Evaluation Complete!\n")


if __name__ == "__main__":
    main()
