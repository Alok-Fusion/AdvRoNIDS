#!/usr/bin/env python3
"""
AdvRoNIDS: Unified Project Pipeline CLI
Executes any pipeline stage or the full end-to-end experimental workflow.

Usage:
    python main.py --step preprocess
    python main.py --step mask
    python main.py --step train
    python main.py --step evaluate
    python main.py --step all
"""

import sys
import argparse
from pathlib import Path

# Add project root and src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src import model, attacks


def run_preprocess():
    from src.preprocess import main as preprocess_main
    print("\n" + "=" * 80 + "\n[STEP 1] Running CICIDS2017 Preprocessing Pipeline (Section 7.2)\n" + "=" * 80)
    preprocess_main()


def run_mask():
    from src.generate_feasibility_mask import build_feasibility_mask
    print("\n" + "=" * 80 + "\n[STEP 2] Building Domain Feasibility Mask & Bounds (Section 6.3)\n" + "=" * 80)
    build_feasibility_mask()


def run_train_clean():
    from src.train_clean import main as train_main
    print("\n" + "=" * 80 + "\n[STEP 3] Training Clean Classifier Model A (Section 6.2)\n" + "=" * 80)
    train_main()


def run_evaluate_attacks():
    from src.evaluate_attacks import main as eval_main
    print("\n" + "=" * 80 + "\n[STEP 4] Evaluating FGSM & Constrained PGD Vulnerability (Section 6.4)\n" + "=" * 80)
    eval_main()


def run_evaluate_feasibility():
    from src.evaluate_feasibility import main as feas_main
    print("\n" + "=" * 80 + "\n[STEP 5] Evaluating Refined Feasibility Violation Metrics (Section 4.3 & 8)\n" + "=" * 80)
    feas_main()


def run_train_robust():
    from src.train_robust import main as robust_train_main
    print("\n" + "=" * 80 + "\n[STEP 6] Min-Max Adversarial Training for Robust Classifier Model B (Section 6.4.4)\n" + "=" * 80)
    robust_train_main()


def run_evaluate_robust():
    from src.evaluate_robust import main as robust_eval_main
    print("\n" + "=" * 80 + "\n[STEP 7] Evaluating Robust Model B & Benchmarking Clean vs. Robust (Section 8)\n" + "=" * 80)
    robust_eval_main()


def main():
    parser = argparse.ArgumentParser(description="AdvRoNIDS Main Pipeline Entrypoint")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["preprocess", "mask", "train", "evaluate", "feasibility", "robust_train", "robust_eval", "all"],
        help="Pipeline step to execute: preprocess, mask, train, evaluate, feasibility, robust_train, robust_eval, or all."
    )
    args = parser.parse_args()

    if args.step == "preprocess":
        run_preprocess()
    elif args.step == "mask":
        run_mask()
    elif args.step == "train":
        run_train_clean()
    elif args.step == "evaluate":
        run_evaluate_attacks()
    elif args.step == "feasibility":
        run_evaluate_feasibility()
    elif args.step == "robust_train":
        run_train_robust()
    elif args.step == "robust_eval":
        run_evaluate_robust()
    elif args.step == "all":
        run_preprocess()
        run_mask()
        run_train_clean()
        run_evaluate_attacks()
        run_evaluate_feasibility()
        run_train_robust()
        run_evaluate_robust()



if __name__ == "__main__":
    main()
