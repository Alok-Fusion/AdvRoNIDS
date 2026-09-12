#!/usr/bin/env python3
"""
AdvRoNIDS Preprocessing Pipeline (Section 7.2 & 7.2.1)
Dataset: CICIDS2017 (8 Parquet files, identifier-free mirror)

This script performs:
1. Ingestion of 8 raw parquet files with lineage tracking (__source_file).
2. Memory optimization: downcasting float64 -> float32 and explicit GC.
3. Label corruption repair (replacement of \ufffd/ with '-').
4. Defensive inf/NaN sanitization and row deduplication.
5. Zero-variance feature detection and removal.
6. Categorical encoding for Protocol (one-hot) while preserving binary flags as numeric.
7. Stratified 70/15/15 train/val/test splitting with rare class handling.
8. Train-only standardization on continuous features (preserving bounded stats for Pi_S).
9. Dual persistence in Parquet and CSV formats + raw files to CSV conversion.
10. Export of preprocessing_stats.json and comprehensive logging to project_documentation.txt.
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
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output in Windows consoles
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


class PreprocessingLogger:
    """Logs pipeline execution to stdout and project_documentation.txt."""

    def __init__(self, doc_path="project_documentation.txt"):
        self.doc_path = Path(doc_path)
        # Clear or initialize doc file
        with open(self.doc_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("AdvRoNIDS - CICIDS2017 Preprocessing Pipeline Detailed Documentation\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

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



RAW_FILES = [
    "Benign-Monday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet"
]

BINARY_FLAG_COLUMNS = [
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count"
]


def find_data_directory(user_data_dir=None):
    """Locates the directory containing the raw 8 parquet files."""
    candidates = []
    if user_data_dir:
        candidates.append(Path(user_data_dir))
    candidates.extend([
        Path("./data/raw"),
        Path("./data"),
        Path("../data/raw"),
        Path("../data")
    ])

    for cand in candidates:
        if cand.exists():
            matched = [f for f in RAW_FILES if (cand / f).exists()]
            if len(matched) == len(RAW_FILES):
                return cand.resolve()
            elif len(matched) > 0:
                # Partial match found
                return cand.resolve()
    
    # Default to ./data if nothing found yet
    return Path("./data").resolve()


def export_raw_to_csv(data_dir, raw_csv_dir, logger):
    """Exports raw parquet files to CSV in raw_csv_dir."""
    raw_csv_dir = Path(raw_csv_dir)
    raw_csv_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Exporting raw parquet files to CSV in '{raw_csv_dir}'...", header=True)

    for filename in RAW_FILES:
        parquet_path = data_dir / filename
        if not parquet_path.exists():
            logger.log(f"  [WARNING] File {filename} not found, skipping CSV export.")
            continue
        
        csv_filename = filename.replace(".parquet", ".csv")
        csv_path = raw_csv_dir / csv_filename
        
        if csv_path.exists() and csv_path.stat().st_size > 0:
            logger.log(f"  Raw CSV already exists: {csv_path.name} (skipping redundant rewrite).")
            continue

        logger.log(f"  Reading {filename} and writing {csv_filename}...")
        df_part = pd.read_parquet(parquet_path)
        df_part.to_csv(csv_path, index=False)
        logger.log(f"  Saved: {csv_path} ({df_part.shape[0]:,} rows x {df_part.shape[1]} cols)")
        del df_part
        gc.collect()

    logger.log("Raw files CSV export complete.\n")



def load_and_merge_data(data_dir, logger):
    """Loads all 8 parquet files, tags __source_file, and merges."""
    logger.log("Step 1: Ingestion & Schema Alignment", section=True)
    dfs = []
    total_raw_rows = 0

    for filename in RAW_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required raw file not found: {filepath}")
        
        logger.log(f"Loading '{filename}'...")
        df_part = pd.read_parquet(filepath)
        row_count = len(df_part)
        total_raw_rows += row_count
        
        # Tag source file for traceability
        df_part["__source_file"] = filename
        
        logger.log(f"  Loaded {row_count:,} rows, {df_part.shape[1]} columns.")
        dfs.append(df_part)
        del df_part
        gc.collect()

    logger.log(f"Concatenating {len(dfs)} dataframes (total raw rows: {total_raw_rows:,})...")
    merged_df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    logger.log(f"Initial merged dataset shape: {merged_df.shape[0]:,} rows x {merged_df.shape[1]} columns")
    return merged_df



def clean_labels(df, logger):
    """Fixes corrupted label characters (\ufffd, corrupted en-dash) in WebAttacks."""
    logger.log("Step 2: Label Encoding Correction", section=True)
    
    raw_labels = df["Label"].value_counts().to_dict()
    logger.log("Raw label distribution before cleaning:")
    for lbl, cnt in raw_labels.items():
        safe_lbl = lbl.encode("ascii", "backslashreplace").decode("ascii")
        logger.log(f"  - {safe_lbl}: {cnt:,} samples")

    # Fix Web Attack corrupted en-dash (\ufffd)
    df["Label"] = df["Label"].astype(str).str.replace(r"Web Attack\s*[\ufffd\u2013-]?\s*", "Web Attack-", regex=True)
    df["Label"] = df["Label"].str.strip()

    cleaned_labels = df["Label"].value_counts().to_dict()
    logger.log("Cleaned label distribution:")
    for lbl, cnt in cleaned_labels.items():
        logger.log(f"  - {lbl}: {cnt:,} samples")

    return df




def handle_inf_and_nan(df, logger):
    """Defensively replaces inf/-inf with NaN and drops rows with NaN."""
    logger.log("Step 3: Defensive Inf and NaN Handling", section=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logger.log(f"Scanning {len(numeric_cols)} numeric columns for infinite values...")
    
    # Replace inf with nan
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    nan_count = df.isna().sum().sum()
    rows_with_nan = df.isna().any(axis=1).sum()
    logger.log(f"Found {nan_count:,} NaN values across {rows_with_nan:,} rows.")
    
    if rows_with_nan > 0:
        logger.log(f"Dropping {rows_with_nan:,} rows containing NaN...")
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        logger.log("Dataset is free of NaN/Inf values. No rows dropped in this step.")

    logger.log(f"Shape after NaN/Inf handling: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def deduplicate_dataset(df, logger):
    """Deduplicates exact duplicate rows excluding __source_file."""
    logger.log("Step 4: Dataset Deduplication", section=True)
    initial_rows = len(df)
    
    # Feature + Label columns (all except __source_file)
    dedup_subset = [c for c in df.columns if c != "__source_file"]
    
    df.drop_duplicates(subset=dedup_subset, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    final_rows = len(df)
    dropped_rows = initial_rows - final_rows
    logger.log(f"Exact duplicates removed: {dropped_rows:,} rows.")
    logger.log(f"Dataset shape after deduplication: {final_rows:,} rows x {df.shape[1]} columns (Expected ~2,231,806 rows).")
    return df


def drop_zero_variance(df, logger):
    """Computes variance on numeric features and drops zero-variance columns."""
    logger.log("Step 5: Zero-Variance Feature Pruning", section=True)
    
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "__source_file"]
    variances = df[numeric_cols].var()
    zero_var_cols = variances[variances == 0].index.tolist()
    
    logger.log(f"Identified {len(zero_var_cols)} zero-variance columns:")
    for col in zero_var_cols:
        logger.log(f"  - {col} (variance = 0.0)")

    df.drop(columns=zero_var_cols, inplace=True)
    logger.log(f"Shape after dropping zero-variance features: {df.shape[0]:,} rows x {df.shape[1]} columns (Expected: 71 cols before OHE/split).")
    return df, zero_var_cols


def encode_features(df, logger):
    """One-hot encodes Protocol, identifies binary flag columns and continuous columns."""
    logger.log("Step 6: Feature Categorization & One-Hot Encoding", section=True)
    
    # Drop __source_file before encoding and splitting
    if "__source_file" in df.columns:
        df.drop(columns=["__source_file"], inplace=True)

    # 1. One-hot encode Protocol only
    if "Protocol" in df.columns:
        unique_protocols = sorted(df["Protocol"].unique().tolist())
        logger.log(f"One-hot encoding multi-category feature 'Protocol' (unique values: {unique_protocols})...")
        
        # OHE Protocol single column slice
        protocol_dummies = pd.get_dummies(df["Protocol"], prefix="Protocol", dtype=np.uint8)
        dummy_col_names = protocol_dummies.columns.tolist()
        logger.log(f"Generated Protocol dummy columns: {dummy_col_names}")
        
        # Drop original Protocol and concatenate dummies
        df.drop(columns=["Protocol"], inplace=True)
        df = pd.concat([df, protocol_dummies], axis=1)
        del protocol_dummies
        gc.collect()
    else:
        dummy_col_names = [c for c in df.columns if c.startswith("Protocol_")]

    # 2. Identify binary flag columns
    active_binary_flags = [c for c in BINARY_FLAG_COLUMNS if c in df.columns]
    logger.log(f"Identified {len(active_binary_flags)} binary flag columns (kept as raw 0/1 numeric):")
    for col in active_binary_flags:
        logger.log(f"  - {col}")

    # 3. Identify categorical/discrete vs continuous features
    discrete_features = dummy_col_names + active_binary_flags
    all_features = [c for c in df.columns if c != "Label"]
    continuous_features = [c for c in all_features if c not in discrete_features]

    logger.log(f"Feature summary: {len(all_features)} total features ({len(continuous_features)} continuous, {len(discrete_features)} discrete/flags/OHE).")
    return df, continuous_features, discrete_features, all_features


def perform_stratified_split(df, logger, random_state=42):
    """Performs stratified 70/15/15 train/val/test split."""
    logger.log("Step 7: Stratified 70/15/15 Train/Val/Test Split", section=True)
    
    X = df.drop(columns=["Label"])
    y = df["Label"]
    
    class_counts = y.value_counts()
    logger.log("Total class counts before splitting:")
    for cls_name, cnt in class_counts.items():
        logger.log(f"  - {cls_name}: {cnt:,} samples")

    # Safety check on rare classes
    rare_classes = class_counts[class_counts < 10].index.tolist()
    if rare_classes:
        logger.log(f"[WARNING] Rare classes with < 10 samples detected: {rare_classes}")

    # Stage 1: Split 70% Train, 30% Temp (Val + Test)
    logger.log("Splitting Train (70%) and Temp (30%)...")
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=random_state, stratify=y
        )
    except ValueError as e:
        logger.log(f"[ERROR] Stratification failed in 70/30 split: {e}. Retrying with non-stratified fallback.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=random_state
        )

    # Stage 2: Split Temp (30%) into 50/50 Val (15%) and Test (15%)
    logger.log("Splitting Temp into Validation (15%) and Test (15%)...")
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
        )
    except ValueError as e:
        logger.log(f"[ERROR] Stratification failed in 50/50 val/test split: {e}. Retrying with non-stratified fallback.")
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=random_state
        )

    del X, y, X_temp, y_temp
    gc.collect()

    logger.log(f"Split sizes:")
    logger.log(f"  Train: {len(X_train):,} samples ({len(X_train)/len(df)*100:.2f}%)")
    logger.log(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(df)*100:.2f}%)")
    logger.log(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.2f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_features(X_train, X_val, X_test, continuous_cols, logger):
    """Computes train-only mean/std and empirical min/max, transforms train, val, test."""
    logger.log("Step 8: Train-Only Feature Standardization (Pi_S Bounded Stats)", section=True)
    
    logger.log(f"Computing statistics on {len(continuous_cols)} continuous features across {len(X_train):,} training samples...")
    
    # Train-only statistics
    train_mean = X_train[continuous_cols].mean(axis=0)
    train_std = X_train[continuous_cols].std(axis=0)
    empirical_min = X_train[continuous_cols].min(axis=0)
    empirical_max = X_train[continuous_cols].max(axis=0)

    # Guard against zero standard deviation
    zero_std_cols = train_std[train_std == 0].index.tolist()
    if zero_std_cols:
        logger.log(f"[WARNING] Zero standard deviation in training continuous features: {zero_std_cols}")
        train_std[zero_std_cols] = 1.0

    # Apply standardization: (x - mean) / std
    logger.log("Applying standardization transform (X - mu_train) / sigma_train to Train split...")
    X_train[continuous_cols] = (X_train[continuous_cols] - train_mean) / train_std

    logger.log("Applying standardization transform to Validation split...")
    X_val[continuous_cols] = (X_val[continuous_cols] - train_mean) / train_std

    logger.log("Applying standardization transform to Test split...")
    X_test[continuous_cols] = (X_test[continuous_cols] - train_mean) / train_std

    stats_dict = {
        "train_mean": train_mean.to_dict(),
        "train_std": train_std.to_dict(),
        "empirical_min": empirical_min.to_dict(),
        "empirical_max": empirical_max.to_dict()
    }

    logger.log("Standardization completed successfully.")
    return X_train, X_val, X_test, stats_dict


def save_processed_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir, save_csv, logger):
    """Saves train/val/test splits to parquet and optionally CSV."""
    logger.log("Step 9: Artifact Persistence", section=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("X_train", X_train),
        ("y_train", pd.DataFrame(y_train, columns=["Label"])),
        ("X_val", X_val),
        ("y_val", pd.DataFrame(y_val, columns=["Label"])),
        ("X_test", X_test),
        ("y_test", pd.DataFrame(y_test, columns=["Label"]))
    ]

    for name, df_split in splits:
        parquet_path = output_dir / f"{name}.parquet"
        logger.log(f"Writing {parquet_path} ({df_split.shape[0]:,} rows x {df_split.shape[1]} cols)...")
        df_split.to_parquet(parquet_path, index=False)

        if save_csv:
            csv_path = output_dir / f"{name}.csv"
            logger.log(f"Writing {csv_path}...")
            df_split.to_csv(csv_path, index=False)

    logger.log("All processed data splits successfully saved to disk.")


def export_metadata(stats_dict, continuous_cols, discrete_cols, all_features,
                    y_train, y_val, y_test, output_dir, total_cleaned_rows, logger):
    """Exports preprocessing_stats.json."""
    logger.log("Step 10: Metadata & Statistics Export", section=True)
    output_dir = Path(output_dir)
    
    # Class proportions summary
    train_dist = y_train.value_counts(normalize=True).to_dict()
    val_dist = y_val.value_counts(normalize=True).to_dict()
    test_dist = y_test.value_counts(normalize=True).to_dict()

    train_counts = y_train.value_counts().to_dict()
    val_counts = y_val.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()

    metadata = {
        "dataset": "CICIDS2017 Preprocessed (AdvRoNIDS Section 7.2)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cleaned_rows": total_cleaned_rows,
        "feature_count": len(all_features),
        "continuous_feature_count": len(continuous_cols),
        "discrete_feature_count": len(discrete_cols),
        "feature_columns": all_features,
        "continuous_columns": continuous_cols,
        "discrete_columns": discrete_cols,
        "train_mean": stats_dict["train_mean"],
        "train_std": stats_dict["train_std"],
        "empirical_min": stats_dict["empirical_min"],
        "empirical_max": stats_dict["empirical_max"],
        "class_counts": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts
        },
        "class_proportions": {
            "train": train_dist,
            "val": val_dist,
            "test": test_dist
        }
    }

    stats_path = output_dir / "preprocessing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.log(f"Preprocessing metadata saved to '{stats_path}'.")


def print_final_summary(y_train, y_val, y_test, all_features, total_start_time, logger):
    """Prints final summary of class distributions and pipeline execution."""
    logger.log("Final Preprocessing Summary", section=True)
    
    all_classes = sorted(list(set(y_train.unique()) | set(y_val.unique()) | set(y_test.unique())))
    
    logger.log(f"{'Class':<28} | {'Train Count':<12} | {'Val Count':<10} | {'Test Count':<10} | {'Train %':<8} | {'Val %':<8} | {'Test %':<8}")
    logger.log("-" * 95)
    
    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)
    
    t_cnt = y_train.value_counts()
    v_cnt = y_val.value_counts()
    te_cnt = y_test.value_counts()

    for cls in all_classes:
        c_tr = t_cnt.get(cls, 0)
        c_va = v_cnt.get(cls, 0)
        c_te = te_cnt.get(cls, 0)
        
        p_tr = (c_tr / n_train) * 100
        p_va = (c_va / n_val) * 100
        p_te = (c_te / n_test) * 100
        
        logger.log(f"{cls:<28} | {c_tr:<12,} | {c_va:<10,} | {c_te:<10,} | {p_tr:<7.3f}% | {p_va:<7.3f}% | {p_te:<7.3f}%")

    logger.log("-" * 95)
    logger.log(f"{'TOTAL':<28} | {n_train:<12,} | {n_val:<10,} | {n_test:<10,} | 100.000% | 100.000% | 100.000%")
    logger.log(f"\nFinal feature count: {len(all_features)}")
    logger.log(f"Total pipeline execution time: {time.time() - total_start_time:.2f} seconds")
    logger.log(f"Peak memory RSS: {get_memory_mb():.1f} MB")
    logger.log("Pipeline finished successfully!\n")


def downcast_float_precision(df, logger):
    """Downcasts float64 features to float32 to minimize memory usage for splitting and scaling."""
    logger.log("Step 5b: Memory Optimization (Float64 -> Float32 Downcasting)", section=True)
    float64_cols = df.select_dtypes(include=["float64"]).columns.tolist()
    if float64_cols:
        logger.log(f"Downcasting {len(float64_cols)} float64 columns to float32: {float64_cols}")
        df[float64_cols] = df[float64_cols].astype(np.float32)
    else:
        logger.log("No float64 columns found to downcast.")
    gc.collect()
    return df


def main():
    parser = argparse.ArgumentParser(description="AdvRoNIDS Preprocessing Pipeline for CICIDS2017")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing raw parquet files")
    parser.add_argument("--output-dir", type=str, default="./data/processed", help="Output directory for processed splits")
    parser.add_argument("--raw-csv-dir", type=str, default="./data/raw_csv", help="Output directory for raw CSV export")
    parser.add_argument("--export-raw-csv", action="store_true", default=True, help="Export raw parquet files to CSV")
    parser.add_argument("--save-csv", action="store_true", default=True, help="Save processed splits in CSV as well as Parquet")
    parser.add_argument("--doc-path", type=str, default="project_documentation.txt", help="Path for documentation output")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for data splitting")

    args = parser.parse_args()
    total_start_time = time.time()
    
    logger = PreprocessingLogger(doc_path=args.doc_path)
    logger.log("Starting AdvRoNIDS Preprocessing Pipeline...", section=True)

    data_dir = find_data_directory(args.data_dir)
    logger.log(f"Resolved raw data directory: {data_dir}")

    # Optional Raw CSV export
    if args.export_raw_csv:
        export_raw_to_csv(data_dir, args.raw_csv_dir, logger)

    # 1. Ingestion
    df = load_and_merge_data(data_dir, logger)

    # 2. Label Cleaning
    df = clean_labels(df, logger)

    # 3. Inf & NaN handling
    df = handle_inf_and_nan(df, logger)

    # 4. Deduplication (on raw precision)
    df = deduplicate_dataset(df, logger)
    total_cleaned_rows = len(df)

    # 5. Zero-Variance Pruning
    df, zero_var_cols = drop_zero_variance(df, logger)

    # 5b. Downcast float64 to float32
    df = downcast_float_precision(df, logger)

    # 6. Feature Categorization & One-Hot Encoding
    df, continuous_cols, discrete_cols, all_features = encode_features(df, logger)

    # 7. Stratified Train/Val/Test Split (70/15/15)
    X_train, X_val, X_test, y_train, y_val, y_test = perform_stratified_split(
        df, logger, random_state=args.random_state
    )
    del df
    gc.collect()

    # 8. Train-Only Feature Standardization
    X_train, X_val, X_test, stats_dict = standardize_features(
        X_train, X_val, X_test, continuous_cols, logger
    )

    # 9. Save Processed Splits
    save_processed_splits(
        X_train, X_val, X_test, y_train, y_val, y_test,
        args.output_dir, args.save_csv, logger
    )

    # 10. Export Metadata & Stats
    export_metadata(
        stats_dict, continuous_cols, discrete_cols, all_features,
        y_train, y_val, y_test, args.output_dir, total_cleaned_rows, logger
    )

    # Print Final Summary
    print_final_summary(y_train, y_val, y_test, all_features, total_start_time, logger)


if __name__ == "__main__":
    main()

