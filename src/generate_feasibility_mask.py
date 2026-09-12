#!/usr/bin/env python3
"""
AdvRoNIDS Feasibility Mask Generator (Section 6.3 & 7.1.2)

Reads the feature column list and statistics from ./data/processed/preprocessing_stats.json,
and generates the domain feasibility mask mapping every feature to either PERTURBABLE (1)
or FROZEN (0), along with identifying the cumulative counter features (delta_i >= 0).

Exports to: ./data/processed/feasibility_mask.json
"""

import json
from pathlib import Path

# Explicit frozen definitions per Section 6.3:
# - Protocol (and one-hot columns Protocol_0, Protocol_6, Protocol_17)
# - All flag-count columns (Fwd PSH Flags, Fwd URG Flags, FIN Flag Count, SYN Flag Count,
#   RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count)
# - Header length fields (any column containing "Header Length")
# - Init window fields (any column containing "Init Win Bytes" or "Init Fwd Win Bytes" / "Init Bwd Win Bytes")
# - Destination Port (if present)

# Cumulative counters per Section 6.3.1 (non-negativity constraint delta_i >= 0):
# - Total Fwd Packets, Total Backward Packets, Fwd Packets Length Total, Bwd Packets Length Total,
# - Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes, Fwd Act Data Packets


def build_feasibility_mask(stats_path="./data/processed/preprocessing_stats.json",
                           output_path="./data/processed/feasibility_mask.json"):
    stats_path = Path(stats_path)
    output_path = Path(output_path)

    if not stats_path.exists():
        raise FileNotFoundError(f"Preprocessing stats file not found at: {stats_path}")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    feature_columns = stats["feature_columns"]
    train_mean = stats["train_mean"]
    train_std = stats["train_std"]
    emp_min = stats["empirical_min"]
    emp_max = stats["empirical_max"]

    mask_dict = {}
    frozen_features = []
    perturbable_features = []
    cumulative_counter_features = []

    frozen_indices = []
    perturbable_indices = []
    cumulative_indices = []

    # Compute standardized empirical min/max for all 71 features
    std_emp_min = {}
    std_emp_max = {}

    for idx, col in enumerate(feature_columns):
        # Check frozen conditions
        is_protocol = col.startswith("Protocol")
        is_flag = any(f in col for f in [
            "PSH Flags", "URG Flags", "FIN Flag", "SYN Flag", "RST Flag",
            "PSH Flag", "ACK Flag", "URG Flag", "CWE Flag", "ECE Flag"
        ])
        is_header_len = "Header Length" in col
        is_init_win = "Init" in col and "Win Bytes" in col
        is_dest_port = "Destination Port" in col or "Dest Port" in col

        is_frozen = is_protocol or is_flag or is_header_len or is_init_win or is_dest_port

        if is_frozen:
            mask_dict[col] = 0
            frozen_features.append(col)
            frozen_indices.append(idx)
        else:
            mask_dict[col] = 1
            perturbable_features.append(col)
            perturbable_indices.append(idx)

            # Check if this perturbable feature is a cumulative counter
            is_cumulative = any(c in col for c in [
                "Total Fwd Packets", "Total Backward Packets",
                "Fwd Packets Length Total", "Bwd Packets Length Total",
                "Subflow Fwd Packets", "Subflow Fwd Bytes",
                "Subflow Bwd Packets", "Subflow Bwd Bytes",
                "Fwd Act Data Packets"
            ])
            if is_cumulative:
                cumulative_counter_features.append(col)
                cumulative_indices.append(idx)

        # Standardized empirical min/max computation
        if col in stats["continuous_columns"]:
            mean_val = train_mean[col]
            std_val = train_std[col]
            std_min = (emp_min[col] - mean_val) / std_val
            std_max = (emp_max[col] - mean_val) / std_val
        else:
            # Discrete/flags/protocol were not z-score normalized
            std_min = 0.0
            std_max = 1.0

        std_emp_min[col] = float(std_min)
        std_emp_max[col] = float(std_max)

    mask_data = {
        "description": "AdvRoNIDS Domain Feasibility Mask & Bounds Configuration (Section 6.3 & 6.3.1)",
        "total_features": len(feature_columns),
        "frozen_count": len(frozen_features),
        "perturbable_count": len(perturbable_features),
        "cumulative_counter_count": len(cumulative_counter_features),
        "mask": mask_dict,
        "frozen_features": frozen_features,
        "perturbable_features": perturbable_features,
        "cumulative_counter_features": cumulative_counter_features,
        "frozen_indices": frozen_indices,
        "perturbable_indices": perturbable_indices,
        "cumulative_indices": cumulative_indices,
        "protocol_indices": [idx for idx, col in enumerate(feature_columns) if col.startswith("Protocol_")],
        "feature_order": feature_columns,
        "std_empirical_min": std_emp_min,
        "std_empirical_max": std_emp_max,
        "std_emp_min_vector": [std_emp_min[col] for col in feature_columns],
        "std_emp_max_vector": [std_emp_max[col] for col in feature_columns],
        "mask_vector": [mask_dict[col] for col in feature_columns],
        "cumulative_mask_vector": [1 if col in cumulative_counter_features else 0 for col in feature_columns]
    }


    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mask_data, f, indent=2)

    print(f"Feasibility mask successfully saved to '{output_path}'.")
    print(f"Summary: {len(feature_columns)} total features = {len(frozen_features)} Frozen (mask=0) + {len(perturbable_features)} Perturbable (mask=1).")
    print(f"Cumulative counter features with non-negativity constraint (delta_i >= 0): {len(cumulative_counter_features)}")

    print("\n--- Frozen Features (delta = 0) ---")
    for f in frozen_features:
        print(f"  [FROZEN] {f}")

    print("\n--- Cumulative Counter Features (delta >= 0) ---")
    for c in cumulative_counter_features:
        print(f"  [CUMULATIVE] {c}")

    return mask_data


if __name__ == "__main__":
    build_feasibility_mask()
