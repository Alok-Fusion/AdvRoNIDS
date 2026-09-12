import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import shap

from src.train_robust import AdvRoNIDS_CNN
from src.attacks import pgd_attack

def prepare_demo_dataset():
    data_dir = Path("data/processed")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("checkpoints")

    with open(data_dir / "label_encoding.json", "r", encoding="utf-8") as f:
        enc_data = json.load(f)
    classes = enc_data["classes"]
    class_to_idx = enc_data["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(classes)

    # Load feasibility mask
    with open(data_dir / "feasibility_mask.json", "r", encoding="utf-8") as f:
        mask_data = json.load(f)
    mask_t = torch.tensor(mask_data["mask_vector"], dtype=torch.float32)
    cum_mask_t = torch.tensor(mask_data["cumulative_mask_vector"], dtype=torch.float32)
    min_t = torch.tensor(mask_data["std_emp_min_vector"], dtype=torch.float32)
    max_t = torch.tensor(mask_data["std_emp_max_vector"], dtype=torch.float32)

    # Load models
    device = torch.device("cpu")
    model_A = AdvRoNIDS_CNN(num_classes=num_classes, in_channels=1, input_features=71, dropout_rate=0.3)
    ckpt_A = torch.load(checkpoints_dir / "clean_model_best.pt", map_location=device)
    model_A.load_state_dict(ckpt_A["model_state_dict"])
    model_A.eval()

    model_B = AdvRoNIDS_CNN(num_classes=num_classes, in_channels=1, input_features=71, dropout_rate=0.3)
    ckpt_B = torch.load(checkpoints_dir / "robust_model_best.pt", map_location=device)
    model_B.load_state_dict(ckpt_B["model_state_dict"])
    model_B.eval()

    # Load test data and background
    df_test_x = pd.read_parquet(data_dir / "X_test.parquet")
    df_test_y = pd.read_parquet(data_dir / "y_test.parquet")
    feature_names = list(df_test_x.columns)
    y_test_np = df_test_y["Label"].map(class_to_idx).to_numpy(dtype=np.int64)
    X_test_np = np.ascontiguousarray(df_test_x.to_numpy(dtype=np.float32))

    df_train_x = pd.read_parquet(data_dir / "X_train.parquet")
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(df_train_x), size=30, replace=False)
    X_bg_t = torch.tensor(df_train_x.iloc[bg_idx].to_numpy(dtype=np.float32))

    # Initialize SHAP explainers
    explainer_A = shap.GradientExplainer(model_A, X_bg_t)
    explainer_B = shap.GradientExplainer(model_B, X_bg_t)

    # Target specific classes for rich demo scenarios
    target_scenarios = [
        {"class": "SSH-Patator", "is_adv": True, "tag": "Evaded Model A (Robust Model B Defended)"},
        {"class": "DoS GoldenEye", "is_adv": True, "tag": "Evaded Model A (Robust Model B Defended)"},
        {"class": "Bot", "is_adv": True, "tag": "Evaded Model A (Robust Model B Defended)"},
        {"class": "Web Attack-Brute Force", "is_adv": True, "tag": "Evaded Model A (Robust Model B Defended)"},
        {"class": "Heartbleed", "is_adv": True, "tag": "Evaded Model A (Robust Model B Defended)"},
        {"class": "DDoS", "is_adv": True, "tag": "Evaded Model A (Robust Model B Defended)"},
        {"class": "Benign", "is_adv": False, "tag": "Clean Operational Traffic (Both Agree)"},
        {"class": "DoS Hulk", "is_adv": False, "tag": "High-Volume Attack (Both Agree)"},
    ]

    demo_flows = []

    for scenario in target_scenarios:
        target_cls_name = scenario["class"]
        target_cls_idx = class_to_idx[target_cls_name]
        is_adv = scenario["is_adv"]
        tag = scenario["tag"]

        matching_indices = np.where(y_test_np == target_cls_idx)[0]
        selected_flow_idx = None
        selected_x_orig = None
        selected_x_eval = None
        selected_pred_a = None
        selected_pred_b = None
        selected_delta = None

        for idx in matching_indices[:100]:
            x_clean = torch.tensor(X_test_np[idx:idx+1])
            y_true = torch.tensor([target_cls_idx])

            with torch.no_grad():
                logits_a_clean = model_A(x_clean)
                logits_b_clean = model_B(x_clean)
                pred_a_clean = torch.argmax(logits_a_clean, dim=1).item()
                pred_b_clean = torch.argmax(logits_b_clean, dim=1).item()

            if is_adv:
                # Target an evasion case
                x_adv_a = pgd_attack(
                    model_A, x_clean, y_true, epsilon=0.10, alpha=0.025, num_steps=7,
                    constrained=True, mask=mask_t, cumulative_mask=cum_mask_t,
                    emp_min=min_t, emp_max=max_t
                )
                x_adv_b = pgd_attack(
                    model_B, x_clean, y_true, epsilon=0.10, alpha=0.025, num_steps=7,
                    constrained=True, mask=mask_t, cumulative_mask=cum_mask_t,
                    emp_min=min_t, emp_max=max_t
                )

                with torch.no_grad():
                    prob_a_adv = F.softmax(model_A(x_adv_a), dim=1)[0]
                    prob_b_adv = F.softmax(model_B(x_adv_b), dim=1)[0]
                    pred_a_adv = torch.argmax(prob_a_adv).item()
                    pred_b_adv = torch.argmax(prob_b_adv).item()

                if pred_a_clean == target_cls_idx and pred_a_adv != target_cls_idx and pred_b_adv == target_cls_idx:
                    selected_flow_idx = int(idx)
                    selected_x_orig = x_clean
                    selected_x_eval_a = x_adv_a
                    selected_x_eval_b = x_adv_b
                    selected_pred_a = {"label": idx_to_class[pred_a_adv], "confidence": float(prob_a_adv[pred_a_adv])}
                    selected_pred_b = {"label": idx_to_class[pred_b_adv], "confidence": float(prob_b_adv[pred_b_adv])}
                    
                    delta_np = (x_adv_a - x_clean).numpy()[0]
                    selected_delta = {
                        feature_names[i]: float(delta_np[i])
                        for i in np.where(np.abs(delta_np) > 1e-4)[0]
                    }
                    break
            else:
                # Clean scenario where both models agree
                with torch.no_grad():
                    prob_a = F.softmax(logits_a_clean, dim=1)[0]
                    prob_b = F.softmax(logits_b_clean, dim=1)[0]
                    p_a = torch.argmax(prob_a).item()
                    p_b = torch.argmax(prob_b).item()

                if p_a == target_cls_idx and p_b == target_cls_idx:
                    selected_flow_idx = int(idx)
                    selected_x_orig = x_clean
                    selected_x_eval_a = x_clean
                    selected_x_eval_b = x_clean
                    selected_pred_a = {"label": idx_to_class[p_a], "confidence": float(prob_a[p_a])}
                    selected_pred_b = {"label": idx_to_class[p_b], "confidence": float(prob_b[p_b])}
                    selected_delta = None
                    break

        if selected_flow_idx is None:
            # Fallback to first instance
            selected_flow_idx = int(matching_indices[0])
            selected_x_orig = torch.tensor(X_test_np[selected_flow_idx:selected_flow_idx+1])
            selected_x_eval_a = selected_x_orig
            selected_x_eval_b = selected_x_orig
            with torch.no_grad():
                prob_a = F.softmax(model_A(selected_x_eval_a), dim=1)[0]
                prob_b = F.softmax(model_B(selected_x_eval_b), dim=1)[0]
                selected_pred_a = {"label": idx_to_class[torch.argmax(prob_a).item()], "confidence": float(torch.max(prob_a))}
                selected_pred_b = {"label": idx_to_class[torch.argmax(prob_b).item()], "confidence": float(torch.max(prob_b))}
            selected_delta = None

        # Compute SHAP for this individual flow
        shap_vals_a = explainer_A.shap_values(selected_x_eval_a)
        shap_vals_b = explainer_B.shap_values(selected_x_eval_b)

        # Extract SHAP array for true class
        if isinstance(shap_vals_a, list):
            attr_a = shap_vals_a[target_cls_idx][0]
            attr_b = shap_vals_b[target_cls_idx][0]
        else:
            attr_a = shap_vals_a[0, :, target_cls_idx]
            attr_b = shap_vals_b[0, :, target_cls_idx]

        top_idx_a = np.argsort(np.abs(attr_a))[::-1][:5]
        top_idx_b = np.argsort(np.abs(attr_b))[::-1][:5]

        top_shap_a = [{"feature": feature_names[i], "shap_value": float(attr_a[i])} for i in top_idx_a]
        top_shap_b = [{"feature": feature_names[i], "shap_value": float(attr_b[i])} for i in top_idx_b]

        demo_flows.append({
            "flow_id": f"flow_{selected_flow_idx}_{target_cls_name.replace(' ', '_')}",
            "raw_flow_index": selected_flow_idx,
            "ground_truth_label": target_cls_name,
            "is_adversarial": is_adv,
            "category_tag": tag,
            "model_a_prediction": selected_pred_a,
            "model_b_prediction": selected_pred_b,
            "top_shap_features_model_a": top_shap_a,
            "top_shap_features_model_b": top_shap_b,
            "perturbation_delta": selected_delta
        })

    demo_path = results_dir / "demo_flows.json"
    with open(demo_path, "w", encoding="utf-8") as f:
        json.dump(demo_flows, f, indent=2)
    print(f"Successfully exported {len(demo_flows)} curated demo flows to {demo_path}")

if __name__ == "__main__":
    prepare_demo_dataset()
