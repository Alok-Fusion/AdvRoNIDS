"""
AdvRoNIDS LLM-Assisted Incident Triage API Gateway (Section 9.2) & Live Attack Simulator
Serves REST API endpoints for:
1. /health: System liveness and local LLM status.
2. /demo-flows: Curated representative evaluation flows.
3. /triage: Real-time LLM incident note narration.
4. /attack-categories: Available attack classes and sample pool sizes.
5. /simulate-attack: Live, step-by-step traced PGD attack simulation against Model A & Model B.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, List

from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure root is in Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.train_robust import AdvRoNIDS_CNN
from src.attacks import pgd_attack_traced
from src.triage import generate_incident_note, clean_feature_name, FEATURE_NAME_MAP


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_runtime_resources()
    yield


app = FastAPI(
    title="AdvRoNIDS Incident Triage & Live Attack Simulator Gateway",
    description="LLM-Assisted Cyber Threat Narration & Real-Time Adversarial Attack Visualizer (Section 9.2)",
    version="1.1.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Global In-Memory State & Cache
# -----------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data" / "processed"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
DEMO_FLOWS_PATH = RESULTS_DIR / "demo_flows.json"

cached_demo_flows: List[Dict[str, Any]] = []

# Telemetry resources
device = torch.device("cpu")
model_A: Optional[AdvRoNIDS_CNN] = None
model_B: Optional[AdvRoNIDS_CNN] = None
class_to_idx: Dict[str, int] = {}
idx_to_class: Dict[int, str] = {}
unique_classes: List[str] = []
feature_names: List[str] = []

X_test_np: Optional[np.ndarray] = None
y_test_np: Optional[np.ndarray] = None

mask_t: Optional[torch.Tensor] = None
cum_mask_t: Optional[torch.Tensor] = None
min_t: Optional[torch.Tensor] = None
max_t: Optional[torch.Tensor] = None
frozen_indices: List[int] = []
perturbable_indices: List[int] = []
frozen_features: List[str] = []
perturbable_features: List[str] = []

# Pre-computed candidate indices per class where clean Model A is correct
clean_correct_indices_by_class: Dict[str, List[int]] = {}


def load_runtime_resources():
    """Initializes models, datasets, and feasibility constraints into memory once on startup."""
    global model_A, model_B, class_to_idx, idx_to_class, unique_classes, feature_names
    global X_test_np, y_test_np, mask_t, cum_mask_t, min_t, max_t
    global frozen_indices, perturbable_indices, frozen_features, perturbable_features
    global clean_correct_indices_by_class

    if model_A is not None:
        return

    # 1. Label encoding
    with open(DATA_DIR / "label_encoding.json", "r", encoding="utf-8") as f:
        enc_data = json.load(f)
    class_to_idx = enc_data["class_to_idx"]
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    unique_classes = enc_data["classes"]
    num_classes = len(unique_classes)

    # 2. Feasibility mask
    with open(DATA_DIR / "feasibility_mask.json", "r", encoding="utf-8") as f:
        mask_data = json.load(f)
    mask_t = torch.tensor(mask_data["mask_vector"], dtype=torch.float32)
    cum_mask_t = torch.tensor(mask_data["cumulative_mask_vector"], dtype=torch.float32)
    min_t = torch.tensor(mask_data["std_emp_min_vector"], dtype=torch.float32)
    max_t = torch.tensor(mask_data["std_emp_max_vector"], dtype=torch.float32)
    frozen_indices = mask_data["frozen_indices"]
    perturbable_indices = mask_data["perturbable_indices"]
    frozen_features = mask_data["frozen_features"]
    perturbable_features = mask_data["perturbable_features"]

    # 3. Load Models
    model_A = AdvRoNIDS_CNN(num_classes=num_classes, in_channels=1, input_features=71, dropout_rate=0.3)
    ckpt_a = torch.load(CHECKPOINTS_DIR / "clean_model_best.pt", map_location=device)
    model_A.load_state_dict(ckpt_a["model_state_dict"])
    model_A.eval()

    model_B = AdvRoNIDS_CNN(num_classes=num_classes, in_channels=1, input_features=71, dropout_rate=0.3)
    ckpt_b = torch.load(CHECKPOINTS_DIR / "robust_model_best.pt", map_location=device)
    model_B.load_state_dict(ckpt_b["model_state_dict"])
    model_B.eval()

    # 4. Ingest Test Parquet
    df_test_x = pd.read_parquet(DATA_DIR / "X_test.parquet")
    df_test_y = pd.read_parquet(DATA_DIR / "y_test.parquet")
    feature_names = list(df_test_x.columns)
    y_test_np = df_test_y["Label"].map(class_to_idx).to_numpy(dtype=np.int64)
    X_test_np = np.ascontiguousarray(df_test_x.to_numpy(dtype=np.float32))

    # 5. Pre-filter candidate clean-correct flows per class for clean evaluation narratives
    clean_correct_indices_by_class = {}
    for cls_name, cls_idx in class_to_idx.items():
        matching = np.where(y_test_np == cls_idx)[0]
        # Evaluate clean Model A on a subset of up to 1000 flows for speed
        eval_sample = matching[:min(len(matching), 1000)]
        x_sub = torch.tensor(X_test_np[eval_sample])
        with torch.no_grad():
            preds_sub = torch.argmax(model_A(x_sub), dim=1).numpy()
        correct_in_sub = eval_sample[np.where(preds_sub == cls_idx)[0]]
        if len(correct_in_sub) > 0:
            clean_correct_indices_by_class[cls_name] = correct_in_sub.tolist()
        else:
            clean_correct_indices_by_class[cls_name] = matching.tolist()


def get_demo_flows() -> List[Dict[str, Any]]:
    global cached_demo_flows
    if not cached_demo_flows and DEMO_FLOWS_PATH.exists():
        with open(DEMO_FLOWS_PATH, "r", encoding="utf-8") as f:
            cached_demo_flows = json.load(f)
    return cached_demo_flows


# -----------------------------------------------------------------------------
# Request & Response Models
# -----------------------------------------------------------------------------
class ModelPrediction(BaseModel):
    label: str
    confidence: float

class FeatureAttribution(BaseModel):
    feature: str
    shap_value: float

class TriageRequest(BaseModel):
    flow_id: str
    ground_truth_label: Optional[str] = None
    is_adversarial: bool = False
    category_tag: Optional[str] = None
    model_a_prediction: ModelPrediction
    model_b_prediction: ModelPrediction
    top_shap_features_model_a: Optional[List[FeatureAttribution]] = None
    top_shap_features_model_b: Optional[List[FeatureAttribution]] = None
    perturbation_delta: Optional[Dict[str, float]] = None

class TriageResponse(BaseModel):
    incident_note: str
    structured_summary: Dict[str, Any]

class SimulationRequest(BaseModel):
    category: str = Field(default="SSH-Patator", description="Target attack category name")
    constrained: bool = Field(default=True, description="Enforce Pi_S domain feasibility constraints")
    epsilon: float = Field(default=0.10, ge=0.001, le=0.50, description="Perturbation budget epsilon")
    num_steps: int = Field(default=7, ge=1, le=30, description="PGD iteration count")
    alpha: Optional[float] = Field(default=None, description="Step size (defaults to epsilon / 4)")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/health")
def health_check():
    """Liveness check and local Ollama status."""
    import requests
    ollama_ready = False
    models = []
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if res.status_code == 200:
            ollama_ready = True
            models = [m["name"] for m in res.json().get("models", [])]
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": "AdvRoNIDS Incident Triage & Live Attack Simulator (Section 9.2)",
        "models_loaded": model_A is not None and model_B is not None,
        "ollama_available": ollama_ready,
        "available_models": models,
        "active_model": "qwen2.5:3b" if "qwen2.5:3b" in models else (models[0] if models else "deterministic-fallback")
    }


@app.get("/demo-flows")
def list_demo_flows():
    """Returns curated representative demo flows."""
    flows = get_demo_flows()
    if not flows:
        raise HTTPException(status_code=404, detail="Demo flows not found. Run prepare_demo_flows.py first.")
    return {
        "count": len(flows),
        "flows": flows
    }


@app.get("/attack-categories")
def list_attack_categories():
    """Returns list of attack categories and their candidate sample pool sizes."""
    load_runtime_resources()
    categories = []
    for cls_name in unique_classes:
        if cls_name == "Benign":
            continue
        pool_size = len(clean_correct_indices_by_class.get(cls_name, []))
        categories.append({
            "name": cls_name,
            "pool_size": pool_size,
            "is_rare": pool_size < 20
        })
    return {
        "categories": categories,
        "total_attack_classes": len(categories)
    }


@app.get("/feature-labels")
def get_feature_labels():
    """Returns plain-language labels and layman tooltips for all 71 network features."""
    labels_file = STATIC_DIR / "feature_labels.json"
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@app.get("/attack-labels")
def get_attack_labels():
    """Returns plain-language display names and descriptions for all 15 traffic classes."""
    labels_file = STATIC_DIR / "attack_labels.json"
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@app.post("/triage", response_model=TriageResponse)
def triage_incident(payload: TriageRequest):
    """
    Synthesizes an incident response note from structured detection telemetry.
    Strictly follows Section 9.2.1 scope boundary: narrating without overriding.
    """
    data = payload.model_dump()
    note = generate_incident_note(data)

    structured_summary = {
        "flow_id": payload.flow_id,
        "ground_truth_label": payload.ground_truth_label,
        "is_adversarial": payload.is_adversarial,
        "decision_concordance": payload.model_a_prediction.label == payload.model_b_prediction.label,
        "model_a": {
            "label": payload.model_a_prediction.label,
            "confidence": payload.model_a_prediction.confidence,
            "evaded": payload.ground_truth_label is not None and payload.model_a_prediction.label != payload.ground_truth_label
        },
        "model_b": {
            "label": payload.model_b_prediction.label,
            "confidence": payload.model_b_prediction.confidence,
            "defended": payload.ground_truth_label is not None and payload.model_b_prediction.label == payload.ground_truth_label
        },
        "top_perturbed_features": [
            {"feature": clean_feature_name(k), "raw_feature": k, "delta": v}
            for k, v in sorted((payload.perturbation_delta or {}).items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        ] if payload.perturbation_delta else []
    }

    return TriageResponse(
        incident_note=note,
        structured_summary=structured_summary
    )


@app.post("/simulate-attack")
def simulate_attack(req: SimulationRequest):
    """
    Simulates a live multi-step PGD attack against Clean Model A and Robust Model B.
    Samples a random candidate flow and tracks prediction confidence and feature shifts at every step.
    """
    load_runtime_resources()

    if req.category not in class_to_idx:
        raise HTTPException(status_code=400, detail=f"Unknown category: '{req.category}'. Available: {unique_classes}")

    cat_name = req.category
    cat_idx = class_to_idx[cat_name]
    candidate_pool = clean_correct_indices_by_class.get(cat_name, [])

    if not candidate_pool:
        candidate_pool = np.where(y_test_np == cat_idx)[0].tolist()

    if not candidate_pool:
        raise HTTPException(status_code=404, detail=f"No test flows available for category '{cat_name}'")

    # -------------------------------------------------------------------------
    # PART D NOTE: We deliberately use unseeded random sampling (random.choice)
    # here so that repeated clicks in the interactive live simulator evaluate
    # distinct flow instances. This is an intentional, scoped exception to the
    # project's global reproducibility conventions for demo interactivity.
    # -------------------------------------------------------------------------
    selected_idx = random.choice(candidate_pool)
    x_flow = torch.tensor(X_test_np[selected_idx:selected_idx + 1])
    y_true = torch.tensor([cat_idx])

    alpha = req.alpha if req.alpha is not None else (req.epsilon / 4.0)

    t0 = time.time()

    # 1. Traced PGD attack against Model A
    x_adv_a, trace_a = pgd_attack_traced(
        model=model_A,
        x=x_flow,
        y=y_true,
        epsilon=req.epsilon,
        alpha=alpha,
        num_steps=req.num_steps,
        constrained=req.constrained,
        mask=mask_t,
        cumulative_mask=cum_mask_t,
        emp_min=min_t,
        emp_max=max_t,
        random_start=False,
        idx_to_class=idx_to_class
    )

    # 2. Traced PGD attack against Model B (independently attacked)
    x_adv_b, trace_b = pgd_attack_traced(
        model=model_B,
        x=x_flow,
        y=y_true,
        epsilon=req.epsilon,
        alpha=alpha,
        num_steps=req.num_steps,
        constrained=req.constrained,
        mask=mask_t,
        cumulative_mask=cum_mask_t,
        emp_min=min_t,
        emp_max=max_t,
        random_start=False,
        idx_to_class=idx_to_class
    )

    elapsed_ms = (time.time() - t0) * 1000.0

    # Step Flip Detection
    flip_step_a = next((t["step"] for t in trace_a if t["prediction"] != cat_name), None)
    flip_step_b = next((t["step"] for t in trace_b if t["prediction"] != cat_name), None)

    # Enhance trace steps with side-by-side perturbation telemetry and L2 norms
    enhanced_trace_a = []
    enhanced_trace_b = []

    for s in range(len(trace_a)):
        item_a = trace_a[s]
        item_b = trace_b[s]
        delta_arr_a = np.array(item_a["delta"], dtype=float)
        delta_arr_b = np.array(item_b["delta"], dtype=float)

        l2_a = float(np.linalg.norm(delta_arr_a))
        l2_b = float(np.linalg.norm(delta_arr_b))

        # Top 6 features by Model A's perturbation magnitude (showing Model B's delta on the same features)
        sorted_indices = sorted(range(len(delta_arr_a)), key=lambda i: abs(delta_arr_a[i]), reverse=True)[:6]
        top_feats = []
        for i in sorted_indices:
            f_name = feature_names[i]
            is_frz = i in frozen_indices
            top_feats.append({
                "feature_name": f_name,
                "clean_name": clean_feature_name(f_name),
                "delta": float(delta_arr_a[i]),
                "delta_a": float(delta_arr_a[i]),
                "delta_b": float(delta_arr_b[i]),
                "is_frozen": is_frz
            })

        enhanced_trace_a.append({
            "step": item_a["step"],
            "prediction": item_a["prediction"],
            "prediction_index": item_a["prediction_index"],
            "confidence": item_a["confidence"],
            "true_class_confidence": item_a["true_class_confidence"],
            "l2_norm": l2_a,
            "l2_norm_a": l2_a,
            "l2_norm_b": l2_b,
            "top_features": top_feats
        })

        enhanced_trace_b.append({
            "step": item_b["step"],
            "prediction": item_b["prediction"],
            "prediction_index": item_b["prediction_index"],
            "confidence": item_b["confidence"],
            "true_class_confidence": item_b["true_class_confidence"],
            "l2_norm": l2_b,
            "l2_norm_a": l2_a,
            "l2_norm_b": l2_b,
            "top_features": top_feats
        })

    # Final perturbation delta dict for incident note triage integration
    delta_final_a = (x_adv_a - x_flow).view(-1).numpy()
    final_delta_dict = {
        feature_names[i]: float(delta_final_a[i])
        for i in np.where(np.abs(delta_final_a) > 1e-4)[0]
    }

    flow_id = f"sim_flow_{selected_idx}_{cat_name.replace(' ', '_')}"

    return {
        "flow_id": flow_id,
        "raw_flow_index": selected_idx,
        "ground_truth_label": cat_name,
        "ground_truth_index": cat_idx,
        "pool_size": len(candidate_pool),
        "pool_size_note": f"{len(candidate_pool)} test instances available for '{cat_name}'" if len(candidate_pool) < 20 else None,
        "epsilon": req.epsilon,
        "alpha": alpha,
        "num_steps": req.num_steps,
        "constrained": req.constrained,
        "simulation_time_ms": round(elapsed_ms, 2),
        "model_a_flip_step": flip_step_a,
        "model_b_flip_step": flip_step_b,
        "model_a_trace": enhanced_trace_a,
        "model_b_trace": enhanced_trace_b,
        "final_delta_dict": final_delta_dict,
        "feature_metadata": {
            "total_features": len(feature_names),
            "frozen_count": len(frozen_indices),
            "perturbable_count": len(perturbable_indices)
        }
    }


@app.post("/autonomous-tick")
def autonomous_tick():
    """
    Autonomous live network feed endpoint for SOC War Room mode.
    Autonomously intercepts a random test traffic flow (weighted 70% Benign / 30% Attacks),
    decides whether an adversarial evasion attempt is underway, and executes the independent
    Model A and Model B evaluations with complete telemetry and formatted terminal logs.
    """
    load_runtime_resources()

    attack_classes = [c for c in unique_classes if c != "Benign"]

    # 70% Benign, 30% Attack distribution
    is_benign = random.random() < 0.65
    if is_benign:
        cat_name = "Benign"
        is_adversarial = False
    else:
        cat_name = random.choice(attack_classes)
        # 85% of attack traffic is under active adversarial evasion attempt (PGD)
        is_adversarial = random.random() < 0.85

    if is_adversarial:
        # Attacked flow
        constrained = random.random() < 0.75
        eps = random.choice([0.05, 0.10, 0.15, 0.20])
        sim_req = SimulationRequest(
            category=cat_name,
            constrained=constrained,
            epsilon=eps,
            num_steps=7
        )
        sim_res = simulate_attack(sim_req)
        sim_res["is_autonomous"] = True
        sim_res["traffic_type"] = "Attack"
        sim_res["is_adversarial"] = True
        sim_res["tick_timestamp"] = time.strftime("%H:%M:%S")

        # Build movie-style terminal log lines
        flow_tag = sim_res["flow_id"].split("_")[2] if len(sim_res["flow_id"].split("_")) > 2 else "4821"
        ts = sim_res["tick_timestamp"]
        final_a = sim_res["model_a_trace"][-1]
        final_b = sim_res["model_b_trace"][-1]
        evaded_a = final_a["prediction"] != cat_name
        evaded_b = final_b["prediction"] != cat_name

        logs = [
            f"[{ts}] Flow #{flow_tag} intercepted at NIDS inspection point — analyzing...",
            f"[{ts}] Threat signature detected: {cat_name} ({'Realistic Domain-Constrained' if constrained else 'Unconstrained Naive'}, eps={eps:.2f})",
        ]
        if evaded_a:
            logs.append(f"[{ts}] Model A (Clean Baseline): CLASSIFICATION COMPROMISED — Evaded into '{final_a['prediction']}' ({final_a['confidence']*100:.1f}%)")
        else:
            logs.append(f"[{ts}] Model A (Clean Baseline): Threat detected — '{final_a['prediction']}' ({final_a['confidence']*100:.1f}%)")

        if not evaded_b:
            logs.append(f"[{ts}] Model B (AdvRoNIDS Robust): THREAT NEUTRALIZED — Verified '{final_b['prediction']}' ({final_b['confidence']*100:.1f}%)")
        else:
            logs.append(f"[{ts}] Model B (AdvRoNIDS Robust): Prediction '{final_b['prediction']}' ({final_b['confidence']*100:.1f}%)")

        logs.append(f"[{ts}] Telemetry recorded. Incident ready for automated AI triage narration.")
        sim_res["log_entries"] = logs
        return sim_res

    else:
        # Clean unattacked flow (Benign or unperturbed attack)
        cat_idx = class_to_idx[cat_name]
        candidate_pool = clean_correct_indices_by_class.get(cat_name, [])
        if not candidate_pool:
            candidate_pool = np.where(y_test_np == cat_idx)[0].tolist()
        selected_idx = random.choice(candidate_pool)
        x_flow = torch.tensor(X_test_np[selected_idx:selected_idx + 1])

        t0 = time.time()
        with torch.no_grad():
            logits_a = model_A(x_flow)
            probs_a = F.softmax(logits_a, dim=1).view(-1).numpy()
            pred_idx_a = int(np.argmax(probs_a))
            pred_a = idx_to_class[pred_idx_a]
            conf_a = float(probs_a[pred_idx_a])
            true_conf_a = float(probs_a[cat_idx])

            logits_b = model_B(x_flow)
            probs_b = F.softmax(logits_b, dim=1).view(-1).numpy()
            pred_idx_b = int(np.argmax(probs_b))
            pred_b = idx_to_class[pred_idx_b]
            conf_b = float(probs_b[pred_idx_b])
            true_conf_b = float(probs_b[cat_idx])
        elapsed_ms = (time.time() - t0) * 1000.0

        flow_id = f"sim_flow_{selected_idx}_{cat_name.replace(' ', '_')}"
        ts = time.strftime("%H:%M:%S")

        # 8-step clean trace for animation
        clean_step_a = {
            "step": 0,
            "prediction": pred_a,
            "prediction_index": pred_idx_a,
            "confidence": conf_a,
            "true_class_confidence": true_conf_a,
            "l2_norm": 0.0,
            "l2_norm_a": 0.0,
            "l2_norm_b": 0.0,
            "top_features": []
        }
        clean_step_b = {
            "step": 0,
            "prediction": pred_b,
            "prediction_index": pred_idx_b,
            "confidence": conf_b,
            "true_class_confidence": true_conf_b,
            "l2_norm": 0.0,
            "l2_norm_a": 0.0,
            "l2_norm_b": 0.0,
            "top_features": []
        }

        trace_a = [dict(clean_step_a, step=s) for s in range(8)]
        trace_b = [dict(clean_step_b, step=s) for s in range(8)]

        logs = [
            f"[{ts}] Flow #{selected_idx} intercepted — analyzing baseline packet...",
            f"[{ts}] Traffic profile: {cat_name} (Unperturbed Standard Traffic)",
            f"[{ts}] Model A: Classified as '{pred_a}' ({conf_a*100:.1f}%) — NORMAL",
            f"[{ts}] Model B: Classified as '{pred_b}' ({conf_b*100:.1f}%) — NORMAL",
            f"[{ts}] Flow cleared. No adversarial anomaly detected."
        ]

        return {
            "flow_id": flow_id,
            "raw_flow_index": selected_idx,
            "ground_truth_label": cat_name,
            "ground_truth_index": cat_idx,
            "pool_size": len(candidate_pool),
            "pool_size_note": None,
            "epsilon": 0.0,
            "alpha": 0.0,
            "num_steps": 7,
            "constrained": True,
            "simulation_time_ms": round(elapsed_ms, 2),
            "model_a_flip_step": None,
            "model_b_flip_step": None,
            "model_a_trace": trace_a,
            "model_b_trace": trace_b,
            "final_delta_dict": {},
            "feature_metadata": {
                "total_features": len(feature_names),
                "frozen_count": len(frozen_indices),
                "perturbable_count": len(perturbable_indices)
            },
            "is_autonomous": True,
            "traffic_type": "Benign" if cat_name == "Benign" else "Attack",
            "is_adversarial": False,
            "tick_timestamp": ts,
            "log_entries": logs
        }


# Serve Static UI Frontend
STATIC_DIR = ROOT_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AdvRoNIDS Gateway Running. static/index.html not found.</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
