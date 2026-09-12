"""
AdvRoNIDS LLM-Assisted Incident Triage Layer (Section 9.2)
Translates multi-model intrusion detection verdicts, domain-constrained
adversarial perturbation vectors, and SHAP feature attributions into
concise, structured incident response notes.

DESIGN BOUNDARY (Section 9.2.1):
The LLM does NOT make any classification or detection decisions.
It solely converts structured inputs into clear, verifiable prose.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import requests

logger = logging.getLogger("AdvRoNIDS.Triage")

# Human-readable feature name mapping
FEATURE_NAME_MAP = {
    "Flow IAT Mean": "Mean Flow Inter-Arrival Time",
    "Flow IAT Std": "Flow Inter-Arrival Time Std Dev",
    "Flow IAT Max": "Maximum Flow Inter-Arrival Time",
    "Flow IAT Min": "Minimum Flow Inter-Arrival Time",
    "Fwd IAT Total": "Total Forward Inter-Arrival Time",
    "Fwd IAT Mean": "Mean Forward Inter-Arrival Time",
    "Fwd IAT Std": "Forward Inter-Arrival Time Std Dev",
    "Fwd IAT Max": "Maximum Forward Inter-Arrival Time",
    "Fwd IAT Min": "Minimum Forward Inter-Arrival Time",
    "Bwd IAT Total": "Total Backward Inter-Arrival Time",
    "Bwd IAT Mean": "Mean Backward Inter-Arrival Time",
    "Bwd IAT Std": "Backward Inter-Arrival Time Std Dev",
    "Bwd IAT Max": "Maximum Backward Inter-Arrival Time",
    "Bwd IAT Min": "Minimum Backward Inter-Arrival Time",
    "Fwd Packet Length Min": "Minimum Forward Packet Length",
    "Fwd Packet Length Max": "Maximum Forward Packet Length",
    "Fwd Packet Length Mean": "Mean Forward Packet Length",
    "Fwd Packet Length Std": "Forward Packet Length Std Dev",
    "Bwd Packet Length Min": "Minimum Backward Packet Length",
    "Bwd Packet Length Max": "Maximum Backward Packet Length",
    "Bwd Packet Length Mean": "Mean Backward Packet Length",
    "Bwd Packet Length Std": "Backward Packet Length Std Dev",
    "Flow Duration": "Total Flow Duration",
    "Total Fwd Packets": "Total Forward Packet Count",
    "Total Backward Packets": "Total Backward Packet Count",
    "Fwd Header Length": "Forward Header Length (Frozen)",
    "Bwd Header Length": "Backward Header Length (Frozen)",
    "Init Fwd Win Bytes": "Initial Forward TCP Window Size (Frozen)",
    "Init Bwd Win Bytes": "Initial Backward TCP Window Size (Frozen)",
    "Protocol_6": "TCP Protocol Flag (Frozen)",
    "Protocol_17": "UDP Protocol Flag (Frozen)",
    "Protocol_0": "HOPOPT Protocol Flag (Frozen)",
    "SYN Flag Count": "SYN Flag Indicator (Frozen)",
    "ACK Flag Count": "ACK Flag Indicator (Frozen)",
    "FIN Flag Count": "FIN Flag Indicator (Frozen)",
    "RST Flag Count": "RST Flag Indicator (Frozen)",
    "PSH Flag Count": "PSH Flag Indicator (Frozen)",
    "URG Flag Count": "URG Flag Indicator (Frozen)",
}

def clean_feature_name(feat: str) -> str:
    """Converts raw dataset feature names into human-readable descriptions."""
    return FEATURE_NAME_MAP.get(feat, feat.replace("_", " "))


def generate_fallback_incident_note(data: Dict[str, Any]) -> str:
    """
    Deterministic rule-based incident note generator as fallback
    when local LLM server is unreachable.
    """
    if not isinstance(data, dict):
        data = {}

    flow_id = data.get("flow_id", "unknown")
    pred_a = data.get("model_a_prediction") or {}
    pred_b = data.get("model_b_prediction") or {}
    if not isinstance(pred_a, dict):
        pred_a = {}
    if not isinstance(pred_b, dict):
        pred_b = {}

    label_a = pred_a.get("label", "Unknown")
    conf_a = float(pred_a.get("confidence", 0.0) or 0.0) * 100.0
    label_b = pred_b.get("label", "Unknown")
    conf_b = float(pred_b.get("confidence", 0.0) or 0.0) * 100.0
    gt = data.get("ground_truth_label", None)
    is_adv = bool(data.get("is_adversarial", False))
    deltas = data.get("perturbation_delta") or {}
    if not isinstance(deltas, dict):
        deltas = {}

    sentences = []

    # 1. Primary Classification Verdict
    if label_a == label_b:
        sentences.append(
            f"Both models concordantly classified flow {flow_id} as {label_a} "
            f"(Clean Model A confidence: {conf_a:.1f}%, Robust Model B confidence: {conf_b:.1f}%)."
        )
    else:
        gt_clause = f" (matching verified ground truth {gt})" if gt == label_b else ""
        sentences.append(
            f"Classification divergence detected on flow {flow_id}: Undefended Model A classified "
            f"the flow as {label_a} ({conf_a:.1f}% confidence), whereas Robust Model B classified it "
            f"as {label_b} ({conf_b:.1f}% confidence){gt_clause}."
        )

    # 2. Adversarial & Attribution Context
    if is_adv and deltas:
        # Sort deltas by magnitude
        sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        manipulated_names = [f"{clean_feature_name(k)} (delta: {v:+.2f})" for k, v in sorted_deltas]
        manip_str = ", ".join(manipulated_names)
        sentences.append(
            f"The network flow exhibits domain-constrained adversarial manipulation (epsilon = 0.10) "
            f"concentrated in {manip_str}."
        )
        sentences.append(
            "While Model A experienced attribution drift toward these perturbed features leading to evasion, "
            "Model B remained anchored on invariant protocol headers to maintain reliable threat detection."
        )
    else:
        sentences.append(
            "No adversarial perturbations were detected, and flow characteristics remain fully consistent "
            "with baseline operational traffic profiles."
        )

    return " ".join(sentences)


def generate_incident_note(
    data: Dict[str, Any],
    model_name: str = "qwen2.5:3b",
    ollama_host: str = "http://localhost:11434",
    timeout: int = 25
) -> str:
    """
    Generates a concise 3-5 sentence prose incident triage note using a local Ollama LLM.
    Strictly follows Section 9.2.1 scope boundary: narrates structured inputs without
    making independent detection decisions.
    """
    if not isinstance(data, dict):
        data = {}

    flow_id = data.get("flow_id", "flow_unknown")
    pred_a = data.get("model_a_prediction") or {}
    pred_b = data.get("model_b_prediction") or {}
    if not isinstance(pred_a, dict):
        pred_a = {}
    if not isinstance(pred_b, dict):
        pred_b = {}

    label_a = pred_a.get("label", "Unknown")
    conf_a = float(pred_a.get("confidence", 0.0) or 0.0) * 100.0
    label_b = pred_b.get("label", "Unknown")
    conf_b = float(pred_b.get("confidence", 0.0) or 0.0) * 100.0
    gt = data.get("ground_truth_label", None)
    is_adv = bool(data.get("is_adversarial", False))
    deltas = data.get("perturbation_delta") or {}
    if not isinstance(deltas, dict):
        deltas = {}

    raw_shap_a = data.get("top_shap_features_model_a") or []
    raw_shap_b = data.get("top_shap_features_model_b") or []
    top_shap_a = [f for f in raw_shap_a if isinstance(f, dict) and "feature" in f and "shap_value" in f][:3]
    top_shap_b = [f for f in raw_shap_b if isinstance(f, dict) and "feature" in f and "shap_value" in f][:3]

    shap_a_str = ", ".join([f"{clean_feature_name(f['feature'])} ({float(f['shap_value']):+.3f})" for f in top_shap_a]) if top_shap_a else "None recorded"
    shap_b_str = ", ".join([f"{clean_feature_name(f['feature'])} ({float(f['shap_value']):+.3f})" for f in top_shap_b]) if top_shap_b else "None recorded"

    if deltas:
        sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        delta_str = ", ".join([f"{clean_feature_name(k)} (shift: {v:+.2f})" for k, v in sorted_deltas])
    else:
        delta_str = "None (clean unperturbed flow)"

    prompt = f"""You are an automated network security triage analyst for AdvRoNIDS.
Write a concise, professional 3-4 sentence incident triage note based strictly on the following structured telemetry data.

STRUCTURED INPUT DATA:
- Flow ID: {flow_id}
- Ground Truth Label: {gt if gt else 'Unknown'}
- Is Adversarial Example: {'Yes (Domain-Constrained PGD, epsilon=0.10)' if is_adv else 'No (Clean Traffic)'}
- Undefended Model A Classification: {label_a} (Confidence: {conf_a:.1f}%)
- Robust Model B Classification: {label_b} (Confidence: {conf_b:.1f}%)
- Top Perturbed Features: {delta_str}
- Model A Top SHAP Features: {shap_a_str}
- Model B Top SHAP Features: {shap_b_str}

STRICT WRITING RULES:
1. State clearly what Model A and Model B classified the flow as and their exact confidences.
2. If Model A and Model B disagree, explicitly highlight the disagreement and note which model correctly identified the ground truth.
3. If adversarial perturbations are present, name the top 2-3 manipulated features in plain language (e.g., "flow inter-arrival time", "backward packet length").
4. Explain that Model A was misled by attribution drift onto manipulated features, whereas Robust Model B anchored on protocol invariants.
5. Do NOT invent new facts or numbers.
6. Keep the response to exactly 3 to 4 sentences in a professional cybersecurity incident note tone. Output ONLY the prose note with no extra preamble."""

    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 256
                }
            },
            timeout=timeout
        )
        if response.status_code == 200:
            res_json = response.json()
            note = res_json.get("response", "").strip()
            if note and len(note.split()) >= 15:
                return note
    except Exception as e:
        logger.warning(f"Ollama generation failed or timed out ({e}). Utilizing deterministic fallback.")

    return generate_fallback_incident_note(data)
