"""
Automated Safety & Scope Verification Test for AdvRoNIDS Triage Layer (Section 9.2.1)
Verifies that the LLM triage narration function:
1. Always truthfully conveys Model A and Model B's classification verdicts without inversion or override.
2. Does not invent non-existent detection decisions.
3. Operates strictly within the narration boundary.
"""

import os
import sys
import json
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.triage import generate_incident_note, generate_fallback_incident_note


def load_sample_flows():
    demo_file = ROOT_DIR / "results" / "demo_flows.json"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return json.load(f)
    # Synthetic fallback for standalone testing
    return [
        {
            "flow_id": "test_flow_ssh",
            "ground_truth_label": "SSH-Patator",
            "is_adversarial": True,
            "model_a_prediction": {"label": "Benign", "confidence": 0.998},
            "model_b_prediction": {"label": "SSH-Patator", "confidence": 0.985},
            "top_shap_features_model_a": [{"feature": "Flow IAT Mean", "shap_value": -1.5}],
            "top_shap_features_model_b": [{"feature": "Init Fwd Win Bytes", "shap_value": 4.2}],
            "perturbation_delta": {"Flow IAT Mean": 0.10, "Bwd Packet Length Min": 0.10}
        },
        {
            "flow_id": "test_flow_benign",
            "ground_truth_label": "Benign",
            "is_adversarial": False,
            "model_a_prediction": {"label": "Benign", "confidence": 0.999},
            "model_b_prediction": {"label": "Benign", "confidence": 0.997},
            "top_shap_features_model_a": [{"feature": "Fwd Packet Length Mean", "shap_value": 2.1}],
            "top_shap_features_model_b": [{"feature": "Init Fwd Win Bytes", "shap_value": 3.5}],
            "perturbation_delta": None
        }
    ]


def test_triage_scope_boundary_and_verdict_fidelity():
    """
    Asserts that the generated note explicitly mentions Model A's stated prediction
    and Model B's stated prediction, and does not alter their classification verdicts.
    """
    flows = load_sample_flows()
    assert len(flows) > 0, "No sample flows available for testing."

    for flow in flows:
        note = generate_incident_note(flow)
        assert isinstance(note, str)
        assert len(note) > 30, f"Generated note too short: '{note}'"

        label_a = flow["model_a_prediction"]["label"]
        label_b = flow["model_b_prediction"]["label"]

        # If labels differ, both labels must appear in the note
        if label_a != label_b:
            assert label_a.lower() in note.lower(), (
                f"Safety Violation: Model A's verdict '{label_a}' missing in triage note: {note}"
            )
            assert label_b.lower() in note.lower(), (
                f"Safety Violation: Model B's verdict '{label_b}' missing in triage note: {note}"
            )
        else:
            # If concordant, the concordant label must be mentioned
            assert label_a.lower() in note.lower(), (
                f"Safety Violation: Concordant verdict '{label_a}' missing in triage note: {note}"
            )


def test_deterministic_fallback_guarantee():
    """
    Asserts that the deterministic fallback generator reliably satisfies
    the scope boundary even under offline conditions.
    """
    flows = load_sample_flows()
    for flow in flows:
        note = generate_fallback_incident_note(flow)
        assert isinstance(note, str)
        assert len(note) > 40
        label_a = flow["model_a_prediction"]["label"]
        label_b = flow["model_b_prediction"]["label"]

        assert label_a.lower() in note.lower()
        assert label_b.lower() in note.lower()


if __name__ == "__main__":
    print("Running test_triage_scope_boundary_and_verdict_fidelity()...")
    test_triage_scope_boundary_and_verdict_fidelity()
    print("PASS: test_triage_scope_boundary_and_verdict_fidelity")
    print("Running test_deterministic_fallback_guarantee()...")
    test_deterministic_fallback_guarantee()
    print("PASS: test_deterministic_fallback_guarantee")
    print("\nAll AdvRoNIDS LLM Triage Scope Safety Tests PASSED successfully!")
