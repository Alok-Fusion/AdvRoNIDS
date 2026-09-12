"""
Generates results/example_incident_notes.md with pre-generated
illustrative incident notes for Section 9.2 of the paper.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.triage import generate_incident_note

def export_notes():
    demo_path = Path("results/demo_flows.json")
    with open(demo_path, "r", encoding="utf-8") as f:
        demo_flows = json.load(f)

    # Pick 3 diverse illustrative examples
    # 1. SSH-Patator (Evasion of Model A, Defended by Model B)
    # 2. DoS GoldenEye (Evasion of Model A, Defended by Model B)
    # 3. Benign (Clean Traffic Agreement)
    selected = [
        demo_flows[0],
        demo_flows[1],
        demo_flows[6]
    ]

    out = [
        "# AdvRoNIDS LLM-Assisted Incident Triage Layer (Section 9.2)",
        "",
        "## Illustrative Incident Triage Notes for Paper (Section 9.2)",
        "",
        "The following incident response notes illustrate how the AdvRoNIDS triage layer converts structured telemetry (multi-model verdicts, domain-constrained perturbation deltas, and SHAP feature attributions) into concise, actionable prose. Per **Section 9.2.1's explicit scope boundary**, the LLM operates strictly in a narration capacity and does not alter or make classification decisions.",
        "",
        "---",
        ""
    ]

    for i, flow in enumerate(selected, 1):
        note = generate_incident_note(flow)
        label_a = flow['model_a_prediction']['label']
        conf_a = flow['model_a_prediction']['confidence'] * 100.0
        label_b = flow['model_b_prediction']['label']
        conf_b = flow['model_b_prediction']['confidence'] * 100.0
        gt = flow['ground_truth_label']
        is_adv = flow['is_adversarial']
        tag = flow['category_tag']

        out.append(f"### Example {i}: `{flow['flow_id']}` ({tag})")
        out.append("")
        out.append(f"- **Ground Truth:** `{gt}`")
        out.append(f"- **Telemetry Condition:** `{'Domain-Constrained PGD (eps = 0.10, Pi_S)' if is_adv else 'Clean Operational Baseline'}`")
        out.append(f"- **Undefended Model A Verdict:** `{label_a}` ({conf_a:.1f}% confidence) " + ("[EVADED]" if label_a != gt else "[ACCURATE]"))
        out.append(f"- **Robust Model B Verdict:** `{label_b}` ({conf_b:.1f}% confidence) " + ("[DEFENDED]" if label_b == gt else "[EVADED]"))
        
        if flow.get('perturbation_delta'):
            deltas = flow['perturbation_delta']
            sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            delta_str = ", ".join([f"`{k}` (Δ = {v:+.2f})" for k, v in sorted_deltas])
            out.append(f"- **Key Manipulated Features:** {delta_str}")
        
        out.append("")
        out.append("> [!NOTE]")
        out.append("> **AdvRoNIDS Incident Note (LLM Telemetry Narration):**")
        out.append(f'> *"{note}"*')
        out.append("")
        out.append("---")
        out.append("")

    out_text = "\n".join(out)
    out_file = Path("results/example_incident_notes.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"Successfully generated {out_file}")

if __name__ == "__main__":
    export_notes()
