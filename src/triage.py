"""
AdvRoNIDS LLM-Assisted Incident Triage & Attack Synthesis Layer (Section 9.2)
Provides:
1. Dynamic Novel Cyber Attack Scenario Generation via local Ollama LLM
2. Attack-Specific Forensics Assessment & Mitigation Playbook Generation
3. Real-time Incident Triage Briefing Synthesis with Deterministic Fallbacks
"""

import json
import logging
import random
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

# Pool of 14 valid benchmark categories
BENCHMARK_ATTACK_CATEGORIES = [
    "DoS Hulk", "DDoS", "DoS GoldenEye", "FTP-Patator", "DoS slowloris",
    "DoS Slowhttptest", "SSH-Patator", "PortScan", "Web Attack-Brute Force",
    "Bot", "Web Attack-XSS", "Infiltration", "Web Attack-Sql Injection", "Heartbleed"
]

FALLBACK_ATTACK_TEMPLATES = [
    {
        "attack_name": "Ghost-SYN Polymorphic Flood",
        "base_category": "DDoS",
        "threat_actor": "APT-28 Cyber Vanguard",
        "mitre_technique": "T1498.001 - Direct Network Flood",
        "target_service": "Edge Border Gateway / Port 443",
        "epsilon": 0.14,
        "num_steps": 8,
        "scenario_brief": "Distributed multi-vector SYN flood manipulating TCP window scaling and burst jitter to mimic bursty legitimate video streaming traffic."
    },
    {
        "attack_name": "Adaptive Slowloris Tunneling",
        "base_category": "DoS slowloris",
        "threat_actor": "Shadow-Broker Swarm",
        "mitre_technique": "T1499.003 - Application Exhaustion Flood",
        "target_service": "NGINX Reverse Proxy / Port 80",
        "epsilon": 0.12,
        "num_steps": 7,
        "scenario_brief": "Low-and-slow HTTP header exhaustion embedding artificial packet interval jitter to bypass traditional rate-limiting heuristics."
    },
    {
        "attack_name": "Credential-Stuffing SSH Storm",
        "base_category": "SSH-Patator",
        "threat_actor": "FIN7 Automated Campaign",
        "mitre_technique": "T1110.001 - Password Guessing",
        "target_service": "SSH Bastion / Port 22",
        "epsilon": 0.15,
        "num_steps": 9,
        "scenario_brief": "Automated brute-force authentication sequence modifying backward packet sizes to masquerade as normal interactive terminal sessions."
    },
    {
        "attack_name": "Subnet Stealth Sweep Probe",
        "base_category": "PortScan",
        "threat_actor": "Lazarus Heuristic Recon",
        "mitre_technique": "T1046 - Network Service Discovery",
        "target_service": "Internal DMZ Subnet (Ports 1-1024)",
        "epsilon": 0.10,
        "num_steps": 6,
        "scenario_brief": "Slow asynchronous TCP half-open port enumeration designed to stay beneath threshold alert triggers by perturbing flow duration."
    },
    {
        "attack_name": "Obfuscated XSS Infiltration Vector",
        "base_category": "Web Attack-XSS",
        "threat_actor": "Magecart Supply Chain",
        "mitre_technique": "T1059.007 - JavaScript Execution",
        "target_service": "Public E-Commerce API / Port 443",
        "epsilon": 0.18,
        "num_steps": 10,
        "scenario_brief": "Payload injection with non-contiguous chunked transfer encoding, shifting backward packet metrics into benign ranges."
    },
    {
        "attack_name": "C2 Heartbeat Exfiltration Beacon",
        "base_category": "Bot",
        "threat_actor": "RedLine Stealer Operator",
        "mitre_technique": "T1071.001 - Web Protocols Command & Control",
        "target_service": "Outbound HTTPS Channel / Port 8443",
        "epsilon": 0.11,
        "num_steps": 7,
        "scenario_brief": "Persistent covert channel embedding periodic ping jitter into standard background telemetry to evade AI anomaly detectors."
    }
]


def clean_feature_name(feat: str) -> str:
    """Converts raw dataset feature names into human-readable descriptions."""
    return FEATURE_NAME_MAP.get(feat, feat.replace("_", " "))


def generate_novel_attack_scenario(
    model_name: str = "qwen2.5:3b",
    ollama_host: str = "http://localhost:11434",
    timeout: int = 8
) -> Dict[str, Any]:
    """
    Prompts Ollama to generate a creative, novel cyber attack scenario.
    Returns structured parameters used directly by the AdvRoNIDS PGD simulation engine.
    """
    prompt = f"""You are a Red Team Cyber Threat Emulator. Invent a realistic, novel adversarial cyber attack scenario for testing an AI Network Intrusion Detection System.

Pick one base category from this exact list:
{json.dumps(BENCHMARK_ATTACK_CATEGORIES)}

Respond with ONLY a valid JSON object matching this schema:
{{
  "attack_name": "Creative Attack Name (e.g., Ghost-SYN Polymorphic Flood)",
  "base_category": "One exact name from the list above",
  "threat_actor": "Threat Actor / APT Group Name",
  "mitre_technique": "MITRE ATT&CK ID and Technique (e.g. T1498.001 - Network Flood)",
  "target_service": "Target Service & Port (e.g. Edge Gateway / Port 443)",
  "epsilon": 0.12,
  "num_steps": 7,
  "scenario_brief": "2 sentence description of how the attacker perturbs packet timing and size features to evade AI detection."
}}
Output ONLY the JSON object. Do not add markdown backticks or explanation."""

    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 300
                }
            },
            timeout=timeout
        )
        if response.status_code == 200:
            res_text = response.json().get("response", "").strip()
            # Clean possible markdown wrapping
            if res_text.startswith("```"):
                res_text = res_text.split("```")[1]
                if res_text.startswith("json"):
                    res_text = res_text[4:]
            res_text = res_text.strip()
            data = json.loads(res_text)
            
            # Validate base category
            if data.get("base_category") in BENCHMARK_ATTACK_CATEGORIES:
                data["epsilon"] = max(0.05, min(0.25, float(data.get("epsilon", 0.12))))
                data["num_steps"] = max(4, min(12, int(data.get("num_steps", 7))))
                return data
    except Exception as e:
        logger.info(f"Ollama attack generation skipped ({e}). Using diverse template pool.")

    # Diverse randomized fallback
    chosen = random.choice(FALLBACK_ATTACK_TEMPLATES).copy()
    chosen["epsilon"] = round(random.uniform(0.08, 0.20), 2)
    chosen["num_steps"] = random.choice([6, 7, 8, 9, 10])
    return chosen


def generate_custom_attack_report(
    attack_info: Dict[str, Any],
    model_name: str = "qwen2.5:3b",
    ollama_host: str = "http://localhost:11434",
    timeout: int = 10
) -> Dict[str, str]:
    """
    Generates tailored threat forensics narrative and firewall mitigation playbooks
    for a specific simulated attack scenario based on actual model findings.
    """
    attack_name = attack_info.get("attack_name", "Adversarial Intrusion")
    base_cat = attack_info.get("base_category", "Unknown")
    pred_a = attack_info.get("model_a_pred", "Benign")
    conf_a = float(attack_info.get("model_a_conf", 0.0)) * 100.0
    pred_b = attack_info.get("model_b_pred", base_cat)
    conf_b = float(attack_info.get("model_b_conf", 0.0)) * 100.0
    evaded_a = attack_info.get("model_a_evaded", True)
    top_features = attack_info.get("top_features", [])

    feat_str = ", ".join([f"{f.get('feature_name', '')} (Δ: {f.get('delta_a', 0.0):+.2f})" for f in top_features[:3]])

    prompt = f"""You are a Principal Cybersecurity SOC Analyst evaluating an adversarial AI evasion attack.
Write a structured threat intelligence briefing based on these findings:

ATTACK PROFILE:
- Attack Scenario: {attack_name} (Base signature: {base_cat})
- Threat Actor: {attack_info.get('threat_actor', 'Unknown')}
- MITRE Technique: {attack_info.get('mitre_technique', 'T1498')}
- Undefended Model A Verdict: {pred_a} ({conf_a:.1f}% confidence) -> {'COMPROMISED / EVADED' if evaded_a else 'NORMAL'}
- AdvRoNIDS Robust Model B Verdict: {pred_b} ({conf_b:.1f}% confidence) -> DEFENDED
- Manipulated Feature Vector: {feat_str}

Respond with ONLY a JSON object containing these 3 fields:
{{
  "executive_summary": "2 sentences describing the attack vector and how Model A was fooled while Model B stood ground.",
  "root_cause_analysis": "2 sentences explaining the mathematical feature perturbation exploit.",
  "mitigation_playbook": "2-3 bullet points with specific firewall / IDS mitigation recommendations."
}}
Output ONLY the JSON object."""

    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 350
                }
            },
            timeout=timeout
        )
        if response.status_code == 200:
            res_text = response.json().get("response", "").strip()
            if res_text.startswith("```"):
                res_text = res_text.split("```")[1]
                if res_text.startswith("json"):
                    res_text = res_text[4:]
            res_text = res_text.strip()
            return json.loads(res_text)
    except Exception as e:
        logger.info(f"Ollama custom report generation fallback ({e}).")

    # High-quality deterministic analysis
    return {
        "executive_summary": (
            f"During the '{attack_name}' simulation, the adversary injected realistic feature perturbations "
            f"into {base_cat} traffic. Standard Model A was successfully blinded, misclassifying the attack as '{pred_a}' ({conf_a:.1f}% confidence), "
            f"while AdvRoNIDS Robust Model B held ground with {conf_b:.1f}% confidence."
        ),
        "root_cause_analysis": (
            f"The adversary manipulated non-critical flow metrics ({feat_str}) to induce gradient attribution drift. "
            f"Because Model A relies on fragile continuous correlations, its decision boundary flipped. Model B utilized domain-preserving "
            f"projection (Π_S) to anchor on structural protocol invariants."
        ),
        "mitigation_playbook": (
            f"1. Deploy AdvRoNIDS invariant neural filters at the edge perimeter to neutralize {base_cat} evasion vectors.\n"
            f"2. Implement stateful packet inspection rules correlating TCP window sizing with flow inter-arrival times.\n"
            f"3. Quarantining traffic matching anomalous perturbation signatures on {attack_info.get('target_service', 'Edge Ports')}."
        )
    }


def generate_incident_note(
    data: Dict[str, Any],
    model_name: str = "qwen2.5:3b",
    ollama_host: str = "http://localhost:11434",
    timeout: int = 15
) -> str:
    """
    Generates concise 3-4 sentence incident triage note using Ollama with deterministic fallback.
    """
    if not isinstance(data, dict):
        data = {}

    flow_id = data.get("flow_id", "flow_unknown")
    pred_a = data.get("model_a_prediction") or {}
    pred_b = data.get("model_b_prediction") or {}
    label_a = pred_a.get("label", "Unknown")
    conf_a = float(pred_a.get("confidence", 0.0) or 0.0) * 100.0
    label_b = pred_b.get("label", "Unknown")
    conf_b = float(pred_b.get("confidence", 0.0) or 0.0) * 100.0
    gt = data.get("ground_truth_label", None)
    is_adv = bool(data.get("is_adversarial", False))
    deltas = data.get("perturbation_delta") or {}

    if deltas:
        sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        delta_str = ", ".join([f"{clean_feature_name(k)} ({v:+.2f})" for k, v in sorted_deltas])
    else:
        delta_str = "None"

    prompt = f"""You are an automated network security triage analyst for AdvRoNIDS.
Write a concise 3-sentence incident triage note based on this telemetry:
- Flow ID: {flow_id}
- Ground Truth Threat: {gt}
- Attack Active: {'Yes (Domain-Constrained PGD)' if is_adv else 'No (Normal Traffic)'}
- Undefended Model A: {label_a} ({conf_a:.1f}%)
- Robust Model B: {label_b} ({conf_b:.1f}%)
- Top Perturbed Features: {delta_str}

Output ONLY the 3-sentence note in a professional cybersecurity analyst tone."""

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
                    "num_predict": 200
                }
            },
            timeout=timeout
        )
        if response.status_code == 200:
            note = response.json().get("response", "").strip()
            if note and len(note.split()) >= 12:
                return note
    except Exception:
        pass

    # Deterministic Fallback
    if label_a == label_b:
        return (
            f"Both models concordantly classified flow {flow_id} as {label_a} "
            f"(Model A: {conf_a:.1f}%, Model B: {conf_b:.1f}%). Flow characteristics remain consistent with expected network traffic."
        )
    else:
        return (
            f"Classification divergence detected on flow {flow_id}: Standard Model A was compromised, misclassifying the intrusion as {label_a} ({conf_a:.1f}%), "
            f"while AdvRoNIDS Robust Model B sustained accurate identification of {label_b} ({conf_b:.1f}%). "
            f"Perturbations concentrated in {delta_str} induced attribution drift on baseline models."
        )
