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


def get_available_ollama_model(ollama_host: str = "http://localhost:11434") -> str:
    """Detects available installed Ollama models and picks the best match."""
    preferred = ["qwen2.5:3b", "llama3:latest", "llama3.2:3b", "deepseek-coder-v2:latest", "mistral:latest"]
    try:
        r = requests.get(f"{ollama_host}/api/tags", timeout=1.5)
        if r.status_code == 200:
            installed = [m.get("name", "") for m in r.json().get("models", [])]
            for p in preferred:
                if p in installed:
                    return p
            if installed:
                return installed[0]
    except Exception:
        pass
    return "qwen2.5:3b"


# High-entropy seed components for procedural generation & LLM creativity priming
ATTACK_PREFIXES = [
    "Ghost-SYN", "Adaptive-Echo", "Subnet-Vortex", "Cipher-Shift", "Quantum-Probe",
    "Shadow-Jitter", "Polymorphic-Pulse", "Stealth-Beacon", "Zero-Day-Strobe", "Chronos-Burst",
    "Deep-Tunnel", "Phantom-Burst", "Neural-Desync", "BGP-Siphon", "Temporal-Drift"
]

ATTACK_SUFFIXES = [
    "Polymorphic Flood", "Timing Desync Infiltration", "Exfiltration Beacon", "Subnet Sweep",
    "Slowloris Header Exhaustion", "Credential Harvest Storm", "Payload Quantization Exploit",
    "Asynchronous Micro-Burst", "Stateful Connection Bleed", "Feature-Masking Tunnel"
]

THREAT_ACTOR_POOLS = [
    "APT-28 (Fancy Bear)", "APT-29 (Cozy Bear)", "Lazarus Group Sub-Cluster", "Sandworm Team",
    "FIN7 Financial Threat Cluster", "Volt Typhoon Infiltration Unit", "DarkSide RaaS Syndicate",
    "Equation Group Unit", "Magecart Supply-Chain Consortium", "BlackCat Cyber Cartel"
]

TARGET_SERVICES_POOL = [
    "Edge BGP Router / Port 179", "Zero-Trust Bastion Host / Port 22", "NGINX SSL Reverse Proxy / Port 443",
    "Internal Kubernetes API / Port 6443", "Core SQL Cluster / Port 3306", "Active Directory Kerberos / Port 88",
    "Public REST Gateway / Port 8080", "Industrial SCADA Modbus / Port 502", "DNS Anycast Resolver / Port 53"
]

MITRE_TECHNIQUE_POOL = [
    ("T1498.001", "Network Denial of Service: Direct Network Flood"),
    ("T1499.003", "Endpoint Denial of Service: Application Exhaustion"),
    ("T1110.001", "Credential Access: Password Guessing / Brute Force"),
    ("T1046", "Discovery: Network Service Scanning"),
    ("T1071.001", "Command and Control: Web Protocols Beaconing"),
    ("T1059.007", "Execution: JavaScript / XSS Code Injection"),
    ("T1190", "Initial Access: Exploit Public-Facing Application"),
    ("T1048.003", "Exfiltration: Exfiltration Over Unencrypted Protocol")
]


def generate_procedural_novel_attack() -> Dict[str, Any]:
    """Generates a rich, high-entropy novel attack scenario procedurally."""
    prefix = random.choice(ATTACK_PREFIXES)
    suffix = random.choice(ATTACK_SUFFIXES)
    base_cat = random.choice(BENCHMARK_ATTACK_CATEGORIES)
    actor = random.choice(THREAT_ACTOR_POOLS)
    target = random.choice(TARGET_SERVICES_POOL)
    mitre_id, mitre_desc = random.choice(MITRE_TECHNIQUE_POOL)
    eps = round(random.uniform(0.08, 0.22), 2)
    steps = random.choice([6, 7, 8, 9, 10])

    evasion_mechanisms = [
        f"Manipulates backward packet inter-arrival times and packet length variance to camouflage adversarial flow as legitimate traffic to {target}.",
        f"Executes sub-threshold micro-burst perturbation with bounded epsilon={eps}, deliberately staying within benign statistical distributions.",
        f"Applies non-contiguous jitter across forward and backward timing streams while strictly preserving invariant protocol flags.",
        f"Perturbs TCP flow duration and backward header metrics using projected gradient ascent to bypass linear decision boundaries."
    ]

    return {
        "attack_name": f"{prefix} {suffix}",
        "base_category": base_cat,
        "threat_actor": actor,
        "mitre_technique": f"{mitre_id} - {mitre_desc}",
        "target_service": target,
        "epsilon": eps,
        "num_steps": steps,
        "scenario_brief": random.choice(evasion_mechanisms)
    }


def clean_feature_name(feat: str) -> str:
    """Converts raw dataset feature names into human-readable descriptions."""
    return FEATURE_NAME_MAP.get(feat, feat.replace("_", " "))


def generate_novel_attack_scenario(
    model_name: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Prompts Ollama to generate a genuinely creative, novel cyber attack scenario.
    Dynamically injects randomized seeds so Ollama never generates duplicate attacks.
    Returns structured parameters used directly by the AdvRoNIDS PGD simulation engine.
    """
    if not model_name:
        model_name = get_available_ollama_model(ollama_host)

    target_category = random.choice(BENCHMARK_ATTACK_CATEGORIES)
    creative_seed_word = random.choice(ATTACK_PREFIXES)
    random_actor_seed = random.choice(THREAT_ACTOR_POOLS)
    random_target_seed = random.choice(TARGET_SERVICES_POOL)

    prompt = f"""You are a Red Team Cyber Threat Researcher inventing a brand new, highly sophisticated adversarial network attack.
INVENT A UNIQUE, CREATIVE ATTACK SCENARIO targeting category: "{target_category}".
Do NOT use generic names. Create a distinct, memorable codename (e.g. incorporating themes like "{creative_seed_word}").

Target Infrastructure Inspiration: {random_target_seed}
Threat Actor Inspiration: {random_actor_seed}

Respond with ONLY a valid JSON object matching this schema:
{{
  "attack_name": "A unique, creative attack title (e.g., '{creative_seed_word} Shadow Injection')",
  "base_category": "{target_category}",
  "threat_actor": "Threat Actor / APT Group Name",
  "mitre_technique": "MITRE ATT&CK ID and Technique (e.g. T1498.001 - Direct Network Flood)",
  "target_service": "Specific Target Service and Port",
  "epsilon": {round(random.uniform(0.08, 0.22), 2)},
  "num_steps": {random.choice([6, 7, 8, 9, 10])},
  "scenario_brief": "2 sentence technical description of how the attacker perturbs packet timing and packet size distributions to evade AI classifiers."
}}
Output ONLY the raw JSON object. Do not include markdown ticks, preamble, or explanation."""

    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.88,
                    "top_p": 0.95,
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
                logger.info(f"Ollama successfully generated attack: {data.get('attack_name')}")
                return data
            else:
                data["base_category"] = target_category
                return data
    except Exception as e:
        logger.info(f"Ollama attack generation fallback ({e}). Using high-entropy procedural generator.")

    return generate_procedural_novel_attack()


def generate_custom_attack_report(
    attack_info: Dict[str, Any],
    model_name: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    timeout: int = 10
) -> Dict[str, str]:
    """
    Generates tailored threat forensics narrative and firewall mitigation playbooks
    for a specific simulated attack scenario based on actual model findings.
    """
    if not model_name:
        model_name = get_available_ollama_model(ollama_host)

    attack_name = attack_info.get("attack_name", "Adversarial Intrusion")
    base_cat = attack_info.get("base_category", "Unknown")
    pred_a = attack_info.get("model_a_pred", "Benign")
    conf_a = float(attack_info.get("model_a_conf", 0.0)) * 100.0
    pred_b = attack_info.get("model_b_pred", base_cat)
    conf_b = float(attack_info.get("model_b_conf", 0.0)) * 100.0
    evaded_a = attack_info.get("model_a_evaded", True)
    top_features = attack_info.get("top_features", [])

    feat_str = ", ".join([f"{f.get('feature_name', '')} (Δ: {f.get('delta_a', 0.0):+.2f})" for f in top_features[:3]])

    prompt = f"""You are a Principal Cybersecurity SOC Forensics Director investigating an adversarial cyber intrusion.
Write a structured, highly technical threat briefing based on these findings:

ATTACK INCIDENT FINDINGS:
- Incident Scenario: {attack_name} (Ground Truth: {base_cat})
- Attributed Actor: {attack_info.get('threat_actor', 'Unknown')}
- MITRE Technique: {attack_info.get('mitre_technique', 'T1498')}
- Target: {attack_info.get('target_service', 'Edge Network')}
- Undefended Model A Result: {pred_a} ({conf_a:.1f}% confidence) -> {'COMPROMISED / MISCLASSIFIED' if evaded_a else 'CORRECTLY IDENTIFIED'}
- AdvRoNIDS Robust Model B Result: {pred_b} ({conf_b:.1f}% confidence) -> {'DEFENSE SUCCEEDED' if pred_b == base_cat else 'FLAGGED'}
- Manipulated Feature Gradients: {feat_str}

Respond with ONLY a JSON object containing these 3 fields:
{{
  "executive_summary": "2 concise sentences explaining the attack vector and why Model A failed while Model B maintained security.",
  "root_cause_analysis": "2 technical sentences explaining how manipulating {feat_str or 'packet timing'} deceived the decision boundary.",
  "mitigation_playbook": "3 actionable bullet points with specific firewall/IDS and edge mitigation actions."
}}
Output ONLY the raw JSON object."""

    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.35,
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
            data = json.loads(res_text)
            return {
                "executive_summary": str(data.get("executive_summary", "")).strip(),
                "root_cause_analysis": str(data.get("root_cause_analysis", "")).strip(),
                "mitigation_playbook": str(data.get("mitigation_playbook", "")).strip()
            }
    except Exception as e:
        logger.info(f"Ollama custom report fallback ({e})")

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
