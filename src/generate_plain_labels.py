"""
AdvRoNIDS Plain-Language Translation Dictionary Generator
Generates comprehensive plain-English label and tooltip mappings for all 71 CICFlowMeter
network features and 15 traffic classes.
Used by frontend visualization console (static/index.html) and API layers.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FEASIBILITY_PATH = ROOT_DIR / "data" / "processed" / "feasibility_mask.json"
FEATURE_LABELS_OUT = ROOT_DIR / "static" / "feature_labels.json"
ATTACK_LABELS_OUT = ROOT_DIR / "static" / "attack_labels.json"

# 15 Traffic Classes
ATTACK_CLASSES = {
    "Benign": {
        "display_name": "Normal Traffic",
        "description": "Ordinary, non-malicious network activity (web browsing, streaming, standard business protocols).",
        "category": "Legitimate Traffic",
        "severity": "None"
    },
    "SSH-Patator": {
        "display_name": "SSH Password Guessing",
        "description": "Automated repeated login attempts against remote SSH administrative access.",
        "category": "Brute Force Attack",
        "severity": "High"
    },
    "FTP-Patator": {
        "display_name": "FTP Password Guessing",
        "description": "Automated repeated credential guessing against FTP file transfer services.",
        "category": "Brute Force Attack",
        "severity": "High"
    },
    "DoS GoldenEye": {
        "display_name": "DoS Attack (GoldenEye)",
        "description": "A denial-of-service attack designed to exhaust web server connection pools and memory.",
        "category": "Denial of Service",
        "severity": "High"
    },
    "DoS Hulk": {
        "display_name": "DoS Attack (HULK)",
        "description": "A high-volume HTTP flood attack designed to bypass web caching layers and crash servers.",
        "category": "Denial of Service",
        "severity": "High"
    },
    "DoS Slowhttptest": {
        "display_name": "DoS Attack (SlowHTTPTest)",
        "description": "A slow-rate HTTP attack that holds server connections open until connection tables fill up.",
        "category": "Denial of Service",
        "severity": "Medium"
    },
    "DoS slowloris": {
        "display_name": "DoS Attack (Slowloris)",
        "description": "A low-bandwidth attack sending incomplete HTTP headers to tie up server worker threads.",
        "category": "Denial of Service",
        "severity": "Medium"
    },
    "DDoS": {
        "display_name": "DDoS Flood (Distributed)",
        "description": "A distributed high-rate traffic flood originating from multiple sources to overwhelm target infrastructure.",
        "category": "Distributed Denial of Service",
        "severity": "Critical"
    },
    "PortScan": {
        "display_name": "Port Scanning Probes",
        "description": "Reconnaissance probing to discover which ports, services, and system versions are accessible.",
        "category": "Reconnaissance",
        "severity": "Medium"
    },
    "Bot": {
        "display_name": "Botnet Activity (C2)",
        "description": "Traffic between a compromised internal device and an external botnet Command-and-Control controller.",
        "category": "Malware & Botnet",
        "severity": "High"
    },
    "Web Attack-Brute Force": {
        "display_name": "Web Login Brute-Forcing",
        "description": "Repeated automated password guessing to unlawfully gain access to a web application account.",
        "category": "Web Application Attack",
        "severity": "High"
    },
    "Web Attack-XSS": {
        "display_name": "Cross-Site Scripting (XSS)",
        "description": "Injecting malicious browser scripts into web parameters to target application visitors.",
        "category": "Web Application Attack",
        "severity": "Medium"
    },
    "Web Attack-Sql Injection": {
        "display_name": "SQL Injection Attack",
        "description": "Injecting database commands to illicitly extract sensitive records or bypass authentication.",
        "category": "Web Application Attack",
        "severity": "Critical"
    },
    "Infiltration": {
        "display_name": "Internal Network Infiltration",
        "description": "Post-exploitation lateral movement and internal discovery following an initial breach.",
        "category": "Lateral Movement",
        "severity": "Critical"
    },
    "Heartbleed": {
        "display_name": "Heartbleed Exploit (CVE-2014-0160)",
        "description": "Exploiting OpenSSL memory disclosure flaw to illicitly read server memory and private keys.",
        "category": "Vulnerability Exploit",
        "severity": "Critical"
    }
}

# 71 Features Dictionary
FEATURES_DICT = {
    # Timing & Inter-Arrival Metrics
    "Flow Duration": {
        "label": "Connection Duration",
        "tooltip": "Total time elapsed from the first packet to the last packet in this network flow (an attacker could delay packets to manipulate this).",
        "category": "Timing Metric"
    },
    "Flow IAT Mean": {
        "label": "Average Pause Between Packets",
        "tooltip": "Average time elapsed between consecutive packets in the flow (manipulable by pacing traffic).",
        "category": "Timing Metric"
    },
    "Flow IAT Std": {
        "label": "Packet Timing Consistency",
        "tooltip": "Variation in timing between packets — steady timing produces low variation while bursty traffic produces high variation.",
        "category": "Timing Metric"
    },
    "Flow IAT Max": {
        "label": "Longest Pause Between Packets",
        "tooltip": "The single longest delay observed between any two packets in the entire flow.",
        "category": "Timing Metric"
    },
    "Flow IAT Min": {
        "label": "Shortest Pause Between Packets",
        "tooltip": "The shortest delay recorded between any two consecutive packets in the flow.",
        "category": "Timing Metric"
    },
    "Fwd IAT Total": {
        "label": "Total Outgoing Packet Delays",
        "tooltip": "The combined time spent waiting between outgoing packets sent by the client.",
        "category": "Timing Metric"
    },
    "Fwd IAT Mean": {
        "label": "Average Outgoing Packet Pause",
        "tooltip": "Average time elapsed between consecutive outgoing packets sent to the server.",
        "category": "Timing Metric"
    },
    "Fwd IAT Std": {
        "label": "Outgoing Packet Timing Variation",
        "tooltip": "How erratic or steady the delays are between outgoing client packets.",
        "category": "Timing Metric"
    },
    "Fwd IAT Max": {
        "label": "Longest Outgoing Packet Pause",
        "tooltip": "Maximum delay recorded between any two consecutive outgoing client packets.",
        "category": "Timing Metric"
    },
    "Fwd IAT Min": {
        "label": "Shortest Outgoing Packet Pause",
        "tooltip": "Minimum delay recorded between any two consecutive outgoing client packets.",
        "category": "Timing Metric"
    },
    "Bwd IAT Total": {
        "label": "Total Incoming Response Delays",
        "tooltip": "The combined time spent waiting between incoming response packets from the server.",
        "category": "Timing Metric"
    },
    "Bwd IAT Mean": {
        "label": "Average Incoming Response Pause",
        "tooltip": "Average time elapsed between consecutive response packets arriving from the server.",
        "category": "Timing Metric"
    },
    "Bwd IAT Std": {
        "label": "Incoming Response Timing Variation",
        "tooltip": "How much the delays between server response packets fluctuate.",
        "category": "Timing Metric"
    },
    "Bwd IAT Max": {
        "label": "Longest Incoming Response Pause",
        "tooltip": "Maximum delay recorded between any two consecutive server response packets.",
        "category": "Timing Metric"
    },
    "Bwd IAT Min": {
        "label": "Shortest Incoming Response Pause",
        "tooltip": "Minimum delay recorded between any two consecutive server response packets.",
        "category": "Timing Metric"
    },

    # Packet Sizes (Forward / Outgoing)
    "Fwd Packet Length Max": {
        "label": "Largest Outgoing Packet Size",
        "tooltip": "Maximum byte payload size among all outgoing packets sent by the client.",
        "category": "Packet Size"
    },
    "Fwd Packet Length Min": {
        "label": "Smallest Outgoing Packet Size",
        "tooltip": "Minimum byte payload size among all outgoing packets sent by the client.",
        "category": "Packet Size"
    },
    "Fwd Packet Length Mean": {
        "label": "Average Outgoing Packet Size",
        "tooltip": "Average payload size in bytes for outgoing client packets (easily padded by attackers).",
        "category": "Packet Size"
    },
    "Fwd Packet Length Std": {
        "label": "Outgoing Packet Size Variation",
        "tooltip": "Standard deviation in outgoing packet sizes — steady fixed-size packets vs. variable payloads.",
        "category": "Packet Size"
    },
    "Fwd Packets Length Total": {
        "label": "Total Outgoing Data Volume",
        "tooltip": "Total sum of all payload bytes sent in the forward (client-to-server) direction.",
        "category": "Volume Metric"
    },

    # Packet Sizes (Backward / Incoming)
    "Bwd Packet Length Max": {
        "label": "Largest Server Response Packet",
        "tooltip": "Maximum byte payload size among all response packets returned by the server.",
        "category": "Packet Size"
    },
    "Bwd Packet Length Min": {
        "label": "Smallest Server Response Packet",
        "tooltip": "Minimum byte payload size among all response packets returned by the server.",
        "category": "Packet Size"
    },
    "Bwd Packet Length Mean": {
        "label": "Average Server Response Size",
        "tooltip": "Average payload size in bytes for response packets returned by the server.",
        "category": "Packet Size"
    },
    "Bwd Packet Length Std": {
        "label": "Server Response Size Variation",
        "tooltip": "Variation in byte sizes of server response packets.",
        "category": "Packet Size"
    },
    "Bwd Packets Length Total": {
        "label": "Total Incoming Data Volume",
        "tooltip": "Total sum of all payload bytes received in the backward (server-to-client) direction.",
        "category": "Volume Metric"
    },

    # Overall Packet Size Distributions
    "Packet Length Min": {
        "label": "Overall Smallest Packet Size",
        "tooltip": "Minimum packet size observed across both flow directions.",
        "category": "Packet Size"
    },
    "Packet Length Max": {
        "label": "Overall Largest Packet Size",
        "tooltip": "Maximum packet size observed across both flow directions.",
        "category": "Packet Size"
    },
    "Packet Length Mean": {
        "label": "Overall Average Packet Size",
        "tooltip": "Average packet length in bytes across the entire bidirectional flow.",
        "category": "Packet Size"
    },
    "Packet Length Std": {
        "label": "Overall Packet Size Variation",
        "tooltip": "Spread of packet sizes across the entire connection.",
        "category": "Packet Size"
    },
    "Packet Length Variance": {
        "label": "Packet Size Variance",
        "tooltip": "Statistical variance in packet lengths throughout the connection.",
        "category": "Packet Size"
    },
    "Avg Packet Size": {
        "label": "Mean Flow Packet Size",
        "tooltip": "Average size of individual packets transmitted during the session.",
        "category": "Packet Size"
    },
    "Avg Fwd Segment Size": {
        "label": "Average Outgoing Segment Size",
        "tooltip": "Mean TCP segment size for forward packets sent by the client.",
        "category": "Packet Size"
    },
    "Avg Bwd Segment Size": {
        "label": "Average Incoming Segment Size",
        "tooltip": "Mean TCP segment size for response packets sent by the server.",
        "category": "Packet Size"
    },

    # Rates & Throughput
    "Flow Bytes/s": {
        "label": "Data Transfer Speed (Bytes/sec)",
        "tooltip": "Total byte transmission rate per second across the connection.",
        "category": "Throughput Metric"
    },
    "Flow Packets/s": {
        "label": "Packet Transmission Rate (Pkts/sec)",
        "tooltip": "Number of packets sent per second across the entire flow.",
        "category": "Throughput Metric"
    },
    "Fwd Packets/s": {
        "label": "Outgoing Packet Rate (Pkts/sec)",
        "tooltip": "Rate of outgoing packets transmitted by the client per second.",
        "category": "Throughput Metric"
    },
    "Bwd Packets/s": {
        "label": "Incoming Packet Rate (Pkts/sec)",
        "tooltip": "Rate of incoming response packets received from the server per second.",
        "category": "Throughput Metric"
    },
    "Down/Up Ratio": {
        "label": "Download vs Upload Ratio",
        "tooltip": "Ratio of received (downstream) packets to sent (upstream) packets.",
        "category": "Traffic Ratio"
    },

    # Packet Counts & Subflow Counters
    "Total Fwd Packets": {
        "label": "Total Outgoing Packets Sent",
        "tooltip": "Total count of individual packets sent by the client.",
        "category": "Volume Metric"
    },
    "Total Backward Packets": {
        "label": "Total Incoming Packets Received",
        "tooltip": "Total count of individual response packets returned by the server.",
        "category": "Volume Metric"
    },
    "Subflow Fwd Packets": {
        "label": "Subflow Outgoing Packet Count",
        "tooltip": "Count of forward packets belonging to individual flow sub-sequences.",
        "category": "Volume Metric"
    },
    "Subflow Fwd Bytes": {
        "label": "Subflow Outgoing Byte Volume",
        "tooltip": "Total forward byte volume in individual flow sub-sequences.",
        "category": "Volume Metric"
    },
    "Subflow Bwd Packets": {
        "label": "Subflow Incoming Packet Count",
        "tooltip": "Count of response packets belonging to individual flow sub-sequences.",
        "category": "Volume Metric"
    },
    "Subflow Bwd Bytes": {
        "label": "Subflow Incoming Byte Volume",
        "tooltip": "Total response byte volume in individual flow sub-sequences.",
        "category": "Volume Metric"
    },
    "Fwd Act Data Packets": {
        "label": "Outgoing Packets with Data Payload",
        "tooltip": "Number of outgoing forward packets that contain at least 1 byte of actual application data payload.",
        "category": "Volume Metric"
    },
    "Fwd Seg Size Min": {
        "label": "Minimum Outgoing TCP Segment Size",
        "tooltip": "Minimum TCP segment size observed for outgoing client packets.",
        "category": "Packet Size"
    },

    # Activity & Idle Windows
    "Active Mean": {
        "label": "Average Active Connection Period",
        "tooltip": "Average duration the flow was actively transmitting data between idle periods.",
        "category": "Activity Metric"
    },
    "Active Std": {
        "label": "Active Period Consistency",
        "tooltip": "Variation in the duration of active transmission periods.",
        "category": "Activity Metric"
    },
    "Active Max": {
        "label": "Longest Active Transmission Period",
        "tooltip": "Maximum time spent actively transmitting data before pausing.",
        "category": "Activity Metric"
    },
    "Active Min": {
        "label": "Shortest Active Transmission Period",
        "tooltip": "Minimum time spent actively transmitting data between pauses.",
        "category": "Activity Metric"
    },
    "Idle Mean": {
        "label": "Average Idle Silence Duration",
        "tooltip": "Average time the connection sat completely quiet with zero packet transmissions.",
        "category": "Activity Metric"
    },
    "Idle Std": {
        "label": "Idle Duration Variation",
        "tooltip": "Variation in how long the connection remained silent between bursts.",
        "category": "Activity Metric"
    },
    "Idle Max": {
        "label": "Longest Idle Silence Period",
        "tooltip": "Maximum duration of complete silence on the network connection.",
        "category": "Activity Metric"
    },
    "Idle Min": {
        "label": "Shortest Idle Silence Period",
        "tooltip": "Minimum duration of complete silence between data bursts.",
        "category": "Activity Metric"
    },

    # FROZEN PROTOCOL INVARIANTS (17 Features)
    "Fwd Header Length": {
        "label": "Client TCP/IP Header Length (Frozen)",
        "tooltip": "A structural protocol header size in bytes — fixed by the TCP/IP stack; cannot be arbitrarily changed without corrupting the connection.",
        "category": "Protocol Invariant"
    },
    "Bwd Header Length": {
        "label": "Server TCP/IP Header Length (Frozen)",
        "tooltip": "A structural protocol header size for server responses — fixed by protocol standards.",
        "category": "Protocol Invariant"
    },
    "Init Fwd Win Bytes": {
        "label": "Initial Client Window Size (Frozen)",
        "tooltip": "Initial TCP receive window size negotiated at connection handshake — a protocol-level invariant.",
        "category": "Protocol Invariant"
    },
    "Init Bwd Win Bytes": {
        "label": "Initial Server Window Size (Frozen)",
        "tooltip": "Initial TCP receive window size set by the server during handshake — a protocol-level invariant.",
        "category": "Protocol Invariant"
    },
    "Protocol_0": {
        "label": "Protocol: HOPOPT (Frozen)",
        "tooltip": "IPv6 Hop-by-Hop Option indicator — fixed structural protocol classification.",
        "category": "Protocol Invariant"
    },
    "Protocol_6": {
        "label": "Protocol: TCP (Frozen)",
        "tooltip": "Transmission Control Protocol indicator — fixed structural protocol classification.",
        "category": "Protocol Invariant"
    },
    "Protocol_17": {
        "label": "Protocol: UDP (Frozen)",
        "tooltip": "User Datagram Protocol indicator — fixed structural protocol classification.",
        "category": "Protocol Invariant"
    },
    "SYN Flag Count": {
        "label": "SYN Handshake Flag (Frozen)",
        "tooltip": "TCP Synchronize flag used strictly during connection establishment — cannot be modified on an active session.",
        "category": "Protocol Invariant"
    },
    "ACK Flag Count": {
        "label": "ACK Acknowledgment Flag (Frozen)",
        "tooltip": "TCP Acknowledgment flag indicating received data packets — mandatory for stateful communication.",
        "category": "Protocol Invariant"
    },
    "FIN Flag Count": {
        "label": "FIN Termination Flag (Frozen)",
        "tooltip": "TCP Finished flag indicating graceful connection teardown — closing the connection.",
        "category": "Protocol Invariant"
    },
    "RST Flag Count": {
        "label": "RST Connection Reset Flag (Frozen)",
        "tooltip": "TCP Reset flag indicating immediate connection abort.",
        "category": "Protocol Invariant"
    },
    "PSH Flag Count": {
        "label": "PSH Push Data Flag (Frozen)",
        "tooltip": "TCP Push flag requesting immediate buffer delivery to the application.",
        "category": "Protocol Invariant"
    },
    "URG Flag Count": {
        "label": "URG Urgent Data Flag (Frozen)",
        "tooltip": "TCP Urgent flag indicating high-priority out-of-band data.",
        "category": "Protocol Invariant"
    },
    "Fwd PSH Flags": {
        "label": "Client Push Flag Counter (Frozen)",
        "tooltip": "Structural counter for TCP Push flags sent by the client.",
        "category": "Protocol Invariant"
    },
    "Fwd URG Flags": {
        "label": "Client Urgent Flag Counter (Frozen)",
        "tooltip": "Structural counter for TCP Urgent flags sent by the client.",
        "category": "Protocol Invariant"
    },
    "CWE Flag Count": {
        "label": "CWE Congestion Window Flag (Frozen)",
        "tooltip": "TCP Congestion Window Reduced flag defined by RFC 3168.",
        "category": "Protocol Invariant"
    },
    "ECE Flag Count": {
        "label": "ECE Explicit Congestion Flag (Frozen)",
        "tooltip": "TCP ECN-Echo flag indicating network router congestion notifications.",
        "category": "Protocol Invariant"
    }
}


def build_translation_catalogs():
    with open(FEASIBILITY_PATH, "r") as f:
        feas_data = json.load(f)
    
    frozen_set = set(feas_data.get("frozen_features", []))
    all_mask_features = list(feas_data["mask"].keys())

    # Verify all 71 features are mapped
    missing = [feat for feat in all_mask_features if feat not in FEATURES_DICT]
    if missing:
        raise ValueError(f"Missing plain-language translations for {len(missing)} features: {missing}")

    full_catalog = {}
    for feat in all_mask_features:
        info = FEATURES_DICT[feat]
        is_frz = feat in frozen_set
        full_catalog[feat] = {
            "raw_name": feat,
            "label": info["label"],
            "tooltip": info["tooltip"],
            "category": info["category"],
            "is_frozen": is_frz,
            "constraint_type": "Frozen Protocol Invariant (Π_S)" if is_frz else "Perturbable Metric"
        }

    # Save catalogs
    with open(FEATURE_LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(full_catalog, f, indent=2)

    with open(ATTACK_LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(ATTACK_CLASSES, f, indent=2)

    print(f"Successfully generated {len(full_catalog)} feature mappings to '{FEATURE_LABELS_OUT}'")
    print(f"Successfully generated {len(ATTACK_CLASSES)} class mappings to '{ATTACK_LABELS_OUT}'")


if __name__ == "__main__":
    build_translation_catalogs()
