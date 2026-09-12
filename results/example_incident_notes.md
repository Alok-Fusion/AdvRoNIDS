# AdvRoNIDS LLM-Assisted Incident Triage Layer (Section 9.2)

## Illustrative Incident Triage Notes for Paper (Section 9.2)

The following incident response notes illustrate how the AdvRoNIDS triage layer converts structured telemetry (multi-model verdicts, domain-constrained perturbation deltas, and SHAP feature attributions) into concise, actionable prose. Per **Section 9.2.1's explicit scope boundary**, the LLM operates strictly in a narration capacity and does not alter or make classification decisions.

---

### Example 1: `flow_418_SSH-Patator` (Evaded Model A (Robust Model B Defended))

- **Ground Truth:** `SSH-Patator`
- **Telemetry Condition:** `Domain-Constrained PGD (eps = 0.10, Pi_S)`
- **Undefended Model A Verdict:** `Benign` (100.0% confidence) [EVADED]
- **Robust Model B Verdict:** `SSH-Patator` (99.5% confidence) [DEFENDED]
- **Key Manipulated Features:** `Bwd Packet Length Min` (Δ = +0.10), `Packet Length Min` (Δ = +0.10), `Bwd Packet Length Max` (Δ = -0.10)

> [!NOTE]
> **AdvRoNIDS Incident Note (LLM Telemetry Narration):**
> *"Model A classified the flow as Benign with 100.0% confidence, while Model B classified it as SSH-Patator with 99.5% confidence. Despite the disagreement, Model B correctly identified the ground truth. Adversarial perturbations shifted the minimum and maximum backward packet lengths, as well as the minimum packet length, which were manipulated by the perturbation. Model A was misled by attribution drift onto these manipulated features, whereas Model B, anchored on protocol invariants, correctly identified the flow as SSH-Patator."*

---

### Example 2: `flow_342_DoS_GoldenEye` (Evaded Model A (Robust Model B Defended))

- **Ground Truth:** `DoS GoldenEye`
- **Telemetry Condition:** `Domain-Constrained PGD (eps = 0.10, Pi_S)`
- **Undefended Model A Verdict:** `Benign` (55.5% confidence) [EVADED]
- **Robust Model B Verdict:** `DoS GoldenEye` (100.0% confidence) [DEFENDED]
- **Key Manipulated Features:** `Flow IAT Mean` (Δ = +0.10), `Flow IAT Min` (Δ = -0.10), `Flow Duration` (Δ = +0.10)

> [!NOTE]
> **AdvRoNIDS Incident Note (LLM Telemetry Narration):**
> *"Flow ID: flow_342_DoS_GoldenEye was classified as Benign (Confidence: 55.5%) by Model A and as DoS GoldenEye (Confidence: 100.0%) by Model B, indicating a clear disagreement. The adversarial perturbations shifted the Mean Flow Inter-Arrival Time by +0.10, Minimum Flow Inter-Arrival Time by ±0.10, and Total Flow Duration by +0.10. Model A was misled by attribution drift onto manipulated features, while Robust Model B correctly identified the flow as DoS GoldenEye by anchoring on protocol invariants."*

---

### Example 3: `flow_0_Benign` (Clean Operational Traffic (Both Agree))

- **Ground Truth:** `Benign`
- **Telemetry Condition:** `Clean Operational Baseline`
- **Undefended Model A Verdict:** `Benign` (100.0% confidence) [ACCURATE]
- **Robust Model B Verdict:** `Benign` (100.0% confidence) [DEFENDED]

> [!NOTE]
> **AdvRoNIDS Incident Note (LLM Telemetry Narration):**
> *"Model A and Model B both classified the flow as Benign with 100.0% confidence. Both models agreed with the ground truth, as they correctly identified the flow as benign. Model A was misled by attribution drift onto manipulated features, while Robust Model B anchored on protocol invariants, correctly identifying the flow as benign. The top manipulated features identified by Model A were Forward Packet Length Std Dev, Avg Fwd Segment Size, and Maximum Backward Packet Length. Model B's top manipulated features were TCP Protocol Flag (Frozen), UDP Protocol Flag (Frozen), and Forward Inter-Arrival Time Std Dev."*

---
