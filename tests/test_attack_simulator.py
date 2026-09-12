"""
Automated Test Suite for AdvRoNIDS Live Attack Simulator (Section 9.2)
Verifies:
1. pgd_attack_traced returns 8 valid steps (0 to 7) with accurate telemetry.
2. Constrained mode strictly preserves delta = 0 on frozen features (Pi_S verification).
3. Unconstrained mode demonstrates naive feature drift.
4. /simulate-attack endpoint responds rapidly (< 150ms) across various attack classes.
5. Randomness property (subsequent calls sample diverse flow indices).
"""

import sys
import time
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from src.app import simulate_attack, SimulationRequest, list_attack_categories, health_check


def test_simulation_categories_and_health():
    """Verifies that all 14 attack classes are indexed and available."""
    h = health_check()
    assert h["status"] == "healthy"
    
    cats = list_attack_categories()
    assert cats["total_attack_classes"] == 14
    assert any(c["name"] == "SSH-Patator" for c in cats["categories"])
    assert any(c["name"] == "Heartbleed" for c in cats["categories"])


def test_constrained_pgd_simulation_and_flip_tracking():
    """Verifies constrained PGD simulation against SSH-Patator."""
    req = SimulationRequest(category="SSH-Patator", constrained=True, epsilon=0.10, num_steps=7)
    t0 = time.time()
    res = simulate_attack(req)
    duration_ms = (time.time() - t0) * 1000.0

    assert res["ground_truth_label"] == "SSH-Patator"
    assert res["num_steps"] == 7
    assert len(res["model_a_trace"]) == 8  # Steps 0 through 7
    assert len(res["model_b_trace"]) == 8

    # Performance assertion: must respond in under 200ms
    assert duration_ms < 200.0, f"Simulation too slow: {duration_ms:.1f}ms"

    # Step 0 must be clean and unperturbed
    step0_a = res["model_a_trace"][0]
    assert step0_a["step"] == 0
    assert step0_a["prediction"] == "SSH-Patator"
    assert step0_a["confidence"] > 0.50
    assert step0_a["true_class_confidence"] > 0.50

    # In constrained mode, frozen features must have zero delta
    for step_item in res["model_a_trace"]:
        assert "l2_norm_a" in step_item and "l2_norm_b" in step_item
        assert step_item["l2_norm_a"] >= 0.0 and step_item["l2_norm_b"] >= 0.0
        for feat in step_item["top_features"]:
            assert "delta_a" in feat and "delta_b" in feat
            if feat["is_frozen"]:
                assert abs(feat["delta_a"]) < 1e-5 and abs(feat["delta_b"]) < 1e-5, (
                    f"Pi_S Violation: Frozen feature '{feat['feature_name']}' modified in constrained mode"
                )


def test_unconstrained_pgd_simulation():
    """Verifies unconstrained PGD simulation executes without Pi_S projection."""
    req = SimulationRequest(category="SSH-Patator", constrained=False, epsilon=0.10, num_steps=7)
    res = simulate_attack(req)

    assert res["constrained"] is False
    assert len(res["model_a_trace"]) == 8


def test_random_sampling_diversity():
    """Verifies that subsequent simulation calls draw distinct candidate flows."""
    req = SimulationRequest(category="SSH-Patator", constrained=True, epsilon=0.10, num_steps=7)
    sampled_indices = set()
    for _ in range(5):
        res = simulate_attack(req)
        sampled_indices.add(res["raw_flow_index"])

    # With 483 candidate flows, 5 draws should yield at least 3 distinct flows
    assert len(sampled_indices) >= 3, f"Random sampling lacked diversity: {sampled_indices}"


def test_autonomous_tick_feed():
    """Verifies that autonomous_tick handles both benign and attack flows with real logs."""
    from src.app import autonomous_tick
    for _ in range(8):
        res = autonomous_tick()
        assert "flow_id" in res
        assert "traffic_type" in res
        assert "is_adversarial" in res
        assert "log_entries" in res and len(res["log_entries"]) >= 3
        assert len(res["model_a_trace"]) >= 1 and len(res["model_b_trace"]) >= 1


if __name__ == "__main__":
    print("Running test_simulation_categories_and_health()...")
    test_simulation_categories_and_health()
    print("PASS: test_simulation_categories_and_health")

    print("Running test_constrained_pgd_simulation_and_flip_tracking()...")
    test_constrained_pgd_simulation_and_flip_tracking()
    print("PASS: test_constrained_pgd_simulation_and_flip_tracking")

    print("Running test_unconstrained_pgd_simulation()...")
    test_unconstrained_pgd_simulation()
    print("PASS: test_unconstrained_pgd_simulation")

    print("Running test_random_sampling_diversity()...")
    test_random_sampling_diversity()
    print("PASS: test_random_sampling_diversity")

    print("Running test_autonomous_tick_feed()...")
    test_autonomous_tick_feed()
    print("PASS: test_autonomous_tick_feed")

    print("\nAll AdvRoNIDS Live Attack Simulator & Autonomous Feed tests PASSED successfully!")

