"""
AdvRoNIDS Attack Module: FGSM, Constrained PGD, and Pi_S Feasibility Projection
Proposal References: Section 6.3 (Domain Feasibility Constraints),
                     Section 6.3.1 (Pi_S Projection Operator),
                     Section 6.4 (Attack Formulations & Threat Model)

This module provides reusable, import-ready functions for:
1. project_feasible (Pi_S): Enforcing frozen features, non-negative cumulative counters,
   and empirical min/max boundary constraints in standardized feature space.
2. fgsm_attack: Single-step Fast Gradient Sign Method (constrained and unconstrained).
3. pgd_attack: Multi-step Projected Gradient Descent with Pi_S domain projection
   or unconstrained L_infinity ball ablation.
4. compute_feasibility_violations: Quantifying real-world feasibility violation rates
   of unconstrained adversarial perturbations (reproducing Sheatsley et al. 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def project_feasible(x_adv, x_original, mask, cumulative_mask, emp_min, emp_max):
    """
    Applies the domain feasibility projection operator Pi_S (Section 6.3.1).

    Order of operations per iteration:
    1. Frozen-feature mask: delta = (x_adv - x_original) * mask (delta_i = 0 for frozen features)
    2. Non-negativity on cumulative counters: delta_i = max(delta_i, 0) for cumulative features
    3. Bounded clipping: x_projected = clamp(x_original + delta, emp_min, emp_max)

    Args:
        x_adv (torch.Tensor): Perturbed feature tensor (batch, features) or (batch, 1, features).
        x_original (torch.Tensor): Original clean feature tensor (same shape as x_adv).
        mask (torch.Tensor): Binary tensor (1 for perturbable, 0 for frozen).
        cumulative_mask (torch.Tensor): Binary tensor (1 for cumulative counters, 0 otherwise).
        emp_min (torch.Tensor): Standardized empirical minimum bounds.
        emp_max (torch.Tensor): Standardized empirical maximum bounds.

    Returns:
        torch.Tensor: Projected feasible adversarial tensor (same shape as x_adv).
    """
    # Compute raw perturbation delta
    delta = x_adv - x_original

    # Reshape masks/bounds to match tensor dimensionality if necessary
    if delta.dim() == 3 and mask.dim() == 1:
        mask = mask.view(1, 1, -1)
        cumulative_mask = cumulative_mask.view(1, 1, -1)
        emp_min = emp_min.view(1, 1, -1)
        emp_max = emp_max.view(1, 1, -1)
    elif delta.dim() == 2 and mask.dim() == 1:
        mask = mask.view(1, -1)
        cumulative_mask = cumulative_mask.view(1, -1)
        emp_min = emp_min.view(1, -1)
        emp_max = emp_max.view(1, -1)

    # 1. Zero out perturbations on frozen features
    delta = delta * mask

    # 2. Enforce non-negativity (delta_i >= 0) on cumulative counter features
    delta = torch.where(cumulative_mask == 1, torch.clamp(delta, min=0.0), delta)

    # 3. Apply bounded clipping in standardized feature space
    x_projected = torch.clamp(x_original + delta, min=emp_min, max=emp_max)

    return x_projected


def fgsm_attack(model, x, y, epsilon, criterion=None, constrained=False,
                mask=None, cumulative_mask=None, emp_min=None, emp_max=None):
    """
    Fast Gradient Sign Method (FGSM) attack (Section 4.1 & 6.4).

    Args:
        model (nn.Module): Target PyTorch model.
        x (torch.Tensor): Input feature tensor (batch, features) or (batch, 1, features).
        y (torch.Tensor): True ground-truth class labels.
        epsilon (float): Maximum perturbation magnitude (L_infinity norm).
        criterion (nn.Module, optional): Loss function (defaults to CrossEntropyLoss).
        constrained (bool): If True, applies Pi_S feasibility projection.
        mask (torch.Tensor, optional): Feasibility mask for Pi_S.
        cumulative_mask (torch.Tensor, optional): Cumulative counter mask for Pi_S.
        emp_min (torch.Tensor, optional): Empirical minimum bounds for Pi_S.
        emp_max (torch.Tensor, optional): Empirical maximum bounds for Pi_S.

    Returns:
        torch.Tensor: Adversarial feature tensor.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    x_adv = x.clone().detach()
    x_adv.requires_grad_(True)

    logits = model(x_adv)
    loss = criterion(logits, y)

    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

    # Signed step
    x_adv = x_adv.detach() + epsilon * torch.sign(grad)

    if constrained:
        if mask is None or cumulative_mask is None or emp_min is None or emp_max is None:
            raise ValueError("Feasibility masks and bounds must be provided when constrained=True.")
        # L_infinity ball clip followed by Pi_S projection
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
        x_adv = project_feasible(x_adv, x, mask, cumulative_mask, emp_min, emp_max)
    else:
        # Standard unconstrained L_infinity ball clipping
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

    return x_adv.detach()


def pgd_attack(model, x, y, epsilon, alpha=None, num_steps=7, criterion=None,
               constrained=True, mask=None, cumulative_mask=None,
               emp_min=None, emp_max=None, random_start=True):
    """
    Projected Gradient Descent (PGD) attack (Section 6.2.4 & 6.4).

    Args:
        model (nn.Module): Target PyTorch model.
        x (torch.Tensor): Input feature tensor (batch, features) or (batch, 1, features).
        y (torch.Tensor): True ground-truth class labels.
        epsilon (float): Maximum perturbation magnitude (L_infinity norm).
        alpha (float, optional): Step size (defaults to epsilon / 4).
        num_steps (int): Number of gradient ascent steps (default: 7).
        criterion (nn.Module, optional): Loss function (defaults to CrossEntropyLoss).
        constrained (bool): If True, applies Pi_S feasibility projection at every step.
        mask (torch.Tensor, optional): Feasibility mask for Pi_S.
        cumulative_mask (torch.Tensor, optional): Cumulative counter mask for Pi_S.
        emp_min (torch.Tensor, optional): Empirical minimum bounds for Pi_S.
        emp_max (torch.Tensor, optional): Empirical maximum bounds for Pi_S.
        random_start (bool): If True, initializes with random uniform perturbation.

    Returns:
        torch.Tensor: Adversarial feature tensor.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if alpha is None:
        alpha = epsilon / 4.0

    x_adv = x.clone().detach()

    # Optional uniform random initialization in L_infinity ball
    if random_start and epsilon > 0:
        noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv + noise
        if constrained:
            if mask is not None and cumulative_mask is not None and emp_min is not None and emp_max is not None:
                x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
                x_adv = project_feasible(x_adv, x, mask, cumulative_mask, emp_min, emp_max)
        else:
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

    for step in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        # Take gradient ascent step
        x_adv = x_adv.detach() + alpha * torch.sign(grad)

        if constrained:
            if mask is None or cumulative_mask is None or emp_min is None or emp_max is None:
                raise ValueError("Feasibility masks and bounds must be provided when constrained=True.")
            # Step 1: L_infinity ball clamp
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            # Step 2: Domain feasibility projection Pi_S
            x_adv = project_feasible(x_adv, x, mask, cumulative_mask, emp_min, emp_max)
        else:
            # Unconstrained ablation: standard L_infinity ball clamp only
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

    return x_adv.detach()


def pgd_attack_traced(model, x, y, epsilon, alpha=None, num_steps=7, criterion=None,
                      constrained=True, mask=None, cumulative_mask=None,
                      emp_min=None, emp_max=None, random_start=False, idx_to_class=None):
    """
    Projected Gradient Descent (PGD) attack with step-by-step telemetry tracing (Section 9.2).
    Records classification prediction, confidence, true-class probability, and feature deltas
    at every gradient ascent and projection iteration.

    Args:
        model (nn.Module): Target PyTorch model.
        x (torch.Tensor): Input feature tensor (1, features) or (1, 1, features).
        y (torch.Tensor): True ground-truth class label tensor ([target_cls]).
        epsilon (float): Maximum perturbation magnitude (L_infinity norm).
        alpha (float, optional): Step size (defaults to epsilon / 4).
        num_steps (int): Number of gradient ascent steps (default: 7).
        criterion (nn.Module, optional): Loss function (defaults to CrossEntropyLoss).
        constrained (bool): If True, applies Pi_S feasibility projection at every step.
        mask (torch.Tensor, optional): Feasibility mask for Pi_S.
        cumulative_mask (torch.Tensor, optional): Cumulative counter mask for Pi_S.
        emp_min (torch.Tensor, optional): Empirical minimum bounds for Pi_S.
        emp_max (torch.Tensor, optional): Empirical maximum bounds for Pi_S.
        random_start (bool): If True, initializes with random uniform perturbation.
        idx_to_class (dict, optional): Mapping from class integer index to name string.

    Returns:
        tuple: (x_adv_final, trace_list)
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if alpha is None:
        alpha = epsilon / 4.0

    y_idx = y.item() if isinstance(y, torch.Tensor) and y.numel() == 1 else int(y[0])
    trace = []

    x_orig = x.clone().detach()
    x_adv = x.clone().detach()

    if random_start and epsilon > 0:
        noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv + noise
        if constrained:
            if mask is not None and cumulative_mask is not None and emp_min is not None and emp_max is not None:
                x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)
                x_adv = project_feasible(x_adv, x_orig, mask, cumulative_mask, emp_min, emp_max)
        else:
            x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)

    # Step 0: Clean unperturbed flow state (or after random start)
    with torch.no_grad():
        logits_0 = model(x_adv)
        probs_0 = F.softmax(logits_0, dim=-1)[0]
        pred_0 = torch.argmax(probs_0).item()
        conf_0 = float(probs_0[pred_0].item())
        true_conf_0 = float(probs_0[y_idx].item()) if y_idx < len(probs_0) else 0.0

    delta_0 = (x_adv - x_orig).view(-1).cpu().tolist()
    trace.append({
        "step": 0,
        "x": x_adv.view(-1).cpu().tolist(),
        "delta": delta_0,
        "prediction": idx_to_class[pred_0] if idx_to_class and pred_0 in idx_to_class else pred_0,
        "prediction_index": pred_0,
        "confidence": conf_0,
        "true_class_confidence": true_conf_0
    })

    # Step 1 to num_steps: Iterative gradient ascent & projection
    for step in range(1, num_steps + 1):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        # Signed step
        x_adv = x_adv.detach() + alpha * torch.sign(grad)

        if constrained:
            if mask is None or cumulative_mask is None or emp_min is None or emp_max is None:
                raise ValueError("Feasibility masks and bounds must be provided when constrained=True.")
            x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)
            x_adv = project_feasible(x_adv, x_orig, mask, cumulative_mask, emp_min, emp_max)
        else:
            x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)

        # Record step telemetry
        with torch.no_grad():
            logits_step = model(x_adv)
            probs_step = F.softmax(logits_step, dim=-1)[0]
            pred_step = torch.argmax(probs_step).item()
            conf_step = float(probs_step[pred_step].item())
            true_conf_step = float(probs_step[y_idx].item()) if y_idx < len(probs_step) else 0.0

        delta_step = (x_adv - x_orig).view(-1).cpu().tolist()
        trace.append({
            "step": step,
            "x": x_adv.view(-1).cpu().tolist(),
            "delta": delta_step,
            "prediction": idx_to_class[pred_step] if idx_to_class and pred_step in idx_to_class else pred_step,
            "prediction_index": pred_step,
            "confidence": conf_step,
            "true_class_confidence": true_conf_step
        })

    return x_adv.detach(), trace


def compute_detailed_feasibility_metrics(x_adv_unconstrained, x_original, mask,
                                         cumulative_mask, protocol_indices, emp_min, emp_max, tol=1e-5):
    """
    Computes three continuous, non-tautological domain-feasibility severity metrics on unconstrained adversarial examples:
    1. Metric 1 (Magnitude-weighted frozen drift): L2 norm of perturbation on frozen features ||delta_F||_2
    2. Metric 2 (Protocol deviation distance): Minimum L2 distance to any valid one-hot protocol vector min_v ||p_adv - v||_2
    3. Metric 3 (Cumulative counter violation severity):
       a) mean_violated_counters: Average count of cumulative counters (out of 9) with delta_i < -tol per sample (0 to 9)
       b) mean_violation_magnitude: Average magnitude |delta_i| across violated counters (in standardized units)
    4. Reference occurrence metrics:
       - protocol_invalid_rate: Percentage with non-discrete/fractional protocol values
       - cumulative_counter_negative_rate: Percentage with at least one negative cumulative counter
       - naive_touched_frozen_feature_rate: Percentage with any non-zero delta on frozen features

    Args:
        x_adv_unconstrained (torch.Tensor): Unconstrained adversarial examples (batch, features) or (batch, 1, features).
        x_original (torch.Tensor): Clean original features.
        mask (torch.Tensor): Feasibility mask tensor (1=perturbable, 0=frozen).
        cumulative_mask (torch.Tensor): Cumulative counter mask tensor (1=cumulative, 0=otherwise).
        protocol_indices (list or torch.Tensor): Feature indices for Protocol one-hot columns (e.g. [68, 69, 70]).
        emp_min (torch.Tensor): Standardized empirical minimum bounds tensor.
        emp_max (torch.Tensor): Standardized empirical maximum bounds tensor.
        tol (float): Numerical tolerance threshold.

    Returns:
        dict: Detailed continuous statistics, distributions, and reference occurrence rates.
    """
    delta = (x_adv_unconstrained - x_original).detach()

    # Align shapes
    if delta.dim() == 3:
        delta = delta.squeeze(1)
        x_adv = x_adv_unconstrained.squeeze(1)
        x_orig = x_original.squeeze(1)
    else:
        x_adv = x_adv_unconstrained.detach()
        x_orig = x_original.detach()

    mask_1d = mask.view(-1)
    cum_mask_1d = cumulative_mask.view(-1)
    batch_size = delta.size(0)

    # 1. Metric 1: Frozen feature drift ||delta_F||_2
    frozen_cols = (mask_1d == 0).nonzero(as_tuple=True)[0]
    if len(frozen_cols) > 0:
        frozen_deltas = delta[:, frozen_cols]
        frozen_drift_per_sample = torch.norm(frozen_deltas, p=2, dim=1).cpu().numpy()
        naive_touched = (torch.abs(frozen_deltas) > tol).any(dim=1)
    else:
        frozen_drift_per_sample = np.zeros(batch_size, dtype=np.float32)
        naive_touched = torch.zeros(batch_size, dtype=torch.bool, device=delta.device)

    # 2. Metric 2: Protocol deviation distance to valid one-hot space
    if protocol_indices is not None and len(protocol_indices) > 0:
        proto_idx_t = torch.as_tensor(protocol_indices, dtype=torch.long, device=x_adv.device)
        proto_vals = x_adv[:, proto_idx_t]  # (batch, num_protocol_cols)
        k_classes = len(protocol_indices)
        
        # Candidate valid one-hot vectors I_k (e.g. [1,0,0], [0,1,0], [0,0,1])
        valid_one_hots = torch.eye(k_classes, device=x_adv.device)  # (k, k)
        
        # Compute L2 distance from each sample's proto_vals to each valid one-hot vector
        # proto_vals: (batch, 1, k), valid_one_hots: (1, k, k)
        diffs = proto_vals.unsqueeze(1) - valid_one_hots.unsqueeze(0)  # (batch, k, k)
        dists_to_valid = torch.norm(diffs, p=2, dim=2)  # (batch, k)
        protocol_deviation_per_sample = torch.min(dists_to_valid, dim=1)[0].cpu().numpy()  # (batch,)

        # Reference binary check
        is_binary = (torch.abs(proto_vals - torch.round(proto_vals)) < 1e-4) & (proto_vals >= -1e-4) & (proto_vals <= 1.0 + 1e-4)
        is_valid_sum = torch.abs(proto_vals.sum(dim=1) - 1.0) < 1e-4
        protocol_valid = is_binary.all(dim=1) & is_valid_sum
        protocol_invalid = ~protocol_valid
    else:
        protocol_deviation_per_sample = np.zeros(batch_size, dtype=np.float32)
        protocol_invalid = torch.zeros(batch_size, dtype=torch.bool, device=x_adv.device)

    # 3. Metric 3: Cumulative counter violation count and magnitude
    cum_cols = (cum_mask_1d == 1).nonzero(as_tuple=True)[0]
    if len(cum_cols) > 0:
        cum_deltas = delta[:, cum_cols]  # (batch, num_cumulative_cols)
        is_neg = (cum_deltas < -tol)  # (batch, num_cumulative_cols) bool
        
        # a) Count of violated counters per sample (0 to 9)
        violated_counts_per_sample = is_neg.sum(dim=1).float().cpu().numpy()  # (batch,)
        
        # b) Mean magnitude of negative deltas per sample for violated counters
        # Mask non-violated with 0 for magnitude, and divide by count
        neg_magnitudes = torch.where(is_neg, torch.abs(cum_deltas), torch.zeros_like(cum_deltas))
        sum_mags = neg_magnitudes.sum(dim=1)  # (batch,)
        counts_t = is_neg.sum(dim=1).float()  # (batch,)
        
        # Per-sample mean violation magnitude (only for samples with >= 1 violation)
        has_violation = counts_t > 0
        sample_mean_mags = torch.where(has_violation, sum_mags / counts_t, torch.zeros_like(sum_mags)).cpu().numpy()
        valid_sample_mags = sample_mean_mags[has_violation.cpu().numpy()]
        
        cum_negative_rate = has_violation.float().mean().item() * 100.0
    else:
        violated_counts_per_sample = np.zeros(batch_size, dtype=np.float32)
        valid_sample_mags = np.zeros(0, dtype=np.float32)
        sample_mean_mags = np.zeros(batch_size, dtype=np.float32)
        cum_negative_rate = 0.0

    protocol_invalid_rate = protocol_invalid.float().mean().item() * 100.0
    naive_touched_rate = naive_touched.float().mean().item() * 100.0

    mean_viol_mag = float(np.mean(valid_sample_mags)) if len(valid_sample_mags) > 0 else 0.0
    std_viol_mag = float(np.std(valid_sample_mags)) if len(valid_sample_mags) > 0 else 0.0

    return {
        "total_samples": batch_size,
        # Metric 1: Frozen drift
        "frozen_drift_samples": frozen_drift_per_sample,
        "frozen_drift_mean": float(np.mean(frozen_drift_per_sample)),
        "frozen_drift_median": float(np.median(frozen_drift_per_sample)),
        "frozen_drift_std": float(np.std(frozen_drift_per_sample)),
        "frozen_drift_min": float(np.min(frozen_drift_per_sample)),
        "frozen_drift_max": float(np.max(frozen_drift_per_sample)),
        # Metric 2: Protocol deviation distance
        "protocol_deviation_samples": protocol_deviation_per_sample,
        "protocol_deviation_mean": float(np.mean(protocol_deviation_per_sample)),
        "protocol_deviation_median": float(np.median(protocol_deviation_per_sample)),
        "protocol_deviation_std": float(np.std(protocol_deviation_per_sample)),
        "protocol_deviation_min": float(np.min(protocol_deviation_per_sample)),
        "protocol_deviation_max": float(np.max(protocol_deviation_per_sample)),
        # Metric 3: Cumulative counter severity
        "violated_counter_counts_samples": violated_counts_per_sample,
        "mean_violated_counters": float(np.mean(violated_counts_per_sample)),
        "mean_violated_counters_std": float(np.std(violated_counts_per_sample)),
        "mean_violation_magnitude": mean_viol_mag,
        "mean_violation_magnitude_std": std_viol_mag,
        # Reference occurrence rates (saturating)
        "protocol_invalid_rate": float(protocol_invalid_rate),
        "cumulative_counter_negative_rate": float(cum_negative_rate),
        "naive_touched_frozen_feature_rate": float(naive_touched_rate)
    }


