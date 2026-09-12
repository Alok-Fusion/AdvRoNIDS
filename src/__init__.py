"""
AdvRoNIDS: Adversarial Robust Network Intrusion Detection Systems
Core package containing models, domain-constrained attacks, preprocessing, and training routines.
"""

from .model import AdvRoNIDS_CNN
from .attacks import (
    project_feasible,
    fgsm_attack,
    pgd_attack,
    compute_detailed_feasibility_metrics
)

__all__ = [
    "AdvRoNIDS_CNN",
    "project_feasible",
    "fgsm_attack",
    "pgd_attack",
    "compute_detailed_feasibility_metrics"
]

