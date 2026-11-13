"""
src/features/__init__.py
================================================================================
CareTrajectory — Clinical Feature Engineering Package Exports
================================================================================
"""

from src.features.clinical_scores import (
    compute_shock_index,
    compute_modified_shock_index,
    compute_age_shock_index,
    compute_pulse_pressure,
    compute_imputed_map,
    compute_bun_creatinine_ratio,
    compute_pao2_fio2_ratio,
    compute_sirs_score,
    compute_qsofa_score,
    compute_all_clinical_scores,
)
from src.features.temporal_features import (
    TemporalFeatureExtractor,
    extract_features,
)

__all__ = [
    "compute_shock_index",
    "compute_modified_shock_index",
    "compute_age_shock_index",
    "compute_pulse_pressure",
    "compute_imputed_map",
    "compute_bun_creatinine_ratio",
    "compute_pao2_fio2_ratio",
    "compute_sirs_score",
    "compute_qsofa_score",
    "compute_all_clinical_scores",
    "TemporalFeatureExtractor",
    "extract_features",
]
