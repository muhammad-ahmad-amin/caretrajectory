"""
src/evaluation/__init__.py
================================================================================
CareTrajectory — Clinical Evaluation & Benchmarking Package Exports
================================================================================
"""

from src.evaluation.metrics import (
    compute_clinical_metrics,
    compute_physionet_utility,
    find_optimal_threshold_f1,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_feature_importances,
    save_metrics_report,
)

__all__ = [
    "compute_clinical_metrics",
    "compute_physionet_utility",
    "find_optimal_threshold_f1",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_calibration_curves",
    "plot_confusion_matrices",
    "plot_feature_importances",
    "save_metrics_report",
]
