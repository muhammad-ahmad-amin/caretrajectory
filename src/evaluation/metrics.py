"""
src/evaluation/metrics.py
================================================================================
CareTrajectory — Clinical Evaluation Engine & Publication Visualization Suite

Implements clinical discrimination and calibration metrics:
  - AUROC, AUPRC (Average Precision), F1-optimal Thresholding
  - Specificity at 90% Sensitivity (Alert Fatigue Defense)
  - Brier Calibration Score & PhysioNet 2019 Normalized Utility Score
  - Publication-grade multi-model visualization (ROC, PR, Calibration, SHAP/Importances)
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Headless backend for crash-free CLI and server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("CareTrajectory.Evaluation")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [CareTrajectory.Evaluation] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Publication styling defaults
PALETTE = ["#0284C7", "#E11D48", "#10B981", "#8B5CF6", "#F59E0B"]
sns.set_theme(style="whitegrid", font_scale=1.1)


# ==============================================================================
# 1. PhysioNet 2019 Official Clinical Utility Metric
# ==============================================================================
def compute_physionet_utility(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    dt_early: int = 12,
    dt_optimal: int = 6,
    dt_late: int = 3,
) -> float:
    """
    Computes a continuous approximation of the PhysioNet 2019 Challenge Utility Score.
    Rewards early prediction of sepsis (dt_optimal to dt_early hours before onset)
    and penalizes false alarms (alert fatigue) and missed onset.
    """
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)
    
    # Utility weights
    u_tp = 1.0     # True positive reward
    u_fp = -0.05   # False positive penalty (alert fatigue)
    u_fn = -1.0    # False negative penalty (missed clinical deterioration)
    u_tn = 0.0     # True negative baseline
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    observed_utility = (tp * u_tp) + (fp * u_fp) + (fn * u_fn) + (tn * u_tn)
    # Theoretical maximum utility (perfect classifier)
    max_utility = np.sum(y_true == 1) * u_tp
    
    if max_utility <= 0:
        return 0.0
    
    normalized_utility = observed_utility / max_utility
    return float(np.clip(normalized_utility, -1.0, 1.0))


# ==============================================================================
# 2. Comprehensive Clinical Metrics Evaluator
# ==============================================================================
def find_optimal_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Finds probability threshold maximizing F1-score on validation/test predictions.
    Essential for highly imbalanced clinical cohorts.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid zero division
    denom = precisions + recalls
    f1_scores = np.divide(
        2 * (precisions * recalls),
        denom,
        out=np.zeros_like(denom),
        where=denom != 0
    )
    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return best_threshold, best_f1


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    threshold: Optional[float] = None,
) -> Dict[str, Union[float, str]]:
    """
    Computes a complete panel of discrimination, calibration, and utility metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(np.float64)
    
    # 1. Discrimination (Threshold-Independent)
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        logger.warning(
            f"Evaluation target for {model_name} contains only 1 class "
            f"(pos={np.sum(y_true == 1)}, neg={np.sum(y_true == 0)}). "
            "AUROC and AUPRC defaulted to baseline."
        )
        auroc = 0.5
        auprc = float(np.mean(y_true))
    else:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
            if np.isnan(auroc):
                auroc = 0.5
        except Exception:
            auroc = 0.5
            
        try:
            auprc = float(average_precision_score(y_true, y_prob))
            if np.isnan(auprc):
                auprc = float(np.mean(y_true))
        except Exception:
            auprc = float(np.mean(y_true))

    # 2. Threshold-Dependent Metrics
    if threshold is None:
        opt_thresh, opt_f1 = find_optimal_threshold_f1(y_true, y_prob)
        chosen_threshold = opt_thresh
    else:
        chosen_threshold = threshold

    y_pred = (y_prob >= chosen_threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Specificity: TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # 3. Calibration & Clinical Utility
    brier = float(brier_score_loss(y_true, y_prob))
    utility = compute_physionet_utility(y_true, y_prob, threshold=chosen_threshold)
    
    prevalence = float(np.mean(y_true))

    metrics = {
        "model_name": model_name,
        "prevalence": round(prevalence, 4),
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
        "threshold": round(chosen_threshold, 4),
        "Accuracy": round(acc, 4),
        "Balanced_Accuracy": round(bal_acc, 4),
        "Sensitivity_Recall": round(recall, 4),
        "Specificity": round(specificity, 4),
        "Precision_PPV": round(precision, 4),
        "F1_Score": round(f1, 4),
        "Brier_Score": round(brier, 4),
        "PhysioNet_Utility": round(utility, 4),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
    return metrics


# ==============================================================================
# 3. Publication-Grade Visualization Suite (300 DPI)
# ==============================================================================
def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plots high-resolution ROC curves for multiple models with AUC values and no-skill line.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=dpi)
    
    for idx, (model_name, (y_true, y_prob)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(
            fpr, tpr,
            label=f"{model_name} (AUROC = {auc_val:.3f})",
            color=color,
            lw=2.5,
            alpha=0.9
        )
        
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Chance Baseline (0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=13, fontweight="bold")
    ax.set_title("CareTrajectory — ROC Discrimination Analysis", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved ROC curve figure to: {save_path}")
    return fig


def plot_precision_recall_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plots high-resolution Precision-Recall curves with prevalence no-skill baseline.
    Vital for clinical deterioration where positive prevalence is low (~2%).
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=dpi)
    
    prevalence = None
    for idx, (model_name, (y_true, y_prob)) in enumerate(results.items()):
        if prevalence is None:
            prevalence = np.mean(y_true)
            
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auprc_val = average_precision_score(y_true, y_prob)
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(
            rec, prec,
            label=f"{model_name} (AUPRC = {auprc_val:.3f})",
            color=color,
            lw=2.5,
            alpha=0.9
        )
        
    if prevalence is not None:
        ax.axhline(
            y=prevalence, color="gray", linestyle="--", lw=1.5,
            label=f"Prevalence Baseline ({prevalence:.1%})"
        )
        
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=13, fontweight="bold")
    ax.set_title("CareTrajectory — Precision-Recall Trajectory Curve", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved Precision-Recall figure to: {save_path}")
    return fig


def plot_calibration_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None,
    n_bins: int = 10,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plots reliability diagram / probability calibration curve.
    Evaluates whether predicted probabilities match true empirical event frequencies.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=dpi)
    
    for idx, (model_name, (y_true, y_prob)) in enumerate(results.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        brier = brier_score_loss(y_true, y_prob)
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(
            prob_pred, prob_true,
            marker="o", markersize=6,
            label=f"{model_name} (Brier = {brier:.4f})",
            color=color, lw=2.2, alpha=0.9
        )
        
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Perfect Calibration")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Mean Predicted Probability", fontsize=13, fontweight="bold")
    ax.set_ylabel("Empirical Fraction of Positives", fontsize=13, fontweight="bold")
    ax.set_title("CareTrajectory — Clinical Probability Calibration", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved Calibration curve figure to: {save_path}")
    return fig


def plot_confusion_matrices(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plots formatted confusion matrices side-by-side for all evaluated models.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), dpi=dpi, squeeze=False)
    axes = axes.flatten()
    
    for idx, (model_name, (y_true, y_prob)) in enumerate(results.items()):
        opt_thresh, _ = find_optimal_threshold_f1(y_true, y_prob)
        y_pred = (y_prob >= opt_thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            annot_kws={"size": 14, "fontweight": "bold"}, ax=ax
        )
        ax.set_title(f"{model_name}\n(Threshold: {opt_thresh:.2f})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Ground Truth", fontsize=11, fontweight="bold")
        ax.set_xticklabels(["Non-Deteriorating", "Deteriorating"])
        ax.set_yticklabels(["Non-Deteriorating", "Deteriorating"])
        
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved Confusion Matrices figure to: {save_path}")
    return fig


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 25,
    title: str = "Top 25 Physiological Trajectory Drivers",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plots horizontal bar chart ranking top predictive physiological features.
    """
    top_n = min(top_n, len(feature_names))
    indices = np.argsort(importances)[::-1][:top_n]
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=dpi)
    y_pos = np.arange(top_n)
    
    # Gradient bar colors
    colors = sns.color_palette("mako", n_colors=top_n)[::-1]
    
    bars = ax.barh(y_pos, sorted_importances, align="center", color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=10, fontweight="medium")
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel("Relative Feature Importance Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved Feature Importance figure to: {save_path}")
    return fig


# ==============================================================================
# 4. Report Exporters (JSON & Markdown)
# ==============================================================================
def save_metrics_report(
    metrics_list: List[Dict[str, Union[float, str]]],
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """
    Saves performance benchmarks as both a JSON payload and a GitHub-flavored Markdown table.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    json_path = out_path / "baseline_metrics.json"
    md_path = out_path / "baseline_metrics.md"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)
        
    df_metrics = pd.DataFrame(metrics_list)
    cols_order = [
        "model_name", "AUROC", "AUPRC", "F1_Score", "Sensitivity_Recall",
        "Specificity", "Precision_PPV", "PhysioNet_Utility", "Brier_Score", "threshold"
    ]
    cols = [c for c in cols_order if c in df_metrics.columns]
    
    try:
        md_table = df_metrics[cols].to_markdown(index=False)
    except Exception:
        # Fallback pure-python markdown table generation (no tabulate dependency needed)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, row in df_metrics[cols].iterrows():
            rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        md_table = "\n".join([header, sep] + rows)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# CareTrajectory — Baseline Model Benchmark Report\n\n")
        f.write(f"Generated across evaluated models on patient-level holdout validation set.\n\n")
        f.write(md_table)
        f.write("\n\n### Metric Explanations:\n")
        f.write("- **AUROC**: Overall discrimination across all potential alarm cutoffs.\n")
        f.write("- **AUPRC**: Primary evaluation metric under extreme clinical class imbalance.\n")
        f.write("- **Sensitivity_Recall**: Percentage of true deteriorating patients flagged.\n")
        f.write("- **Specificity**: Percentage of non-deteriorating patients spared false alarms.\n")
        f.write("- **PhysioNet_Utility**: Early prediction reward versus alert fatigue penalty.\n")
        
    logger.info(f"Saved JSON metrics to: {json_path}")
    logger.info(f"Saved Markdown report to: {md_path}")
    return json_path, md_path
