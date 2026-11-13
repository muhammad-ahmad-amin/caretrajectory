# CareTrajectory — Clinical Deterioration & Sepsis Prediction

GPU-Accelerated, Non-Lookahead Clinical Trajectory Intelligence Platform for early detection of sepsis onset (PhysioNet 2019) and acute deterioration (MIMIC-IV Demo).

---

## 🎯 Deliverable 2: Baseline Models & Feature Engineering

This repository contains the production-grade implementation of:
1. **High-Throughput Feature Engineering (`src/features/`)**:
   - Multi-horizon non-lookahead rolling statistics (Mean, Min, Max, Std) over **6h, 12h, and 24h** windows.
   - Physiological velocity & acceleration (rate of change over **1h, 3h, and 6h**).
   - Bedside clinical risk scores: **Shock Index (SI)**, **Modified Shock Index (MSI)**, **Age-SI**, **Pulse Pressure (PP)**, **Imputed MAP**, **BUN/Creatinine Ratio**, **PaO2/FiO2 (Horowitz Index)**, **SIRS**, and **qSOFA**.
   - Missingness indicators and observation frequency counters (modeling Missingness-Not-At-Random clinical signals).
   - Strict patient boundary preservation (zero data bleeding across patient stays).
   - Memory optimization: Automatic downcasting of float channels to `float32`.

2. **Baseline Training & Benchmarking Pipeline (`scripts/train_baseline.py`)**:
   - **Patient-Level Group K-Fold / GroupShuffleSplit**: Guaranteed isolation of patient stays between train, validation, and test partitions (eliminating patient-identity leakage).
   - **Severe Imbalance Defenses**: Dynamic calculation of `scale_pos_weight` for XGBoost and `class_weight='balanced'` for Logistic Regression.
   - **Fast Gradient Boosting**: Histogram-based tree binning (`tree_method='hist'`) with early stopping on validation AUPRC.
   - **Publication-Grade Reporting Suite (`reports/`)**: Automatically outputs 300 DPI multi-model ROC curves, Precision-Recall curves, Calibration reliability diagrams, Confusion matrices, and Top-25 Feature Importance rankings.

---

## 📁 Repository Structure

```
caretrajectory/
├── configs/
│   └── base.yaml                 # Master configuration (windows, hyperparameters, metrics)
├── src/
│   ├── features/
│   │   ├── __init__.py           # Clean package exports
│   │   ├── clinical_scores.py    # SI, MSI, SIRS, qSOFA, P/F ratio, imputed MAP
│   │   └── temporal_features.py  # Non-lookahead rolling windows, deltas & missingness
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py            # AUROC, AUPRC, PhysioNet utility, publication 300 DPI plots
│   ├── io/                       # Data loading and format parsers
│   └── utils/                    # Common logging and configuration helpers
├── scripts/
│   └── train_baseline.py         # End-to-end reproducible baseline runner
├── tests/
│   ├── __init__.py
│   └── test_features.py          # Unit tests: zero-lookahead, patient isolation, scores
├── reports/
│   ├── baseline_metrics.json     # Quantitative benchmark results
│   ├── baseline_metrics.md       # Formatted Markdown report
│   └── figures/                  # 300 DPI publication plots (ROC, PR, Calibration, Importances)
├── models/                       # Serialized model checkpoints (.joblib)
├── requirements.txt              # Production dependency specifications
└── README.md
```

---

## 🚀 Execution Guide for the Team

### 1. Environment Setup
Activate the pre-configured virtual environment:
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 2. Run Pipeline with Google Drive Dataset
When running on the real PhysioNet 2019 dataset from Drive (e.g., extracted into `data/physionet_2019`):
```bash
python scripts/train_baseline.py --data_dir data/physionet_2019 --config configs/base.yaml
```

### 3. Run Pipeline in Synthetic / Simulation Mode
To run, benchmark, or verify the entire pipeline without downloading gigabytes of raw data:
```bash
python scripts/train_baseline.py --use_synthetic --n_patients 150
```
*This synthesizes 150 realistic ICU patient trajectories with clinical deterioration dynamics, runs the complete feature engineering pipeline, trains Logistic Regression and XGBoost, and generates all publication-grade figures in `reports/figures/`.*

### 4. Run Automated Test Suite
To verify zero lookahead leakage, patient boundary isolation, and clinical calculations:
```bash
python -m unittest tests/test_features.py
```

---

## 📊 Evaluation & Artifacts

All evaluation outputs are saved to `reports/`:
- `reports/baseline_metrics.json`: Full benchmark dictionary (AUROC, AUPRC, F1, Sensitivity, Specificity, Brier score, PhysioNet Utility).
- `reports/baseline_metrics.md`: Ready-to-copy Markdown table for course reports and papers.
- `reports/figures/roc_curves.png`: Multi-model ROC comparison.
- `reports/figures/precision_recall_curves.png`: Precision-Recall curves with prevalence baseline.
- `reports/figures/calibration_curves.png`: Reliability diagram assessing probability calibration.
- `reports/figures/confusion_matrices.png`: Formatted side-by-side confusion heatmaps.
- `reports/figures/feature_importance.png`: Top-25 physiological predictors from gradient boosting.
