"""
scripts/train_baseline.py
================================================================================
CareTrajectory — Reproducible Classical ML Baseline Training & Benchmark Runner

Trains and validates baseline clinical models (Logistic Regression & XGBoost)
on ICU patient deterioration trajectories. Enforces strict patient-level grouping
to prevent data leakage, handles class imbalance, and generates publication-grade
evaluation artifacts and visual reports.
================================================================================
"""

import argparse
import glob
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Project Imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.temporal_features import TemporalFeatureExtractor
from src.evaluation.metrics import (
    compute_clinical_metrics,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_feature_importances,
    plot_precision_recall_curves,
    plot_roc_curves,
    save_metrics_report,
)

logger = logging.getLogger("CareTrajectory.BaselineTrainer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [CareTrajectory.Trainer] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ==============================================================================
# 1. Synthetic ICU Data Generator (For Testing & Immediate Verification)
# ==============================================================================
def generate_synthetic_icu_cohort(
    n_patients: int = 120,
    seed: int = 42
) -> pd.DataFrame:
    """
    Synthesizes realistic clinical time-series for n_patients with ICU lengths of stay.
    Simulates physiological deterioration trajectories (rising HR, falling MAP, spiking Lactate)
    for patients that develop sepsis, mimicking PhysioNet 2019 dataset distributions.
    """
    np.random.seed(seed)
    records = []
    
    for pid in range(1, n_patients + 1):
        patient_id = f"p_{pid:05d}"
        los = np.random.randint(18, 50)  # ICU length of stay between 18 and 50 hours
        age = float(np.random.normal(63.0, 12.0))
        gender = int(np.random.choice([0, 1]))
        is_septic = np.random.rand() < 0.12  # ~12% patient-level prevalence
        
        onset_hour = np.random.randint(10, los - 4) if is_septic else 9999
        
        # Baselines
        hr_base = np.random.normal(76.0, 8.0)
        map_base = np.random.normal(86.0, 9.0)
        temp_base = np.random.normal(36.9, 0.4)
        resp_base = np.random.normal(16.0, 2.5)
        o2_base = np.random.normal(98.0, 1.2)
        lactate_base = np.random.normal(1.2, 0.3)
        creat_base = np.random.normal(0.9, 0.2)
        wbc_base = np.random.normal(7.5, 1.5)
        
        for t in range(1, los + 1):
            # Check if patient is in deteriorating phase
            if is_septic and (t >= onset_hour - 6):
                # Escalating shock & infection trajectory
                hrs_since_deterioration = max(0, t - (onset_hour - 6))
                hr_val = hr_base + 3.2 * hrs_since_deterioration + np.random.normal(0, 3.0)
                map_val = map_base - 2.8 * hrs_since_deterioration + np.random.normal(0, 3.0)
                temp_val = temp_base + 0.18 * hrs_since_deterioration + np.random.normal(0, 0.2)
                resp_val = resp_base + 1.1 * hrs_since_deterioration + np.random.normal(0, 1.5)
                o2_val = max(84.0, o2_base - 0.9 * hrs_since_deterioration + np.random.normal(0, 0.8))
                lactate_val = lactate_base + 0.35 * hrs_since_deterioration + np.random.normal(0, 0.2)
                creat_val = creat_base + 0.12 * hrs_since_deterioration + np.random.normal(0, 0.05)
                wbc_val = wbc_base + 0.8 * hrs_since_deterioration + np.random.normal(0, 0.5)
            else:
                hr_val = hr_base + np.random.normal(0, 3.5)
                map_val = map_base + np.random.normal(0, 4.0)
                temp_val = temp_base + np.random.normal(0, 0.25)
                resp_val = resp_base + np.random.normal(0, 1.5)
                o2_val = min(100.0, max(90.0, o2_base + np.random.normal(0, 1.0)))
                lactate_val = max(0.5, lactate_base + np.random.normal(0, 0.15))
                creat_val = max(0.4, creat_base + np.random.normal(0, 0.08))
                wbc_val = max(3.0, wbc_base + np.random.normal(0, 0.6))
                
            # Ground truth label (active sepsis)
            label = 1 if (is_septic and t >= onset_hour) else 0
            
            # Simulate clinical lab missingness (labs drawn periodically, not hourly)
            row = {
                "patient_id": patient_id,
                "ICULOS": t,
                "Age": age,
                "Gender": gender,
                "HR": max(35.0, min(190.0, hr_val)),
                "MAP": max(30.0, min(140.0, map_val)),
                "SBP": max(50.0, min(210.0, map_val * 1.35)),
                "DBP": max(25.0, min(120.0, map_val * 0.75)),
                "Temp": max(34.0, min(41.5, temp_val)),
                "Resp": max(8.0, min(45.0, resp_val)),
                "O2Sat": max(70.0, min(100.0, o2_val)),
                "FiO2": 0.40 if (t % 8 == 0 or (is_septic and t >= onset_hour - 3)) else np.nan,
                "Lactate": lactate_val if (t % 6 == 0 or (is_septic and t >= onset_hour - 4)) else np.nan,
                "Creatinine": creat_val if (t % 12 == 0) else np.nan,
                "BUN": (creat_val * 15.0) if (t % 12 == 0) else np.nan,
                "WBC": wbc_val if (t % 12 == 0) else np.nan,
                "Platelets": np.random.normal(210.0, 30.0) if (t % 12 == 0) else np.nan,
                "SepsisLabel": label,
            }
            records.append(row)
            
    df = pd.DataFrame(records)
    logger.info(f"Synthesized ICU Cohort: {n_patients} patients, {len(df)} total hours, {df['SepsisLabel'].sum()} sepsis events ({df['SepsisLabel'].mean():.2%}).")
    return df


# ==============================================================================
# 2. Multi-Dataset Loading & Checkpointed Feature Extraction Engine
# ==============================================================================
def discover_patient_files(
    data_sources: Union[str, Path, List[Union[str, Path]]],
    max_files: Optional[int] = None
) -> List[Tuple[Path, str]]:
    """Discovers all .psv and .csv patient files across provided sources."""
    if isinstance(data_sources, (str, Path)):
        sources = [Path(data_sources)]
    else:
        sources = [Path(s) for s in data_sources]

    all_files: List[Tuple[Path, str]] = []
    for src in sources:
        if not src.exists():
            logger.warning(f"Data source path does not exist: {src}")
            continue
        
        source_tag = src.name if src.is_dir() else src.parent.name
        if src.is_dir():
            found = sorted(list(src.glob("*.psv")) + list(src.glob("**/*.psv")) +
                           list(src.glob("*.csv")) + list(src.glob("**/*.csv")))
            for f in found:
                all_files.append((f, source_tag))
        elif src.is_file():
            all_files.append((src, source_tag))

    if not all_files:
        raise FileNotFoundError(f"No .psv or .csv patient files found across sources: {data_sources}")

    if max_files:
        all_files = all_files[:max_files]
        
    return all_files


def load_file_batch(file_batch: List[Tuple[Path, str]]) -> pd.DataFrame:
    """Reads a batch of patient files into a combined DataFrame with source tagging."""
    batch_dfs = []
    for fpath, src_tag in file_batch:
        pid = fpath.stem
        sep = "|" if fpath.suffix.lower() == ".psv" else ","
        try:
            p_df = pd.read_csv(fpath, sep=sep, engine="c", low_memory=False)
            p_df["patient_id"] = f"{src_tag}_{pid}"
            p_df["hospital_source"] = src_tag
            batch_dfs.append(p_df)
        except Exception as e:
            logger.debug(f"Failed to parse {fpath}: {e}")
            continue
    if not batch_dfs:
        return pd.DataFrame()
    return pd.concat(batch_dfs, ignore_index=True)


def load_or_extract_features_with_checkpoints(
    extractor: TemporalFeatureExtractor,
    data_sources: Union[str, Path, List[Union[str, Path]]],
    processed_features_path: Path,
    checkpoint_dir: Path,
    chunk_size: int = 500,
    max_files: Optional[int] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Two-Level Checkpoint & Resume Architecture:
      Level 1: Checks for master cached parquet (data_processed/engineered_features.parquet).
               If present and not force_recompute, loads instantly in 1-2 seconds.
      Level 2: If extracting from raw files, processes in incremental chunks.
               Each completed chunk is written to checkpoint_dir/chunk_XXXX.parquet.
               If interrupted, resumes exactly from the last completed chunk without data loss.
    """
    # 1. Master Cache Level
    if not force_recompute and processed_features_path.exists():
        logger.info(f"⚡ [RESUME LEVEL 1] Found master feature cache: {processed_features_path}")
        logger.info("Loading precomputed engineered features from disk (skipping raw ingestion)...")
        feature_df = pd.read_parquet(processed_features_path)
        logger.info(f"Loaded feature matrix: {feature_df.shape[0]:,} rows, {feature_df.shape[1]:,} columns.")
        return feature_df

    # 2. Incremental Chunk Checkpoints Level
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_files = discover_patient_files(data_sources, max_files=max_files)
    n_files = len(all_files)
    logger.info(f"Discovered {n_files} patient records. Processing in chunks of {chunk_size} with auto-checkpoints...")

    chunks = [all_files[i:i + chunk_size] for i in range(0, n_files, chunk_size)]
    chunk_feature_dfs = []

    from tqdm import tqdm
    for idx, chunk in enumerate(tqdm(chunks, desc="Processing Patient Chunks", unit="chunk")):
        chunk_file = checkpoint_dir / f"chunk_{idx:04d}.parquet"
        
        # Check if this intermediate chunk was already completed in a prior run
        if not force_recompute and chunk_file.exists():
            logger.info(f"⚡ [RESUME LEVEL 2] Checkpoint found for chunk {idx + 1}/{len(chunks)} ({chunk_file.name}). Loading...")
            chunk_df = pd.read_parquet(chunk_file)
            chunk_feature_dfs.append(chunk_df)
            continue

        # Ingest and transform current batch
        batch_raw_df = load_file_batch(chunk)
        if batch_raw_df.empty:
            continue

        chunk_features = extractor.transform(batch_raw_df)
        
        # Preserve hospital source if available
        if "hospital_source" in batch_raw_df.columns:
            chunk_features["hospital_source"] = batch_raw_df["hospital_source"].values

        # Save chunk checkpoint immediately
        chunk_features.to_parquet(chunk_file, index=False, compression="snappy")
        logger.info(f"💾 Checkpoint saved: {chunk_file.name} ({len(chunk_features)} rows)")
        chunk_feature_dfs.append(chunk_features)

    # 3. Assemble and save Master Feature Cache
    logger.info("Assembling all chunk checkpoints into master feature dataset...")
    full_feature_df = pd.concat(chunk_feature_dfs, ignore_index=True)
    processed_features_path.parent.mkdir(parents=True, exist_ok=True)
    full_feature_df.to_parquet(processed_features_path, index=False, compression="snappy")
    logger.info(f"💾 Master feature cache saved to: {processed_features_path.resolve()}")
    return full_feature_df


# ==============================================================================
# 3. Model Training Engine
# ==============================================================================
def train_and_evaluate_baselines(
    config_path: str = "configs/base.yaml",
    data_dir: Optional[Union[str, List[str]]] = None,
    output_dir: str = "reports",
    use_synthetic: bool = False,
    n_synthetic_patients: int = 150,
    max_files: Optional[int] = None,
    cross_hospital_split: bool = False,
    checkpoint_dir: str = "data_processed/checkpoints",
    processed_features_path: str = "data_processed/engineered_features.parquet",
    chunk_size: int = 500,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Executes end-to-end baseline training and evaluation with multi-stage checkpointing:
      - Resumes from data_processed/engineered_features.parquet if present.
      - Resumes from incremental chunk checkpoints (data_processed/checkpoints/chunk_XXXX.parquet)
        if raw extraction was interrupted.
      - Performs patient-level group split or cross-hospital domain transfer split.
      - Trains Logistic Regression (balanced) and XGBoost (scale_pos_weight).
      - Computes AUROC, AUPRC, F1, Brier, and PhysioNet Utility.
      - Generates publication-grade figures and markdown report.
    """
    # Load configuration
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    extractor = TemporalFeatureExtractor(
        windows_hours=cfg.get("features", {}).get("windows_hours", [6, 12, 24]),
        aggregations=cfg.get("features", {}).get("aggregations", ["mean", "min", "max", "std"]),
        delta_hours=cfg.get("features", {}).get("delta_hours", [1, 3, 6]),
        include_clinical_scores=cfg.get("features", {}).get("include_clinical_scores", True),
        include_missingness=cfg.get("features", {}).get("include_missingness", True),
        include_deltas=cfg.get("features", {}).get("include_deltas", True),
        patient_id_col="patient_id",
        time_col="ICULOS",
        target_col="SepsisLabel",
    )

    # 1. Dataset Acquisition & Checkpointed Feature Extraction
    if use_synthetic or not data_dir:
        logger.info("Initializing baseline training using synthesized high-fidelity ICU cohort...")
        raw_df = generate_synthetic_icu_cohort(n_patients=n_synthetic_patients, seed=seed)
        feature_matrix = extractor.transform(raw_df)
    else:
        logger.info(f"Targeting data source(s): {data_dir}")
        feature_matrix = load_or_extract_features_with_checkpoints(
            extractor=extractor,
            data_sources=data_dir,
            processed_features_path=Path(processed_features_path),
            checkpoint_dir=Path(checkpoint_dir),
            chunk_size=chunk_size,
            max_files=max_files,
            force_recompute=force_recompute,
        )

    exclude_cols = {"patient_id", "ICULOS", "SepsisLabel", "hospital_source"}
    feature_cols = [c for c in feature_matrix.columns if c not in exclude_cols]
    target_col = "SepsisLabel"
    patient_id_col = "patient_id"

    X = feature_matrix[feature_cols].copy()
    y = feature_matrix[target_col].values.astype(int)
    groups = feature_matrix[patient_id_col].values

    # 3. Train/Validation/Test Partitioning
    hospitals = feature_matrix.get("hospital_source", None)
    if cross_hospital_split and hospitals is not None and len(hospitals.unique()) >= 2:
        unique_hosp = sorted(hospitals.unique())
        train_hosp = unique_hosp[0]
        test_hosp = unique_hosp[1]
        logger.info(f"Cross-Hospital Domain Transfer: Training on '{train_hosp}', Testing on '{test_hosp}'")
        
        train_val_mask = (hospitals == train_hosp).values
        test_mask = (hospitals == test_hosp).values
        
        X_train_val = X.iloc[train_val_mask]
        y_train_val = y[train_val_mask]
        groups_train_val = groups[train_val_mask]
        
        X_test = X.iloc[test_mask]
        y_test = y[test_mask]
        groups_test = groups[test_mask]
    else:
        logger.info(f"Executing patient-level GroupShuffleSplit on '{patient_id_col}'...")
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

        X_train_val = X.iloc[train_val_idx]
        y_train_val = y[train_val_idx]
        groups_train_val = groups[train_val_idx]

        X_test = X.iloc[test_idx]
        y_test = y[test_idx]
        groups_test = groups[test_idx]

    # Split train and validation for early stopping
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val[train_idx]

    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val[val_idx]

    logger.info(
        f"Data Partitioning: Train={len(X_train)} rows ({len(np.unique(groups[train_val_idx][train_idx]))} pts) | "
        f"Val={len(X_val)} rows ({len(np.unique(groups[train_val_idx][val_idx]))} pts) | "
        f"Test={len(X_test)} rows ({len(np.unique(groups_test))} pts)"
    )

    # 4. Handle Missing Values
    # Median imputation fitted solely on training partition to prevent distribution leakage
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # Class imbalance scale factor
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos = max(1.0, float(n_neg) / max(1.0, float(n_pos)))
    logger.info(f"Class Distribution: {n_pos} Positive vs {n_neg} Negative (scale_pos_weight = {scale_pos:.2f})")

    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    metrics_summary: List[Dict[str, Union[float, str]]] = []

    # ==========================================================================
    # Model 1: Logistic Regression (L2 Regularized with Balanced Class Weight)
    # ==========================================================================
    logger.info("Training Baseline 1: Regularized Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    lr_model = LogisticRegression(
        penalty="l2",
        C=0.1,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    results_dict["Logistic Regression"] = (y_test, lr_probs)
    lr_metrics = compute_clinical_metrics(y_test, lr_probs, model_name="Logistic Regression")
    metrics_summary.append(lr_metrics)
    logger.info(f"Logistic Regression — AUROC: {lr_metrics['AUROC']:.3f} | AUPRC: {lr_metrics['AUPRC']:.3f} | F1: {lr_metrics['F1_Score']:.3f}")

    # ==========================================================================
    # Model 2: XGBoost Classifier (Histogram-based, Fast Gradient Boosting)
    # ==========================================================================
    logger.info("Training Baseline 2: XGBoost Gradient Boosting Classifier...")
    xgb_params = cfg.get("models", {}).get("xgboost", {})
    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_params.get("n_estimators", 300),
        learning_rate=xgb_params.get("learning_rate", 0.05),
        max_depth=xgb_params.get("max_depth", 5),
        subsample=xgb_params.get("subsample", 0.8),
        colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
        scale_pos_weight=scale_pos,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=xgb_params.get("early_stopping_rounds", 30),
        random_state=seed,
        n_jobs=-1,
    )
    
    xgb_model.fit(
        X_train_imp, y_train,
        eval_set=[(X_val_imp, y_val)],
        verbose=False,
    )
    xgb_probs = xgb_model.predict_proba(X_test_imp)[:, 1]
    
    results_dict["XGBoost"] = (y_test, xgb_probs)
    xgb_metrics = compute_clinical_metrics(y_test, xgb_probs, model_name="XGBoost")
    metrics_summary.append(xgb_metrics)
    logger.info(f"XGBoost Classifier — AUROC: {xgb_metrics['AUROC']:.3f} | AUPRC: {xgb_metrics['AUPRC']:.3f} | F1: {xgb_metrics['F1_Score']:.3f}")

    # ==========================================================================
    # 5. Export Reports and Publication Figures
    # ==========================================================================
    out_path = Path(output_dir)
    figures_path = out_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    # 1. Save Metrics JSON & Markdown Report
    save_metrics_report(metrics_summary, output_dir=out_path)

    # 2. Publication Figures (300 DPI)
    dpi = cfg.get("evaluation", {}).get("dpi", 300)
    plot_roc_curves(results_dict, save_path=figures_path / "roc_curves.png", dpi=dpi)
    plot_precision_recall_curves(results_dict, save_path=figures_path / "precision_recall_curves.png", dpi=dpi)
    plot_calibration_curves(results_dict, save_path=figures_path / "calibration_curves.png", dpi=dpi)
    plot_confusion_matrices(results_dict, save_path=figures_path / "confusion_matrices.png", dpi=dpi)
    
    # Feature Importance Plot
    plot_feature_importances(
        feature_names=feature_cols,
        importances=xgb_model.feature_importances_,
        top_n=25,
        title="Top 25 Physiological Trajectory Predictors (XGBoost)",
        save_path=figures_path / "feature_importance.png",
        dpi=dpi,
    )

    # 3. Save Trained Model Weights
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_model, models_dir / "logistic_regression_baseline.joblib")
    joblib.dump(xgb_model, models_dir / "xgboost_baseline.joblib")
    joblib.dump(imputer, models_dir / "feature_imputer.joblib")
    joblib.dump(scaler, models_dir / "feature_scaler.joblib")
    logger.info(f"Saved model checkpoints and scalers to: {models_dir.resolve()}")

    logger.info("=== Baseline Training and Evaluation Pipeline Finished Successfully ===")
    return {m["model_name"]: m for m in metrics_summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CareTrajectory Baseline Training & Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML configuration")
    parser.add_argument(
        "--data_dir", nargs="+", default=None,
        help="One or more directories containing raw patient PSV or CSV files (e.g., data/setA data/setB)"
    )
    parser.add_argument("--output_dir", type=str, default="reports", help="Output directory for reports & figures")
    parser.add_argument("--use_synthetic", action="store_true", help="Force execution on synthetic ICU cohort")
    parser.add_argument("--n_patients", type=int, default=150, help="Number of synthetic patients if synthesizing")
    parser.add_argument("--max_files", type=int, default=None, help="Optional maximum number of patient files to load for quick trial")
    parser.add_argument(
        "--cross_hospital", action="store_true",
        help="Enable cross-hospital domain transfer evaluation (train on hospital A, test on hospital B)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="data_processed/checkpoints",
        help="Directory to store intermediate batch checkpoints for seamless resumption"
    )
    parser.add_argument(
        "--processed_features", type=str, default="data_processed/engineered_features.parquet",
        help="Path to master precomputed parquet feature table"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=500,
        help="Number of patient files to process per checkpoint chunk"
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Ignore existing checkpoints and force full re-computation from scratch"
    )
    
    args = parser.parse_args()
    train_and_evaluate_baselines(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_synthetic=args.use_synthetic,
        n_synthetic_patients=args.n_patients,
        max_files=args.max_files,
        cross_hospital_split=args.cross_hospital,
        checkpoint_dir=args.checkpoint_dir,
        processed_features_path=args.processed_features,
        chunk_size=args.chunk_size,
        force_recompute=args.force_recompute,
    )
