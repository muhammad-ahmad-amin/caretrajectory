"""
CareTrajectory — Reproducible Classical ML Baseline Training & Benchmark Runner

Trains and validates baseline clinical models (Logistic Regression & XGBoost)
on ICU patient deterioration trajectories. Enforces strict patient-level grouping
to prevent data leakage, handles class imbalance, and generates publication-grade
evaluation artifacts and visual reports.
"""

import argparse
import gc
import logging
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ==============================================================================
# 1. Synthetic ICU Data Generator
# ==============================================================================

def generate_synthetic_icu_cohort(
    n_patients: int = 120,
    seed: int = 42
) -> pd.DataFrame:

    np.random.seed(seed)
    records = []

    for pid in range(1, n_patients + 1):

        patient_id = f"p_{pid:05d}"

        los = np.random.randint(18, 50)
        age = float(np.random.normal(63.0, 12.0))
        gender = int(np.random.choice([0, 1]))
        is_septic = np.random.rand() < 0.12

        onset_hour = (
            np.random.randint(10, los - 4)
            if is_septic
            else 9999
        )

        hr_base = np.random.normal(76.0, 8.0)
        map_base = np.random.normal(86.0, 9.0)
        temp_base = np.random.normal(36.9, 0.4)
        resp_base = np.random.normal(16.0, 2.5)
        o2_base = np.random.normal(98.0, 1.2)
        lactate_base = np.random.normal(1.2, 0.3)
        creat_base = np.random.normal(0.9, 0.2)
        wbc_base = np.random.normal(7.5, 1.5)

        for t in range(1, los + 1):

            if is_septic and (t >= onset_hour - 6):

                hrs_since_deterioration = max(
                    0,
                    t - (onset_hour - 6)
                )

                hr_val = (
                    hr_base
                    + 3.2 * hrs_since_deterioration
                    + np.random.normal(0, 3.0)
                )

                map_val = (
                    map_base
                    - 2.8 * hrs_since_deterioration
                    + np.random.normal(0, 3.0)
                )

                temp_val = (
                    temp_base
                    + 0.18 * hrs_since_deterioration
                    + np.random.normal(0, 0.2)
                )

                resp_val = (
                    resp_base
                    + 1.1 * hrs_since_deterioration
                    + np.random.normal(0, 1.5)
                )

                o2_val = max(
                    84.0,
                    o2_base
                    - 0.9 * hrs_since_deterioration
                    + np.random.normal(0, 0.8)
                )

                lactate_val = (
                    lactate_base
                    + 0.35 * hrs_since_deterioration
                    + np.random.normal(0, 0.2)
                )

                creat_val = (
                    creat_base
                    + 0.12 * hrs_since_deterioration
                    + np.random.normal(0, 0.05)
                )

                wbc_val = (
                    wbc_base
                    + 0.8 * hrs_since_deterioration
                    + np.random.normal(0, 0.5)
                )

            else:

                hr_val = hr_base + np.random.normal(0, 3.5)
                map_val = map_base + np.random.normal(0, 4.0)
                temp_val = temp_base + np.random.normal(0, 0.25)
                resp_val = resp_base + np.random.normal(0, 1.5)

                o2_val = min(
                    100.0,
                    max(
                        90.0,
                        o2_base + np.random.normal(0, 1.0)
                    )
                )

                lactate_val = max(
                    0.5,
                    lactate_base + np.random.normal(0, 0.15)
                )

                creat_val = max(
                    0.4,
                    creat_base + np.random.normal(0, 0.08)
                )

                wbc_val = max(
                    3.0,
                    wbc_base + np.random.normal(0, 0.6)
                )

            label = (
                1
                if (is_septic and t >= onset_hour)
                else 0
            )

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
                "FiO2": (
                    0.40
                    if (
                        t % 8 == 0
                        or (is_septic and t >= onset_hour - 3)
                    )
                    else np.nan
                ),
                "Lactate": (
                    lactate_val
                    if (
                        t % 6 == 0
                        or (is_septic and t >= onset_hour - 4)
                    )
                    else np.nan
                ),
                "Creatinine": (
                    creat_val
                    if t % 12 == 0
                    else np.nan
                ),
                "BUN": (
                    creat_val * 15.0
                    if t % 12 == 0
                    else np.nan
                ),
                "WBC": (
                    wbc_val
                    if t % 12 == 0
                    else np.nan
                ),
                "Platelets": (
                    np.random.normal(210.0, 30.0)
                    if t % 12 == 0
                    else np.nan
                ),
                "SepsisLabel": label,
            }

            records.append(row)

    df = pd.DataFrame(records)

    logger.info(
        f"Synthesized ICU Cohort: {n_patients} patients, "
        f"{len(df)} total hours, "
        f"{df['SepsisLabel'].sum()} sepsis events "
        f"({df['SepsisLabel'].mean():.2%})."
    )

    return df


# ==============================================================================
# 2. Multi-Dataset Loading
# ==============================================================================

def discover_patient_files(
    data_sources: Union[str, Path, List[Union[str, Path]]],
    max_files: Optional[int] = None
) -> List[Tuple[Path, str]]:

    if isinstance(data_sources, (str, Path)):
        sources = [Path(data_sources)]
    else:
        sources = [Path(s) for s in data_sources]

    all_files = []

    for src in sources:

        if not src.exists():
            logger.warning(
                f"Data source path does not exist: {src}"
            )
            continue

        source_tag = (
            src.name
            if src.is_dir()
            else src.parent.name
        )

        if src.is_dir():

            found = sorted(
                list(src.glob("*.psv"))
                + list(src.glob("**/*.psv"))
                + list(src.glob("*.csv"))
                + list(src.glob("**/*.csv"))
            )

            for f in found:
                all_files.append((f, source_tag))

        elif src.is_file():
            all_files.append((src, source_tag))

    if not all_files:
        raise FileNotFoundError(
            f"No .psv or .csv patient files found across sources: "
            f"{data_sources}"
        )

    if max_files:
        all_files = all_files[:max_files]

    return all_files


PHYSIONET_DTYPES = {
    "HR": "float32",
    "O2Sat": "float32",
    "Temp": "float32",
    "SBP": "float32",
    "MAP": "float32",
    "DBP": "float32",
    "Resp": "float32",
    "BaseExcess": "float32",
    "HCO3": "float32",
    "FiO2": "float32",
    "pH": "float32",
    "PaCO2": "float32",
    "SaO2": "float32",
    "AST": "float32",
    "BUN": "float32",
    "Alkalinephos": "float32",
    "Calcium": "float32",
    "Chloride": "float32",
    "Creatinine": "float32",
    "Bilirubin_direct": "float32",
    "Glucose": "float32",
    "Lactate": "float32",
    "Magnesium": "float32",
    "Phosphate": "float32",
    "Potassium": "float32",
    "Bilirubin_total": "float32",
    "TroponinI": "float32",
    "Hct": "float32",
    "Hgb": "float32",
    "PTT": "float32",
    "WBC": "float32",
    "Fibrinogen": "float32",
    "Platelets": "float32",
    "Age": "float32",
    "Gender": "float32",
    "Unit1": "float32",
    "Unit2": "float32",
    "HospAdmTime": "float32",
    "ICULOS": "float32",
    "SepsisLabel": "int32",
}


def _read_single_file_fast(
    item: Tuple[Path, str]
) -> Optional[pd.DataFrame]:

    fpath, src_tag = item

    sep = "|" if fpath.suffix.lower() == ".psv" else ","

    try:

        p_df = pd.read_csv(
            fpath,
            sep=sep,
            engine="c",
            dtype=PHYSIONET_DTYPES,
            low_memory=False,
        )

        p_df["patient_id"] = f"{src_tag}_{fpath.stem}"
        p_df["hospital_source"] = src_tag

        return p_df

    except Exception:

        try:

            p_df = pd.read_csv(
                fpath,
                sep=sep,
                engine="c",
                low_memory=False,
            )

            p_df["patient_id"] = f"{src_tag}_{fpath.stem}"
            p_df["hospital_source"] = src_tag

            return p_df

        except Exception:

            return None


def load_file_batch(
    file_batch: List[Tuple[Path, str]],
    max_workers: Optional[int] = None
) -> pd.DataFrame:

    workers = (
        max_workers
        or min(32, (os.cpu_count() or 4) * 4)
    )

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        results = list(
            executor.map(
                _read_single_file_fast,
                file_batch
            )
        )

    valid_dfs = [
        df
        for df in results
        if df is not None and not df.empty
    ]

    if not valid_dfs:
        return pd.DataFrame()

    return pd.concat(
        valid_dfs,
        ignore_index=True
    )


def load_raw_dataset(
    data_sources: Union[str, Path, List[Union[str, Path]]],
    max_files: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:

    all_files = discover_patient_files(
        data_sources,
        max_files=max_files
    )

    workers = (
        max_workers
        or min(32, (os.cpu_count() or 4) * 4)
    )

    logger.info(
        f"Fast Parallel Ingestion: reading "
        f"{len(all_files):,} records across "
        f"{workers} I/O threads..."
    )

    from tqdm import tqdm

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        results = list(
            tqdm(
                executor.map(
                    _read_single_file_fast,
                    all_files
                ),
                total=len(all_files),
                desc="Ingesting Records",
                unit="pt",
            )
        )

    valid_dfs = [
        df
        for df in results
        if df is not None and not df.empty
    ]

    if not valid_dfs:
        raise ValueError(
            "Could not successfully read any patient files."
        )

    combined_df = pd.concat(
        valid_dfs,
        ignore_index=True
    )

    logger.info(
        f"Ingested {len(valid_dfs):,} patients "
        f"({len(combined_df):,} hourly records) "
        f"in parallel."
    )

    return combined_df


def load_or_extract_features_with_checkpoints(
    extractor: TemporalFeatureExtractor,
    data_sources: Union[str, Path, List[Union[str, Path]]],
    processed_features_path: Path,
    checkpoint_dir: Path,
    chunk_size: int = 500,
    max_files: Optional[int] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:

    # Level 1: Master cache

    if (
        not force_recompute
        and processed_features_path.exists()
    ):

        logger.info(
            f"Found master feature cache: "
            f"{processed_features_path}"
        )

        logger.info(
            "Loading precomputed engineered features "
            "from disk (skipping raw ingestion)..."
        )

        feature_df = pd.read_parquet(
            processed_features_path
        )

        logger.info(
            f"Loaded feature matrix: "
            f"{feature_df.shape[0]:,} rows, "
            f"{feature_df.shape[1]:,} columns."
        )

        return feature_df

    # Level 2: Chunk checkpoints

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_files = discover_patient_files(
        data_sources,
        max_files=max_files
    )

    n_files = len(all_files)

    logger.info(
        f"Discovered {n_files} patient records. "
        f"Processing in chunks of {chunk_size}..."
    )

    chunks = [
        all_files[i:i + chunk_size]
        for i in range(
            0,
            n_files,
            chunk_size
        )
    ]

    chunk_feature_dfs = []

    from tqdm import tqdm

    for idx, chunk in enumerate(
        tqdm(
            chunks,
            desc="Processing Patient Chunks",
            unit="chunk"
        )
    ):

        chunk_file = (
            checkpoint_dir
            / f"chunk_{idx:04d}.parquet"
        )

        if (
            not force_recompute
            and chunk_file.exists()
        ):

            logger.info(
                f"Loading checkpoint "
                f"{idx + 1}/{len(chunks)}: "
                f"{chunk_file.name}"
            )

            chunk_df = pd.read_parquet(
                chunk_file
            )

            chunk_feature_dfs.append(
                chunk_df
            )

            continue

        batch_raw_df = load_file_batch(chunk)

        if batch_raw_df.empty:
            continue

        chunk_features = extractor.transform(
            batch_raw_df
        )

        if "hospital_source" in batch_raw_df.columns:

            chunk_features[
                "hospital_source"
            ] = batch_raw_df[
                "hospital_source"
            ].values

        chunk_features.to_parquet(
            chunk_file,
            index=False,
            compression="snappy"
        )

        logger.info(
            f"Checkpoint saved: "
            f"{chunk_file.name} "
            f"({len(chunk_features)} rows)"
        )

        chunk_feature_dfs.append(
            chunk_features
        )

    logger.info(
        "Assembling all chunk checkpoints "
        "into master feature dataset..."
    )

    full_feature_df = pd.concat(
        chunk_feature_dfs,
        ignore_index=True
    )

    processed_features_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    full_feature_df.to_parquet(
        processed_features_path,
        index=False,
        compression="snappy"
    )

    logger.info(
        f"Master feature cache saved to: "
        f"{processed_features_path.resolve()}"
    )

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

    # Load configuration

    if Path(config_path).exists():

        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as f:

            cfg = yaml.safe_load(f)

    else:

        cfg = {}

    seed = cfg.get("seed", 42)

    np.random.seed(seed)

    extractor = TemporalFeatureExtractor(
        windows_hours=cfg.get(
            "features", {}
        ).get(
            "windows_hours",
            [6, 12, 24]
        ),

        aggregations=cfg.get(
            "features", {}
        ).get(
            "aggregations",
            ["mean", "min", "max", "std"]
        ),

        delta_hours=cfg.get(
            "features", {}
        ).get(
            "delta_hours",
            [1, 3, 6]
        ),

        include_clinical_scores=cfg.get(
            "features", {}
        ).get(
            "include_clinical_scores",
            True
        ),

        include_missingness=cfg.get(
            "features", {}
        ).get(
            "include_missingness",
            True
        ),

        include_deltas=cfg.get(
            "features", {}
        ).get(
            "include_deltas",
            True
        ),

        patient_id_col="patient_id",
        time_col="ICULOS",
        target_col="SepsisLabel",
    )

    # ==========================================================================
    # Dataset acquisition
    # ==========================================================================

    if use_synthetic or not data_dir:

        logger.info(
            "Initializing baseline training using "
            "synthesized ICU cohort..."
        )

        raw_df = generate_synthetic_icu_cohort(
            n_patients=n_synthetic_patients,
            seed=seed
        )

        feature_matrix = extractor.transform(
            raw_df
        )

    else:

        logger.info(
            f"Targeting data source(s): {data_dir}"
        )

        feature_matrix = (
            load_or_extract_features_with_checkpoints(
                extractor=extractor,
                data_sources=data_dir,
                processed_features_path=Path(
                    processed_features_path
                ),
                checkpoint_dir=Path(
                    checkpoint_dir
                ),
                chunk_size=chunk_size,
                max_files=max_files,
                force_recompute=force_recompute,
            )
        )

    # ==========================================================================
    # Feature preparation
    # ==========================================================================

    exclude_cols = {
        "patient_id",
        "ICULOS",
        "SepsisLabel",
        "hospital_source",
    }

    feature_cols = [
        c
        for c in feature_matrix.columns
        if c not in exclude_cols
    ]

    target_col = "SepsisLabel"
    patient_id_col = "patient_id"

    float_cols = (
        feature_matrix[feature_cols]
        .select_dtypes(
            include=["float64"]
        )
        .columns
    )

    if len(float_cols) > 0:

        logger.info(
            f"Downcasting {len(float_cols)} "
            f"float64 columns to float32..."
        )

        feature_matrix[
            float_cols
        ] = feature_matrix[
            float_cols
        ].astype(
            np.float32
        )

    gc.collect()

    y = feature_matrix[
        target_col
    ].values.astype(
        np.int32
    )

    patient_ids = feature_matrix[
        patient_id_col
    ].values

    # ==========================================================================
    # Patient-level partitioning
    # ==========================================================================

    logger.info(
        "Partitioning cohort by unique patient IDs..."
    )

    hospitals = feature_matrix.get(
        "hospital_source",
        None
    )

    if (
        cross_hospital_split
        and hospitals is not None
        and len(hospitals.unique()) >= 2
    ):

        unique_hosp = sorted(
            hospitals.unique()
        )

        train_hosp = unique_hosp[0]
        test_hosp = unique_hosp[1]

        logger.info(
            f"Cross-Hospital Domain Transfer: "
            f"Training on '{train_hosp}', "
            f"Testing on '{test_hosp}'"
        )

        train_val_mask = (
            hospitals == train_hosp
        ).values

        test_mask = (
            hospitals == test_hosp
        ).values

        train_val_pids = np.unique(
            patient_ids[
                train_val_mask
            ]
        )

        np.random.seed(seed)

        np.random.shuffle(
            train_val_pids
        )

        split_pt = int(
            len(train_val_pids) * 0.85
        )

        train_pids = set(
            train_val_pids[:split_pt]
        )

        val_pids = set(
            train_val_pids[split_pt:]
        )

        train_mask = (
            train_val_mask
            & pd.Series(
                patient_ids
            ).isin(
                train_pids
            ).values
        )

        val_mask = (
            train_val_mask
            & pd.Series(
                patient_ids
            ).isin(
                val_pids
            ).values
        )

    else:

        unique_pids = np.unique(
            patient_ids
        )

        np.random.seed(seed)

        np.random.shuffle(
            unique_pids
        )

        n_total_pts = len(
            unique_pids
        )

        n_train_pts = int(
            n_total_pts * 0.70
        )

        n_val_pts = int(
            n_total_pts * 0.15
        )

        train_pids = set(
            unique_pids[
                :n_train_pts
            ]
        )

        val_pids = set(
            unique_pids[
                n_train_pts:
                n_train_pts + n_val_pts
            ]
        )

        test_pids = set(
            unique_pids[
                n_train_pts + n_val_pts:
            ]
        )

        pid_series = pd.Series(
            patient_ids
        )

        train_mask = (
            pid_series
            .isin(train_pids)
            .values
        )

        val_mask = (
            pid_series
            .isin(val_pids)
            .values
        )

        test_mask = (
            pid_series
            .isin(test_pids)
            .values
        )

    y_train = y[
        train_mask
    ]

    y_val = y[
        val_mask
    ]

    y_test = y[
        test_mask
    ]

    logger.info(
        f"Data Partitioning: "
        f"Train={np.sum(train_mask):,} rows | "
        f"Val={np.sum(val_mask):,} rows | "
        f"Test={np.sum(test_mask):,} rows"
    )

    # ==========================================================================
    # Class imbalance
    # ==========================================================================

    n_neg = int(
        np.sum(
            y_train == 0
        )
    )

    n_pos = int(
        np.sum(
            y_train == 1
        )
    )

    scale_pos = max(
        1.0,
        float(n_neg)
        / max(
            1.0,
            float(n_pos)
        )
    )

    logger.info(
        f"Class Distribution: "
        f"{n_pos:,} Positive vs "
        f"{n_neg:,} Negative "
        f"(scale_pos_weight = {scale_pos:.2f})"
    )

    results_dict = {}
    metrics_summary = []

    # ==========================================================================
    # Model 1: XGBoost
    # ==========================================================================

    logger.info(
        "Training Model 1: XGBoost Gradient Boosting Classifier..."
    )

    xgb_params = cfg.get(
        "models",
        {}
    ).get(
        "xgboost",
        {}
    )

    xgb_model = xgb.XGBClassifier(

        n_estimators=xgb_params.get(
            "n_estimators",
            100
        ),

        learning_rate=xgb_params.get(
            "learning_rate",
            0.05
        ),

        max_depth=xgb_params.get(
            "max_depth",
            5
        ),

        subsample=xgb_params.get(
            "subsample",
            0.8
        ),

        colsample_bytree=xgb_params.get(
            "colsample_bytree",
            0.8
        ),

        scale_pos_weight=scale_pos,

        tree_method="hist",

        eval_metric="aucpr",

        early_stopping_rounds=xgb_params.get(
            "early_stopping_rounds",
            30
        ),

        random_state=seed,

        n_jobs=4,
    )

    # --------------------------------------------------------------------------
    # Create train / validation / test feature matrices
    # --------------------------------------------------------------------------

    X_train_xgb = feature_matrix.loc[
        train_mask,
        feature_cols
    ]

    X_val_xgb = feature_matrix.loc[
        val_mask,
        feature_cols
    ]

    X_test_xgb = feature_matrix.loc[
        test_mask,
        feature_cols
    ]

    # --------------------------------------------------------------------------
    # MEMORY GUARD FOR XGBOOST
    # --------------------------------------------------------------------------

    max_xgb_train_rows = 200000

    if len(y_train) > max_xgb_train_rows:

        logger.info(
            f"Memory Guard: Subsampling "
            f"{max_xgb_train_rows:,} rows "
            f"from {len(y_train):,} "
            f"training rows for XGBoost..."
        )

        rng = np.random.default_rng(
            seed
        )

        xgb_indices = rng.choice(
            len(y_train),
            size=max_xgb_train_rows,
            replace=False
        )

        X_train_xgb = (
            X_train_xgb.iloc[
                xgb_indices
            ]
        )

        y_train_xgb = (
            y_train[
                xgb_indices
            ]
        )

    else:

        y_train_xgb = y_train

    logger.info(
        f"XGBoost training rows: "
        f"{len(y_train_xgb):,}"
    )

    logger.info(
        f"XGBoost validation rows: "
        f"{len(y_val):,}"
    )

    # --------------------------------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------------------------------

    xgb_model.fit(
        X_train_xgb,
        y_train_xgb,
        eval_set=[
            (
                X_val_xgb,
                y_val
            )
        ],
        verbose=False,
    )

    xgb_probs = (
        xgb_model
        .predict_proba(
            X_test_xgb
        )[:, 1]
    )

    results_dict[
        "XGBoost"
    ] = (
        y_test,
        xgb_probs
    )

    xgb_metrics = (
        compute_clinical_metrics(
            y_test,
            xgb_probs,
            model_name="XGBoost"
        )
    )

    metrics_summary.append(
        xgb_metrics
    )

    logger.info(
        f"XGBoost Classifier — "
        f"AUROC: {xgb_metrics['AUROC']:.3f} | "
        f"AUPRC: {xgb_metrics['AUPRC']:.3f} | "
        f"F1: {xgb_metrics['F1_Score']:.3f}"
    )

    # Free XGBoost feature matrices

    del X_val_xgb
    del X_test_xgb
    del X_train_xgb
    del y_train_xgb

    gc.collect()

    # ==========================================================================
    # Model 2: Logistic Regression
    # ==========================================================================

    logger.info(
        "Training Model 2: Regularized Logistic Regression..."
    )

    max_lr_train_rows = 150000

    # Recreate train/test feature matrices

    X_train_lr_source = feature_matrix.loc[
        train_mask,
        feature_cols
    ]

    X_test_lr_source = feature_matrix.loc[
        test_mask,
        feature_cols
    ]

    if len(y_train) > max_lr_train_rows:

        logger.info(
            f"Memory Guard: Subsampling "
            f"{max_lr_train_rows:,} rows "
            f"from training set for "
            f"Logistic Regression..."
        )

        rng = np.random.default_rng(
            seed
        )

        sub_indices = rng.choice(
            len(y_train),
            size=max_lr_train_rows,
            replace=False
        )

        X_train_lr = (
            X_train_lr_source.iloc[
                sub_indices
            ]
        )

        y_train_lr = (
            y_train[
                sub_indices
            ]
        )

    else:

        X_train_lr = X_train_lr_source
        y_train_lr = y_train

    # Median imputation

    col_medians = (
        X_train_lr
        .median()
        .astype(np.float32)
    )

    X_train_lr_imp = (
        X_train_lr
        .fillna(col_medians)
        .values
        .astype(np.float32)
    )

    X_test_lr_imp = (
        X_test_lr_source
        .fillna(col_medians)
        .values
        .astype(np.float32)
    )

    X_train_lr = np.nan_to_num(X_train_lr, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_lr = np.nan_to_num(X_val_lr, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_lr = np.nan_to_num(X_test_lr, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train_lr_imp
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_lr_imp
        )
    )

    del X_train_lr
    del X_train_lr_imp
    del X_test_lr_source
    del X_test_lr_imp
    del X_train_lr_source

    gc.collect()

    lr_model = LogisticRegression(
        penalty="l2",
        C=0.1,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )

    lr_model.fit(
        X_train_scaled,
        y_train_lr
    )

    lr_probs = (
        lr_model
        .predict_proba(
            X_test_scaled
        )[:, 1]
    )

    results_dict[
        "Logistic Regression"
    ] = (
        y_test,
        lr_probs
    )

    lr_metrics = (
        compute_clinical_metrics(
            y_test,
            lr_probs,
            model_name="Logistic Regression"
        )
    )

    metrics_summary.append(
        lr_metrics
    )

    logger.info(
        f"Logistic Regression — "
        f"AUROC: {lr_metrics['AUROC']:.3f} | "
        f"AUPRC: {lr_metrics['AUPRC']:.3f} | "
        f"F1: {lr_metrics['F1_Score']:.3f}"
    )

    del X_train_scaled
    del X_test_scaled

    gc.collect()

    # ==========================================================================
    # 5. Export Reports
    # ==========================================================================

    out_path = Path(
        output_dir
    )

    figures_path = (
        out_path
        / "figures"
    )

    figures_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # Metrics report

    save_metrics_report(
        metrics_summary,
        output_dir=out_path
    )

    # Figures

    dpi = cfg.get(
        "evaluation",
        {}
    ).get(
        "dpi",
        300
    )

    plot_roc_curves(
        results_dict,
        save_path=(
            figures_path
            / "roc_curves.png"
        ),
        dpi=dpi
    )

    plot_precision_recall_curves(
        results_dict,
        save_path=(
            figures_path
            / "precision_recall_curves.png"
        ),
        dpi=dpi
    )

    plot_calibration_curves(
        results_dict,
        save_path=(
            figures_path
            / "calibration_curves.png"
        ),
        dpi=dpi
    )

    plot_confusion_matrices(
        results_dict,
        save_path=(
            figures_path
            / "confusion_matrices.png"
        ),
        dpi=dpi
    )

    plot_feature_importances(
        feature_names=feature_cols,
        importances=xgb_model.feature_importances_,
        top_n=25,
        title=(
            "Top 25 Physiological "
            "Trajectory Predictors (XGBoost)"
        ),
        save_path=(
            figures_path
            / "feature_importance.png"
        ),
        dpi=dpi
    )

    # ==========================================================================
    # Save models
    # ==========================================================================

    models_dir = Path(
        "models"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        lr_model,
        models_dir
        / "logistic_regression_baseline.joblib"
    )

    joblib.dump(
        xgb_model,
        models_dir
        / "xgboost_baseline.joblib"
    )

    joblib.dump(
        col_medians,
        models_dir
        / "feature_medians.joblib"
    )

    joblib.dump(
        scaler,
        models_dir
        / "feature_scaler.joblib"
    )

    logger.info(
        f"Saved model checkpoints and scalers to: "
        f"{models_dir.resolve()}"
    )

    logger.info(
        "=== Baseline Training and Evaluation "
        "Pipeline Finished Successfully ==="
    )

    return {
        m["model_name"]: m
        for m in metrics_summary
    }


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "CareTrajectory Baseline Training "
            "& Benchmark Runner"
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration"
    )

    parser.add_argument(
        "--data_dir",
        nargs="+",
        default=None,
        help=(
            "One or more directories containing "
            "raw patient PSV or CSV files"
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Output directory for reports & figures"
    )

    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Force execution on synthetic ICU cohort"
    )

    parser.add_argument(
        "--n_patients",
        type=int,
        default=150,
        help=(
            "Number of synthetic patients "
            "if synthesizing"
        )
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help=(
            "Optional maximum number of patient "
            "files to load for quick trial"
        )
    )

    parser.add_argument(
        "--cross_hospital",
        action="store_true",
        help=(
            "Enable cross-hospital domain transfer "
            "evaluation"
        )
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="data_processed/checkpoints",
        help=(
            "Directory to store intermediate "
            "batch checkpoints"
        )
    )

    parser.add_argument(
        "--processed_features",
        type=str,
        default=(
            "data_processed/"
            "engineered_features.parquet"
        ),
        help=(
            "Path to master precomputed "
            "parquet feature table"
        )
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help=(
            "Number of patient files to process "
            "per checkpoint chunk"
        )
    )

    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help=(
            "Ignore existing checkpoints and "
            "force full re-computation"
        )
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

        processed_features_path=(
            args.processed_features
        ),

        chunk_size=args.chunk_size,

        force_recompute=args.force_recompute,
    )
