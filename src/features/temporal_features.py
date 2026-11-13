"""
src/features/temporal_features.py
================================================================================
CareTrajectory — High-Throughput Temporal Feature Extraction Engine

Extracts non-lookahead multi-horizon rolling statistics (mean, min, max, std),
physiological deltas (rates of change), and measurement missingness dynamics
across patient ICU timelines. Grouping strictly preserves patient boundaries.
================================================================================
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.features.clinical_scores import compute_all_clinical_scores

logger = logging.getLogger("CareTrajectory.Features")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [CareTrajectory.Features] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class TemporalFeatureExtractor:
    """
    High-performance temporal feature extractor for clinical time series.
    
    Generates:
      1. Rolling aggregations (Mean, Min, Max, Std) over configurable windows (6h, 12h, 24h).
      2. Temporal deltas (velocity/acceleration) over 1h, 3h, and 6h horizons.
      3. Missingness indicators and observation frequency counters.
      4. Bedside clinical risk scores (Shock Index, SIRS, qSOFA, P/F ratio).
      
    All operations are strictly backward-looking (closed='right') to eliminate data leakage.
    """
    def __init__(
        self,
        windows_hours: Optional[List[int]] = None,
        aggregations: Optional[List[str]] = None,
        delta_hours: Optional[List[int]] = None,
        vital_columns: Optional[List[str]] = None,
        lab_columns: Optional[List[str]] = None,
        demographic_columns: Optional[List[str]] = None,
        include_clinical_scores: bool = True,
        include_missingness: bool = True,
        include_deltas: bool = True,
        patient_id_col: str = "patient_id",
        time_col: str = "ICULOS",
        target_col: str = "SepsisLabel",
    ):
        self.windows_hours = windows_hours or [6, 12, 24]
        self.aggregations = aggregations or ["mean", "min", "max", "std"]
        self.delta_hours = delta_hours or [1, 3, 6]
        
        self.vital_columns = vital_columns or [
            "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"
        ]
        self.lab_columns = lab_columns or [
            "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
            "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
            "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
            "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
            "Fibrinogen", "Platelets"
        ]
        self.demographic_columns = demographic_columns or [
            "Age", "Gender", "Unit1", "Unit2", "HospAdmTime"
        ]
        
        self.include_clinical_scores = include_clinical_scores
        self.include_missingness = include_missingness
        self.include_deltas = include_deltas
        
        self.patient_id_col = patient_id_col
        self.time_col = time_col
        self.target_col = target_col
        
        # Track feature column names after extraction
        self.feature_names_: List[str] = []

    def _extract_single_patient(self, p_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal features for one patient trajectory.
        Guarantees strictly backward-looking calculations without lookahead leakage.
        """
        # Ensure chronological ordering by ICU length of stay
        if self.time_col in p_df.columns:
            p_df = p_df.sort_values(by=self.time_col).copy()
        else:
            p_df = p_df.copy()

        n_rows = len(p_df)
        features_dict: Dict[str, Union[pd.Series, np.ndarray]] = {}

        # 1. Preserve Identifiers & Target (if present)
        for col in [self.patient_id_col, self.time_col, self.target_col]:
            if col in p_df.columns:
                features_dict[col] = p_df[col].values

        # 2. Preserve Demographics (static / time-invariant per stay)
        for col in self.demographic_columns:
            if col in p_df.columns:
                features_dict[col] = p_df[col].astype(np.float32).values

        # 3. Clinical Risk Indices & Physiological Derivations
        if self.include_clinical_scores:
            clinical_scores = compute_all_clinical_scores(p_df)
            for col in clinical_scores.columns:
                features_dict[col] = clinical_scores[col].values

        # Identify numerical channels available for temporal windowing
        time_series_cols = [
            c for c in (self.vital_columns + self.lab_columns) if c in p_df.columns
        ]

        # 4. Forward-Fill for Rolling Window Continuity (within patient only)
        # Clinical reasoning: Vitals/labs persist until a new lab is drawn
        ffill_df = p_df[time_series_cols].ffill()

        # 5. Multi-Horizon Rolling Windows (6h, 12h, 24h)
        for w in self.windows_hours:
            # Using rolling over integer count of rows (each row = 1 hour in PhysioNet)
            rolling_obj = ffill_df.rolling(window=w, min_periods=1, closed="right")
            
            if "mean" in self.aggregations:
                mean_vals = rolling_obj.mean().astype(np.float32)
                for col in time_series_cols:
                    features_dict[f"{col}_mean_{w}h"] = mean_vals[col].values

            if "min" in self.aggregations:
                min_vals = rolling_obj.min().astype(np.float32)
                for col in time_series_cols:
                    features_dict[f"{col}_min_{w}h"] = min_vals[col].values

            if "max" in self.aggregations:
                max_vals = rolling_obj.max().astype(np.float32)
                for col in time_series_cols:
                    features_dict[f"{col}_max_{w}h"] = max_vals[col].values

            if "std" in self.aggregations:
                std_vals = rolling_obj.std().fillna(0.0).astype(np.float32)
                for col in time_series_cols:
                    features_dict[f"{col}_std_{w}h"] = std_vals[col].values

        # 6. Physiological Velocity & Acceleration (Deltas)
        if self.include_deltas:
            for k in self.delta_hours:
                # Absolute change: X_t - X_{t-k}
                delta_abs = (ffill_df - ffill_df.shift(k)).astype(np.float32)
                # Relative change: (X_t - X_{t-k}) / (|X_{t-k}| + eps)
                delta_rel = (delta_abs / (ffill_df.shift(k).abs() + 1e-6)).astype(np.float32)
                
                # Apply deltas to high-priority vitals & critical labs
                priority_channels = [
                    c for c in ["HR", "MAP", "SBP", "Resp", "Temp", "O2Sat", "Lactate", "Creatinine", "WBC"]
                    if c in time_series_cols
                ]
                for col in priority_channels:
                    features_dict[f"{col}_delta_{k}h"] = delta_abs[col].fillna(0.0).values
                    features_dict[f"{col}_reldelta_{k}h"] = delta_rel[col].fillna(0.0).values

        # 7. Missingness Dynamics & Measurement Frequency
        if self.include_missingness:
            # Binary mask: was the lab/vital measured at this specific hour?
            measured_mask = p_df[time_series_cols].notna().astype(np.float32)
            
            for col in time_series_cols:
                features_dict[f"{col}_is_measured"] = measured_mask[col].values

            # Observation frequency over rolling windows
            for w in self.windows_hours:
                freq_w = measured_mask.rolling(window=w, min_periods=1, closed="right").sum().astype(np.float32)
                # Select key labs where ordering frequency directly signals clinical urgency
                freq_channels = [
                    c for c in ["Lactate", "Creatinine", "WBC", "Platelets", "BUN", "FiO2", "pH", "PaCO2"]
                    if c in time_series_cols
                ]
                for col in freq_channels:
                    features_dict[f"{col}_meas_count_{w}h"] = freq_w[col].values

            # Time since last measurement (elapsed hours)
            for col in ["Lactate", "Creatinine", "WBC", "MAP", "HR"]:
                if col in time_series_cols:
                    elapsed = np.zeros(n_rows, dtype=np.float32)
                    counter = 0.0
                    col_vals = p_df[col].values
                    for i in range(n_rows):
                        if not np.isnan(col_vals[i]):
                            counter = 0.0
                        else:
                            counter += 1.0
                        elapsed[i] = counter
                    features_dict[f"{col}_hours_since_last_meas"] = elapsed

        # Assemble extracted features into DataFrame
        out_df = pd.DataFrame(features_dict, index=p_df.index)
        return out_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a DataFrame containing single or multiple patient trajectories.
        Strictly groups by patient_id to prevent any cross-patient state bleeding.
        
        Args:
            df: Raw DataFrame containing patient time-series records.
            
        Returns:
            pd.DataFrame: Feature-engineered matrix with rolling windows, deltas, and scores.
        """
        if self.patient_id_col in df.columns:
            groups = [group for _, group in df.groupby(self.patient_id_col, sort=False)]
            n_patients = len(groups)
            if n_patients > 10:
                workers = min(16, (os.cpu_count() or 4) * 2)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    patient_dfs = list(executor.map(self._extract_single_patient, groups))
            else:
                patient_dfs = [self._extract_single_patient(g) for g in groups]
            
            transformed_df = pd.concat(patient_dfs, axis=0, ignore_index=True)
        else:
            logger.info("No patient_id column detected. Processing as a single trajectory...")
            transformed_df = self._extract_single_patient(df)

        # Cache feature column list (excluding identifiers and target)
        exclude_cols = {self.patient_id_col, self.time_col, self.target_col}
        self.feature_names_ = [c for c in transformed_df.columns if c not in exclude_cols]
        
        # Optimize memory footprint by downcasting float64 to float32
        float_cols = transformed_df.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            transformed_df[float_cols] = transformed_df[float_cols].astype(np.float32)

        logger.info(
            f"Feature extraction complete: {transformed_df.shape[0]} rows, "
            f"{len(self.feature_names_)} engineered features generated."
        )
        return transformed_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform compatibility with scikit-learn API."""
        return self.transform(df)


def extract_features(
    df: pd.DataFrame,
    windows_hours: Optional[List[int]] = None,
    include_clinical_scores: bool = True,
    include_missingness: bool = True,
    include_deltas: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience functional wrapper for TemporalFeatureExtractor.
    Returns the enriched DataFrame along with the list of engineered feature names.
    """
    extractor = TemporalFeatureExtractor(
        windows_hours=windows_hours,
        include_clinical_scores=include_clinical_scores,
        include_missingness=include_missingness,
        include_deltas=include_deltas,
    )
    result_df = extractor.transform(df)
    return result_df, extractor.feature_names_
