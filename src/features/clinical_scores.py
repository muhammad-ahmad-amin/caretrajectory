"""
src/features/clinical_scores.py
================================================================================
CareTrajectory — Clinical Risk Indices & Physiological Derivations

Computes validated bedside clinical indicators and composite physiological indices
from continuous vital signs and laboratory measurements. All operations are
fully vectorized, strictly non-lookahead, and numerically stabilized.
================================================================================
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_shock_index(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.Series:
    """
    Computes Shock Index (SI) = Heart Rate / Systolic Blood Pressure.
    Normal physiological range: 0.5 - 0.7. SI > 0.9 indicates acute occult shock.
    """
    if "HR" not in df.columns or "SBP" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    
    sbp_safe = df["SBP"].replace(0, np.nan)
    return (df["HR"] / (sbp_safe + epsilon)).astype(np.float32)


def compute_modified_shock_index(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.Series:
    """
    Computes Modified Shock Index (MSI) = Heart Rate / Mean Arterial Pressure.
    Normal range: 0.7 - 1.3. MSI > 1.3 is a strong predictor of ICU mortality.
    """
    if "HR" not in df.columns or "MAP" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    
    map_safe = df["MAP"].replace(0, np.nan)
    return (df["HR"] / (map_safe + epsilon)).astype(np.float32)


def compute_age_shock_index(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.Series:
    """
    Computes Age Shock Index (Age-SI) = Age * (HR / SBP).
    Threshold > 50.0 identifies geriatric and high-acuity decompensation.
    """
    if "Age" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    
    si = compute_shock_index(df, epsilon=epsilon)
    return (df["Age"] * si).astype(np.float32)


def compute_pulse_pressure(df: pd.DataFrame) -> pd.Series:
    """
    Computes Pulse Pressure (PP) = SBP - DBP.
    PP < 30 mmHg signifies narrow pulse pressure (hypovolemic or cardiogenic shock).
    """
    if "SBP" not in df.columns or "DBP" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    
    return (df["SBP"] - df["DBP"]).astype(np.float32)


def compute_imputed_map(df: pd.DataFrame) -> pd.Series:
    """
    Reconstructs Mean Arterial Pressure (MAP) if missing:
    MAP = DBP + 1/3 * (SBP - DBP).
    Preserves original MAP where already recorded.
    """
    if "MAP" not in df.columns:
        if "SBP" in df.columns and "DBP" in df.columns:
            return (df["DBP"] + (df["SBP"] - df["DBP"]) / 3.0).astype(np.float32)
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    
    map_col = df["MAP"].copy()
    if "SBP" in df.columns and "DBP" in df.columns:
        calculated_map = df["DBP"] + (df["SBP"] - df["DBP"]) / 3.0
        map_col = map_col.combine_first(calculated_map)
    return map_col.astype(np.float32)


def compute_bun_creatinine_ratio(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.Series:
    """
    Computes Blood Urea Nitrogen to Creatinine Ratio (BUN/Cr).
    Ratio > 20:1 points toward pre-renal azotemia, dehydration, or septic renal hypoperfusion.
    """
    if "BUN" not in df.columns or "Creatinine" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    
    cr_safe = df["Creatinine"].replace(0, np.nan)
    return (df["BUN"] / (cr_safe + epsilon)).astype(np.float32)


def compute_pao2_fio2_ratio(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.Series:
    """
    Computes PaO2 / FiO2 (Horowitz Index / P/F ratio).
    FiO2 is converted from percentage (21-100) to fraction (0.21-1.0) if needed.
    P/F <= 300 mmHg indicates Berlin criteria ARDS / acute lung injury.
    """
    if "FiO2" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    
    # Try PaO2 if present, or infer from SaO2/O2Sat via Severinghaus inversion approximation if absent
    if "PaO2" in df.columns:
        pao2 = df["PaO2"]
    elif "SaO2" in df.columns:
        pao2 = df["SaO2"]
    elif "O2Sat" in df.columns:
        pao2 = df["O2Sat"]
    else:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    
    fio2 = df["FiO2"].copy()
    # Normalize FiO2 percentage if scaled > 1.0 (e.g., 21.0 -> 0.21)
    fio2_normalized = np.where(fio2 > 1.0, fio2 / 100.0, fio2)
    fio2_normalized = np.where(fio2_normalized <= 0.0, np.nan, fio2_normalized)
    
    ratio = pao2 / (fio2_normalized + epsilon)
    return pd.Series(ratio, index=df.index, dtype=np.float32)


def compute_sirs_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Systemic Inflammatory Response Syndrome (SIRS) criteria:
      1. Temp > 38.3 C or Temp < 36.0 C
      2. HR > 90 bpm
      3. Resp > 20 breaths/min or PaCO2 < 32 mmHg
      4. WBC > 12.0 or WBC < 4.0 (*10^9 / L)
      
    Returns DataFrame with individual binary flags and aggregate sirs_count (0-4).
    """
    sirs_df = pd.DataFrame(index=df.index)
    
    # Criterion 1: Temperature
    if "Temp" in df.columns:
        sirs_df["sirs_temp"] = ((df["Temp"] > 38.3) | (df["Temp"] < 36.0)).astype(np.float32)
    else:
        sirs_df["sirs_temp"] = 0.0
        
    # Criterion 2: Heart Rate
    if "HR" in df.columns:
        sirs_df["sirs_hr"] = (df["HR"] > 90.0).astype(np.float32)
    else:
        sirs_df["sirs_hr"] = 0.0
        
    # Criterion 3: Respiratory Rate / PaCO2
    crit_resp = pd.Series(False, index=df.index)
    if "Resp" in df.columns:
        crit_resp = crit_resp | (df["Resp"] > 20.0)
    if "PaCO2" in df.columns:
        crit_resp = crit_resp | (df["PaCO2"] < 32.0)
    sirs_df["sirs_resp"] = crit_resp.astype(np.float32)
    
    # Criterion 4: White Blood Cell count
    if "WBC" in df.columns:
        sirs_df["sirs_wbc"] = ((df["WBC"] > 12.0) | (df["WBC"] < 4.0)).astype(np.float32)
    else:
        sirs_df["sirs_wbc"] = 0.0
        
    sirs_df["sirs_score"] = (
        sirs_df["sirs_temp"] + sirs_df["sirs_hr"] + sirs_df["sirs_resp"] + sirs_df["sirs_wbc"]
    ).astype(np.float32)
    
    sirs_df["sirs_positive"] = (sirs_df["sirs_score"] >= 2.0).astype(np.float32)
    return sirs_df


def compute_qsofa_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates quick Sequential Organ Failure Assessment (qSOFA) criteria:
      1. Respiratory Rate >= 22
      2. SBP <= 100 mmHg
      
    Returns DataFrame with flags and aggregate qsofa_score (0-2).
    """
    qsofa_df = pd.DataFrame(index=df.index)
    
    if "Resp" in df.columns:
        qsofa_df["qsofa_resp"] = (df["Resp"] >= 22.0).astype(np.float32)
    else:
        qsofa_df["qsofa_resp"] = 0.0
        
    if "SBP" in df.columns:
        qsofa_df["qsofa_sbp"] = (df["SBP"] <= 100.0).astype(np.float32)
    else:
        qsofa_df["qsofa_sbp"] = 0.0
        
    qsofa_df["qsofa_score"] = (qsofa_df["qsofa_resp"] + qsofa_df["qsofa_sbp"]).astype(np.float32)
    return qsofa_df


def compute_all_clinical_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline: computes all bedside physiological and clinical scores,
    returning a clean DataFrame of derived clinical indicators.
    """
    scores = pd.DataFrame(index=df.index)
    
    scores["score_shock_index"] = compute_shock_index(df)
    scores["score_modified_shock_index"] = compute_modified_shock_index(df)
    scores["score_age_shock_index"] = compute_age_shock_index(df)
    scores["score_pulse_pressure"] = compute_pulse_pressure(df)
    scores["score_imputed_map"] = compute_imputed_map(df)
    scores["score_bun_cr_ratio"] = compute_bun_creatinine_ratio(df)
    scores["score_pf_ratio"] = compute_pao2_fio2_ratio(df)
    
    # Merge SIRS criteria
    sirs_df = compute_sirs_score(df)
    for col in sirs_df.columns:
        scores[col] = sirs_df[col]
        
    # Merge qSOFA criteria
    qsofa_df = compute_qsofa_score(df)
    for col in qsofa_df.columns:
        scores[col] = qsofa_df[col]
        
    return scores.astype(np.float32)
