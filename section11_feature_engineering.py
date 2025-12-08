from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Keep missing-token logic consistent with Sections 4/5
DEFAULT_MISSING_TOKENS: Tuple[str, ...] = (
    "", "?", "NA", "N/A", "NULL", "NAN", "UNKNOWN", "UNKNOWN/INVALID"
)

# Medication columns in the UCI diabetes dataset (we'll use intersection with df.columns for robustness)
DEFAULT_MED_COLS: Tuple[str, ...] = (
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide",
    "glipizide","glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose","miglitol",
    "troglitazone","tolazamide","examide","citoglipton","insulin",
    "glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone",
)

DIAG_COLS_DEFAULT: Tuple[str, ...] = ("diag_1", "diag_2", "diag_3")


def _norm_token(x: object) -> str:
    return str(x).strip().upper()


def is_missing_token(x: object, *, missing_tokens: Sequence[str] = DEFAULT_MISSING_TOKENS) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x)
    if s.strip() == "":
        return True
    toks = {_norm_token(t) for t in missing_tokens}
    return _norm_token(s) in toks


# -------------------------
# Age: interval -> midpoint
# -------------------------
def age_midpoint(age_str: object) -> float:
    """
    Convert age bucket like "[50-60)" -> 55.0.
    Missing/unknown -> np.nan.
    """
    if is_missing_token(age_str):
        return np.nan
    s = str(age_str).strip()
    # Expect format "[a-b)"
    if not (s.startswith("[") and "-" in s):
        return np.nan
    try:
        inside = s.strip("[]()")
        a, b = inside.split("-")
        a = float(a)
        b = float(b)
        return (a + b) / 2.0
    except Exception:
        return np.nan


# -------------------------
# Fixed binning (no leakage)
# -------------------------
def bin_numeric_fixed(
    x: pd.Series,
    bins: Sequence[float],
    labels: Sequence[str],
    *,
    include_lowest: bool = True,
    right: bool = True,
) -> pd.Series:
    """
    Deterministic, fixed-threshold binning via pd.cut.
    Keeps reproducibility and avoids data-driven thresholds (leakage).
    """
    x_num = pd.to_numeric(x, errors="coerce")
    return pd.cut(x_num, bins=bins, labels=labels, include_lowest=include_lowest, right=right)


# -------------------------
# ICD-9 grouping (high-level)
# -------------------------
def _icd9_numeric_prefix(code: object) -> Optional[float]:
    """
    Parse ICD-9 code:
      - '250.83' -> 250.83
      - '276' -> 276.0
      - 'V27'/'E849' -> None (handled as special)
      - missing -> None
    """
    if is_missing_token(code):
        return None
    s = str(code).strip().upper()
    if s.startswith(("V", "E")):
        return None
    # numeric part possibly with decimal
    try:
        return float(s)
    except Exception:
        # sometimes codes have trailing stuff; try leading numeric
        num = ""
        for ch in s:
            if ch.isdigit() or ch == ".":
                num += ch
            else:
                break
        try:
            return float(num) if num else None
        except Exception:
            return None


def icd9_group(code: object) -> str:
    """
    High-level diagnosis grouping (compact categorical feature).
    This is a common, clinically motivated way to reduce sparsity in diag_1/2/3.
    """
    if is_missing_token(code):
        return "unknown"

    s = str(code).strip().upper()
    if s.startswith("V"):
        return "supplementary"
    if s.startswith("E"):
        return "external_causes"

    num = _icd9_numeric_prefix(s)
    if num is None:
        return "other"

    # Diabetes specifically
    if 250.0 <= num < 251.0:
        return "diabetes"

    # Broad ICD-9 chapters (approximate but standard in practice)
    if 1.0 <= num < 140.0:
        return "infectious"
    if 140.0 <= num < 240.0:
        return "neoplasms"
    if 240.0 <= num < 280.0:
        return "endocrine_metabolic"  # includes many metabolic disorders (excluding diabetes handled above)
    if 280.0 <= num < 290.0:
        return "blood"
    if 290.0 <= num < 320.0:
        return "mental"
    if 320.0 <= num < 390.0:
        return "nervous"
    if 390.0 <= num < 460.0 or int(num) == 785:
        return "circulatory"
    if 460.0 <= num < 520.0 or int(num) == 786:
        return "respiratory"
    if 520.0 <= num < 580.0 or int(num) == 787:
        return "digestive"
    if 580.0 <= num < 630.0 or int(num) == 788:
        return "genitourinary"
    if 630.0 <= num < 680.0:
        return "pregnancy"
    if 680.0 <= num < 710.0:
        return "skin"
    if 710.0 <= num < 740.0:
        return "musculoskeletal"
    if 740.0 <= num < 800.0:
        return "congenital_perinatal_other"
    if 800.0 <= num < 1000.0:
        return "injury_poisoning"

    return "other"


def add_diag_group_features(df: pd.DataFrame, diag_cols: Sequence[str] = DIAG_COLS_DEFAULT) -> pd.DataFrame:
    out = df.copy()
    present = [c for c in diag_cols if c in out.columns]
    for c in present:
        out[f"{c}_group"] = out[c].map(icd9_group)

    # small, useful aggregates
    group_cols = [f"{c}_group" for c in present]
    if group_cols:
        out["n_diabetes_diags"] = (out[group_cols] == "diabetes").sum(axis=1).astype(int)
        out["primary_diag_is_diabetes"] = (out.get("diag_1_group", pd.Series(["unknown"] * len(out))) == "diabetes").astype(int)

    return out


# -------------------------
# Labs & tests (A1C / glucose)
# -------------------------
def add_lab_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "A1Cresult" in out.columns:
        a1c = out["A1Cresult"].astype("object")
        out["a1c_measured"] = (~a1c.map(is_missing_token) & (a1c.astype(str).str.strip().str.upper() != "NONE")).astype(int)
        out["a1c_high"] = a1c.astype(str).str.strip().str.upper().isin({">7", ">8"}).astype(int)

    if "max_glu_serum" in out.columns:
        glu = out["max_glu_serum"].astype("object")
        out["glu_measured"] = (~glu.map(is_missing_token) & (glu.astype(str).str.strip().str.upper() != "NONE")).astype(int)
        out["glu_high"] = glu.astype(str).str.strip().str.upper().isin({">200", ">300"}).astype(int)

    # Optional bins for lab procedure count (fixed thresholds)
    if "num_lab_procedures" in out.columns:
        out["num_lab_procedures_bin"] = bin_numeric_fixed(
            out["num_lab_procedures"],
            bins=[-np.inf, 20, 40, 60, np.inf],
            labels=["low", "medium", "high", "very_high"],
        )

    return out


# -------------------------
# Utilization / burden features
# -------------------------
def add_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ensure numeric coercion for arithmetic
    for c in ["number_outpatient", "number_emergency", "number_inpatient", "time_in_hospital", "num_medications", "number_diagnoses"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if all(c in out.columns for c in ["number_outpatient", "number_emergency", "number_inpatient"]):
        out["total_visits"] = (out["number_outpatient"].fillna(0) +
                              out["number_emergency"].fillna(0) +
                              out["number_inpatient"].fillna(0)).astype(float)

        out["any_emergency"] = (out["number_emergency"].fillna(0) > 0).astype(int)
        out["any_inpatient"] = (out["number_inpatient"].fillna(0) > 0).astype(int)
        out["any_outpatient"] = (out["number_outpatient"].fillna(0) > 0).astype(int)

        out["total_visits_bin"] = bin_numeric_fixed(
            out["total_visits"],
            bins=[-np.inf, 0, 2, 5, np.inf],
            labels=["none", "low", "medium", "high"],
        )

    if "time_in_hospital" in out.columns:
        out["stay_bin"] = bin_numeric_fixed(
            out["time_in_hospital"],
            bins=[-np.inf, 3, 7, 14, np.inf],
            labels=["short", "medium", "long", "very_long"],
        )

    if "num_medications" in out.columns:
        out["num_medications_bin"] = bin_numeric_fixed(
            out["num_medications"],
            bins=[-np.inf, 5, 10, 20, np.inf],
            labels=["low", "medium", "high", "very_high"],
        )
        out["polypharmacy"] = (out["num_medications"].fillna(0) >= 10).astype(int)

    if "number_diagnoses" in out.columns:
        out["number_diagnoses_bin"] = bin_numeric_fixed(
            out["number_diagnoses"],
            bins=[-np.inf, 3, 6, np.inf],
            labels=["low", "medium", "high"],
        )

    return out


# -------------------------
# Medication burden from per-drug columns
# -------------------------
def add_medication_burden_features(
    df: pd.DataFrame,
    *,
    med_cols: Sequence[str] = DEFAULT_MED_COLS,
) -> pd.DataFrame:
    out = df.copy()
    present = [c for c in med_cols if c in out.columns]
    if not present:
        return out

    meds = out[present].astype("object")

    # Normalize
    meds_norm = meds.applymap(lambda v: "MISSING" if is_missing_token(v) else str(v).strip().upper())

    active_mask = meds_norm.isin({"STEADY", "UP", "DOWN"})
    changed_mask = meds_norm.isin({"UP", "DOWN"})

    out["n_active_diabetes_meds"] = active_mask.sum(axis=1).astype(int)
    out["n_changed_diabetes_meds"] = changed_mask.sum(axis=1).astype(int)

    # insulin-specific flags (often predictive / clinically meaningful)
    if "insulin" in out.columns:
        insulin = out["insulin"].astype("object").map(lambda v: "MISSING" if is_missing_token(v) else str(v).strip().upper())
        out["insulin_active"] = insulin.isin({"STEADY", "UP", "DOWN"}).astype(int)
        out["insulin_changed"] = insulin.isin({"UP", "DOWN"}).astype(int)

    return out


# -------------------------
# Simple interactions (numeric only; low risk)
# -------------------------
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # age midpoint as numeric (enables non-linearities + interactions without exploding one-hot)
    if "age" in out.columns:
        out["age_mid"] = out["age"].map(age_midpoint)
        out["elderly"] = (out["age_mid"].fillna(-1) >= 65).astype(int)

    # interactions that often capture “burden × duration”
    for c in ["time_in_hospital", "num_medications", "number_diagnoses"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if all(c in out.columns for c in ["time_in_hospital", "num_medications"]):
        out["stay_x_meds"] = (out["time_in_hospital"].fillna(0) * out["num_medications"].fillna(0)).astype(float)

    if all(c in out.columns for c in ["time_in_hospital", "number_diagnoses"]):
        out["stay_x_diagnoses"] = (out["time_in_hospital"].fillna(0) * out["number_diagnoses"].fillna(0)).astype(float)

    if all(c in out.columns for c in ["total_visits", "time_in_hospital"]):
        out["visits_x_stay"] = (pd.to_numeric(out["total_visits"], errors="coerce").fillna(0) *
                                out["time_in_hospital"].fillna(0)).astype(float)

    return out


# -------------------------
# Main entry point
# -------------------------
@dataclass(frozen=True)
class FeatureEngineeringOutput:
    df: pd.DataFrame
    added_columns: List[str]


def apply_feature_engineering(
    df: pd.DataFrame,
    *,
    diag_cols: Sequence[str] = DIAG_COLS_DEFAULT,
    med_cols: Sequence[str] = DEFAULT_MED_COLS,
) -> FeatureEngineeringOutput:
    """
    Apply Section 11 transformations and return:
      - df with engineered features added
      - list of added column names (useful for report + debugging)

    IMPORTANT: run this BEFORE building the Section 5 preprocessing pipeline,
    so the new columns are included in modeling.
    """
    before = set(df.columns)

    out = df.copy()
    out = add_diag_group_features(out, diag_cols=diag_cols)
    out = add_lab_features(out)
    out = add_utilization_features(out)
    out = add_medication_burden_features(out, med_cols=med_cols)
    out = add_interaction_features(out)

    after = set(out.columns)
    added = sorted(list(after - before))
    return FeatureEngineeringOutput(df=out, added_columns=added)
