from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Missing token handling
# -------------------------
DEFAULT_MISSING_TOKENS: Tuple[str, ...] = (
    "", "?", "NA", "N/A", "NULL", "NAN", "UNKNOWN", "UNKNOWN/INVALID"
)

def _norm_token(x: object) -> str:
    return str(x).strip().upper()

_DEFAULT_MISSING_TOKENS_UP = frozenset(_norm_token(t) for t in DEFAULT_MISSING_TOKENS)

@lru_cache(maxsize=64)
def _missing_token_set(tokens: Tuple[str, ...]) -> frozenset[str]:
    return frozenset(_norm_token(t) for t in tokens)

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

    # Fast path for default tokens
    up = _norm_token(s)
    if missing_tokens is DEFAULT_MISSING_TOKENS:
        return up in _DEFAULT_MISSING_TOKENS_UP

    toks = _missing_token_set(tuple(missing_tokens))
    return up in toks


# Medication columns in the UCI diabetes dataset (intersection with df.columns for robustness)
DEFAULT_MED_COLS: Tuple[str, ...] = (
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide",
    "glipizide","glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose","miglitol",
    "troglitazone","tolazamide","examide","citoglipton","insulin",
    "glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone",
)

DIAG_COLS_DEFAULT: Tuple[str, ...] = ("diag_1", "diag_2", "diag_3")


# -------------------------
# Helpers
# -------------------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _df_elementwise_map(df: pd.DataFrame, func):
    # pandas>=2.1: DataFrame.map exists; older: use applymap
    return df.map(func) if hasattr(df, "map") else df.applymap(func)


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
    if not s:
        return np.nan

    # Expect formats like "[a-b)" or "(a-b]" etc.
    if "-" not in s:
        return np.nan

    try:
        inside = s.strip("[]()")
        a, b = inside.split("-", 1)
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
    return pd.cut(
        x_num,
        bins=bins,
        labels=labels,
        include_lowest=include_lowest,
        right=right,
        ordered=True,
    )


# -------------------------
# ICD-9 grouping (high-level)
# -------------------------
def _diag_token(v: object) -> str | None:
    """Normalize and validate a diagnosis token."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    s = str(v).strip()
    if s == "":
        return None

    up = s.upper()
    if up in _DEFAULT_MISSING_TOKENS_UP:
        return None

    return up

def _extract_leading_numeric(tok: str) -> float | None:
    """
    Extract a leading numeric value from an ICD-9 token (handles '250.13', '401', '530.81').
    Returns None if it can't parse a leading numeric prefix.
    """
    # Keep digits and first dot only until a non-digit/non-dot appears.
    buf = []
    dot_used = False
    for ch in tok:
        if ch.isdigit():
            buf.append(ch)
        elif ch == "." and not dot_used:
            buf.append(ch)
            dot_used = True
        else:
            break
    if not buf:
        return None
    try:
        return float("".join(buf))
    except Exception:
        return None

def icd9_group(code: object) -> str:
    """
    Robust ICD-9 grouping:
      - V* -> supplementary_v
      - E* -> external_e
      - numeric ranges -> major ICD-9 chapter
      - 250.* -> diabetes (special case)
    """
    tok = _diag_token(code)
    if tok is None:
        return "__MISSING__"

    # Non-numeric ICD-9 families
    if tok.startswith("V"):
        return "supplementary_v"
    if tok.startswith("E"):
        return "external_e"

    num = _extract_leading_numeric(tok)
    if num is None:
        return "other"

    # Special case diabetes: 250.xx
    if 250.0 <= num < 251.0:
        return "diabetes"

    # Major ICD-9 chapters (coarse)
    if 1.0 <= num <= 139.0:
        return "infectious"
    if 140.0 <= num <= 239.0:
        return "neoplasms"
    if 240.0 <= num <= 279.0:
        return "endocrine_metabolic"
    if 280.0 <= num <= 289.0:
        return "blood"
    if 290.0 <= num <= 319.0:
        return "mental"
    if 320.0 <= num <= 389.0:
        return "nervous"
    if 390.0 <= num <= 459.0:
        return "circulatory"
    if 460.0 <= num <= 519.0:
        return "respiratory"
    if 520.0 <= num <= 579.0:
        return "digestive"
    if 580.0 <= num <= 629.0:
        return "genitourinary"
    if 630.0 <= num <= 679.0:
        return "pregnancy"
    if 680.0 <= num <= 709.0:
        return "skin"
    if 710.0 <= num <= 739.0:
        return "musculoskeletal"
    if 740.0 <= num <= 759.0:
        return "congenital"
    if 760.0 <= num <= 779.0:
        return "perinatal"
    if 780.0 <= num <= 799.0:
        return "symptoms"
    if 800.0 <= num <= 999.0:
        return "injury"

    return "other"

def add_diag_group_features(df: pd.DataFrame, diag_cols: Sequence[str] = DIAG_COLS_DEFAULT) -> pd.DataFrame:
    out = df.copy()
    present = [c for c in diag_cols if c in out.columns]
    for c in present:
        out[f"{c}_group"] = out[c].map(icd9_group)

    group_cols = [f"{c}_group" for c in present]
    if group_cols:
        # Counts / flags
        out["n_diabetes_diags"] = (out[group_cols] == "diabetes").sum(axis=1).astype(int)

        # Robust primary flag (keeps index)
        if "diag_1_group" in out.columns:
            out["primary_diag_is_diabetes"] = (out["diag_1_group"] == "diabetes").astype(int)
        else:
            out["primary_diag_is_diabetes"] = pd.Series(0, index=out.index, dtype=int)

        # Diversity of diagnosis groups (excluding missing)
        tmp = out[group_cols].replace("__MISSING__", np.nan)
        out["n_unique_diag_groups"] = tmp.nunique(axis=1, dropna=True).fillna(0).astype(int)

        # Whether any diag group equals primary (useful redundancy indicator)
        if "diag_1_group" in out.columns:
            out["n_diags_same_as_primary_group"] = (
                out[group_cols].eq(out["diag_1_group"], axis=0).sum(axis=1).astype(int)
            )

    return out


# -------------------------
# Labs & tests (A1C / glucose)
# -------------------------
_A1C_MAP = {"NONE": 0, "NORM": 1, ">7": 2, ">8": 3}
_GLU_MAP = {"NONE": 0, "NORM": 1, ">200": 2, ">300": 3}

def add_lab_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "A1Cresult" in out.columns:
        a1c_raw = out["A1Cresult"].astype("object")
        a1c_norm = a1c_raw.map(lambda v: "MISSING" if is_missing_token(v) else str(v).strip().upper())
        out["a1c_measured"] = (a1c_norm.notna() & (a1c_norm != "MISSING") & (a1c_norm != "NONE")).astype(int)
        out["a1c_high"] = a1c_norm.isin({">7", ">8"}).astype(int)
        out["a1c_level"] = a1c_norm.map(_A1C_MAP).fillna(0).astype(int)

    if "max_glu_serum" in out.columns:
        glu_raw = out["max_glu_serum"].astype("object")
        glu_norm = glu_raw.map(lambda v: "MISSING" if is_missing_token(v) else str(v).strip().upper())
        out["glu_measured"] = (glu_norm.notna() & (glu_norm != "MISSING") & (glu_norm != "NONE")).astype(int)
        out["glu_high"] = glu_norm.isin({">200", ">300"}).astype(int)
        out["glu_level"] = glu_norm.map(_GLU_MAP).fillna(0).astype(int)

    # Optional bins for lab procedure count (fixed thresholds)
    if "num_lab_procedures" in out.columns:
        out["num_lab_procedures"] = _to_num(out["num_lab_procedures"])
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

    num_cols = [
        "number_outpatient", "number_emergency", "number_inpatient",
        "time_in_hospital", "num_medications", "number_diagnoses"
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = _to_num(out[c])

    if all(c in out.columns for c in ["number_outpatient", "number_emergency", "number_inpatient"]):
        out["total_visits"] = (
            out["number_outpatient"].fillna(0)
            + out["number_emergency"].fillna(0)
            + out["number_inpatient"].fillna(0)
        ).astype(float)

        out["any_emergency"] = (out["number_emergency"].fillna(0) > 0).astype(int)
        out["any_inpatient"] = (out["number_inpatient"].fillna(0) > 0).astype(int)
        out["any_outpatient"] = (out["number_outpatient"].fillna(0) > 0).astype(int)
        out["any_prior_visit"] = (out["total_visits"].fillna(0) > 0).astype(int)

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
    meds_norm = _df_elementwise_map(
        meds, lambda v: "MISSING" if is_missing_token(v) else str(v).strip().upper()
    )

    active_mask = meds_norm.isin({"STEADY", "UP", "DOWN"})
    changed_mask = meds_norm.isin({"UP", "DOWN"})

    out["n_active_diabetes_meds"] = active_mask.sum(axis=1).astype(int)
    out["n_changed_diabetes_meds"] = changed_mask.sum(axis=1).astype(int)
    out["any_active_diabetes_med"] = (out["n_active_diabetes_meds"] > 0).astype(int)
    out["any_changed_diabetes_med"] = (out["n_changed_diabetes_meds"] > 0).astype(int)

    # insulin-specific flags
    if "insulin" in out.columns:
        insulin = out["insulin"].astype("object").map(
            lambda v: "MISSING" if is_missing_token(v) else str(v).strip().upper()
        )
        out["insulin_active"] = insulin.isin({"STEADY", "UP", "DOWN"}).astype(int)
        out["insulin_changed"] = insulin.isin({"UP", "DOWN"}).astype(int)

    return out


# -------------------------
# Simple interactions (numeric only; low risk)
# -------------------------
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # age midpoint
    if "age" in out.columns:
        out["age_mid"] = out["age"].map(age_midpoint)
        out["elderly"] = (out["age_mid"].fillna(-1) >= 65).astype(int)

    # ensure numeric coercion for interactions
    for c in ["time_in_hospital", "num_medications", "number_diagnoses", "total_visits"]:
        if c in out.columns:
            out[c] = _to_num(out[c])

    if all(c in out.columns for c in ["time_in_hospital", "num_medications"]):
        out["stay_x_meds"] = (out["time_in_hospital"].fillna(0) * out["num_medications"].fillna(0)).astype(float)

    if all(c in out.columns for c in ["time_in_hospital", "number_diagnoses"]):
        out["stay_x_diagnoses"] = (out["time_in_hospital"].fillna(0) * out["number_diagnoses"].fillna(0)).astype(float)

    if all(c in out.columns for c in ["total_visits", "time_in_hospital"]):
        out["visits_x_stay"] = (out["total_visits"].fillna(0) * out["time_in_hospital"].fillna(0)).astype(float)

    if all(c in out.columns for c in ["age_mid", "num_medications"]):
        out["age_x_meds"] = (out["age_mid"].fillna(0) * out["num_medications"].fillna(0)).astype(float)

    if all(c in out.columns for c in ["age_mid", "total_visits"]):
        out["age_x_visits"] = (out["age_mid"].fillna(0) * out["total_visits"].fillna(0)).astype(float)

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
    if not isinstance(df, pd.DataFrame):
        raise TypeError("apply_feature_engineering expects a pandas DataFrame.")

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
