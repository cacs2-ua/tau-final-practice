from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


DEFAULT_MISSING_TOKENS: Tuple[str, ...] = (
    "", "?", "NA", "N/A", "NULL", "NAN", "UNKNOWN", "UNKNOWN/INVALID"
)

DEFAULT_ID_COLUMNS: Tuple[str, ...] = (
    "id",
    "url",
    "encounter_id",
    "patient_nbr",
    "patient_id",
    "paper_id",
)

DEFAULT_TEXT_COLUMNS: Tuple[str, ...] = (
    "title",
    "abstract",
    "text",
)

DEFAULT_INTERVAL_COLUMNS: Tuple[str, ...] = (
    "year",
)


def _is_missing_value(x: Any, *, treat_empty_as_missing: bool = True) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if treat_empty_as_missing and isinstance(x, str) and x.strip() == "":
        return True
    return False


def _normalize_token(s: str) -> str:
    return s.strip().upper()


def _count_missing_like_tokens(
    series: pd.Series,
    *,
    missing_tokens: Sequence[str] = DEFAULT_MISSING_TOKENS,
) -> Dict[str, int]:
    """
    Count occurrences of missing/unknown encodings in an object/string-like column.
    Counting is case-insensitive and strips whitespace.
    """
    if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
        return {}

    toks = {_normalize_token(t) for t in missing_tokens}
    counts: Dict[str, int] = {t: 0 for t in toks}

    for v in series.astype("object").tolist():
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass

        if isinstance(v, str):
            key = _normalize_token(v)
        else:
            key = _normalize_token(str(v))

        if key in counts:
            counts[key] += 1

    # Keep only tokens that actually appear (>0)
    return {k: v for k, v in counts.items() if v > 0}


def _unique_ratio(series: pd.Series) -> float:
    n = len(series)
    if n == 0:
        return 0.0
    return float(series.nunique(dropna=True)) / float(n)


def _avg_text_length(series: pd.Series) -> float:
    vals = []
    for v in series.tolist():
        if _is_missing_value(v, treat_empty_as_missing=True):
            continue
        vals.append(len(str(v)))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def infer_kind_and_scale(
    series: pd.Series,
    col: str,
    *,
    text_columns: Sequence[str] = DEFAULT_TEXT_COLUMNS,
    interval_columns: Sequence[str] = DEFAULT_INTERVAL_COLUMNS,
    ordinal_columns: Sequence[str] = (),
) -> Tuple[str, str]:
    """
    Returns (kind, scale) where:
      - kind  in {"numeric","categorical","text"}
      - scale in {"nominal","ordinal","interval","ratio","text"}
    """
    col_low = col.strip().lower()

    # Forced text columns by name (common in this project)
    if col_low in {c.lower() for c in text_columns}:
        return "text", "text"

    # Numeric
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
        if col_low in {c.lower() for c in interval_columns}:
            return "numeric", "interval"
        return "numeric", "ratio"

    # Bool
    if pd.api.types.is_bool_dtype(series.dtype):
        return "categorical", "nominal"

    # Otherwise object/string => decide text vs categorical by heuristic
    avg_len = _avg_text_length(series)
    if avg_len >= 30.0:
        return "text", "text"

    # Ordinal hint by name override
    if col_low in {c.lower() for c in ordinal_columns}:
        return "categorical", "ordinal"

    return "categorical", "nominal"


def infer_role(
    df: pd.DataFrame,
    col: str,
    kind: str,
    *,
    id_columns: Sequence[str] = DEFAULT_ID_COLUMNS,
    derived_text_column: str = "text",
    id_like_unique_ratio_threshold: float = 0.98,
) -> str:
    """
    Returns a role label to help the report/checklist:
      - "identifier"
      - "raw_text_feature"
      - "derived_feature"
      - "metadata_feature"
      - "numeric_feature"
      - "categorical_feature"
    """
    col_low = col.strip().lower()
    s = df[col]

    if col_low in {c.lower() for c in id_columns}:
        return "identifier"

    # Name-based derived text (common: text = title + abstract)
    if col_low == derived_text_column.lower():
        return "derived_feature"

    # ID-like heuristic (very high uniqueness ratio)
    if _unique_ratio(s) >= id_like_unique_ratio_threshold and kind != "numeric":
        return "identifier"

    # Heuristic: metadata-like
    if col_low in {"year", "venue"}:
        return "metadata_feature"

    if kind == "text":
        return "raw_text_feature"
    if kind == "numeric":
        return "numeric_feature"
    return "categorical_feature"


@dataclass(frozen=True)
class DatasetUnderstandingReport:
    data_dictionary: pd.DataFrame
    missing_summary: pd.DataFrame
    missing_tokens_found: pd.DataFrame
    duplicates_summary: pd.DataFrame
    leakage_risks: pd.DataFrame
    quality_risks: pd.DataFrame


def build_data_dictionary(
    df: pd.DataFrame,
    *,
    text_columns: Sequence[str] = DEFAULT_TEXT_COLUMNS,
    interval_columns: Sequence[str] = DEFAULT_INTERVAL_COLUMNS,
    ordinal_columns: Sequence[str] = (),
    id_columns: Sequence[str] = DEFAULT_ID_COLUMNS,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        kind, scale = infer_kind_and_scale(
            s,
            col,
            text_columns=text_columns,
            interval_columns=interval_columns,
            ordinal_columns=ordinal_columns,
        )
        role = infer_role(df, col, kind, id_columns=id_columns)

        miss = int(s.isna().sum())
        empty = 0
        if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
            empty = int((s.astype("object").map(lambda x: isinstance(x, str) and x.strip() == "")).sum())

        nun = int(s.nunique(dropna=True))
        ur = float(nun) / float(n) if n else 0.0

        # Example values (up to 5) excluding missing/empty
        examples = []
        for v in s.tolist():
            if _is_missing_value(v, treat_empty_as_missing=True):
                continue
            examples.append(v)
            if len(examples) >= 5:
                break

        rows.append(
            {
                "column": col,
                "pandas_dtype": str(s.dtype),
                "kind": kind,               # numeric / categorical / text
                "scale": scale,             # nominal / ordinal / interval / ratio / text
                "role": role,               # identifier / feature etc
                "n_rows": n,
                "n_unique": nun,
                "unique_ratio": ur,
                "missing_count": miss,
                "missing_rate": (miss / n) if n else 0.0,
                "empty_string_count": empty,
                "examples": examples,
            }
        )

    return pd.DataFrame(rows).sort_values(["role", "kind", "column"]).reset_index(drop=True)


def summarize_missingness(
    df: pd.DataFrame,
    *,
    treat_empty_as_missing: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        na = int(s.isna().sum())
        empty = 0
        if treat_empty_as_missing and (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
            empty = int((s.astype("object").map(lambda x: isinstance(x, str) and x.strip() == "")).sum())

        miss_total = na + empty
        rows.append(
            {
                "column": col,
                "na_count": na,
                "empty_string_count": empty,
                "missing_total": miss_total,
                "missing_rate": (miss_total / n) if n else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False).reset_index(drop=True)


def detect_missing_tokens(
    df: pd.DataFrame,
    *,
    missing_tokens: Sequence[str] = DEFAULT_MISSING_TOKENS,
) -> pd.DataFrame:
    """
    Detects tokens like '?', 'N/A', 'Unknown', etc., per column (string/object columns).
    Returns a long-form DataFrame: column, token, count
    """
    rows: List[Dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        counts = _count_missing_like_tokens(series, missing_tokens=missing_tokens)
        for token, cnt in sorted(counts.items()):
            rows.append({"column": col, "token": token, "count": int(cnt)})
    return pd.DataFrame(rows)


def duplicates_report(
    df: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("url",),
    content_columns: Sequence[str] = ("title", "abstract"),
) -> pd.DataFrame:
    """
    Returns a compact table of duplicate signals:
      - duplicates by key_columns (e.g., URL duplicates)
      - duplicates by content_columns (title+abstract duplicates)
    """
    rows: List[Dict[str, Any]] = []

    def _dup_stats(subset: Sequence[str], name: str) -> None:
        present = [c for c in subset if c in df.columns]
        if not present:
            rows.append(
                {
                    "scope": name,
                    "subset": list(subset),
                    "available_subset": [],
                    "duplicate_rows": 0,
                    "duplicate_groups": 0,
                }
            )
            return

        dup_mask = df.duplicated(subset=present, keep=False)
        duplicate_rows = int(dup_mask.sum())
        # number of duplicated keys/groups (excluding non-duplicated)
        duplicate_groups = int(df.loc[dup_mask, present].drop_duplicates().shape[0])

        rows.append(
            {
                "scope": name,
                "subset": list(subset),
                "available_subset": present,
                "duplicate_rows": duplicate_rows,
                "duplicate_groups": duplicate_groups,
            }
        )

    _dup_stats(key_columns, "key_duplicates")
    _dup_stats(content_columns, "content_duplicates")

    return pd.DataFrame(rows)


def leakage_risk_report(
    df: pd.DataFrame,
    *,
    id_columns: Sequence[str] = DEFAULT_ID_COLUMNS,
    id_like_unique_ratio_threshold: float = 0.98,
) -> pd.DataFrame:
    """
    Flags columns that are likely to cause leakage / trivial memorization in supervised setups
    (or trivial retrieval shortcuts in IR): identifier-like columns, near-unique columns, etc.
    """
    risks: List[Dict[str, Any]] = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        col_low = col.strip().lower()
        ur = _unique_ratio(s)
        nun = int(s.nunique(dropna=True))

        # 1) Name-based ID columns
        if col_low in {c.lower() for c in id_columns}:
            risks.append(
                {
                    "column": col,
                    "risk_type": "identifier_like",
                    "severity": "high",
                    "reason": "Column name matches a known identifier field (e.g., 'url'/'id').",
                    "unique_ratio": ur,
                    "n_unique": nun,
                    "n_rows": n,
                }
            )
            continue

        # 2) ID-like by uniqueness ratio (very close to 1)
        if ur >= id_like_unique_ratio_threshold and not pd.api.types.is_numeric_dtype(s.dtype):
            risks.append(
                {
                    "column": col,
                    "risk_type": "identifier_like",
                    "severity": "medium",
                    "reason": f"Very high uniqueness ratio (>= {id_like_unique_ratio_threshold}).",
                    "unique_ratio": ur,
                    "n_unique": nun,
                    "n_rows": n,
                }
            )

    return pd.DataFrame(risks).sort_values(["severity", "unique_ratio"], ascending=[True, False]).reset_index(drop=True)


def quality_risk_report(
    df: pd.DataFrame,
    *,
    missing_rate_warn: float = 0.20,
    high_cardinality_warn: float = 0.50,
    treat_empty_as_missing: bool = True,
) -> pd.DataFrame:
    """
    Heuristic quality risks: high missingness, high cardinality, constant columns.
    """
    risks: List[Dict[str, Any]] = []
    n = len(df)

    for col in df.columns:
        s = df[col]

        # Missingness risk
        miss_rate = float(
            (_is_missing_value_count(s, treat_empty_as_missing=treat_empty_as_missing) / n) if n else 0.0
        )
        if miss_rate >= missing_rate_warn:
            risks.append(
                {
                    "column": col,
                    "risk_type": "high_missingness",
                    "severity": "medium",
                    "metric": "missing_rate",
                    "value": miss_rate,
                    "threshold": missing_rate_warn,
                    "note": "Consider explicit handling in preprocessing (drop/impute/tokenize missing).",
                }
            )

        # Cardinality risk (mostly for categorical/text columns)
        ur = float(s.nunique(dropna=True) / n) if n else 0.0
        if ur >= high_cardinality_warn and not pd.api.types.is_numeric_dtype(s.dtype):
            risks.append(
                {
                    "column": col,
                    "risk_type": "high_cardinality",
                    "severity": "low",
                    "metric": "unique_ratio",
                    "value": ur,
                    "threshold": high_cardinality_warn,
                    "note": "May lead to sparse features / overfitting if one-hot encoded.",
                }
            )

        # Constant column risk
        nun = int(s.nunique(dropna=True))
        if nun <= 1:
            risks.append(
                {
                    "column": col,
                    "risk_type": "constant_or_almost_constant",
                    "severity": "low",
                    "metric": "n_unique",
                    "value": nun,
                    "threshold": 1,
                    "note": "Usually safe to drop (no predictive signal).",
                }
            )

    return pd.DataFrame(risks).sort_values(["severity", "risk_type", "column"]).reset_index(drop=True)


def _is_missing_value_count(series: pd.Series, *, treat_empty_as_missing: bool) -> int:
    na = int(series.isna().sum())
    if not treat_empty_as_missing:
        return na
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        empty = int((series.astype("object").map(lambda x: isinstance(x, str) and x.strip() == "")).sum())
        return na + empty
    return na


def profile_dataset(
    df_or_records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    *,
    expected_keys: Sequence[str] = ("title", "abstract", "url", "venue", "year"),
    text_columns: Sequence[str] = DEFAULT_TEXT_COLUMNS,
    interval_columns: Sequence[str] = DEFAULT_INTERVAL_COLUMNS,
    ordinal_columns: Sequence[str] = (),
    id_columns: Sequence[str] = DEFAULT_ID_COLUMNS,
    missing_tokens: Sequence[str] = DEFAULT_MISSING_TOKENS,
) -> DatasetUnderstandingReport:
    """
    End-to-end helper for Section 3:
      - Builds data dictionary (types/scales/roles)
      - Summarizes missingness + detects missing tokens
      - Detects duplicates
      - Flags leakage risks
      - Flags quality risks

    Accepts either a DataFrame or the raw list[dict] records.
    """
    if isinstance(df_or_records, pd.DataFrame):
        df = df_or_records.copy()
    else:
        # records -> dataframe
        if not isinstance(df_or_records, (list, tuple)):
            raise TypeError(f"df_or_records must be a DataFrame or list/tuple of dicts, got {type(df_or_records)}")
        rows = []
        for i, rec in enumerate(df_or_records):
            if not isinstance(rec, dict):
                raise TypeError(f"Each record must be a dict. Found {type(rec)} at index {i}.")
            rows.append({k: rec.get(k, None) for k in expected_keys})
        df = pd.DataFrame(rows)

    data_dict = build_data_dictionary(
        df,
        text_columns=text_columns,
        interval_columns=interval_columns,
        ordinal_columns=ordinal_columns,
        id_columns=id_columns,
    )
    missing_summary = summarize_missingness(df, treat_empty_as_missing=True)
    missing_tokens_found = detect_missing_tokens(df, missing_tokens=missing_tokens)
    duplicates_summary = duplicates_report(df)
    leakage_risks = leakage_risk_report(df, id_columns=id_columns)
    quality_risks = quality_risk_report(df)

    return DatasetUnderstandingReport(
        data_dictionary=data_dict,
        missing_summary=missing_summary,
        missing_tokens_found=missing_tokens_found,
        duplicates_summary=duplicates_summary,
        leakage_risks=leakage_risks,
        quality_risks=quality_risks,
    )
