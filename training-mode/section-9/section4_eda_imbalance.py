from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# 4.1  Class imbalance after binarization
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ClassImbalanceSummary:
    """Compact summary of class imbalance for the binary target.

    The positive class is the early readmission (<30 days), i.e. y=1.
    This supports the discussion of why accuracy can be misleading and
    why precision/recall/F1 (for the minority class) are preferred
    under imbalance (see PDF: confusion matrix + metrics). 
    """
    N: int
    n_pos: int
    n_neg: int
    pos_rate: float
    neg_rate: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "N": self.N,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "pos_rate": self.pos_rate,
            "neg_rate": self.neg_rate,
        }


def compute_class_imbalance(
    y: Iterable[int],
    *,
    positive_label: int = 1,
    negative_label: int = 0,
) -> ClassImbalanceSummary:
    """Quantify class imbalance for a binary target y={0,1}.

    Parameters
    ----------
    y:
        Iterable with the binary labels (after binarization).
    positive_label:
        Label considered "positive" (early readmission).
    negative_label:
        Label considered "negative".

    Returns
    -------
    ClassImbalanceSummary
        N, n_pos, n_neg and prevalence rates.
    """
    y_series = pd.Series(list(y))
    N = int(len(y_series))
    counts = y_series.value_counts(dropna=False).to_dict()
    n_pos = int(counts.get(positive_label, 0))
    n_neg = int(counts.get(negative_label, 0))
    pos_rate = float(n_pos / N) if N else 0.0
    neg_rate = float(n_neg / N) if N else 0.0
    return ClassImbalanceSummary(N=N, n_pos=n_pos, n_neg=n_neg,
                                 pos_rate=pos_rate, neg_rate=neg_rate)


# ---------------------------------------------------------------------
# 4.2  Missingness per feature and per record
# ---------------------------------------------------------------------

MISSING_TOKENS: Tuple[str, ...] = (
    "", "?", "NA", "N/A", "NONE", "NULL", "NAN", "UNKNOWN", "UNKNOWN/INVALID"
)


def is_missing(x: Any, missing_tokens: Sequence[str] = MISSING_TOKENS) -> bool:
    """Unified predicate for 'missing' used in Section 4.

    Treats as missing:
    - None / NaN (according to pandas.isna)
    - empty strings (after strip)
    - tokens like '?', 'NA', 'UNKNOWN', 'UNKNOWN/INVALID', etc.
    """
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        # If pandas cannot decide, ignore and continue
        pass

    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return True
        s_up = s.upper()
        tokens_up = {t.upper() for t in missing_tokens}
        return s_up in tokens_up

    return False


def _series_missing_mask(series: pd.Series,
                         missing_tokens: Sequence[str]) -> pd.Series:
    """Return a boolean mask indicating missing values in a Series."""
    return series.map(lambda v: is_missing(v, missing_tokens=missing_tokens))


def missingness_by_feature(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    missing_tokens: Sequence[str] = MISSING_TOKENS,
) -> pd.DataFrame:
    """Column-level missingness summary.

    Parameters
    ----------
    df:
        Full dataframe including target and features.
    feature_cols:
        Columns to treat as features. If None, all columns except common
        target columns are used.
    missing_tokens:
        Encodings to be treated as missing (in addition to NaN / empty).

    Returns
    -------
    DataFrame
        Index = column name, columns:
        - missing_rate
        - missing_count
    """
    if feature_cols is None:
        excluded = {"y", "readmitted", "readmitted_30d", "readmitted_3class"}
        feature_cols = [c for c in df.columns if c not in excluded]

    n = len(df)
    if n == 0:
        raise ValueError("DataFrame is empty; cannot summarize missingness.")

    def _col_missing_rate(col: pd.Series) -> float:
        return float(_series_missing_mask(col, missing_tokens).mean())

    missing_rate_by_col = (
        df[feature_cols]
        .apply(_col_missing_rate)
        .sort_values(ascending=False)
        .to_frame("missing_rate")
    )
    missing_rate_by_col["missing_count"] = (
        (missing_rate_by_col["missing_rate"] * n).round().astype(int)
    )
    return missing_rate_by_col


@dataclass(frozen=True)
class MissingnessByRowResult:
    """Result of row-level missingness analysis."""
    missing_per_row: pd.Series
    missing_rate_per_row: pd.Series
    summary_counts: Dict[str, float]
    high_missing_fraction: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "missing_per_row_summary": self.summary_counts,
            "high_missing_fraction": self.high_missing_fraction,
        }


def missingness_by_row(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    missing_tokens: Sequence[str] = MISSING_TOKENS,
    high_missing_threshold: float = 0.30,
) -> MissingnessByRowResult:
    """Row-level missingness (per encounter/patient record).

    Parameters
    ----------
    df:
        DataFrame with at least the feature columns.
    feature_cols:
        Subset of df columns to treat as features. If None, all except target.
    missing_tokens:
        Tokens to treat as missing.
    high_missing_threshold:
        Threshold on missing *rate* to consider a row 'highly incomplete'.

    Returns
    -------
    MissingnessByRowResult
        - missing_per_row: number of missing values in each row.
        - missing_rate_per_row: fraction of missing per row.
        - summary_counts: describe() of missing_per_row (as dict).
        - high_missing_fraction: proportion of rows with missing_rate >= threshold.
    """
    if feature_cols is None:
        excluded = {"y", "readmitted", "readmitted_30d", "readmitted_3class"}
        feature_cols = [c for c in df.columns if c not in excluded]

    if not feature_cols:
        raise ValueError("No feature columns provided for missingness_by_row.")

    miss_matrix = df[feature_cols].applymap(
        lambda v: is_missing(v, missing_tokens=missing_tokens)
    )
    missing_per_row = miss_matrix.sum(axis=1)
    missing_rate_per_row = missing_per_row / float(len(feature_cols))

    summary_counts = missing_per_row.describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
    ).to_dict()
    high_missing_fraction = float((missing_rate_per_row >= high_missing_threshold).mean())

    return MissingnessByRowResult(
        missing_per_row=missing_per_row,
        missing_rate_per_row=missing_rate_per_row,
        summary_counts=summary_counts,
        high_missing_fraction=high_missing_fraction,
    )


# ---------------------------------------------------------------------
# 4.3  Distributions and high-cardinality categoricals
# ---------------------------------------------------------------------


def topk_table(
    series: pd.Series,
    *,
    k: int = 20,
    missing_tokens: Sequence[str] = MISSING_TOKENS,
    missing_label: str = "__MISSING__",
) -> Tuple[pd.DataFrame, int, float]:
    """Frequency table for high-cardinality categorical variables.

    Parameters
    ----------
    series:
        Categorical column (e.g., diag_1, medical_specialty, payer_code).
    k:
        Top-k categories to display.
    missing_tokens:
        Encodings considered missing.
    missing_label:
        Label used to group all missing values into a single bucket.

    Returns
    -------
    (top_df, n_unique, coverage)
        - top_df: DataFrame with columns ['count', 'rate'] for top-k items.
        - n_unique: total number of distinct categories (including missing_label).
        - coverage: fraction of rows covered by the top-k categories.
    """
    s = series.copy()
    s = s.where(~s.map(lambda v: is_missing(v, missing_tokens=missing_tokens)),
                other=missing_label)

    vc = s.value_counts(dropna=False)
    top = vc.head(k).to_frame("count")
    if len(s) > 0:
        top["rate"] = top["count"] / float(len(s))
        coverage = float(top["count"].sum() / float(len(s)))
    else:
        top["rate"] = 0.0
        coverage = 0.0

    return top, int(vc.shape[0]), coverage


# ---------------------------------------------------------------------
# 4.4  Clinically meaningful errors (FN vs FP) – dummy baseline
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DummyBaselineResult:
    """Result of a most_frequent DummyClassifier baseline.

    This baseline is meant to *illustrate* the effect of class imbalance:
    it usually predicts only the majority class, leading to apparently
    reasonable accuracy but zero recall and F1 for the minority class, as
    discussed in the PDF for imbalanced problems. 
    """
    confusion_terms: Dict[str, int]
    metrics: Dict[str, float]

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "confusion_terms": self.confusion_terms,
            "metrics": self.metrics,
        }


def run_dummy_most_frequent_baseline(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str = "y",
    test_size: float = 0.2,
    random_state: int = 42,
) -> DummyBaselineResult:
    """Run a 'most_frequent' DummyClassifier to illustrate imbalance.

    Parameters
    ----------
    df:
        DataFrame with features and binary target.
    feature_cols:
        Columns used as input X (raw; preprocessing comes later).
    target_col:
        Name of the binary target column (e.g., 'readmitted_30d' or 'y').
    test_size:
        Fraction reserved for the illustrative test split.
    random_state:
        Seed for the train/test partition (kept fixed for reproducibility).

    Returns
    -------
    DummyBaselineResult
        Confusion-matrix terms TN/FP/FN/TP and metrics:
        accuracy, precision, recall, F1.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in df.")

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

    metrics_dict = {
        "accuracy": float(metrics.accuracy_score(y_test, y_pred)),
        "precision": float(metrics.precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(metrics.f1_score(y_test, y_pred, zero_division=0)),
    }

    confusion_terms = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return DummyBaselineResult(confusion_terms=confusion_terms, metrics=metrics_dict)
