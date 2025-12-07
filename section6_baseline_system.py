from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

import section5_preprocessing_pipeline as s5


# ---------------------------------------------------------------------
# Section 6 — Baseline System (DummyClassifier sanity-check)
# ---------------------------------------------------------------------
# Required by the practice:
#   - Baseline ROC-AUC from a DummyClassifier
# Also report clinically interpretable metrics:
#   - Precision / Recall / F1 (minority class = 1)
#
# We evaluate using Stratified K-Fold CV (same philosophy as later sections),
# and we keep random_state fixed for reproducibility and fair comparisons.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineCVResult:
    strategy: str
    n_splits: int
    fold_scores: Dict[str, np.ndarray]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]

    def as_row(self) -> Dict[str, float]:
        row: Dict[str, float] = {"strategy": self.strategy, "n_splits": float(self.n_splits)}
        for k, v in self.mean_scores.items():
            row[f"{k}_mean"] = float(v)
        for k, v in self.std_scores.items():
            row[f"{k}_std"] = float(v)
        return row


def _validate_binary_target(y: pd.Series) -> pd.Series:
    if y.isna().any():
        raise ValueError("Target contains missing values; baseline evaluation requires a clean binary target.")
    uniq = set(pd.Series(y).astype(int).unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"Target must be binary in {{0,1}}, found values: {sorted(list(uniq))}")
    return pd.Series(y).astype(int)


def _default_feature_frame(df: pd.DataFrame, *, target_col: str, config: s5.DiabetesPreprocessConfig) -> pd.DataFrame:
    drop_cols = [c for c in config.target_cols if c in df.columns]
    if target_col in df.columns and target_col not in drop_cols:
        drop_cols.append(target_col)
    X = df.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No feature columns available after dropping target columns.")
    return X


def _scorers_binary() -> Dict[str, object]:
    # zero_division=0 avoids warnings when a baseline predicts no positives
    return {
        "roc_auc": "roc_auc",
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }


def evaluate_dummy_baseline_cv(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    strategy: str = "most_frequent",
    n_splits: int = 10,
    cv_random_state: int = 42,
    dummy_random_state: int = 42,
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
) -> BaselineCVResult:
    """
    Evaluate a DummyClassifier baseline under Stratified K-Fold CV.

    Parameters
    ----------
    df:
        DataFrame containing features + a binary target column.
    target_col:
        Name of the binary target (1 = readmitted <30 days, 0 = otherwise).
    strategy:
        DummyClassifier strategy, e.g. "most_frequent" or "stratified".
    n_splits:
        Number of CV folds (10 required later; baseline can already match it).
    cv_random_state:
        Seed for fold generation (must be fixed for reproducibility).
    dummy_random_state:
        Seed for DummyClassifier randomness (relevant for "stratified"/"uniform").
    preprocess_config:
        If provided, used to build the exact same preprocessing pipeline as later models.

    Returns
    -------
    BaselineCVResult:
        Per-fold scores + mean/std for:
        roc_auc, accuracy, precision, recall, f1
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    y = _validate_binary_target(df[target_col])
    X = _default_feature_frame(df, target_col=target_col, config=preprocess_config)

    # Build preprocessing “single source of truth” (Section 5),
    # then attach the dummy baseline classifier.
    pre = s5.build_preprocessor(df, preprocess_config)

    clf = DummyClassifier(strategy=strategy, random_state=dummy_random_state)
    pipe = Pipeline(steps=[("pre", pre), ("dummy", clf)])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
    scoring = _scorers_binary()

    out = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=None,
    )

    fold_scores: Dict[str, np.ndarray] = {}
    for key in scoring.keys():
        fold_scores[key] = np.asarray(out[f"test_{key}"], dtype=float)

    mean_scores = {k: float(np.mean(v)) for k, v in fold_scores.items()}
    std_scores = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in fold_scores.items()}

    return BaselineCVResult(
        strategy=strategy,
        n_splits=n_splits,
        fold_scores=fold_scores,
        mean_scores=mean_scores,
        std_scores=std_scores,
    )


def run_dummy_baselines(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    strategies: Sequence[str] = ("most_frequent", "stratified"),
    n_splits: int = 10,
    cv_random_state: int = 42,
    dummy_random_state: int = 42,
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
) -> pd.DataFrame:
    """
    Convenience helper: evaluate multiple DummyClassifier baselines
    and return a compact results table (mean ± std across folds).
    """
    rows: List[Dict[str, float]] = []
    for strat in strategies:
        res = evaluate_dummy_baseline_cv(
            df,
            target_col=target_col,
            strategy=strat,
            n_splits=n_splits,
            cv_random_state=cv_random_state,
            dummy_random_state=dummy_random_state,
            preprocess_config=preprocess_config,
        )
        rows.append(res.as_row())

    table = pd.DataFrame(rows)
    # Nice ordering for the report/table
    preferred = [
        "strategy",
        "n_splits",
        "roc_auc_mean",
        "roc_auc_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "accuracy_mean",
        "accuracy_std",
    ]
    cols = [c for c in preferred if c in table.columns] + [c for c in table.columns if c not in preferred]
    return table[cols]
