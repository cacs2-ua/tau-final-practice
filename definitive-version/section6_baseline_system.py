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
    cv_splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> BaselineCVResult:
    """
    Evaluate a DummyClassifier baseline under Stratified K-Fold CV.
      - If cv_splits is provided, those exact folds are used (identical partitions across experiments).
      - Otherwise, a StratifiedKFold is created as before.
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    y = _validate_binary_target(df[target_col])
    X = _default_feature_frame(df, target_col=target_col, config=preprocess_config)

    pre = s5.build_preprocessor(df, preprocess_config)

    clf = DummyClassifier(strategy=strategy, random_state=dummy_random_state)
    pipe = Pipeline(steps=[("pre", pre), ("dummy", clf)])

    if cv_splits is not None:
        cv_used = list(cv_splits)
        n_splits_effective = len(cv_used)
    else:
        cv_used = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
        n_splits_effective = n_splits

    scoring = _scorers_binary()

    out = cross_validate(
        pipe,
        X,
        y,
        cv=cv_used,
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
        n_splits=n_splits_effective,
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
    cv_splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """
    Convenience helper: evaluate multiple DummyClassifier baselines
    and return a compact results table (mean ± std across folds).
    cv_splits can be passed to force identical folds across baselines and later models.
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
            cv_splits=cv_splits,
        )
        rows.append(res.as_row())

    table = pd.DataFrame(rows)
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
