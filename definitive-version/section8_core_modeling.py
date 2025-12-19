from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7

class SparseToCSC(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to CSC (DecisionTree often expects CSC when input is sparse)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            from scipy import sparse  
        except Exception as e:
            raise RuntimeError(
            ) from e

        if sparse.issparse(X):
            return X.tocsc()
        return X


def _validate_binary_target(y: pd.Series) -> np.ndarray:
    if y.isna().any():
        raise ValueError("Target contains missing values; Section 8 requires a clean binary target.")
    y_arr = np.asarray(pd.Series(y).astype(int).tolist(), dtype=int)
    uniq = set(np.unique(y_arr).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"Target must be binary in {{0,1}}, found values: {sorted(list(uniq))}")
    return y_arr


def _default_X_y(
    df: pd.DataFrame,
    *,
    target_col: str,
    preprocess_config: s5.DiabetesPreprocessConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    y_arr = _validate_binary_target(df[target_col])

    drop_cols = [c for c in preprocess_config.target_cols if c in df.columns]
    if target_col in df.columns and target_col not in drop_cols:
        drop_cols.append(target_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No feature columns available after dropping target columns.")
    return X, y_arr


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    require_csc: bool = False


def build_section8_model_specs(
    *,
    random_state: int = 42,
) -> List[ModelSpec]:
    """
    Two conceptually different classifiers:
      1) Logistic Regression (linear, probabilistic)
      2) Decision Tree (rule-based, non-linear)

    We pick a LR solver compatible with sparse one-hot features ("saga").
    """
    lr = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=2000,
        random_state=random_state,
        n_jobs=-1,
    )
    dt = DecisionTreeClassifier(
        random_state=random_state,
    )

    return [
        ModelSpec(name="logistic_regression", estimator=lr, require_csc=False),
        ModelSpec(name="decision_tree", estimator=dt, require_csc=True),
    ]


def build_section8_pipelines(
    df: pd.DataFrame,
    *,
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    model_random_state: int = 42,
) -> Dict[str, Pipeline]:
    """
    Build model pipelines = preprocessing (Section 5) + optional format adapter + classifier.
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()

    pre = s5.build_preprocessor(df, preprocess_config)
    specs = build_section8_model_specs(random_state=model_random_state)

    pipes: Dict[str, Pipeline] = {}
    for spec in specs:
        steps = [("pre", pre)]
        if spec.require_csc:
            steps.append(("to_csc", SparseToCSC()))
        steps.append(("clf", spec.estimator))
        pipes[spec.name] = Pipeline(steps=steps)

    return pipes


@dataclass(frozen=True)
class Section8RunResult:
    """
    Holds everything you need for the report:
      - identical-fold CV metrics (mean/std + per-fold vectors)
      - fold fingerprint for reproducibility
    """
    target_col: str
    n_splits: int
    cv_random_state: int
    split_fingerprint: str
    results_by_model: Dict[str, s7.CVEvaluationResult]
    summary_table: pd.DataFrame


def _summary_table_from_results(
    results_by_model: Dict[str, s7.CVEvaluationResult],
    *,
    model_order: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for name in model_order:
        r = results_by_model[name]
        row: Dict[str, float] = {"model": name}
        for m in metrics:
            row[f"{m}_mean"] = float(r.mean_scores[m])
            row[f"{m}_std"] = float(r.std_scores[m])
        rows.append(row)

    cols = ["model"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    return pd.DataFrame(rows)[cols]


def run_section8_core_models(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    n_splits: int = 10,
    cv_random_state: int = 42,
    model_random_state: int = 42,
    metrics: Sequence[str] = ("roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"),
    return_oof: bool = True,
    cv_splits: Optional[Sequence[s7.Split]] = None,
) -> Section8RunResult:
    """
    End-to-end Section 8 runner:
      - builds preprocessing + LR + DT pipelines
      - creates/reuses identical stratified splits (10-fold by default)
      - evaluates each model with the manual CV loop (Section 7)
      - returns a compact mean±std table + full per-fold vectors
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()

    X, y = _default_X_y(df, target_col=target_col, preprocess_config=preprocess_config)

    if cv_splits is None:
        cv_splits = s7.make_stratified_cv_splits(
            y,
            n_splits=n_splits,
            shuffle=True,
            random_state=cv_random_state,
        )
    else:
        s7.validate_cv_splits(cv_splits, n_samples=len(y), n_splits_expected=n_splits)

    fp = s7.cv_splits_fingerprint(cv_splits)

    pipes = build_section8_pipelines(
        df,
        preprocess_config=preprocess_config,
        model_random_state=model_random_state,
    )

    # Evaluate with identical folds
    results_by_model: Dict[str, s7.CVEvaluationResult] = {}
    for name, pipe in pipes.items():
        res = s7.evaluate_binary_pipeline_cv(
            pipe,
            X,
            y,
            cv_splits=cv_splits,
            metrics=metrics,
            return_oof=return_oof,
        )
        results_by_model[name] = res

    model_order = list(pipes.keys())
    table = _summary_table_from_results(results_by_model, model_order=model_order, metrics=metrics)

    return Section8RunResult(
        target_col=target_col,
        n_splits=n_splits,
        cv_random_state=cv_random_state,
        split_fingerprint=fp,
        results_by_model=results_by_model,
        summary_table=table,
    )
