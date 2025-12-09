from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7

Split = s7.Split

TuneMode = Literal["manual", "grid", "random", "nested_grid", "nested_random"]


# -----------------------------
# Small utilities (consistent with Section 7 contract)
# -----------------------------

def _safe_index(X, idx: np.ndarray):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    return X[idx]


def _ensure_binary_1d(y: Iterable[int]) -> np.ndarray:
    y_arr = np.asarray(list(y), dtype=int).ravel()
    if y_arr.size == 0:
        raise ValueError("y is empty.")
    uniq = set(np.unique(y_arr).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"y must be binary in {{0,1}}. Found: {sorted(list(uniq))}")
    return y_arr


def _get_score_vector(estimator, X, *, positive_label: int = 1) -> np.ndarray:
    """
    Continuous scores for ROC-AUC:
    - predict_proba[:, pos] if available
    - else decision_function if available
    """
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba must return [n_samples, 2+] for binary ROC-AUC.")
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            pos_idx = 1
        else:
            classes = np.asarray(classes)
            if positive_label in set(classes.tolist()):
                pos_idx = int(np.where(classes == positive_label)[0][0])
            else:
                pos_idx = 1
        return proba[:, pos_idx].astype(float)

    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        return np.asarray(s, dtype=float).ravel()

    raise ValueError("Estimator must expose predict_proba or decision_function for ROC-AUC.")


class SparseToCSC:
    """Convert sparse matrices to CSC (tree-based estimators often prefer CSC)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy import sparse
        if sparse.issparse(X):
            return X.tocsc()
        return X


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


# -----------------------------
# Pipeline builders (tunable)
# -----------------------------

TunableModel = Literal[
    "logistic_regression",
    "random_forest",
    "linear_svm",
    "gradient_boosting_svd",
]


def _make_preprocess_variants(
    base: s5.DiabetesPreprocessConfig,
) -> Dict[str, s5.DiabetesPreprocessConfig]:
    """
    Deterministic preprocessing variants:
    - scaled: for scale-sensitive models (LR, SVM)
    - tree:   for tree models (no scaling needed)
    """
    cfg_scaled = dc_replace(base, scale_numeric=True, scaler="standard")
    cfg_tree = dc_replace(base, scale_numeric=False)
    return {"scaled": cfg_scaled, "tree": cfg_tree}


def build_tunable_pipeline(
    df: pd.DataFrame,
    *,
    model: TunableModel,
    preprocess_config_base: Optional[s5.DiabetesPreprocessConfig] = None,
    model_random_state: int = 42,
    svd_components: int = 100,
    fast_mode: bool = False,
) -> Pipeline:
    """
    Build a single sklearn Pipeline = [preprocessing] (+ adapters) + [classifier]
    to be tuned with manual/grid/random/nested CV.

    NOTE: Preprocessing is inside the pipeline => no leakage during CV/search.
    """
    if preprocess_config_base is None:
        preprocess_config_base = s5.DiabetesPreprocessConfig()

    cfgs = _make_preprocess_variants(preprocess_config_base)

    model = str(model).strip().lower()
    if model == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        pre = s5.build_preprocessor(df, cfgs["scaled"])
        max_iter = 300 if fast_mode else 2000
        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=int(max_iter),
            random_state=int(model_random_state),
            n_jobs=-1,
        )
        return Pipeline(steps=[("pre", pre), ("clf", clf)])

    if model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        pre = s5.build_preprocessor(df, cfgs["tree"])
        n_estimators = 80 if fast_mode else 300
        clf = RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(model_random_state),
            n_jobs=-1,
        )
        return Pipeline(steps=[("pre", pre), ("to_csc", SparseToCSC()), ("clf", clf)])

    if model == "linear_svm":
        from sklearn.svm import LinearSVC
        pre = s5.build_preprocessor(df, cfgs["scaled"])
        clf = LinearSVC(C=1.0, random_state=int(model_random_state))
        return Pipeline(steps=[("pre", pre), ("clf", clf)])

    if model == "gradient_boosting_svd":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.ensemble import GradientBoostingClassifier
        pre = s5.build_preprocessor(df, cfgs["tree"])
        n_estimators = 80 if fast_mode else 200
        clf = GradientBoostingClassifier(
            n_estimators=int(n_estimators),
            learning_rate=0.1,
            random_state=int(model_random_state),
        )
        return Pipeline(
            steps=[
                ("pre", pre),
                ("svd", TruncatedSVD(n_components=int(svd_components), random_state=int(model_random_state))),
                ("clf", clf),
            ]
        )

    raise ValueError(f"Unknown model '{model}'. Allowed: logistic_regression, random_forest, linear_svm, gradient_boosting_svd.")


def default_param_search_space(
    model: TunableModel,
    *,
    fast_mode: bool = False,
) -> Union[Dict[str, List[Any]], Dict[str, Any]]:
    """
    Ready-to-use search spaces (reasonable, rubric-safe).
    Keys use the Pipeline prefix 'clf__'.

    You can override these in your notebook when you want more extensive tuning.
    """
    model = str(model).strip().lower()

    if model == "logistic_regression":
        Cs = [0.1, 1.0, 3.0] if fast_mode else [0.01, 0.1, 1.0, 3.0, 10.0]
        return {"clf__C": Cs}

    if model == "random_forest":
        if fast_mode:
            return {
                "clf__max_depth": [None, 8],
                "clf__min_samples_leaf": [1, 5],
                "clf__max_features": ["sqrt"],
            }
        return {
            "clf__max_depth": [None, 8, 16],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__max_features": ["sqrt", "log2"],
        }

    if model == "linear_svm":
        Cs = [0.1, 1.0, 3.0] if fast_mode else [0.01, 0.1, 1.0, 3.0, 10.0]
        return {"clf__C": Cs}

    if model == "gradient_boosting_svd":
        if fast_mode:
            return {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]}
        return {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_depth": [2, 3, 4]}

    raise ValueError(f"Unknown model '{model}'.")


# -----------------------------
# Manual sweep (Basic requirement)
# -----------------------------

@dataclass(frozen=True)
class ManualSweepResult:
    mode: TuneMode
    split_fingerprint: str
    metric_refit: str
    results_table: pd.DataFrame
    best_params: Dict[str, Any]
    best_score: float


def manual_hyperparameter_sweep(
    pipeline: Pipeline,
    X,
    y: Iterable[int],
    *,
    cv_splits: Sequence[Split],
    param_list: Sequence[Dict[str, Any]],
    metrics: Sequence[str] = ("roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"),
    refit_metric: str = "roc_auc",
) -> ManualSweepResult:
    """
    Manual exploration:
      - evaluate each params dict using the SAME fixed splits (Section 7 contract)
      - produce mean±std table
      - pick best by refit_metric mean
    """
    y_arr = _ensure_binary_1d(y)
    s7.validate_cv_splits(cv_splits, n_samples=len(y_arr))
    fp = s7.cv_splits_fingerprint(cv_splits)

    rows: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}

    for i, params in enumerate(param_list):
        pipe_i = clone(pipeline).set_params(**params)

        res = s7.evaluate_binary_pipeline_cv(
            pipe_i,
            X,
            y_arr,
            cv_splits=cv_splits,
            metrics=metrics,
            return_oof=False,
        )

        row: Dict[str, Any] = {"trial": int(i)}
        for k, v in params.items():
            row[k] = v

        for m in metrics:
            row[f"{m}_mean"] = float(res.mean_scores[m])
            row[f"{m}_std"] = float(res.std_scores[m])

        rows.append(row)

        score_i = float(row[f"{refit_metric}_mean"])
        if score_i > best_score:
            best_score = score_i
            best_params = dict(params)

    table = pd.DataFrame(rows).sort_values(f"{refit_metric}_mean", ascending=False).reset_index(drop=True)

    return ManualSweepResult(
        mode="manual",
        split_fingerprint=fp,
        metric_refit=refit_metric,
        results_table=table,
        best_params=best_params,
        best_score=float(best_score),
    )


# -----------------------------
# Automated search (Advanced requirement)
# -----------------------------

@dataclass(frozen=True)
class SearchCVResult:
    mode: TuneMode
    split_fingerprint: str
    metric_refit: str
    best_params: Dict[str, Any]
    best_score: float
    cv_results_table: pd.DataFrame
    best_estimator: Any


def grid_search_fixed_splits(
    pipeline: Pipeline,
    X,
    y: Iterable[int],
    *,
    cv_splits: Sequence[Split],
    param_grid: Dict[str, List[Any]],
    scoring: str = "roc_auc",
    n_jobs: Optional[int] = None,
    refit: bool = True,
    error_score: Union[str, float] = "raise",
) -> SearchCVResult:
    """
    GridSearchCV using the SAME fixed splits (identical folds across comparisons).
    NOTE: this is not nested; use nested_cv_* for unbiased performance.
    """
    y_arr = _ensure_binary_1d(y)
    s7.validate_cv_splits(cv_splits, n_samples=len(y_arr))
    fp = s7.cv_splits_fingerprint(cv_splits)

    gscv = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=list(cv_splits),
        refit=refit,
        n_jobs=n_jobs,
        error_score=error_score,
        return_train_score=False,
    )
    gscv.fit(X, y_arr)

    tbl = pd.DataFrame(gscv.cv_results_).sort_values("rank_test_score").reset_index(drop=True)

    return SearchCVResult(
        mode="grid",
        split_fingerprint=fp,
        metric_refit=scoring,
        best_params=dict(gscv.best_params_),
        best_score=float(gscv.best_score_),
        cv_results_table=tbl,
        best_estimator=gscv.best_estimator_,
    )


def random_search_fixed_splits(
    pipeline: Pipeline,
    X,
    y: Iterable[int],
    *,
    cv_splits: Sequence[Split],
    param_distributions: Dict[str, Any],
    n_iter: int = 20,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: Optional[int] = None,
    refit: bool = True,
    error_score: Union[str, float] = "raise",
) -> SearchCVResult:
    """
    RandomizedSearchCV using the SAME fixed splits.
    NOTE: this is not nested; use nested_cv_* for unbiased performance.
    """
    y_arr = _ensure_binary_1d(y)
    s7.validate_cv_splits(cv_splits, n_samples=len(y_arr))
    fp = s7.cv_splits_fingerprint(cv_splits)

    rscv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=int(n_iter),
        scoring=scoring,
        cv=list(cv_splits),
        refit=refit,
        random_state=int(random_state),
        n_jobs=n_jobs,
        error_score=error_score,
        return_train_score=False,
    )
    rscv.fit(X, y_arr)

    tbl = pd.DataFrame(rscv.cv_results_).sort_values("rank_test_score").reset_index(drop=True)

    return SearchCVResult(
        mode="random",
        split_fingerprint=fp,
        metric_refit=scoring,
        best_params=dict(rscv.best_params_),
        best_score=float(rscv.best_score_),
        cv_results_table=tbl,
        best_estimator=rscv.best_estimator_,
    )


# -----------------------------
# Nested CV (Top-grade requirement)
# -----------------------------

@dataclass(frozen=True)
class NestedCVResult:
    mode: TuneMode
    outer_split_fingerprint: str
    inner_n_splits: int
    scoring: str
    outer_fold_scores: np.ndarray
    mean_score: float
    std_score: float
    outer_metrics_mean: Dict[str, float]
    outer_metrics_std: Dict[str, float]
    best_params_per_fold: List[Dict[str, Any]]


def nested_cv_grid_search(
    pipeline: Pipeline,
    X,
    y: Iterable[int],
    *,
    outer_splits: Sequence[Split],
    param_grid: Dict[str, List[Any]],
    scoring: str = "roc_auc",
    inner_n_splits: int = 5,
    inner_random_state: int = 123,
    n_jobs: Optional[int] = None,
) -> NestedCVResult:
    """
    Nested CV:
      - Outer folds: fixed (identical across model comparisons)
      - Inner loop: GridSearchCV only on training split
      - Report unbiased outer-fold performance of the tuned model
    """
    y_arr = _ensure_binary_1d(y)
    s7.validate_cv_splits(outer_splits, n_samples=len(y_arr))
    fp_outer = s7.cv_splits_fingerprint(outer_splits)

    outer_scores: List[float] = []
    outer_metrics: Dict[str, List[float]] = {k: [] for k in ["roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"]}
    best_params_per_fold: List[Dict[str, Any]] = []

    for fold_i, (tr, te) in enumerate(outer_splits):
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)

        X_tr = _safe_index(X, tr)
        y_tr = y_arr[tr]
        X_te = _safe_index(X, te)
        y_te = y_arr[te]

        inner_cv = StratifiedKFold(
            n_splits=int(inner_n_splits),
            shuffle=True,
            random_state=int(inner_random_state + fold_i),
        )

        gs = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            refit=True,
            n_jobs=n_jobs,
            error_score="raise",
            return_train_score=False,
        )
        gs.fit(X_tr, y_tr)

        best_params_per_fold.append(dict(gs.best_params_))

        best_est = gs.best_estimator_
        y_pred = np.asarray(best_est.predict(X_te), dtype=int)
        y_score = _get_score_vector(best_est, X_te, positive_label=1)

        m = _compute_metrics(y_te, y_pred, y_score)
        outer_scores.append(float(m["roc_auc"]))
        for k in outer_metrics.keys():
            outer_metrics[k].append(float(m[k]))

    outer_arr = np.asarray(outer_scores, dtype=float)
    metrics_mean = {k: float(np.mean(v)) for k, v in outer_metrics.items()}
    metrics_std = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in outer_metrics.items()}

    return NestedCVResult(
        mode="nested_grid",
        outer_split_fingerprint=fp_outer,
        inner_n_splits=int(inner_n_splits),
        scoring=scoring,
        outer_fold_scores=outer_arr,
        mean_score=float(np.mean(outer_arr)),
        std_score=float(np.std(outer_arr, ddof=1)) if len(outer_arr) > 1 else 0.0,
        outer_metrics_mean=metrics_mean,
        outer_metrics_std=metrics_std,
        best_params_per_fold=best_params_per_fold,
    )


def nested_cv_random_search(
    pipeline: Pipeline,
    X,
    y: Iterable[int],
    *,
    outer_splits: Sequence[Split],
    param_distributions: Dict[str, Any],
    n_iter: int = 25,
    scoring: str = "roc_auc",
    inner_n_splits: int = 5,
    inner_random_state: int = 123,
    n_jobs: Optional[int] = None,
) -> NestedCVResult:
    """
    Nested CV with RandomizedSearchCV inner loop.
    """
    y_arr = _ensure_binary_1d(y)
    s7.validate_cv_splits(outer_splits, n_samples=len(y_arr))
    fp_outer = s7.cv_splits_fingerprint(outer_splits)

    outer_scores: List[float] = []
    outer_metrics: Dict[str, List[float]] = {k: [] for k in ["roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"]}
    best_params_per_fold: List[Dict[str, Any]] = []

    for fold_i, (tr, te) in enumerate(outer_splits):
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)

        X_tr = _safe_index(X, tr)
        y_tr = y_arr[tr]
        X_te = _safe_index(X, te)
        y_te = y_arr[te]

        inner_cv = StratifiedKFold(
            n_splits=int(inner_n_splits),
            shuffle=True,
            random_state=int(inner_random_state + fold_i),
        )

        rs = RandomizedSearchCV(
            estimator=clone(pipeline),
            param_distributions=param_distributions,
            n_iter=int(n_iter),
            scoring=scoring,
            cv=inner_cv,
            refit=True,
            random_state=int(inner_random_state + fold_i),
            n_jobs=n_jobs,
            error_score="raise",
            return_train_score=False,
        )
        rs.fit(X_tr, y_tr)

        best_params_per_fold.append(dict(rs.best_params_))

        best_est = rs.best_estimator_
        y_pred = np.asarray(best_est.predict(X_te), dtype=int)
        y_score = _get_score_vector(best_est, X_te, positive_label=1)

        m = _compute_metrics(y_te, y_pred, y_score)
        outer_scores.append(float(m["roc_auc"]))
        for k in outer_metrics.keys():
            outer_metrics[k].append(float(m[k]))

    outer_arr = np.asarray(outer_scores, dtype=float)
    metrics_mean = {k: float(np.mean(v)) for k, v in outer_metrics.items()}
    metrics_std = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in outer_metrics.items()}

    return NestedCVResult(
        mode="nested_random",
        outer_split_fingerprint=fp_outer,
        inner_n_splits=int(inner_n_splits),
        scoring=scoring,
        outer_fold_scores=outer_arr,
        mean_score=float(np.mean(outer_arr)),
        std_score=float(np.std(outer_arr, ddof=1)) if len(outer_arr) > 1 else 0.0,
        outer_metrics_mean=metrics_mean,
        outer_metrics_std=metrics_std,
        best_params_per_fold=best_params_per_fold,
    )
