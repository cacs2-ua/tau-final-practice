
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

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
from sklearn.utils.class_weight import compute_sample_weight

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7

Split = s7.Split


# ---------------------------------------------------------------------
# Section 10 — Class Imbalance Handling Experiments
#
# Goals (rubric):
# - Apply and compare ≥1 explicit imbalance method (we implement 4):
#   (A) cost-sensitive learning via sample weights ("class_weight_balanced")
#   (B) random oversampling ("random_over")
#   (C) random undersampling ("random_under")
#   (D) threshold moving to maximize F1 on training fold ("threshold_f1")
#
# Contract:
# - SAME fixed CV splits (cv_splits) across all experiments (Section 7).
# - NO leakage: resampling + threshold selection happen inside each fold using TRAIN only.
# - Report ROC-AUC + precision/recall/F1 (minority class behavior), plus accuracy/balanced_accuracy.
# ---------------------------------------------------------------------


ImbalanceMethod = Literal[
    "none",
    "class_weight_balanced",
    "random_over",
    "random_under",
    "threshold_f1",
]


@dataclass(frozen=True)
class ImbalanceExperimentSpec:
    model: str
    method: ImbalanceMethod
    # For sampling methods:
    sampling_ratio: float = 1.0  # minority/majority after sampling (1.0 = balanced)
    random_state: int = 42


@dataclass(frozen=True)
class Section10EvaluationResult:
    spec: ImbalanceExperimentSpec
    n_splits: int
    split_fingerprint: str
    fold_scores: Dict[str, np.ndarray]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    fold_thresholds: Optional[np.ndarray] = None  # only for threshold_f1

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "model": self.spec.model,
            "method": self.spec.method,
            "sampling_ratio": float(self.spec.sampling_ratio),
            "n_splits": int(self.n_splits),
            "split_fingerprint": self.split_fingerprint,
        }
        if self.fold_thresholds is not None:
            row["threshold_mean"] = float(np.mean(self.fold_thresholds))
            row["threshold_std"] = float(np.std(self.fold_thresholds, ddof=1)) if len(self.fold_thresholds) > 1 else 0.0

        for m, v in self.mean_scores.items():
            row[f"{m}_mean"] = float(v)
        for m, v in self.std_scores.items():
            row[f"{m}_std"] = float(v)
        return row


# ------------------------------ helpers ------------------------------


def _ensure_1d_binary(y: Iterable[int]) -> np.ndarray:
    y_arr = np.asarray(list(y), dtype=int).ravel()
    if y_arr.size == 0:
        raise ValueError("y is empty.")
    uniq = set(np.unique(y_arr).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"Binary target must be in {{0,1}}. Found: {sorted(list(uniq))}")
    return y_arr


def _safe_index(X, idx: np.ndarray):
    # Works for pandas, numpy, scipy sparse
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    return X[idx]


def _get_score_vector(estimator, X, *, positive_label: int = 1) -> np.ndarray:
    """
    Continuous scores for ROC-AUC + threshold moving:
    - predict_proba[:, pos] if available
    - else decision_function
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba must return [n_samples, 2+] for binary tasks.")
        # find column for positive_label if possible
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

    raise ValueError("Estimator must provide predict_proba or decision_function for ROC-AUC/thresholding.")


def _to_csc_if_sparse(X):
    try:
        from scipy import sparse
    except Exception as e:
        raise RuntimeError("scipy is required for sparse matrix operations. Install with: pip install scipy") from e
    if sparse.issparse(X):
        return X.tocsc()
    return X


def random_oversample(X, y: np.ndarray, *, ratio: float = 1.0, random_state: int = 42):
    """
    Randomly oversample the minority class with replacement to reach:
        n_minority_new ~= ratio * n_majority
    """
    y = _ensure_1d_binary(y)
    if not (ratio > 0.0):
        raise ValueError("ratio must be > 0.")

    rng = np.random.RandomState(int(random_state))
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return X, y

    # identify minority/majority
    if len(idx1) <= len(idx0):
        idx_min, idx_maj = idx1, idx0
        min_label = 1
    else:
        idx_min, idx_maj = idx0, idx1
        min_label = 0

    n_maj = len(idx_maj)
    n_min = len(idx_min)
    target_min = int(np.round(float(ratio) * n_maj))
    if target_min <= n_min:
        # already at/above target
        new_idx = np.concatenate([idx_maj, idx_min])
    else:
        extra = rng.choice(idx_min, size=(target_min - n_min), replace=True)
        new_idx = np.concatenate([idx_maj, idx_min, extra])

    rng.shuffle(new_idx)
    X_new = _safe_index(X, new_idx)
    y_new = y[new_idx]
    return X_new, y_new


def random_undersample(X, y: np.ndarray, *, ratio: float = 1.0, random_state: int = 42):
    """
    Randomly undersample the majority class without replacement to reach:
        n_majority_new ~= n_minority / ratio
    where ratio = minority/majority after sampling (ratio=1 => balanced).
    """
    y = _ensure_1d_binary(y)
    if not (ratio > 0.0):
        raise ValueError("ratio must be > 0.")

    rng = np.random.RandomState(int(random_state))
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return X, y

    # identify minority/majority
    if len(idx1) <= len(idx0):
        idx_min, idx_maj = idx1, idx0
    else:
        idx_min, idx_maj = idx0, idx1

    n_min = len(idx_min)
    n_maj = len(idx_maj)

    # want: n_min / n_maj_new ~= ratio  => n_maj_new ~= n_min / ratio
    target_maj = int(np.round(float(n_min) / float(ratio)))
    target_maj = max(1, min(target_maj, n_maj))

    keep_maj = rng.choice(idx_maj, size=target_maj, replace=False)
    new_idx = np.concatenate([idx_min, keep_maj])
    rng.shuffle(new_idx)

    X_new = _safe_index(X, new_idx)
    y_new = y[new_idx]
    return X_new, y_new


def best_threshold_max_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    positive_label: int = 1,
    n_grid: int = 200,
) -> float:
    """
    Choose threshold that maximizes F1 on the provided (y_true, scores).
    Uses quantile grid for stability and speed.
    """
    y_true = _ensure_1d_binary(y_true)
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size != y_true.size:
        raise ValueError("scores and y_true must have the same length.")

    # Build candidate thresholds
    if scores.size <= 500:
        candidates = np.unique(scores)
    else:
        qs = np.linspace(0.0, 1.0, int(n_grid))
        candidates = np.unique(np.quantile(scores, qs))

    best_t = float(candidates[0])
    best_f1 = -1.0
    best_recall = -1.0

    for t in candidates:
        y_pred = (scores >= t).astype(int)
        f1 = float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_recall = float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
        elif f1 == best_f1:
            # tie-break: prefer higher recall, then lower threshold (catch more positives)
            rec = float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
            if rec > best_recall or (rec == best_recall and float(t) < best_t):
                best_t = float(t)
                best_recall = rec

    return best_t


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _supports_sample_weight(estimator) -> bool:
    # Conservative check: many sklearn estimators accept it; this avoids hard failures.
    import inspect
    try:
        sig = inspect.signature(estimator.fit)
        return "sample_weight" in sig.parameters
    except Exception:
        return False


# -------------------------- model factory ----------------------------


def build_section10_estimator(model: str, *, random_state: int = 42, fast_mode: bool = False):
    """
    Minimal, rubric-safe set for Section 10:
    - logistic_regression (probabilistic; classic for class weights + threshold moving)
    - decision_tree (PDF notes sensitivity to imbalance)
    - random_forest (ensemble/bagging; can respond well to imbalance methods)
    """
    model = str(model).strip().lower()
    if model == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            solver="saga",
            penalty="l2",
            max_iter=2000,
            random_state=int(random_state),
            n_jobs=-1,
        )

    if model == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=int(random_state))

    if model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = 80 if fast_mode else 300
        return RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=-1,
        )

    if model == "linear_svm":
        from sklearn.svm import LinearSVC
        return LinearSVC(C=1.0, random_state=int(random_state))

    raise ValueError(f"Unknown model '{model}'. Use: logistic_regression, decision_tree, random_forest, linear_svm.")


# ----------------------------- main CV -------------------------------


def evaluate_section10_experiment_cv(
    df: pd.DataFrame,
    *,
    target_col: str,
    preprocess_config: Optional[s5.DiabetesPreprocessConfig],
    cv_splits: Sequence[Split],
    spec: ImbalanceExperimentSpec,
    positive_label: int = 1,
    fast_mode: bool = False,
) -> Section10EvaluationResult:
    """
    Manual fold loop to guarantee:
    - resampling inside TRAIN only
    - threshold selection inside TRAIN only
    - identical folds reused across all experiments
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in df.")

    y = _ensure_1d_binary(df[target_col].astype(int).to_numpy())

    drop_cols = [c for c in preprocess_config.target_cols if c in df.columns]
    if target_col not in drop_cols:
        drop_cols.append(target_col)
    X = df.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No feature columns available after dropping target columns.")

    s7.validate_cv_splits(cv_splits, n_samples=len(y))
    fp = s7.cv_splits_fingerprint(cv_splits)

    # Preprocessor template (cloned each fold)
    pre_template = s5.build_preprocessor(df, preprocess_config)

    fold_metrics: Dict[str, List[float]] = {k: [] for k in ["roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"]}
    fold_thresholds: List[float] = []

    for tr, te in cv_splits:
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)

        X_tr_raw = _safe_index(X, tr)
        y_tr = y[tr]
        X_te_raw = _safe_index(X, te)
        y_te = y[te]

        # Fit preprocessing on TRAIN only
        pre = clone(pre_template)
        pre.fit(X_tr_raw, y_tr)
        X_tr = pre.transform(X_tr_raw)
        X_te = pre.transform(X_te_raw)

        # Apply imbalance method inside the fold (TRAIN only)
        sample_weight = None
        if spec.method == "class_weight_balanced":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_tr).astype(float)

        if spec.method == "random_over":
            X_tr, y_tr = random_oversample(X_tr, y_tr, ratio=spec.sampling_ratio, random_state=spec.random_state)
            sample_weight = None  # do not mix by default

        if spec.method == "random_under":
            X_tr, y_tr = random_undersample(X_tr, y_tr, ratio=spec.sampling_ratio, random_state=spec.random_state)
            sample_weight = None

        # Build estimator for this fold
        est = build_section10_estimator(spec.model, random_state=spec.random_state, fast_mode=fast_mode)

        # Some tree models prefer CSC if sparse
        if spec.model in {"decision_tree", "random_forest"}:
            X_tr_fit = _to_csc_if_sparse(X_tr)
            X_te_fit = _to_csc_if_sparse(X_te)
        else:
            X_tr_fit, X_te_fit = X_tr, X_te

        # Fit with or without sample_weight
        if sample_weight is not None:
            if not _supports_sample_weight(est):
                raise RuntimeError(f"Estimator '{spec.model}' does not support sample_weight; cannot run class_weight_balanced.")
            est.fit(X_tr_fit, y_tr, sample_weight=sample_weight)
        else:
            est.fit(X_tr_fit, y_tr)

        # Scores for ROC-AUC and (optionally) threshold moving
        tr_scores = _get_score_vector(est, X_tr_fit, positive_label=positive_label)
        te_scores = _get_score_vector(est, X_te_fit, positive_label=positive_label)

        if spec.method == "threshold_f1":
            thr = best_threshold_max_f1(y_tr, tr_scores, positive_label=positive_label)
            fold_thresholds.append(float(thr))
            y_pred = (te_scores >= thr).astype(int)
        else:
            # default classifier decision (usually 0.5 for proba models, 0 for margin-based)
            y_pred = np.asarray(est.predict(X_te_fit), dtype=int)

        m = _compute_metrics(y_te, y_pred, te_scores)
        for k in fold_metrics.keys():
            fold_metrics[k].append(float(m[k]))

    fold_scores = {k: np.asarray(v, dtype=float) for k, v in fold_metrics.items()}
    mean_scores = {k: float(np.mean(v)) for k, v in fold_scores.items()}
    std_scores = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in fold_scores.items()}

    thr_arr = np.asarray(fold_thresholds, dtype=float) if spec.method == "threshold_f1" else None

    return Section10EvaluationResult(
        spec=spec,
        n_splits=len(cv_splits),
        split_fingerprint=fp,
        fold_scores=fold_scores,
        mean_scores=mean_scores,
        std_scores=std_scores,
        fold_thresholds=thr_arr,
    )


def run_section10_imbalance_experiments(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    cv_splits: Sequence[Split],
    models: Sequence[str] = ("logistic_regression", "decision_tree", "random_forest"),
    methods: Sequence[ImbalanceMethod] = ("none", "class_weight_balanced", "random_over", "random_under", "threshold_f1"),
    sampling_ratio: float = 1.0,
    random_state: int = 42,
    fast_mode: bool = False,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Section10EvaluationResult]]:
    """
    Returns:
      - results_table: mean ± std per (model, method) for ROC-AUC + precision/recall/F1 (+ accuracy, balanced_accuracy)
      - results_map: raw per-fold vectors (needed later for Wilcoxon / deeper analysis)
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()

    rows: List[Dict[str, Any]] = []
    results_map: Dict[Tuple[str, str], Section10EvaluationResult] = {}

    for model in models:
        for method in methods:
            spec = ImbalanceExperimentSpec(
                model=str(model),
                method=method,
                sampling_ratio=float(sampling_ratio),
                random_state=int(random_state),
            )
            res = evaluate_section10_experiment_cv(
                df,
                target_col=target_col,
                preprocess_config=preprocess_config,
                cv_splits=cv_splits,
                spec=spec,
                fast_mode=fast_mode,
            )
            rows.append(res.as_row())
            results_map[(spec.model, spec.method)] = res

    table = pd.DataFrame(rows)

    # Nice, report-ready column ordering
    metric_order = ["roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"]
    cols = (
        ["model", "method", "sampling_ratio", "n_splits", "split_fingerprint"]
        + (["threshold_mean", "threshold_std"] if "threshold_mean" in table.columns else [])
        + [f"{m}_mean" for m in metric_order]
        + [f"{m}_std" for m in metric_order]
    )
    cols = [c for c in cols if c in table.columns] + [c for c in table.columns if c not in cols]
    table = table[cols].sort_values(["model", "method"]).reset_index(drop=True)

    return table, results_map
