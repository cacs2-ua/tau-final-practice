from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import hashlib
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
from sklearn.model_selection import StratifiedKFold


Split = Tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------
# Section 7 — Single, reusable 10-fold CV protocol (identical folds)
# ---------------------------------------------------------------------


def make_stratified_cv_splits(
    y: Iterable[int],
    *,
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
) -> List[Split]:
    """
    Create one fixed set of Stratified K-Fold splits (train_idx, test_idx),
    to be REUSED across all experiments (baseline, models, imbalance methods,
    selection, PCA, tuning outer loop).

    This is the "identical folds across comparisons" contract required by the practice.
    """
    y_arr = np.asarray(list(y), dtype=int)
    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D iterable of binary labels.")
    if len(y_arr) == 0:
        raise ValueError("y is empty.")
    uniq = set(np.unique(y_arr).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"y must be binary in {{0,1}}. Found: {sorted(list(uniq))}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits: List[Split] = []
    X_dummy = np.zeros((len(y_arr), 1), dtype=float)  # splitter needs X, but doesn't use it
    for tr, te in cv.split(X_dummy, y_arr):
        splits.append((np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)))

    validate_cv_splits(splits, n_samples=len(y_arr), n_splits_expected=n_splits)
    return splits


def validate_cv_splits(
    splits: Sequence[Split],
    *,
    n_samples: int,
    n_splits_expected: Optional[int] = None,
) -> None:
    """
    Sanity checks:
      - correct number of folds (if provided)
      - each test fold disjoint
      - union of test indices covers all samples exactly once
      - train/test disjoint per fold
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if n_splits_expected is not None and len(splits) != int(n_splits_expected):
        raise ValueError(f"Expected {n_splits_expected} folds, got {len(splits)}.")

    seen_test = np.zeros(n_samples, dtype=int)

    for i, (tr, te) in enumerate(splits):
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)

        if tr.ndim != 1 or te.ndim != 1:
            raise ValueError(f"Fold {i}: indices must be 1D arrays.")
        if len(te) == 0:
            raise ValueError(f"Fold {i}: empty test fold.")
        if np.intersect1d(tr, te).size != 0:
            raise ValueError(f"Fold {i}: train/test overlap detected.")

        if tr.min() < 0 or te.min() < 0 or tr.max() >= n_samples or te.max() >= n_samples:
            raise ValueError(f"Fold {i}: indices out of bounds for n_samples={n_samples}.")

        # track test coverage
        for idx in te:
            seen_test[idx] += 1

    if not np.all(seen_test == 1):
        bad = np.where(seen_test != 1)[0]
        raise ValueError(
            "Test folds must cover every sample exactly once. "
            f"Violations at indices: {bad[:20].tolist()} (showing up to 20)."
        )


def cv_splits_fingerprint(splits: Sequence[Split]) -> str:
    """
    Stable hash identifying the exact fold partitioning.
    Useful to assert fold identity across experiments and for reproducibility logs.
    """
    h = hashlib.sha256()
    h.update(str(len(splits)).encode("utf-8"))
    for tr, te in splits:
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)
        h.update(tr.tobytes())
        h.update(b"|")
        h.update(te.tobytes())
        h.update(b";")
    return h.hexdigest()


def _final_estimator(pipeline_or_estimator):
    if hasattr(pipeline_or_estimator, "steps") and pipeline_or_estimator.steps:
        return pipeline_or_estimator.steps[-1][1]
    return pipeline_or_estimator


def _get_score_vector(pipeline_or_estimator, X, *, positive_label: int = 1) -> np.ndarray:
    """
    Get continuous scores for ROC-AUC:
      - predict_proba[:, pos] if available
      - else decision_function if available
      - else raise (to avoid meaningless AUC from hard labels)
    """
    if hasattr(pipeline_or_estimator, "predict_proba"):
        proba = pipeline_or_estimator.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba output must be [n_samples, 2+] for binary ROC-AUC.")
        est = _final_estimator(pipeline_or_estimator)
        classes = getattr(est, "classes_", None)
        if classes is None:
            # fallback: assume column 1 is positive
            pos_idx = 1
        else:
            classes = np.asarray(classes)
            if positive_label in set(classes.tolist()):
                pos_idx = int(np.where(classes == positive_label)[0][0])
            else:
                pos_idx = 1
        return proba[:, pos_idx].astype(float)

    if hasattr(pipeline_or_estimator, "decision_function"):
        s = pipeline_or_estimator.decision_function(X)
        return np.asarray(s, dtype=float).ravel()

    raise ValueError("Estimator must expose predict_proba or decision_function to compute ROC-AUC.")


@dataclass(frozen=True)
class CVEvaluationResult:
    n_splits: int
    split_fingerprint: str
    fold_indices: List[Split]
    fold_scores: Dict[str, np.ndarray]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    oof_pred: Optional[np.ndarray] = None
    oof_score: Optional[np.ndarray] = None


def evaluate_binary_pipeline_cv(
    pipeline,
    X,
    y: Iterable[int],
    *,
    cv_splits: Sequence[Split],
    positive_label: int = 1,
    metrics: Sequence[str] = ("roc_auc", "accuracy", "precision", "recall", "f1", "balanced_accuracy"),
    return_oof: bool = True,
) -> CVEvaluationResult:
    """
    Manual CV loop to guarantee:
      - identical folds (cv_splits provided)
      - per-fold metric vectors (needed later for Wilcoxon)
      - optional out-of-fold predictions/scores (useful for error analysis / XAI)
    """
    y_arr = np.asarray(list(y), dtype=int)
    n = len(y_arr)
    validate_cv_splits(cv_splits, n_samples=n)
    fp = cv_splits_fingerprint(cv_splits)

    wanted = set(metrics)
    allowed = {"roc_auc", "accuracy", "precision", "recall", "f1", "balanced_accuracy"}
    if not wanted.issubset(allowed):
        raise ValueError(f"Unsupported metrics requested: {sorted(list(wanted - allowed))}")

    fold_scores: Dict[str, List[float]] = {m: [] for m in metrics}

    oof_pred = np.full(n, fill_value=-1, dtype=int) if return_oof else None
    oof_score = np.full(n, fill_value=np.nan, dtype=float) if return_oof else None

    for tr, te in cv_splits:
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)

        model = clone(pipeline)
        model.fit(_safe_index(X, tr), y_arr[tr])

        y_pred = np.asarray(model.predict(_safe_index(X, te)), dtype=int)

        if "roc_auc" in wanted:
            y_score = _get_score_vector(model, _safe_index(X, te), positive_label=positive_label)
            fold_scores["roc_auc"].append(float(roc_auc_score(y_arr[te], y_score)))
        else:
            y_score = None

        if "accuracy" in wanted:
            fold_scores["accuracy"].append(float(accuracy_score(y_arr[te], y_pred)))
        if "precision" in wanted:
            fold_scores["precision"].append(float(precision_score(y_arr[te], y_pred, pos_label=positive_label, zero_division=0)))
        if "recall" in wanted:
            fold_scores["recall"].append(float(recall_score(y_arr[te], y_pred, pos_label=positive_label, zero_division=0)))
        if "f1" in wanted:
            fold_scores["f1"].append(float(f1_score(y_arr[te], y_pred, pos_label=positive_label, zero_division=0)))
        if "balanced_accuracy" in wanted:
            fold_scores["balanced_accuracy"].append(float(balanced_accuracy_score(y_arr[te], y_pred)))

        if return_oof:
            oof_pred[te] = y_pred
            if y_score is not None:
                oof_score[te] = np.asarray(y_score, dtype=float)

    fold_scores_arr: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in fold_scores.items()}
    mean_scores = {k: float(np.mean(v)) for k, v in fold_scores_arr.items()}
    std_scores = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in fold_scores_arr.items()}

    return CVEvaluationResult(
        n_splits=len(cv_splits),
        split_fingerprint=fp,
        fold_indices=[(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)) for tr, te in cv_splits],
        fold_scores=fold_scores_arr,
        mean_scores=mean_scores,
        std_scores=std_scores,
        oof_pred=oof_pred,
        oof_score=oof_score,
    )


def _safe_index(X, idx: np.ndarray):
    """
    Index X robustly whether it's a pandas DataFrame/Series or a numpy array / sparse matrix.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    return X[idx]
