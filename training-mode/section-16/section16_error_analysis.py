from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_auc_score

# -------------------------
# Missingness (consistent with your earlier sections)
# -------------------------
MISSING_TOKENS = ("", "?", "NA", "N/A", "NULL", "NAN", "UNKNOWN", "UNKNOWN/INVALID")

def is_missing(v: object, missing_tokens: Sequence[str] = MISSING_TOKENS) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return True
        return s.upper() in {t.upper() for t in missing_tokens}
    return False


def _safe_index(X, idx: np.ndarray):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    return X[idx]


def get_score_vector(estimator, X, *, positive_label: int = 1) -> np.ndarray:
    """
    Continuous scores for AUC and confidence:
    - predict_proba[:, pos] if available
    - else decision_function
    """
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba must return [n_samples, 2+] for binary tasks.")
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            pos_idx = 1
        else:
            classes = np.asarray(classes)
            pos_idx = int(np.where(classes == positive_label)[0][0]) if positive_label in set(classes.tolist()) else 1
        return proba[:, pos_idx].astype(float)

    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        return np.asarray(s, dtype=float).ravel()

    raise ValueError("Estimator must provide predict_proba or decision_function.")


def confidence_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    Model-agnostic confidence proxy:
    - if scores look like probabilities in [0,1], use |p-0.5|
    - else use |z| after standardization
    """
    s = np.asarray(scores, dtype=float).ravel()
    if np.all(s >= 0.0) and np.all(s <= 1.0):
        return np.abs(s - 0.5)
    mu = float(np.nanmean(s))
    sd = float(np.nanstd(s) + 1e-12)
    z = (s - mu) / sd
    return np.abs(z)


def row_missingness_counts(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    miss = df[feature_cols].applymap(is_missing)
    miss_count = miss.sum(axis=1).to_numpy(dtype=int)
    miss_rate = (miss_count / float(len(feature_cols))).astype(float)
    return miss_count, miss_rate


@dataclass(frozen=True)
class OOFResult:
    oof_pred: np.ndarray
    oof_score: np.ndarray
    auc: float
    cm: np.ndarray


def compute_oof_for_model(
    pipeline,
    X,
    y: np.ndarray,
    *,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    positive_label: int = 1,
) -> OOFResult:
    """
    Manual CV (no leakage): fit on TRAIN fold, predict on TEST fold.
    Produces OOF predictions and OOF scores for every sample.
    """
    y = np.asarray(y, dtype=int).ravel()
    n = len(y)

    oof_pred = np.full(n, -1, dtype=int)
    oof_score = np.full(n, np.nan, dtype=float)

    for tr, te in cv_splits:
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)

        model = clone(pipeline)
        model.fit(_safe_index(X, tr), y[tr])

        yhat = np.asarray(model.predict(_safe_index(X, te)), dtype=int)
        scr = get_score_vector(model, _safe_index(X, te), positive_label=positive_label)

        oof_pred[te] = yhat
        oof_score[te] = np.asarray(scr, dtype=float)

    if np.any(oof_pred < 0) or np.any(np.isnan(oof_score)):
        raise RuntimeError("OOF predictions/scores incomplete. Check your cv_splits coverage.")

    auc = float(roc_auc_score(y, oof_score))
    cm = confusion_matrix(y, oof_pred, labels=[0, 1])
    return OOFResult(oof_pred=oof_pred, oof_score=oof_score, auc=auc, cm=cm)


def select_systematic_failures(
    y: np.ndarray,
    *,
    best_model: str,
    oof_by_model: Dict[str, OOFResult],
    candidate_models: Sequence[str],
    k: int = 3,
    min_models: int = 2,
) -> List[int]:
    """
    "Systematic failure" = misclassified by best model AND misclassified by >= min_models-1 other models
    among candidate_models (top models).
    """
    y = np.asarray(y, dtype=int).ravel()

    best = oof_by_model[best_model]
    mis_best = (best.oof_pred != y)

    # consensus count among candidate models
    mis_counts = np.zeros_like(y, dtype=int)
    for m in candidate_models:
        r = oof_by_model[m]
        mis_counts += (r.oof_pred != y).astype(int)

    # need best model wrong AND consensus threshold
    keep = np.where(mis_best & (mis_counts >= int(min_models)))[0]

    if keep.size == 0:
        # relax: at least best model wrong (fallback)
        keep = np.where(mis_best)[0]

    # prioritize "high-confidence wrong" in best model
    conf = confidence_from_scores(best.oof_score)
    order = keep[np.argsort(-conf[keep])]  # descending confidence

    return order[: int(k)].astype(int).tolist()


def rare_category_flags(df: pd.DataFrame, idx: int, cols: Sequence[str], min_count: int = 50) -> Dict[str, Tuple[Any, int]]:
    out: Dict[str, Tuple[Any, int]] = {}
    for c in cols:
        if c not in df.columns:
            continue
        v = df.loc[idx, c]
        # bucket missing
        vv = "__MISSING__" if is_missing(v) else str(v)
        counts = df[c].astype("object").map(lambda x: "__MISSING__" if is_missing(x) else str(x)).value_counts()
        out[c] = (vv, int(counts.get(vv, 0)))
    # filter to only rare ones
    return {c: (v, n) for c, (v, n) in out.items() if n < min_count}


def hypothesize_case(
    df: pd.DataFrame,
    idx: int,
    *,
    feature_cols: Sequence[str],
    rare_cols: Sequence[str],
    rare_min_count: int = 50,
) -> List[str]:
    reasons: List[str] = []

    miss_count, miss_rate = row_missingness_counts(df.loc[[idx]], feature_cols)
    miss_count = int(miss_count[0])
    miss_rate = float(miss_rate[0])
    if miss_rate >= 0.30:
        reasons.append(f"High missingness: {miss_count}/{len(feature_cols)} features missing ({miss_rate:.1%}).")

    rares = rare_category_flags(df, idx, rare_cols, min_count=rare_min_count)
    if rares:
        msg = "Rare categories: " + ", ".join([f"{c}={v} (count={n})" for c, (v, n) in rares.items()])
        reasons.append(msg)

    # simple clinically-inspired checks if engineered features exist
    if "a1c_high" in df.columns and "insulin_active" in df.columns:
        a1c_high = int(pd.to_numeric(df.loc[idx, "a1c_high"], errors="coerce") or 0)
        ins_act = int(pd.to_numeric(df.loc[idx, "insulin_active"], errors="coerce") or 0)
        if a1c_high == 1 and ins_act == 0:
            reasons.append("Possible lab/therapy mismatch: A1C high but insulin not active (could confuse model).")

    if "total_visits" in df.columns and "time_in_hospital" in df.columns:
        tv = float(pd.to_numeric(df.loc[idx, "total_visits"], errors="coerce") or 0.0)
        tih = float(pd.to_numeric(df.loc[idx, "time_in_hospital"], errors="coerce") or 0.0)
        if tv >= 5 and tih <= 3:
            reasons.append("Utilization pattern: high prior visits but short stay (non-typical combination).")

    if "n_changed_diabetes_meds" in df.columns:
        nch = float(pd.to_numeric(df.loc[idx, "n_changed_diabetes_meds"], errors="coerce") or 0.0)
        if nch >= 2:
            reasons.append("Multiple medication changes (complex management) may increase uncertainty.")

    return reasons


# -------------------------
# Plotting helpers (matplotlib, no seaborn required)
# -------------------------
def plot_confusion_matrix(cm: np.ndarray, *, title: str = "Confusion matrix (OOF)") -> None:
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=int)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.show()


def plot_fp_fn_by_fold(
    y: np.ndarray,
    oof_pred: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    title: str = "False Positives / False Negatives by fold",
) -> None:
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=int).ravel()
    oof_pred = np.asarray(oof_pred, dtype=int).ravel()

    fps, fns = [], []
    for _, te in cv_splits:
        te = np.asarray(te, dtype=np.int64)
        yt = y[te]
        yp = oof_pred[te]
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fps.append(fp)
        fns.append(fn)

    x = np.arange(len(fps))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x, fps, label="FP")
    ax.bar(x, fns, bottom=fps, label="FN")
    ax.set_title(title)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Count")
    ax.legend()
    plt.show()


def plot_missingness_distributions(
    df: pd.DataFrame,
    y: np.ndarray,
    oof_pred: np.ndarray,
    *,
    feature_cols: Sequence[str],
    title: str = "Row missingness: correct vs misclassified",
) -> None:
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=int).ravel()
    oof_pred = np.asarray(oof_pred, dtype=int).ravel()
    mis = (oof_pred != y)

    miss_count, miss_rate = row_missingness_counts(df, feature_cols)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(miss_rate[~mis], bins=30, alpha=0.7, label="Correct")
    ax.hist(miss_rate[mis], bins=30, alpha=0.7, label="Misclassified")
    ax.set_title(title)
    ax.set_xlabel("Missing-rate per row")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()


def plot_case_profile(
    df: pd.DataFrame,
    idx: int,
    *,
    numeric_cols: Sequence[str],
    title: Optional[str] = None,
) -> None:
    """
    Compare case numeric features vs dataset medians (overall + by class if available).
    """
    import matplotlib.pyplot as plt

    present = [c for c in numeric_cols if c in df.columns]
    if not present:
        return

    vals = []
    meds = []
    for c in present:
        v = pd.to_numeric(df.loc[idx, c], errors="coerce")
        vals.append(float(v) if not pd.isna(v) else np.nan)
        meds.append(float(pd.to_numeric(df[c], errors="coerce").median(skipna=True)))

    x = np.arange(len(present))
    fig = plt.figure(figsize=(max(8, len(present) * 0.6), 4))
    ax = fig.add_subplot(111)
    ax.bar(x - 0.2, vals, width=0.4, label="Case")
    ax.bar(x + 0.2, meds, width=0.4, label="Dataset median")
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title or f"Case profile idx={idx}")
    ax.legend()
    plt.tight_layout()
    plt.show()
