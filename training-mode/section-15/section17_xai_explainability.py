from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union, Literal, Callable

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Section 17 — XAI / Explainability (Global + Local)
#
# Goals:
# - Global importance (overall + by subgroup) using a robust method that works
#   for ANY pipeline: permutation importance on RAW input columns.
# - Global importance (embedded/model-specific) when available:
#     * Linear models: |coef|
#     * Tree-based: feature_importances_
#   (Only when we can map transformed feature names safely.)
# - Local explanations for specific instances:
#     * Transparent models: linear contribution table, decision-tree rules
#     * Opaque models: SHAP (optional external library; installed separately)
#
# Notes:
# - We intentionally support RAW-level permutation importance because the
#   diabetes dataset has many one-hot features and custom transformers,
#   making transformed feature-name reconstruction brittle.
# - IMPORTANT: We DO NOT use make_scorer(needs_threshold=...) because in some
#   sklearn versions that kwarg can be forwarded to roc_auc_score and crash.
#   Instead we pass built-in scorer strings like "roc_auc".
# ---------------------------------------------------------------------


class XAIError(RuntimeError):
    """Raised when an XAI operation cannot be completed."""


Split = Tuple[np.ndarray, np.ndarray]
ImportanceLevel = Literal["raw", "transformed"]
GlobalMethod = Literal["auto", "permutation_raw", "embedded_transformed"]
ScoringType = Union[str, Callable[..., float]]


# ------------------------------ small utils ------------------------------


def _safe_index(X, idx: np.ndarray):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    return X[idx]


def _ensure_1d_binary(y: Iterable[int]) -> np.ndarray:
    y_arr = np.asarray(list(y), dtype=int).ravel()
    if y_arr.size == 0:
        raise ValueError("y is empty.")
    uniq = set(np.unique(y_arr).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"Binary target must be in {{0,1}}. Found: {sorted(list(uniq))}")
    return y_arr


def _has_two_classes(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2


def _is_probability_like(scores: np.ndarray) -> bool:
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        return False
    return float(np.nanmin(scores)) >= 0.0 and float(np.nanmax(scores)) <= 1.0


def _get_feature_pipeline_and_estimator(pipeline: Any) -> Tuple[Optional[Pipeline], Any]:
    """
    If input is a sklearn Pipeline, split into:
      - feature_pipe = all steps except last
      - estimator = last step
    Else:
      - feature_pipe = None
      - estimator = pipeline
    """
    if isinstance(pipeline, Pipeline) and len(pipeline.steps) >= 2:
        feature_pipe = Pipeline(steps=pipeline.steps[:-1])
        estimator = pipeline.steps[-1][1]
        return feature_pipe, estimator
    return None, pipeline


def _get_decision_scores(fitted_model: Any, X) -> np.ndarray:
    """
    Decision scores for ROC-AUC and confidence ordering:
      - predict_proba[:, pos] if available
      - decision_function if available
      - else raise
    """
    if hasattr(fitted_model, "predict_proba"):
        proba = np.asarray(fitted_model.predict_proba(X), dtype=float)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise XAIError("predict_proba must return [n_samples, 2+] for binary tasks.")
        # positive class index (best-effort)
        classes = getattr(fitted_model, "classes_", None)
        if classes is None:
            pos_idx = 1
        else:
            classes = np.asarray(classes)
            pos_idx = int(np.where(classes == 1)[0][0]) if 1 in set(classes.tolist()) else 1
        return proba[:, pos_idx]

    if hasattr(fitted_model, "decision_function"):
        s = np.asarray(fitted_model.decision_function(X), dtype=float).ravel()
        return s

    raise XAIError("Model must expose predict_proba or decision_function to get decision scores.")


# ------------------------------ results types ------------------------------


@dataclass(frozen=True)
class GlobalImportanceResult:
    level: ImportanceLevel
    method: str
    importance: pd.DataFrame  # columns: feature, importance_mean, importance_std
    fold_importances: Optional[pd.DataFrame] = None  # optional: fold x feature table


@dataclass(frozen=True)
class HardErrorCase:
    index: int
    y_true: int
    y_pred: int
    error_type: str  # "FP" or "FN"
    score: float
    confidence: float


@dataclass(frozen=True)
class LocalLinearExplanation:
    index: int
    pred_label: int
    pred_score: float
    base_value: float
    contributions: pd.DataFrame  # feature, value, coef, contribution


@dataclass(frozen=True)
class LocalTreeExplanation:
    index: int
    pred_label: int
    pred_score: float
    rules_text: str


@dataclass(frozen=True)
class LocalShapExplanation:
    index: int
    pred_label: int
    pred_score: float
    shap_top: pd.DataFrame  # feature, shap_value
    base_value: Optional[float] = None


# ------------------------------ selecting “hard errors” ------------------------------


def select_hard_errors_from_oof(
    *,
    y_true: Union[np.ndarray, Sequence[int]],
    y_pred: Union[np.ndarray, Sequence[int]],
    y_score: Union[np.ndarray, Sequence[float]],
    top_n: int = 3,
) -> List[HardErrorCase]:
    """
    Pick the “hardest” misclassifications from out-of-fold predictions:
    - wrong predictions
    - highest confidence (far from threshold)

    If y_score is probability-like -> threshold=0.5
    Else (margin) -> threshold=0.0
    """
    yt = _ensure_1d_binary(y_true)
    yp = np.asarray(y_pred, dtype=int).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()

    if yt.size != yp.size or yt.size != ys.size:
        raise ValueError("y_true, y_pred, y_score must have the same length.")

    prob_like = _is_probability_like(ys)
    thresh = 0.5 if prob_like else 0.0

    wrong = (yp != yt)
    idxs = np.where(wrong)[0].tolist()
    if not idxs:
        return []

    cases: List[HardErrorCase] = []
    for i in idxs:
        score = float(ys[i])
        confidence = abs(score - thresh)
        err_type = "FP" if (yt[i] == 0 and yp[i] == 1) else "FN"
        cases.append(
            HardErrorCase(
                index=int(i),
                y_true=int(yt[i]),
                y_pred=int(yp[i]),
                error_type=err_type,
                score=score,
                confidence=float(confidence),
            )
        )

    cases.sort(key=lambda c: c.confidence, reverse=True)
    return cases[: int(top_n)]


# ------------------------------ GLOBAL: permutation importance on RAW columns ------------------------------


def compute_global_importance_permutation_cv_raw(
    pipeline: Any,
    X: pd.DataFrame,
    y: Iterable[int],
    *,
    cv_splits: Sequence[Split],
    scoring: ScoringType = "roc_auc",
    n_repeats: int = 3,
    random_state: int = 42,
    max_features: Optional[int] = 30,
) -> GlobalImportanceResult:
    """
    Robust global importance that works for ANY sklearn pipeline:
    permutation importance on RAW input columns within each CV fold.

    Returns mean ± std importance across folds.

    IMPORTANT: Use sklearn's built-in scorer strings (e.g. "roc_auc")
    instead of make_scorer(needs_threshold=...), to avoid API mismatches.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame for RAW-level permutation importance.")
    y_arr = _ensure_1d_binary(y)

    if len(cv_splits) == 0:
        raise ValueError("cv_splits is empty.")

    if scoring is None:
        scoring = "roc_auc"

    fold_imps: List[np.ndarray] = []
    used_folds = 0

    for tr, te in cv_splits:
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)

        model = clone(pipeline)
        model.fit(_safe_index(X, tr), y_arr[tr])

        y_te = y_arr[te]
        if not _has_two_classes(y_te):
            # Can happen for tiny datasets or subgroup subsets; skip the fold
            continue

        pi = permutation_importance(
            model,
            _safe_index(X, te),
            y_te,
            scoring=scoring,  # ✅ string/callable scorer (no needs_threshold kwarg)
            n_repeats=int(n_repeats),
            random_state=int(random_state),
            n_jobs=None,
        )
        fold_imps.append(np.asarray(pi.importances_mean, dtype=float))
        used_folds += 1

    if used_folds == 0:
        raise XAIError("No fold had both classes in the test split; cannot compute permutation importance.")

    M = np.vstack(fold_imps)  # [n_folds_used, n_features]
    mean_imp = M.mean(axis=0)
    std_imp = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mean_imp)

    df_imp = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": mean_imp,
            "importance_std": std_imp,
        }
    ).sort_values("importance_mean", ascending=False)

    if max_features is not None:
        df_imp = df_imp.head(int(max_features)).reset_index(drop=True)
    else:
        df_imp = df_imp.reset_index(drop=True)

    fold_df = pd.DataFrame(M, columns=list(X.columns))
    return GlobalImportanceResult(
        level="raw",
        method="permutation_cv_raw",
        importance=df_imp,
        fold_importances=fold_df,
    )


def compute_global_importance_permutation_raw(
    fitted_pipeline: Any,
    X: pd.DataFrame,
    y: Iterable[int],
    *,
    scoring: ScoringType = "roc_auc",
    n_repeats: int = 5,
    random_state: int = 42,
    max_features: Optional[int] = 30,
) -> GlobalImportanceResult:
    """
    Single-fit permutation importance (no CV aggregation).
    Useful for subgroup analysis once you have a fitted model.

    IMPORTANT: Use sklearn's built-in scorer strings (e.g. "roc_auc")
    instead of make_scorer(needs_threshold=...), to avoid API mismatches.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    y_arr = _ensure_1d_binary(y)
    if not _has_two_classes(y_arr):
        raise ValueError("Need at least two classes to compute ROC-AUC permutation importance.")

    if scoring is None:
        scoring = "roc_auc"

    pi = permutation_importance(
        fitted_pipeline,
        X,
        y_arr,
        scoring=scoring,  # ✅ string/callable scorer (no needs_threshold kwarg)
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        n_jobs=None,
    )

    mean_imp = np.asarray(pi.importances_mean, dtype=float)
    std_imp = np.asarray(pi.importances_std, dtype=float)

    df_imp = pd.DataFrame(
        {"feature": list(X.columns), "importance_mean": mean_imp, "importance_std": std_imp}
    ).sort_values("importance_mean", ascending=False)

    if max_features is not None:
        df_imp = df_imp.head(int(max_features)).reset_index(drop=True)
    else:
        df_imp = df_imp.reset_index(drop=True)

    return GlobalImportanceResult(
        level="raw",
        method="permutation_raw",
        importance=df_imp,
        fold_importances=None,
    )


# ------------------------------ GLOBAL: embedded/model-specific on TRANSFORMED features (best-effort) ------------------------------


def _try_get_transformed_feature_names_from_section5_preprocessor(
    fitted_pipeline: Any,
) -> Optional[List[str]]:
    """
    Best-effort feature-name extraction for your Section 5 preprocessor structure:
      Pipeline(pre -> ColumnTransformer(features=...))
    Works even with custom transformers by using known block behavior:
      - num: 1 feature per numeric column
      - ord: 1 feature per ordinal column
      - cat: use OneHotEncoder categories_

    Returns None if structure is not compatible.
    """
    if not isinstance(fitted_pipeline, Pipeline):
        return None
    if "pre" not in fitted_pipeline.named_steps:
        return None

    pre = fitted_pipeline.named_steps["pre"]
    if not isinstance(pre, Pipeline):
        return None
    if "features" not in pre.named_steps:
        return None

    ct = pre.named_steps["features"]
    if not hasattr(ct, "transformers_"):
        return None

    names: List[str] = []
    for block_name, trans, cols in ct.transformers_:
        if trans == "drop":
            continue
        cols = list(cols) if isinstance(cols, (list, tuple, np.ndarray)) else [cols]

        if block_name == "num":
            names.extend([f"num__{c}" for c in cols])
            continue

        if block_name == "ord":
            names.extend([f"ord__{c}" for c in cols])
            continue

        if block_name == "cat":
            # Expect a Pipeline with a OneHotEncoder at named_steps["onehot"]
            try:
                onehot = trans.named_steps["onehot"]
            except Exception:
                return None

            # Preferred
            if hasattr(onehot, "get_feature_names_out"):
                try:
                    outn = onehot.get_feature_names_out(cols).tolist()
                    names.extend([f"cat__{n}" for n in outn])
                    continue
                except Exception:
                    pass

            # Fallback using categories_
            cats = getattr(onehot, "categories_", None)
            if cats is None:
                return None
            for c, cat_list in zip(cols, cats):
                for v in cat_list:
                    names.append(f"cat__{c}={v}")
            continue

        # Unknown block: cannot safely infer
        return None

    return names if names else None


def compute_global_importance_embedded_transformed(
    fitted_pipeline: Any,
    X_train: pd.DataFrame,
    y_train: Iterable[int],
    *,
    max_features: int = 30,
) -> GlobalImportanceResult:
    """
    Uses model-internal importance on transformed features when possible:
      - Linear models: abs(coef_)
      - Tree models: feature_importances_

    Requires:
      - fitted_pipeline is a sklearn Pipeline with steps ending in an estimator
      - we can infer transformed feature names (best-effort)
    """
    if not isinstance(fitted_pipeline, Pipeline):
        raise XAIError("Embedded transformed importance requires a sklearn Pipeline.")

    # Ensure it is fitted (if not, fit it quickly)
    try:
        _ = fitted_pipeline.predict(X_train.head(2))
    except Exception:
        fitted_pipeline.fit(X_train, _ensure_1d_binary(y_train))

    feature_pipe, estimator = _get_feature_pipeline_and_estimator(fitted_pipeline)
    if feature_pipe is None:
        raise XAIError("Pipeline too short to compute transformed importance.")

    names = _try_get_transformed_feature_names_from_section5_preprocessor(fitted_pipeline)
    if names is None:
        raise XAIError(
            "Could not infer transformed feature names (likely due to a non-Section5 preprocessor or custom steps). "
            "Use RAW permutation importance instead."
        )

    # Transform X to match estimator feature space
    Xt = feature_pipe.transform(X_train)
    n_features = Xt.shape[1]
    if len(names) != n_features:
        raise XAIError(
            f"Transformed feature-name length mismatch: names={len(names)} vs Xt_dim={n_features}."
        )

    # Fit estimator alone (to ensure coef_/feature_importances_ exist and match Xt)
    est = clone(estimator)
    est.fit(Xt, _ensure_1d_binary(y_train))

    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[0]
        imp = np.abs(coef)
        method = "embedded_abs_coef"
    elif hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        method = "embedded_tree_importance"
    else:
        raise XAIError("Estimator has neither coef_ nor feature_importances_.")

    df_imp = pd.DataFrame(
        {"feature": names, "importance_mean": imp, "importance_std": np.zeros_like(imp)}
    ).sort_values("importance_mean", ascending=False)

    df_imp = df_imp.head(int(max_features)).reset_index(drop=True)

    return GlobalImportanceResult(
        level="transformed",
        method=method,
        importance=df_imp,
        fold_importances=None,
    )


# ------------------------------ LOCAL: transparent explanations ------------------------------


def explain_instance_linear(
    fitted_pipeline: Any,
    X_row: pd.DataFrame,
    *,
    top_k: int = 15,
) -> LocalLinearExplanation:
    """
    Local explanation for linear classifiers: contribution = value * coef (in transformed space).
    Works best for LogisticRegression / linear SVM with coef_.
    """
    if not isinstance(fitted_pipeline, Pipeline):
        raise XAIError("Linear local explanation expects a fitted sklearn Pipeline.")
    if not isinstance(X_row, pd.DataFrame) or len(X_row) != 1:
        raise ValueError("X_row must be a single-row DataFrame.")

    # Split features vs estimator
    feature_pipe, estimator = _get_feature_pipeline_and_estimator(fitted_pipeline)
    if feature_pipe is None:
        raise XAIError("Pipeline too short for linear explanation.")

    # Predict label & score from full pipeline (raw)
    pred_label = int(fitted_pipeline.predict(X_row)[0])
    try:
        pred_score = float(_get_decision_scores(fitted_pipeline, X_row)[0])
    except Exception:
        pred_score = float("nan")

    # Names in transformed space (best-effort)
    names = _try_get_transformed_feature_names_from_section5_preprocessor(fitted_pipeline)
    if names is None:
        raise XAIError(
            "Could not infer transformed feature names. "
            "If this is not a Section5-based pipeline, use SHAP for local explanations."
        )

    Xt = feature_pipe.transform(X_row)  # (1, d)
    Xt = np.asarray(Xt.todense() if hasattr(Xt, "todense") else Xt, dtype=float)
    if Xt.ndim != 2 or Xt.shape[0] != 1:
        raise XAIError("Unexpected transformed shape for X_row.")

    # Access fitted estimator directly
    fitted_est = fitted_pipeline.steps[-1][1]
    if not hasattr(fitted_est, "coef_"):
        raise XAIError("Final estimator has no coef_. Not a linear model with coefficients.")

    coef = np.asarray(fitted_est.coef_, dtype=float)
    coef = coef[0] if coef.ndim == 2 else coef
    if coef.shape[0] != Xt.shape[1]:
        raise XAIError("coef_ dimension does not match transformed feature dimension.")

    intercept = float(np.asarray(getattr(fitted_est, "intercept_", [0.0]), dtype=float).ravel()[0])

    vals = Xt.ravel()
    contrib = vals * coef

    df = pd.DataFrame(
        {
            "feature": names,
            "value": vals,
            "coef": coef,
            "contribution": contrib,
        }
    )
    df["abs_contribution"] = df["contribution"].abs()
    df = df.sort_values("abs_contribution", ascending=False).drop(columns=["abs_contribution"])
    df = df.head(int(top_k)).reset_index(drop=True)

    return LocalLinearExplanation(
        index=int(X_row.index[0]),
        pred_label=pred_label,
        pred_score=float(pred_score),
        base_value=intercept,
        contributions=df,
    )


def explain_instance_decision_tree(
    fitted_pipeline: Any,
    X_row: pd.DataFrame,
    *,
    max_depth: int = 4,
) -> LocalTreeExplanation:
    """
    Local explanation for a DecisionTreeClassifier: exports text rules.
    Requires the final estimator to be DecisionTreeClassifier-like.
    """
    if not isinstance(fitted_pipeline, Pipeline):
        raise XAIError("Tree local explanation expects a fitted sklearn Pipeline.")
    if not isinstance(X_row, pd.DataFrame) or len(X_row) != 1:
        raise ValueError("X_row must be a single-row DataFrame.")

    pred_label = int(fitted_pipeline.predict(X_row)[0])
    try:
        pred_score = float(_get_decision_scores(fitted_pipeline, X_row)[0])
    except Exception:
        pred_score = float("nan")

    feature_pipe, _ = _get_feature_pipeline_and_estimator(fitted_pipeline)
    if feature_pipe is None:
        raise XAIError("Pipeline too short for tree explanation.")
    Xt = feature_pipe.transform(X_row)

    names = _try_get_transformed_feature_names_from_section5_preprocessor(fitted_pipeline)
    if names is None:
        raise XAIError("Could not infer transformed feature names for tree rule export.")

    tree_est = fitted_pipeline.steps[-1][1]
    if not hasattr(tree_est, "tree_"):
        raise XAIError("Final estimator does not look like a DecisionTreeClassifier.")

    try:
        from sklearn.tree import export_text
        _ = np.asarray(Xt.todense() if hasattr(Xt, "todense") else Xt, dtype=float)
        rules = export_text(tree_est, feature_names=names, max_depth=int(max_depth))
    except Exception as e:
        raise XAIError(f"Failed to export decision-tree rules: {e}") from e

    return LocalTreeExplanation(
        index=int(X_row.index[0]),
        pred_label=pred_label,
        pred_score=float(pred_score),
        rules_text=str(rules),
    )


# ------------------------------ LOCAL: SHAP (opaque models) ------------------------------


def stratified_sample_indices(
    y: Iterable[int],
    *,
    n: int = 200,
    random_state: int = 42,
) -> np.ndarray:
    """
    Stratified sampling indices for background sets (SHAP).
    """
    y_arr = _ensure_1d_binary(y)
    n = int(n)
    rng = np.random.RandomState(int(random_state))

    idx0 = np.where(y_arr == 0)[0]
    idx1 = np.where(y_arr == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        # fall back to uniform
        take = min(n, len(y_arr))
        return rng.choice(np.arange(len(y_arr)), size=take, replace=False)

    n0 = max(1, int(round(n * (len(idx0) / len(y_arr)))))
    n1 = max(1, n - n0)

    n0 = min(n0, len(idx0))
    n1 = min(n1, len(idx1))

    s0 = rng.choice(idx0, size=n0, replace=False)
    s1 = rng.choice(idx1, size=n1, replace=False)
    out = np.concatenate([s0, s1])
    rng.shuffle(out)
    return out


def explain_instance_shap(
    fitted_pipeline: Any,
    X_background: pd.DataFrame,
    X_row: pd.DataFrame,
    *,
    top_k: int = 15,
) -> LocalShapExplanation:
    """
    SHAP local explanation on RAW features (works as black-box for pipelines).

    Requires external dependency:
      pip install shap

    We use shap.Explainer with a background dataset (masker).
    """
    if not isinstance(X_background, pd.DataFrame) or len(X_background) < 5:
        raise ValueError("X_background must be a DataFrame with at least 5 rows.")
    if not isinstance(X_row, pd.DataFrame) or len(X_row) != 1:
        raise ValueError("X_row must be a single-row DataFrame.")

    try:
        import shap  # type: ignore
    except Exception as e:
        raise XAIError(
            "SHAP is not installed. Install it with: pip install shap"
        ) from e

    pred_label = int(fitted_pipeline.predict(X_row)[0])
    try:
        pred_score = float(_get_decision_scores(fitted_pipeline, X_row)[0])
    except Exception:
        pred_score = float("nan")

    # Build explainer (black-box over the pipeline)
    explainer = shap.Explainer(fitted_pipeline, X_background)

    sv = explainer(X_row)  # shap.Explanation
    values = sv.values
    base_value = None

    if isinstance(sv.base_values, np.ndarray):
        bv = sv.base_values
        base_value = float(np.ravel(bv)[0])

    # normalize to (n_features,)
    if np.asarray(values).ndim == 3:
        values_ = np.asarray(values)[0, :, -1]  # positive class if present
    else:
        values_ = np.asarray(values)[0, :]

    feat_names = list(X_row.columns)
    df = pd.DataFrame({"feature": feat_names, "shap_value": values_.astype(float)})
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns=["abs"]).head(int(top_k)).reset_index(drop=True)

    return LocalShapExplanation(
        index=int(X_row.index[0]),
        pred_label=pred_label,
        pred_score=float(pred_score),
        shap_top=df,
        base_value=base_value,
    )


# ------------------------------ plotting helpers (matplotlib only) ------------------------------


def plot_global_importance_bar(
    imp: pd.DataFrame,
    *,
    title: str = "Global feature importance",
    max_features: int = 20,
):
    """
    Simple matplotlib bar plot.
    """
    import matplotlib.pyplot as plt

    df = imp.copy()
    df = df.head(int(max_features))

    plt.figure()
    plt.barh(df["feature"][::-1], df["importance_mean"][::-1])
    plt.xlabel("Importance (mean)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_local_contributions_bar(
    contrib: pd.DataFrame,
    *,
    title: str = "Local linear contributions (top features)",
):
    """
    Simple matplotlib bar plot for linear local explanations.
    """
    import matplotlib.pyplot as plt

    df = contrib.copy()
    plt.figure()
    plt.barh(df["feature"][::-1], df["contribution"][::-1])
    plt.xlabel("Contribution (value × coef)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
