from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import section5_preprocessing_pipeline as s5


# -----------------------------
# Small utilities
# -----------------------------
class SparseToCSC(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to CSC (tree estimators often prefer CSC)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            from scipy import sparse
        except Exception as e:
            raise RuntimeError("scipy is required. Install with: pip install scipy") from e
        return X.tocsc() if sparse.issparse(X) else X


def _safe_series_mode(s: pd.Series):
    s2 = s.dropna()
    if s2.empty:
        return np.nan
    return s2.mode(dropna=True).iloc[0]


def _predict_score_pos1(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return continuous score for positive class (1): proba if available else decision_function."""
    if hasattr(pipe, "predict_proba"):
        proba = np.asarray(pipe.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba must return [n_samples, 2+].")
        # assume class 1 is the positive (your target is 0/1)
        # sklearn orders classes_ ascending, so column for class 1 is usually index 1
        return proba[:, 1].astype(float)

    if hasattr(pipe, "decision_function"):
        return np.asarray(pipe.decision_function(X), dtype=float).ravel()

    raise ValueError("Pipeline must expose predict_proba or decision_function.")


def _get_preprocessed_feature_names(pre: Pipeline) -> List[str]:
    """
    Robust feature-name extraction for your Section 5 preprocessor:
      - num: num__{col}
      - ord: ord__{col}
      - cat: onehot feature names if available, else cat__{col} (fallback)
    """
    if "features" not in pre.named_steps:
        raise ValueError("Expected preprocessor Pipeline to contain a 'features' ColumnTransformer step.")

    ct = pre.named_steps["features"]

    # ColumnTransformer fitted transformers live in ct.transformers_
    names: List[str] = []
    for name, trans, cols in ct.transformers_:
        cols = list(cols) if isinstance(cols, (list, tuple)) else [cols]

        if name == "num":
            names.extend([f"num__{c}" for c in cols])
            continue

        if name == "ord":
            # OrdinalEncoder outputs 1 column per input column
            names.extend([f"ord__{c}" for c in cols])
            continue

        if name == "cat":
            # Expect Pipeline(..., ("onehot", OneHotEncoder))
            try:
                onehot = ct.named_transformers_["cat"].named_steps["onehot"]
                # sklearn >= 1.0
                oh_names = list(onehot.get_feature_names_out(cols))
                names.extend([f"cat__{n}" for n in oh_names])
            except Exception:
                # fallback: one name per raw categorical col (less precise)
                names.extend([f"cat__{c}" for c in cols])
            continue

        # any other blocks
        names.extend([f"{name}__{c}" for c in cols])

    return names


def _plot_top_barh(df_imp: pd.DataFrame, *, title: str, top_n: int = 25):
    if df_imp is None or df_imp.empty:
        print(f"[plot skipped] {title}: empty importance table.")
        return
    d = df_imp.head(top_n).iloc[::-1]  # reverse for barh nice ordering
    plt.figure(figsize=(10, max(4, int(0.22 * len(d) + 2))))
    plt.barh(d["feature"].astype(str), d["importance"].astype(float))
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def _global_importance_logreg(pipe: Pipeline, feature_names: List[str], top_n: int = 30) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    coef = np.asarray(clf.coef_).ravel()
    if len(coef) != len(feature_names):
        raise ValueError(f"coef dim {len(coef)} != feature_names dim {len(feature_names)}")

    imp = np.abs(coef)
    out = pd.DataFrame(
        {"feature": feature_names, "importance": imp, "coef": coef}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return out.head(top_n)


def _global_importance_tree(pipe: Pipeline, feature_names: List[str], top_n: int = 30) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise ValueError("Tree-based estimator has no feature_importances_.")
    imp = np.asarray(clf.feature_importances_, dtype=float).ravel()
    if len(imp) != len(feature_names):
        raise ValueError(f"feature_importances dim {len(imp)} != feature_names dim {len(feature_names)}")

    out = pd.DataFrame({"feature": feature_names, "importance": imp})
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    return out.head(top_n)


def _compute_elderly_from_age_bucket(age_series: pd.Series) -> pd.Series:
    """
    If you don't have 'elderly' yet, infer from 'age' buckets:
      elderly = midpoint(age) >= 65
    """
    def midpoint(v) -> float:
        if v is None:
            return np.nan
        s = str(v).strip()
        if not s.startswith("[") or "-" not in s:
            return np.nan
        try:
            inside = s.strip("[]()")
            a, b = inside.split("-")
            return (float(a) + float(b)) / 2.0
        except Exception:
            return np.nan

    mid = age_series.map(midpoint)
    return (mid.fillna(-1) >= 65).astype(int)


def _perm_importance_raw_cols(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    top_n: int = 25,
) -> pd.DataFrame:
    # Need both classes for ROC-AUC
    if len(np.unique(y_test)) < 2:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    r = permutation_importance(
        pipe,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        n_jobs=None,
    )
    out = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "importance_mean": r.importances_mean.astype(float),
            "importance_std": r.importances_std.astype(float),
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return out.head(top_n)


def _pick_top_confident_errors(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    k: int = 3,
) -> pd.DataFrame:
    scores = _predict_score_pos1(pipe, X_test)
    pred = (scores >= 0.5).astype(int)
    wrong = pred != y_test

    if wrong.sum() == 0:
        return pd.DataFrame(columns=["row_ix", "y_true", "y_pred", "score", "confidence"])

    confidence = np.abs(scores - 0.5)
    idx = np.where(wrong)[0]
    # pick the most confident wrong
    idx_sorted = idx[np.argsort(confidence[idx])[::-1]]
    idx_pick = idx_sorted[: int(k)]

    return pd.DataFrame(
        {
            "row_ix": idx_pick,
            "y_true": y_test[idx_pick],
            "y_pred": pred[idx_pick],
            "score": scores[idx_pick],
            "confidence": confidence[idx_pick],
        }
    ).sort_values("confidence", ascending=False).reset_index(drop=True)


def _local_explain_logreg_exact(
    pipe: Pipeline,
    X_row: pd.DataFrame,
    feature_names: List[str],
    *,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Exact local explanation for LogisticRegression on the *preprocessed* feature space:
      contribution_i = coef_i * x_i
    """
    pre = pipe.named_steps["pre"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    Xz = pre.transform(X_row)
    # make dense 1D for this single row only
    try:
        x = np.asarray(Xz.toarray()).ravel()
    except Exception:
        x = np.asarray(Xz).ravel()

    coef = np.asarray(clf.coef_).ravel()
    if x.shape[0] != coef.shape[0] or len(feature_names) != coef.shape[0]:
        raise ValueError("Dimension mismatch in local_explain_logreg_exact.")

    contrib = x * coef
    out = pd.DataFrame(
        {"feature": feature_names, "x_value": x, "coef": coef, "contribution": contrib, "abs_contribution": np.abs(contrib)}
    ).sort_values("abs_contribution", ascending=False).reset_index(drop=True)

    return out.head(top_n)[["feature", "x_value", "coef", "contribution"]]


def _baseline_values_from_train(X_train: pd.DataFrame) -> Dict[str, object]:
    """
    Baseline replacement values for counterfactual-style local explanation:
      - numeric -> median
      - categorical/object -> mode
    """
    base: Dict[str, object] = {}
    for c in X_train.columns:
        s = X_train[c]
        if pd.api.types.is_numeric_dtype(s):
            base[c] = float(pd.to_numeric(s, errors="coerce").median())
        else:
            base[c] = _safe_series_mode(s.astype("object"))
    return base


def _local_counterfactual_deltas(
    pipe: Pipeline,
    X_row: pd.DataFrame,
    *,
    baseline_values: Dict[str, object],
    candidate_features: List[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Model-agnostic local explanation in RAW feature space:
      For each raw feature f:
        - replace f with baseline(train) value
        - delta = score(original) - score(replaced)
      Positive delta => feature pushes score towards class 1.
    """
    if len(X_row) != 1:
        raise ValueError("X_row must be a single-row DataFrame.")

    base_score = float(_predict_score_pos1(pipe, X_row)[0])

    rows = []
    for f in candidate_features:
        if f not in X_row.columns:
            continue
        x_cf = X_row.copy()
        x_cf.iloc[0, x_cf.columns.get_loc(f)] = baseline_values.get(f, np.nan)
        s_cf = float(_predict_score_pos1(pipe, x_cf)[0])
        rows.append(
            {
                "feature": f,
                "orig_value": X_row.iloc[0][f],
                "baseline_value": baseline_values.get(f, np.nan),
                "delta_score": base_score - s_cf,
                "abs_delta": abs(base_score - s_cf),
            }
        )

    out = pd.DataFrame(rows).sort_values("abs_delta", ascending=False).reset_index(drop=True)
    return out.head(top_n)[["feature", "orig_value", "baseline_value", "delta_score"]]


@dataclass(frozen=True)
class Section17XAIResult:
    split_random_state: int
    test_size: float
    models: Dict[str, Pipeline]
    feature_names: List[str]
    global_importances_encoded: Dict[str, pd.DataFrame]   # per model, encoded-space
    global_importance_raw_perm: pd.DataFrame              # best model, raw-space
    global_importance_raw_perm_elderly: Optional[pd.DataFrame]
    global_importance_raw_perm_nonelderly: Optional[pd.DataFrame]
    hard_errors: pd.DataFrame                             # picked on best model
    local_logreg_exact: Dict[int, pd.DataFrame]           # row_ix -> table
    local_best_counterfactual: Dict[int, pd.DataFrame]    # row_ix -> table


def run_section17_xai(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    top_n_global: int = 25,
    n_repeats_perm: int = 5,
    top_n_local: int = 15,
) -> Section17XAIResult:
    """
    End-to-end XAI runner:
      - Fit 3 model families: Logistic Regression, Decision Tree, Random Forest
      - Global (encoded-space): coef / feature_importances_
      - Global (raw-space): permutation importance on raw columns (model-agnostic)
      - Subgroup (elderly vs non-elderly): raw permutation importance
      - Local: 3 high-confidence wrong predictions on the best model (RF):
          * LR exact contributions (encoded)
          * RF model-agnostic counterfactual deltas (raw)
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=50, scale_numeric=True, scaler="standard")

    if target_col not in df.columns:
        raise KeyError(f"Target col '{target_col}' not found in df.")

    y = df[target_col].astype(int).to_numpy()

    drop_cols = [c for c in preprocess_config.target_cols if c in df.columns]
    if target_col not in drop_cols:
        drop_cols.append(target_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No features left after dropping targets.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_size),
        stratify=y,
        random_state=int(random_state),
    )

    # Shared preprocessor template
    pre = s5.build_preprocessor(df, preprocess_config)

    # Models (compare transparent vs opaque)
    lr = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=4000,
        n_jobs=-1,
        random_state=int(random_state),
    )
    dt = DecisionTreeClassifier(random_state=int(random_state))
    rf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=int(random_state),
    )

    pipes: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline([("pre", pre), ("clf", lr)]),
        "decision_tree": Pipeline([("pre", pre), ("to_csc", SparseToCSC()), ("clf", dt)]),
        "random_forest": Pipeline([("pre", pre), ("to_csc", SparseToCSC()), ("clf", rf)]),
    }

    # Fit all
    for _, p in pipes.items():
        p.fit(X_train, y_train)

    # Feature names from the fitted preprocessor of LR (same design as others)
    feature_names = _get_preprocessed_feature_names(pipes["logistic_regression"].named_steps["pre"])

    # Global importances on encoded feature space
    global_encoded: Dict[str, pd.DataFrame] = {
        "logistic_regression": _global_importance_logreg(pipes["logistic_regression"], feature_names, top_n=top_n_global),
        "decision_tree": _global_importance_tree(pipes["decision_tree"], feature_names, top_n=top_n_global),
        "random_forest": _global_importance_tree(pipes["random_forest"], feature_names, top_n=top_n_global),
    }

    # Global importance on RAW columns (model-agnostic): use the best/opaque model (RF)
    perm_raw = _perm_importance_raw_cols(
        pipes["random_forest"], X_test, y_test,
        n_repeats=int(n_repeats_perm),
        random_state=int(random_state),
        top_n=top_n_global,
    )

    # Subgroup permutation importance: elderly vs non-elderly (if possible)
    if "elderly" in X_test.columns:
        elderly = X_test["elderly"].astype(int)
    elif "age_mid" in X_test.columns:
        elderly = (pd.to_numeric(X_test["age_mid"], errors="coerce").fillna(-1) >= 65).astype(int)
    elif "age" in X_test.columns:
        elderly = _compute_elderly_from_age_bucket(X_test["age"].astype("object"))
    else:
        elderly = None

    perm_elderly = None
    perm_nonelderly = None
    if elderly is not None:
        mask_e = elderly.to_numpy().astype(int) == 1
        mask_ne = ~mask_e

        if mask_e.sum() >= 50 and len(np.unique(y_test[mask_e])) >= 2:
            perm_elderly = _perm_importance_raw_cols(
                pipes["random_forest"], X_test.loc[mask_e], y_test[mask_e],
                n_repeats=int(n_repeats_perm),
                random_state=int(random_state),
                top_n=top_n_global,
            )
        if mask_ne.sum() >= 50 and len(np.unique(y_test[mask_ne])) >= 2:
            perm_nonelderly = _perm_importance_raw_cols(
                pipes["random_forest"], X_test.loc[mask_ne], y_test[mask_ne],
                n_repeats=int(n_repeats_perm),
                random_state=int(random_state),
                top_n=top_n_global,
            )

    # Pick 3 high-confidence wrong predictions on RF (hard errors)
    hard = _pick_top_confident_errors(pipes["random_forest"], X_test, y_test, k=3)

    # Local explanations
    baseline_values = _baseline_values_from_train(X_train)

    # Use top raw features from permutation importance as candidates for local counterfactual deltas
    candidate_raw = perm_raw["feature"].astype(str).tolist() if perm_raw is not None and not perm_raw.empty else list(X_test.columns)[:30]

    local_lr: Dict[int, pd.DataFrame] = {}
    local_best: Dict[int, pd.DataFrame] = {}

    for _, row in hard.iterrows():
        ix = int(row["row_ix"])
        X_one = X_test.iloc[[ix]]

        # Logistic exact contributions (encoded space)
        try:
            local_lr[ix] = _local_explain_logreg_exact(
                pipes["logistic_regression"], X_one, feature_names, top_n=int(top_n_local)
            )
        except Exception as e:
            local_lr[ix] = pd.DataFrame({"error": [str(e)]})

        # RF counterfactual deltas (raw space)
        try:
            local_best[ix] = _local_counterfactual_deltas(
                pipes["random_forest"],
                X_one,
                baseline_values=baseline_values,
                candidate_features=candidate_raw,
                top_n=int(top_n_local),
            )
        except Exception as e:
            local_best[ix] = pd.DataFrame({"error": [str(e)]})

    return Section17XAIResult(
        split_random_state=int(random_state),
        test_size=float(test_size),
        models=pipes,
        feature_names=feature_names,
        global_importances_encoded=global_encoded,
        global_importance_raw_perm=perm_raw,
        global_importance_raw_perm_elderly=perm_elderly,
        global_importance_raw_perm_nonelderly=perm_nonelderly,
        hard_errors=hard,
        local_logreg_exact=local_lr,
        local_best_counterfactual=local_best,
    )


def plot_section17_global(result: Section17XAIResult, *, top_n: int = 25):
    # Encoded-space plots
    for model_name, df_imp in result.global_importances_encoded.items():
        _plot_top_barh(df_imp, title=f"Global importance (encoded) — {model_name}", top_n=top_n)

    # Raw permutation importance plot (best model)
    if result.global_importance_raw_perm is not None and not result.global_importance_raw_perm.empty:
        d = result.global_importance_raw_perm.head(top_n).iloc[::-1]
        plt.figure(figsize=(10, max(4, int(0.22 * len(d) + 2))))
        plt.barh(d["feature"].astype(str), d["importance_mean"].astype(float))
        plt.title("Global importance (RAW permutation) — random_forest")
        plt.xlabel("Mean decrease in ROC-AUC (higher = more important)")
        plt.tight_layout()
        plt.show()


def plot_section17_local_tables(result: Section17XAIResult):
    print("\n=== Hard errors picked on best model (random_forest) ===")
    print(result.hard_errors)

    for _, row in result.hard_errors.iterrows():
        ix = int(row["row_ix"])
        print(f"\n--- Case row_ix={ix} | y_true={int(row['y_true'])} | y_pred={int(row['y_pred'])} | score={float(row['score']):.4f} ---")

        print("\n[Local — Logistic Regression exact contributions (encoded space)]")
        print(result.local_logreg_exact.get(ix, pd.DataFrame()).head(20))

        print("\n[Local — Random Forest counterfactual deltas (raw space)]")
        print(result.local_best_counterfactual.get(ix, pd.DataFrame()).head(20))
