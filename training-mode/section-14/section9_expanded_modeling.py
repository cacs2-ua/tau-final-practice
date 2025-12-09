from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7

Split = s7.Split


class SparseToCSC(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to CSC (tree-based estimators often prefer CSC when input is sparse)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            from scipy import sparse
        except Exception as e:
            raise RuntimeError(
                "scipy is required for sparse matrix conversion. Install with: pip install scipy"
            ) from e

        if sparse.issparse(X):
            return X.tocsc()
        return X


def _validate_binary_target(y: pd.Series) -> np.ndarray:
    if y.isna().any():
        raise ValueError("Target contains missing values; Section 9 requires a clean binary target.")
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

    y = _validate_binary_target(df[target_col])

    drop_cols = [c for c in preprocess_config.target_cols if c in df.columns]
    if target_col not in drop_cols:
        drop_cols.append(target_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No feature columns available after dropping target columns.")
    return X, y


@dataclass(frozen=True)
class ModelSpec:
    name: str
    pipeline: Pipeline
    track: str  # helps your report: scaled / tree / nb_nonneg / svd_dense


def _make_configs(
    base: s5.DiabetesPreprocessConfig,
) -> Dict[str, s5.DiabetesPreprocessConfig]:
    """
    Section 9 allows deterministic preprocessing tracks:
      - scaled: for scale-sensitive models (SVM, MLP)
      - tree: for tree-based models (DT, RF) (scaling not required)
      - nb_nonneg: ensures non-negative features for MultinomialNB:
          * disables ordinal encoding (age becomes one-hot categorical)
          * disables numeric scaling
    """
    cfg_scaled = replace(base, scale_numeric=True, scaler="standard")
    cfg_tree = replace(base, scale_numeric=False)

    # IMPORTANT: MultinomialNB requires non-negative inputs.
    # Our default ordinal encoding can yield -1 for unknowns, so for NB we remove ordinal encoding.
    cfg_nb = replace(cfg_tree, ordinal_cols={})

    return {"scaled": cfg_scaled, "tree": cfg_tree, "nb_nonneg": cfg_nb}


def build_section9_model_specs(
    df: pd.DataFrame,
    *,
    preprocess_config_base: Optional[s5.DiabetesPreprocessConfig] = None,
    model_random_state: int = 42,
    svd_components: int = 100,
    fast_mode: bool = False,
) -> List[ModelSpec]:
    """
    Expanded suite (>=5, includes ensembles):
      - decision_tree (anchor)
      - naive_bayes (MultinomialNB; non-negative track)
      - svm_linear (LinearSVC; margin-based)
      - mlp_svd (MLP on TruncatedSVD components; avoids densifying huge one-hot)
      - random_forest (bagging)
      - gradient_boosting_svd (boosting on SVD components; avoids sparse incompatibility)

    fast_mode reduces compute for unit tests / quick sanity runs.
    """
    if preprocess_config_base is None:
        preprocess_config_base = s5.DiabetesPreprocessConfig()

    cfgs = _make_configs(preprocess_config_base)

    pre_tree = s5.build_preprocessor(df, cfgs["tree"])
    pre_scaled = s5.build_preprocessor(df, cfgs["scaled"])
    pre_nb = s5.build_preprocessor(df, cfgs["nb_nonneg"])

    # ---- hyperparameters (safe defaults; tuning belongs in Section 14) ----
    rf_estimators = 80 if fast_mode else 300
    gb_estimators = 100 if fast_mode else 200
    mlp_max_iter = 60 if fast_mode else 200
    mlp_hidden = (32,) if fast_mode else (64,)

    # ---- pipelines ----
    dt_pipe = Pipeline(
        steps=[
            ("pre", pre_tree),
            ("to_csc", SparseToCSC()),
            ("clf", DecisionTreeClassifier(random_state=model_random_state)),
        ]
    )

    nb_pipe = Pipeline(
        steps=[
            ("pre", pre_nb),
            ("clf", MultinomialNB(alpha=1.0)),
        ]
    )

    svm_pipe = Pipeline(
        steps=[
            ("pre", pre_scaled),
            ("clf", LinearSVC(C=1.0, random_state=model_random_state)),
        ]
    )

    # MLP cannot consume sparse one-hot reliably; we compress sparse -> dense with TruncatedSVD.
    mlp_pipe = Pipeline(
        steps=[
            ("pre", pre_scaled),
            ("svd", TruncatedSVD(n_components=int(svd_components), random_state=model_random_state)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=mlp_hidden,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                max_iter=int(mlp_max_iter),
                random_state=model_random_state,
                early_stopping=True,
                n_iter_no_change=10,
            )),
        ]
    )

    rf_pipe = Pipeline(
        steps=[
            ("pre", pre_tree),
            ("to_csc", SparseToCSC()),
            ("clf", RandomForestClassifier(
                n_estimators=int(rf_estimators),
                random_state=model_random_state,
                n_jobs=-1,
            )),
        ]
    )

    # GradientBoostingClassifier does not accept sparse matrices; we again use SVD to get dense components.
    gb_pipe = Pipeline(
        steps=[
            ("pre", pre_tree),
            ("svd", TruncatedSVD(n_components=int(svd_components), random_state=model_random_state)),
            ("clf", GradientBoostingClassifier(
                n_estimators=int(gb_estimators),
                learning_rate=0.1,
                random_state=model_random_state,
            )),
        ]
    )

    return [
        ModelSpec("decision_tree", dt_pipe, track="tree"),
        ModelSpec("naive_bayes", nb_pipe, track="nb_nonneg"),
        ModelSpec("svm_linear", svm_pipe, track="scaled"),
        ModelSpec("mlp_svd", mlp_pipe, track="svd_dense"),
        ModelSpec("random_forest", rf_pipe, track="tree"),
        ModelSpec("gradient_boosting_svd", gb_pipe, track="svd_dense"),
    ]


@dataclass(frozen=True)
class Section9RunResult:
    target_col: str
    n_splits: int
    cv_random_state: int
    split_fingerprint: str
    results_by_model: Dict[str, s7.CVEvaluationResult]
    summary_table: pd.DataFrame


def _summary_table(
    specs: Sequence[ModelSpec],
    results_by_model: Dict[str, s7.CVEvaluationResult],
    *,
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        r = results_by_model[spec.name]
        row: Dict[str, Any] = {"model": spec.name, "track": spec.track}
        for m in metrics:
            row[f"{m}_mean"] = float(r.mean_scores[m])
            row[f"{m}_std"] = float(r.std_scores[m])
        rows.append(row)

    cols = ["model", "track"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    return pd.DataFrame(rows)[cols]


def run_section9_expanded_suite(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    preprocess_config_base: Optional[s5.DiabetesPreprocessConfig] = None,
    n_splits: int = 10,
    cv_random_state: int = 42,
    model_random_state: int = 42,
    metrics: Sequence[str] = ("roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"),
    return_oof: bool = False,
    cv_splits: Optional[Sequence[Split]] = None,
    svd_components: int = 100,
    fast_mode: bool = False,
) -> Section9RunResult:
    """
    End-to-end Section 9:
      - builds 6-model suite (>=5, includes ensembles)
      - uses identical StratifiedKFold splits (must be reused across comparisons)
      - reports mean ± std across folds (table-ready)

    IMPORTANT: For Wilcoxon later, pass the SAME cv_splits you used elsewhere.
    """
    if preprocess_config_base is None:
        preprocess_config_base = s5.DiabetesPreprocessConfig()

    X, y = _default_X_y(df, target_col=target_col, preprocess_config=preprocess_config_base)

    if cv_splits is None:
        cv_splits = s7.make_stratified_cv_splits(y, n_splits=n_splits, random_state=cv_random_state)
    else:
        s7.validate_cv_splits(cv_splits, n_samples=len(y), n_splits_expected=n_splits)

    fp = s7.cv_splits_fingerprint(cv_splits)

    specs = build_section9_model_specs(
        df,
        preprocess_config_base=preprocess_config_base,
        model_random_state=model_random_state,
        svd_components=svd_components,
        fast_mode=fast_mode,
    )

    results_by_model: Dict[str, s7.CVEvaluationResult] = {}
    for spec in specs:
        res = s7.evaluate_binary_pipeline_cv(
            spec.pipeline,
            X,
            y,
            cv_splits=cv_splits,
            metrics=metrics,
            return_oof=return_oof,
        )
        results_by_model[spec.name] = res

    table = _summary_table(specs, results_by_model, metrics=metrics)

    return Section9RunResult(
        target_col=target_col,
        n_splits=n_splits,
        cv_random_state=cv_random_state,
        split_fingerprint=fp,
        results_by_model=results_by_model,
        summary_table=table,
    )
