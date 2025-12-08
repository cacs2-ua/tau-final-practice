from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7

Split = s7.Split


# ---------------------------------------------------------------------
# 12.1  Helpers to introspect the Section 5 preprocessor
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessedFeatureInfo:
    """
    Description of features after applying the Section 5 ColumnTransformer.

    It lets you:
      - know how many features come from numeric / ordinal / categorical groups
      - build a *second* transformer that does supervised selection on the
        final numeric block (embedded / filter type).
    """
    n_num_features: int
    n_ord_features: int
    n_cat_features: int

    num_slice: slice
    ord_slice: slice
    cat_slice: slice

    total_dim: int


def inspect_preprocessor_feature_slices(
    df: pd.DataFrame,
    config: Optional[s5.DiabetesPreprocessConfig] = None,
) -> PreprocessedFeatureInfo:
    """
    Fit the Section 5 preprocessor on a tiny sample and recover the dimensionality
    of each block (numeric / ordinal / categorical).

    We don't need actual values, only how many columns each transformer produces.
    """
    if config is None:
        config = s5.DiabetesPreprocessConfig()

    pre = s5.build_preprocessor(df, config)
    # Take a small subset just to fit (faster)
    sample = df.head(200).copy()
    X = sample.drop(columns=[c for c in config.target_cols if c in sample.columns], errors="ignore")
    pre.fit(X)

    # ColumnTransformer is the step "features" in the preprocessor pipeline
    ct: ColumnTransformer = pre.named_steps["features"]

    n_num = 0
    n_ord = 0
    n_cat = 0

    # Order in ct.transformers_ defines the concatenation order
    for name, trans, cols in ct.transformers_:
        if name == "num":
            # numeric pipeline: we can infer n_features by transforming a tiny batch
            Z = trans.transform(X[cols].head(5))
            n_num = Z.shape[1]
        elif name == "ord":
            Z = trans.transform(X[cols].head(5))
            n_ord = Z.shape[1]
        elif name == "cat":
            Z = trans.transform(X[cols].head(5))
            n_cat = Z.shape[1]

    # Build slices
    start = 0
    num_slice = slice(start, start + n_num)
    start += n_num
    ord_slice = slice(start, start + n_ord)
    start += n_ord
    cat_slice = slice(start, start + n_cat)
    total = start

    return PreprocessedFeatureInfo(
        n_num_features=int(n_num),
        n_ord_features=int(n_ord),
        n_cat_features=int(n_cat),
        num_slice=num_slice,
        ord_slice=ord_slice,
        cat_slice=cat_slice,
        total_dim=int(total),
    )


# ---------------------------------------------------------------------
# 12.2  Supervised selector on the *numeric block* (filter + embedded)
# ---------------------------------------------------------------------


class NumericBlockSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to keep only the numeric block inside the feature space.
    It assumes the input is the output of the Section 5 ColumnTransformer.

    We will use it ONLY inside a Pipeline, *after* the preprocessor.
    """

    def __init__(self, num_slice: slice):
        self.num_slice = num_slice

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X can be numpy array or sparse matrix
        return X[:, self.num_slice]


def _make_kbest_chi2_selector(k: int) -> SelectKBest:
    # Chi-square requires non-negative features; Section 5 numeric block is scaled
    # via MinMax/StandardScaler => typically OK. If some values end slightly
    # negative (e.g., StandardScaler), you can switch to f_classif.
    return SelectKBest(score_func=f_classif, k=k)


# ---------------------------------------------------------------------
# 12.3  Pipeline builders: full-features vs selected-features
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """
    Config for Section 12 experiments.

    selector_type:
      - "none": baseline all-features model
      - "filter_numeric_kbest": ANOVA F-test on numeric block
      - (extensible later: "embedded_l1", etc.)
    """
    selector_type: Literal["none", "filter_numeric_kbest"] = "none"
    k_numeric: int = 10          # how many numeric features to keep
    model_random_state: int = 42


def build_logistic_pipeline_with_selection(
    df: pd.DataFrame,
    *,
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    fs_config: Optional[FeatureSelectionConfig] = None,
) -> Pipeline:
    """
    Build a pipeline:

        [preprocess] -> [optional numeric selection] -> [LogisticRegression]

    The logistic model is the same family as in Sections 8–9, but here we
    focus on feature selection.
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()
    if fs_config is None:
        fs_config = FeatureSelectionConfig(selector_type="none")

    pre = s5.build_preprocessor(df, preprocess_config)

    # Base classifier
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=2000,
        random_state=fs_config.model_random_state,
        n_jobs=-1,
    )

    steps: List[Tuple[str, Any]] = [("pre", pre)]

    if fs_config.selector_type == "none":
        steps.append(("clf", clf))
        return Pipeline(steps=steps)

    # For selector_type != "none" we introspect the preprocessor to know
    # the numeric slice, then add:
    #   - NumericBlockSelector
    #   - SelectKBest on that numeric block
    # and *concatenate* with the rest via a small ColumnTransformer-like trick.
    info = inspect_preprocessor_feature_slices(df, preprocess_config)

    if fs_config.selector_type == "filter_numeric_kbest":
        # After preprocessing we have: [num | ord | cat]
        # We build a small "post-selection" transformer that:
        #   1) splits X into [num, ord+cat]
        #   2) applies SelectKBest to num
        #   3) concatenates [num_selected, ord+cat] again

        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import FunctionTransformer

        num_sel = Pipeline(
            steps=[
                ("slice_num", FunctionTransformer(lambda X: X[:, info.num_slice], accept_sparse=True)),
                ("kbest", _make_kbest_chi2_selector(k=fs_config.k_numeric)),
            ]
        )

        rest = FunctionTransformer(
            lambda X: X[:, slice(info.num_slice.stop, info.total_dim)],
            accept_sparse=True,
        )

        # FeatureUnion concatenates along feature axis:
        union = FeatureUnion(
            transformer_list=[
                ("num_sel", num_sel),
                ("rest", rest),
            ]
        )

        steps.append(("select", union))
        steps.append(("clf", clf))
        return Pipeline(steps=steps)

    raise ValueError(f"Unknown selector_type: {fs_config.selector_type}")


# ---------------------------------------------------------------------
# 12.4  End-to-end CV comparison: all-features vs selected-features
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Section12RunResult:
    target_col: str
    n_splits: int
    cv_random_state: int
    split_fingerprint: str
    configs: Dict[str, FeatureSelectionConfig]
    results_by_name: Dict[str, s7.CVEvaluationResult]
    summary_table: pd.DataFrame


def _default_X_y(
    df: pd.DataFrame,
    *,
    target_col: str,
    preprocess_config: s5.DiabetesPreprocessConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")
    y = df[target_col].astype(int).to_numpy()
    drop_cols = [c for c in preprocess_config.target_cols if c in df.columns]
    if target_col not in drop_cols:
        drop_cols.append(target_col)
    X = df.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No features left after dropping target columns.")
    return X, y


def run_section12_feature_selection(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    n_splits: int = 10,
    cv_random_state: int = 42,
    model_random_state: int = 42,
    metrics: Sequence[str] = ("roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"),
    k_numeric: int = 10,
    cv_splits: Optional[Sequence[Split]] = None,
) -> Section12RunResult:
    """
    Compare:
      - logistic_full: logistic regression on all preprocessed features
      - logistic_fs:   logistic regression with numeric filter selection (k best)

    using the SAME StratifiedKFold splits as in previous sections.
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()

    X, y = _default_X_y(df, target_col=target_col, preprocess_config=preprocess_config)

    if cv_splits is None:
        cv_splits = s7.make_stratified_cv_splits(y, n_splits=n_splits, random_state=cv_random_state)
    else:
        s7.validate_cv_splits(cv_splits, n_samples=len(y), n_splits_expected=n_splits)

    fp = s7.cv_splits_fingerprint(cv_splits)

    cfg_full = FeatureSelectionConfig(
        selector_type="none",
        k_numeric=k_numeric,
        model_random_state=model_random_state,
    )
    cfg_fs = FeatureSelectionConfig(
        selector_type="filter_numeric_kbest",
        k_numeric=k_numeric,
        model_random_state=model_random_state,
    )
    configs: Dict[str, FeatureSelectionConfig] = {
        "logistic_full": cfg_full,
        "logistic_fs": cfg_fs,
    }

    results_by_name: Dict[str, s7.CVEvaluationResult] = {}
    for name, cfg in configs.items():
        pipe = build_logistic_pipeline_with_selection(
            df,
            preprocess_config=preprocess_config,
            fs_config=cfg,
        )
        res = s7.evaluate_binary_pipeline_cv(
            pipe,
            X,
            y,
            cv_splits=cv_splits,
            metrics=metrics,
            return_oof=False,
        )
        results_by_name[name] = res

    # Build compact summary table
    rows: List[Dict[str, Any]] = []
    for name, cfg in configs.items():
        r = results_by_name[name]
        row: Dict[str, Any] = {"model": name, "selector_type": cfg.selector_type, "k_numeric": cfg.k_numeric}
        for m in metrics:
            row[f"{m}_mean"] = float(r.mean_scores[m])
            row[f"{m}_std"] = float(r.std_scores[m])
        rows.append(row)

    cols = ["model", "selector_type", "k_numeric"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    summary_table = pd.DataFrame(rows)[cols]

    return Section12RunResult(
        target_col=target_col,
        n_splits=n_splits,
        cv_random_state=cv_random_state,
        split_fingerprint=fp,
        configs=configs,
        results_by_name=results_by_name,
        summary_table=summary_table,
    )
