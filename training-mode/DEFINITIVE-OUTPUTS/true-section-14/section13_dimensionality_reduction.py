
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7

Split = s7.Split
ReductionMethod = Literal["none", "svd"]


def _validate_binary_target(y: pd.Series) -> np.ndarray:
    if pd.Series(y).isna().any():
        raise ValueError("Target contains missing values; Section 13 requires a clean binary target.")
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


def _validate_components(components: Sequence[int]) -> List[int]:
    out: List[int] = []
    for c in components:
        try:
            ci = int(c)
        except Exception:
            raise ValueError(f"All n_components must be integers. Got: {c!r}")
        if ci <= 0:
            raise ValueError(f"All n_components must be > 0. Got: {ci}")
        out.append(ci)
    # de-duplicate but keep deterministic order
    seen = set()
    uniq = []
    for ci in out:
        if ci not in seen:
            uniq.append(ci)
            seen.add(ci)
    return uniq


@dataclass(frozen=True)
class DimRedVariant:
    name: str
    method: ReductionMethod
    n_components: Optional[int] = None


@dataclass(frozen=True)
class Section13RunResult:
    """
    Section 13: Dimensionality Reduction ablation.

    We implement "PCA-like" reduction for sparse one-hot data via TruncatedSVD:
    - After one-hot encoding, X is typically a high-dimensional sparse matrix.
    - Classic PCA requires centering and generally dense matrices.
    - TruncatedSVD is the standard practical alternative on sparse matrices,
      and is commonly used as a PCA analogue for sparse feature spaces.

    Returns:
      - per-variant CV results with identical folds (for fair comparisons)
      - a compact mean±std summary table for the report
    """
    target_col: str
    n_splits: int
    cv_random_state: int
    split_fingerprint: str
    variants: List[DimRedVariant]
    results_by_variant: Dict[str, s7.CVEvaluationResult]
    summary_table: pd.DataFrame


def build_section13_pipelines(
    df: pd.DataFrame,
    *,
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    model_random_state: int = 42,
    svd_components: Sequence[int] = (50, 100, 200),
) -> Tuple[List[DimRedVariant], Dict[str, Pipeline]]:
    """
    Build pipelines for:
      - no reduction (baseline)
      - TruncatedSVD(k) for each k in svd_components
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()

    comps = _validate_components(svd_components)

    pre = s5.build_preprocessor(df, preprocess_config)

    clf = LogisticRegression(
        solver="saga",         # sparse-friendly; also fine on dense SVD outputs
        penalty="l2",
        max_iter=3000,
        random_state=model_random_state,
        n_jobs=-1,
    )

    variants: List[DimRedVariant] = []
    pipes: Dict[str, Pipeline] = {}

    # Baseline: no reduction
    v0 = DimRedVariant(name="no_reduction", method="none", n_components=None)
    variants.append(v0)
    pipes[v0.name] = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # SVD variants
    for k in comps:
        vk = DimRedVariant(name=f"svd_{k}", method="svd", n_components=int(k))
        variants.append(vk)
        pipes[vk.name] = Pipeline(
            steps=[
                ("pre", pre),
                ("svd", TruncatedSVD(n_components=int(k), random_state=model_random_state)),
                ("clf", clf),
            ]
        )

    return variants, pipes


def _summary_table(
    variants: Sequence[DimRedVariant],
    results_by_variant: Dict[str, s7.CVEvaluationResult],
    *,
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for v in variants:
        r = results_by_variant[v.name]
        row: Dict[str, Any] = {
            "variant": v.name,
            "method": v.method,
            "n_components": (np.nan if v.n_components is None else int(v.n_components)),
        }
        for m in metrics:
            row[f"{m}_mean"] = float(r.mean_scores[m])
            row[f"{m}_std"] = float(r.std_scores[m])
        rows.append(row)

    cols = ["variant", "method", "n_components"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    return pd.DataFrame(rows)[cols]


def run_section13_dimensionality_reduction(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    svd_components: Sequence[int] = (50, 100, 200),
    metrics: Sequence[str] = ("roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"),
    n_splits: int = 10,
    cv_random_state: int = 42,
    model_random_state: int = 42,
    cv_splits: Optional[Sequence[Split]] = None,
    return_oof: bool = False,
) -> Section13RunResult:
    """
    End-to-end runner for Section 13 (ablation):
      - builds baseline (no reduction) + SVD(k) variants
      - evaluates each variant with the SAME cv_splits (required for fair comparison)
      - returns mean±std table + fold vectors (usable later for Wilcoxon)
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()

    X, y = _default_X_y(df, target_col=target_col, preprocess_config=preprocess_config)

    if cv_splits is None:
        cv_splits = s7.make_stratified_cv_splits(y, n_splits=n_splits, shuffle=True, random_state=cv_random_state)
    else:
        s7.validate_cv_splits(cv_splits, n_samples=len(y), n_splits_expected=n_splits)

    fp = s7.cv_splits_fingerprint(cv_splits)

    variants, pipes = build_section13_pipelines(
        df,
        preprocess_config=preprocess_config,
        model_random_state=model_random_state,
        svd_components=svd_components,
    )

    results_by_variant: Dict[str, s7.CVEvaluationResult] = {}
    for v in variants:
        res = s7.evaluate_binary_pipeline_cv(
            pipes[v.name],
            X,
            y,
            cv_splits=cv_splits,
            metrics=metrics,
            return_oof=return_oof,
        )
        results_by_variant[v.name] = res

    table = _summary_table(variants, results_by_variant, metrics=metrics)

    return Section13RunResult(
        target_col=target_col,
        n_splits=n_splits,
        cv_random_state=cv_random_state,
        split_fingerprint=fp,
        variants=list(variants),
        results_by_variant=results_by_variant,
        summary_table=table,
    )


def pick_best_variant_by_auc(out: Section13RunResult) -> str:
    """Convenience: return the variant name with best mean ROC-AUC."""
    if "roc_auc_mean" not in out.summary_table.columns:
        raise ValueError("summary_table has no roc_auc_mean column. Did you include roc_auc in metrics?")
    best_idx = int(out.summary_table["roc_auc_mean"].astype(float).idxmax())
    return str(out.summary_table.loc[best_idx, "variant"])
