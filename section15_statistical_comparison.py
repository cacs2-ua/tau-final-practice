from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata, wilcoxon


@dataclass(frozen=True)
class WilcoxonSignedRankReport:
    model_a: str
    model_b: str
    metric: str
    n_pairs: int
    alpha: float
    alternative: str
    statistic: float
    p_value: float
    mean_diff: float
    median_diff: float
    effect_r_rank_biserial: float
    split_fingerprint: Optional[str]
    reject_h0: bool
    winner: str

    def to_text(self) -> str:
        h0 = f"H0: median({self.model_a} − {self.model_b}) = 0"
        h1 = f"H1 ({self.alternative}): median difference ≠ 0" if self.alternative == "two-sided" else f"H1 ({self.alternative})"
        decision = "REJECT H0" if self.reject_h0 else "FAIL TO REJECT H0"
        return (
            f"Wilcoxon signed-rank test on '{self.metric}' (paired folds)\n"
            f"- Models: A={self.model_a} vs B={self.model_b}\n"
            f"- {h0}\n"
            f"- {h1}\n"
            f"- alpha={self.alpha:.3f}, n_pairs_used={self.n_pairs}\n"
            f"- statistic={self.statistic:.6g}, p_value={self.p_value:.6g}\n"
            f"- mean(A−B)={self.mean_diff:.6g}, median(A−B)={self.median_diff:.6g}\n"
            f"- rank-biserial r={self.effect_r_rank_biserial:.6g} (≈effect size)\n"
            f"- split_fingerprint={self.split_fingerprint}\n"
            f"- decision: {decision}\n"
            f"- winner (by median diff): {self.winner}\n"
        )


def _as_1d_float(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError("Input must be 1D.")
    return arr


def _rank_biserial_from_differences(d: np.ndarray) -> Tuple[float, int]:
    """
    Rank-biserial correlation for Wilcoxon signed-rank:
      r = (W+ - W-) / (W+ + W-)
    where W+ is sum of ranks for positive diffs, W- for negative diffs,
    after removing zeros.
    """
    d = np.asarray(d, dtype=float).ravel()
    nz = d != 0
    d_nz = d[nz]
    n = int(d_nz.size)
    if n == 0:
        return 0.0, 0

    ranks = rankdata(np.abs(d_nz), method="average")
    w_plus = float(ranks[d_nz > 0].sum())
    w_minus = float(ranks[d_nz < 0].sum())
    denom = w_plus + w_minus
    if denom == 0:
        return 0.0, n
    r = (w_plus - w_minus) / denom
    return float(r), n


def wilcoxon_compare_fold_scores(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    model_a_name: str = "model_A",
    model_b_name: str = "model_B",
    metric: str = "roc_auc",
    alpha: float = 0.05,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
    correction: bool = False,
    split_fingerprint: Optional[str] = None,
) -> WilcoxonSignedRankReport:
    """
    Paired Wilcoxon signed-rank test comparing two models across identical CV folds.
    """
    a = _as_1d_float(scores_a)
    b = _as_1d_float(scores_b)
    if a.size != b.size:
        raise ValueError(f"Fold score vectors must have same length. Got {a.size} vs {b.size}.")

    d = a - b
    effect_r, n_used = _rank_biserial_from_differences(d)

    # SciPy Wilcoxon (paired); zero_method='wilcox' removes zero diffs
    stat, p = wilcoxon(
        a,
        b,
        alternative=alternative,
        zero_method=zero_method,
        correction=correction,
        method="auto",
    )

    mean_diff = float(np.mean(d))
    median_diff = float(np.median(d))
    reject = bool(float(p) < float(alpha))

    if median_diff > 0:
        winner = model_a_name
    elif median_diff < 0:
        winner = model_b_name
    else:
        winner = "tie"

    return WilcoxonSignedRankReport(
        model_a=model_a_name,
        model_b=model_b_name,
        metric=metric,
        n_pairs=n_used,
        alpha=float(alpha),
        alternative=str(alternative),
        statistic=float(stat),
        p_value=float(p),
        mean_diff=mean_diff,
        median_diff=median_diff,
        effect_r_rank_biserial=float(effect_r),
        split_fingerprint=split_fingerprint,
        reject_h0=reject,
        winner=winner,
    )


def _get_attr(obj: Any, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    raise AttributeError(f"Object has no attribute '{name}'.")


def compare_two_cv_results(
    res_a: Any,
    res_b: Any,
    *,
    metric: str = "roc_auc",
    model_a_name: str = "model_A",
    model_b_name: str = "model_B",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> WilcoxonSignedRankReport:
    """
    Convenience wrapper to compare two result objects that expose:
      - fold_scores: Dict[str, np.ndarray]
      - split_fingerprint: str (optional but recommended)

    Works with your Section 7 CVEvaluationResult and also with other section results
    that store fold_scores similarly (as long as they keep identical folds).
    """
    fold_scores_a: Dict[str, np.ndarray] = _get_attr(res_a, "fold_scores")
    fold_scores_b: Dict[str, np.ndarray] = _get_attr(res_b, "fold_scores")

    if metric not in fold_scores_a or metric not in fold_scores_b:
        raise KeyError(f"Metric '{metric}' not found in both fold_scores dicts.")

    fp_a = getattr(res_a, "split_fingerprint", None)
    fp_b = getattr(res_b, "split_fingerprint", None)
    if fp_a is not None and fp_b is not None and fp_a != fp_b:
        raise ValueError(
            "split_fingerprint mismatch: folds are NOT identical. "
            "You must reuse the SAME cv_splits for Wilcoxon.\n"
            f"fp_a={fp_a}\nfp_b={fp_b}"
        )

    return wilcoxon_compare_fold_scores(
        fold_scores_a[metric],
        fold_scores_b[metric],
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        metric=metric,
        alpha=alpha,
        alternative=alternative,
        split_fingerprint=fp_a if fp_a is not None else fp_b,
    )


def pick_best_two(
    summary_table,
    *,
    id_col: str,
    metric_col: str = "roc_auc_mean",
    higher_is_better: bool = True,
) -> Tuple[str, str]:
    """
    Pick top-2 rows from a summary_table by metric_col.
    Example:
      pick_best_two(out9.summary_table, id_col="model", metric_col="roc_auc_mean")
    """
    df = summary_table.copy()
    if id_col not in df.columns or metric_col not in df.columns:
        raise KeyError(f"summary_table must have columns '{id_col}' and '{metric_col}'.")

    df[metric_col] = df[metric_col].astype(float)
    df = df.sort_values(metric_col, ascending=not higher_is_better).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to pick best two models.")
    return str(df.loc[0, id_col]), str(df.loc[1, id_col])
