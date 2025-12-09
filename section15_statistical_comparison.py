from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

# SciPy is required for Wilcoxon signed-rank
try:
    from scipy.stats import wilcoxon
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scipy is required for Section 15 (Wilcoxon signed-rank). "
        "Install with: pip install scipy"
    ) from e

try:
    from scipy.stats import rankdata
except Exception:  # pragma: no cover
    rankdata = None  # fallback handled below


Alternative = Literal["two-sided", "greater", "less"]


@dataclass(frozen=True)
class WilcoxonSignedRankResult:
    """
    Section 15 — Wilcoxon Signed-Rank (paired, non-parametric).

    H0: median(scores_A - scores_B) = 0 (no performance difference).
    Decision rule: reject H0 if p_value < alpha.
    """
    model_a: str
    model_b: str
    metric: str
    alpha: float
    alternative: Alternative

    n_folds: int
    n_nonzero: int

    statistic: float
    p_value: float

    mean_diff: float
    median_diff: float
    effect_size_rbc: Optional[float]  # rank-biserial correlation in [-1,1] (None if undefined)

    reject_null: bool
    interpretation: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "alpha": float(self.alpha),
            "alternative": self.alternative,
            "n_folds": int(self.n_folds),
            "n_nonzero": int(self.n_nonzero),
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "mean_diff": float(self.mean_diff),
            "median_diff": float(self.median_diff),
            "effect_size_rbc": None if self.effect_size_rbc is None else float(self.effect_size_rbc),
            "reject_null": bool(self.reject_null),
            "interpretation": self.interpretation,
        }


def _to_1d_float_array(x: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0][:10].tolist()
        raise ValueError(f"{name} contains non-finite values at indices {bad} (showing up to 10).")
    return arr


def _rank_biserial_from_diffs(diffs: np.ndarray) -> Optional[float]:
    """
    Rank-biserial correlation for Wilcoxon signed-rank:
      RBC = (2*W+ - T) / T, where T = n(n+1)/2 and W+ is sum of ranks of positive diffs.

    Returns None if n==0.
    """
    diffs = np.asarray(diffs, dtype=float).ravel()
    diffs = diffs[diffs != 0.0]
    n = int(diffs.size)
    if n == 0:
        return None

    abs_d = np.abs(diffs)
    if rankdata is not None:
        ranks = rankdata(abs_d, method="average")
    else:
        # Fallback: average ranks using numpy argsort with tie-handling via stable grouping
        order = np.argsort(abs_d, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
                j += 1
            # average rank for tie block (1-indexed ranks)
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg_rank
            i = j + 1

    w_pos = float(np.sum(ranks[diffs > 0.0]))
    t = float(n * (n + 1) / 2.0)
    rbc = (2.0 * w_pos - t) / t
    # numerical safety
    return float(max(-1.0, min(1.0, rbc)))


def wilcoxon_signed_rank(
    scores_a: Sequence[float] | np.ndarray,
    scores_b: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
    correction: bool = False,
) -> Tuple[float, float]:
    """
    Thin compatibility wrapper over scipy.stats.wilcoxon.
    Returns: (statistic, p_value)
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0,1). Got: {alpha}")

    a = _to_1d_float_array(scores_a, name="scores_a")
    b = _to_1d_float_array(scores_b, name="scores_b")
    if a.size != b.size:
        raise ValueError(f"scores_a and scores_b must have same length. Got {a.size} vs {b.size}.")

    diffs = a - b
    if np.all(diffs == 0.0):
        # perfectly identical paired scores => no evidence of difference
        return 0.0, 1.0

    # SciPy signature differs across versions; handle gracefully.
    try:
        res = wilcoxon(
            diffs,
            zero_method=zero_method,
            correction=correction,
            alternative=alternative,
            mode="auto",
        )
    except TypeError:
        res = wilcoxon(
            diffs,
            zero_method=zero_method,
            correction=correction,
            alternative=alternative,
        )

    stat = float(getattr(res, "statistic", res[0]))
    p = float(getattr(res, "pvalue", res[1]))
    return stat, p


def compare_two_models_wilcoxon(
    model_a: str,
    model_b: str,
    scores_a: Sequence[float] | np.ndarray,
    scores_b: Sequence[float] | np.ndarray,
    *,
    metric: str = "roc_auc",
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> WilcoxonSignedRankResult:
    """
    Core Section 15 routine: compare two paired fold-score vectors.
    """
    a = _to_1d_float_array(scores_a, name="scores_a")
    b = _to_1d_float_array(scores_b, name="scores_b")
    if a.size != b.size:
        raise ValueError(f"Fold vectors must have same length. Got {a.size} vs {b.size}.")

    diffs = a - b
    n_folds = int(a.size)
    n_nonzero = int(np.sum(diffs != 0.0))

    stat, p = wilcoxon_signed_rank(
        a, b,
        alpha=alpha,
        alternative=alternative,
        zero_method="wilcox",
        correction=False,
    )

    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    rbc = _rank_biserial_from_diffs(diffs)

    reject = bool(p < float(alpha))
    if reject:
        interp = (
            f"Reject H0 at α={alpha}: evidence of a statistically significant difference "
            f"between {model_a} and {model_b} for metric '{metric}' (Wilcoxon signed-rank, "
            f"alternative='{alternative}', p={p:.6g})."
        )
    else:
        interp = (
            f"Fail to reject H0 at α={alpha}: insufficient evidence of a statistically significant "
            f"difference between {model_a} and {model_b} for metric '{metric}' "
            f"(Wilcoxon signed-rank, alternative='{alternative}', p={p:.6g})."
        )

    return WilcoxonSignedRankResult(
        model_a=str(model_a),
        model_b=str(model_b),
        metric=str(metric),
        alpha=float(alpha),
        alternative=alternative,
        n_folds=n_folds,
        n_nonzero=n_nonzero,
        statistic=float(stat),
        p_value=float(p),
        mean_diff=mean_diff,
        median_diff=median_diff,
        effect_size_rbc=rbc,
        reject_null=reject,
        interpretation=interp,
    )


def _get_fold_scores(obj: Any, metric: str) -> np.ndarray:
    """
    Extract fold-score vector from either:
      - section7_validation_protocol.CVEvaluationResult
      - section10_imbalance_handling.Section10EvaluationResult
      - any object exposing `.fold_scores[metric]`
    """
    if not hasattr(obj, "fold_scores"):
        raise TypeError("Object has no attribute 'fold_scores'. Provide an object with per-fold scores.")
    fs = getattr(obj, "fold_scores")
    if not isinstance(fs, Mapping):
        raise TypeError("fold_scores must be a mapping (e.g., dict).")
    if metric not in fs:
        raise KeyError(f"Metric '{metric}' not found in fold_scores. Available: {list(fs.keys())}")
    return _to_1d_float_array(fs[metric], name=f"fold_scores[{metric}]")


def _get_fingerprint(obj: Any) -> Optional[str]:
    fp = getattr(obj, "split_fingerprint", None)
    return None if fp is None else str(fp)


def compare_results_objects_wilcoxon(
    name_a: str,
    res_a: Any,
    name_b: str,
    res_b: Any,
    *,
    metric: str = "roc_auc",
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    strict_same_folds: bool = True,
) -> WilcoxonSignedRankResult:
    """
    Compare two *result objects* that contain per-fold vectors and (optionally) a split fingerprint.
    If strict_same_folds=True, we enforce identical folds (required by the rubric for coherent testing).
    """
    fpa = _get_fingerprint(res_a)
    fpb = _get_fingerprint(res_b)
    if strict_same_folds and (fpa is not None) and (fpb is not None) and (fpa != fpb):
        raise ValueError(
            "Cannot run Wilcoxon: fold partitions differ (split_fingerprint mismatch). "
            f"{name_a} fingerprint={fpa}, {name_b} fingerprint={fpb}."
        )

    scores_a = _get_fold_scores(res_a, metric)
    scores_b = _get_fold_scores(res_b, metric)

    return compare_two_models_wilcoxon(
        name_a, name_b,
        scores_a, scores_b,
        metric=metric,
        alpha=alpha,
        alternative=alternative,
    )


def pick_top2_by_mean_metric(
    results_by_name: Mapping[str, Any],
    *,
    metric: str = "roc_auc",
    higher_is_better: bool = True,
) -> Tuple[str, str]:
    """
    Select the top-2 models by mean of the fold-score vector for `metric`.
    """
    if len(results_by_name) < 2:
        raise ValueError("Need at least two results to pick top-2.")

    rows = []
    for name, res in results_by_name.items():
        scores = _get_fold_scores(res, metric)
        rows.append((str(name), float(np.mean(scores))))

    rows.sort(key=lambda x: x[1], reverse=bool(higher_is_better))
    return rows[0][0], rows[1][0]


def run_section15_wilcoxon_best_two(
    results_by_name: Mapping[str, Any],
    *,
    metric: str = "roc_auc",
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    strict_same_folds: bool = True,
) -> WilcoxonSignedRankResult:
    """
    End-to-end helper for Section 15:
      1) pick best two models by mean(metric)
      2) run Wilcoxon signed-rank on their paired 10-fold scores
    """
    best, second = pick_top2_by_mean_metric(results_by_name, metric=metric, higher_is_better=True)
    return compare_results_objects_wilcoxon(
        best, results_by_name[best],
        second, results_by_name[second],
        metric=metric,
        alpha=alpha,
        alternative=alternative,
        strict_same_folds=strict_same_folds,
    )
