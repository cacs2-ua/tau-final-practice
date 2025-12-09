from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import os
import numpy as np
import pandas as pd
import matplotlib

# Force a safe, non-IPython backend to avoid importing matplotlib_inline/IPython,
# which can crash if the project has a local "code.py" shadowing stdlib "code".
try:
    matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.pyplot as plt

# If pyplot was imported elsewhere first, still try to switch backend safely.
try:
    plt.switch_backend("Agg")
except Exception:
    pass


from sklearn.metrics import confusion_matrix

import section7_validation_protocol as s7

Split = s7.Split


# -----------------------------
# Core: error typing (FP vs FN)
# -----------------------------

def infer_decision_threshold(scores: np.ndarray, threshold: Optional[float] = None) -> float:
    """
    Infer a sensible decision threshold:
      - If scores look like probabilities in [0,1] -> 0.5
      - Else (decision_function / margins) -> 0.0
    """
    if threshold is not None:
        return float(threshold)

    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        return 0.5

    smin, smax = float(np.nanmin(s)), float(np.nanmax(s))
    if (smin >= -1e-9) and (smax <= 1.0 + 1e-9):
        return 0.5
    return 0.0


def build_error_table(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    scores: Iterable[float],
    *,
    sample_ids: Optional[Sequence[Any]] = None,
    cv_splits: Optional[Sequence[Split]] = None,
    threshold: Optional[float] = None,
    positive_label: int = 1,
) -> pd.DataFrame:
    """
    Build a per-sample error table (OOF-style) with:
      - fold_id (if cv_splits provided)
      - error_type in {TP,TN,FP,FN}
      - confidence = |score - threshold|
    """
    yt = np.asarray(list(y_true), dtype=int).ravel()
    yp = np.asarray(list(y_pred), dtype=int).ravel()
    sc = np.asarray(list(scores), dtype=float).ravel()

    if not (len(yt) == len(yp) == len(sc)):
        raise ValueError("y_true, y_pred, and scores must have the same length.")

    n = len(yt)
    if sample_ids is None:
        sample_ids = list(range(n))
    if len(sample_ids) != n:
        raise ValueError("sample_ids must have the same length as y_true.")

    thr = infer_decision_threshold(sc, threshold=threshold)

    # fold id mapping (each sample appears once in test across StratifiedKFold)
    fold_id = np.full(n, fill_value=-1, dtype=int)
    if cv_splits is not None:
        s7.validate_cv_splits(cv_splits, n_samples=n)
        for i, (_, te) in enumerate(cv_splits):
            fold_id[np.asarray(te, dtype=np.int64)] = int(i)

    # error typing
    err_type = np.empty(n, dtype=object)
    for i in range(n):
        if yt[i] == positive_label and yp[i] == positive_label:
            err_type[i] = "TP"
        elif yt[i] != positive_label and yp[i] != positive_label:
            err_type[i] = "TN"
        elif yt[i] != positive_label and yp[i] == positive_label:
            err_type[i] = "FP"
        else:
            err_type[i] = "FN"

    is_error = np.isin(err_type, ["FP", "FN"])
    confidence = np.abs(sc - thr)

    df = pd.DataFrame(
        {
            "sample_id": list(sample_ids),
            "fold_id": fold_id,
            "y_true": yt,
            "y_pred": yp,
            "score": sc,
            "threshold": float(thr),
            "error_type": err_type,
            "is_error": is_error.astype(bool),
            "confidence": confidence.astype(float),
        }
    )
    return df


def select_systematic_failures(
    errors_df: pd.DataFrame,
    *,
    n_cases: int = 3,
    prefer_balance_fp_fn: bool = True,
) -> pd.DataFrame:
    """
    Select ≥3 *high-confidence* misclassifications (systematic failures under CV):
      - sorts errors by confidence (far from threshold but wrong)
      - optionally tries to include both FP and FN if available
    """
    df = errors_df.copy()
    df = df[df["is_error"]].sort_values(["confidence"], ascending=False).reset_index(drop=True)
    if df.empty:
        return df.head(0)

    if not prefer_balance_fp_fn or n_cases <= 1:
        return df.head(n_cases).reset_index(drop=True)

    fp = df[df["error_type"] == "FP"]
    fn = df[df["error_type"] == "FN"]

    picks = []
    if not fp.empty:
        picks.append(fp.iloc[0])
    if not fn.empty and len(picks) < n_cases:
        picks.append(fn.iloc[0])

    # fill remaining by global confidence order, avoiding duplicates
    picked_ids = {p["sample_id"] for p in picks}
    for _, row in df.iterrows():
        if len(picks) >= n_cases:
            break
        if row["sample_id"] in picked_ids:
            continue
        picks.append(row)
        picked_ids.add(row["sample_id"])

    return pd.DataFrame(picks).reset_index(drop=True)


def confusion_matrix_from_oof(errors_df: pd.DataFrame) -> np.ndarray:
    yt = errors_df["y_true"].to_numpy(dtype=int)
    yp = errors_df["y_pred"].to_numpy(dtype=int)
    return confusion_matrix(yt, yp, labels=[0, 1])


# -----------------------------
# Visualizations (required)
# -----------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    title: str = "Confusion Matrix (OOF)",
    normalize: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    cm = np.asarray(cm, dtype=float)
    disp = cm.copy()
    if normalize:
        row_sums = disp.sum(axis=1, keepdims=True)
        disp = np.divide(disp, np.maximum(row_sums, 1e-12))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(disp, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (not <30)", "1 (<30)"])
    ax.set_yticklabels(["0 (not <30)", "1 (<30)"])

    # annotate cells with raw counts (and normalized if requested)
    for i in range(2):
        for j in range(2):
            if normalize:
                txt = f"{int(cm[i, j])}\n({disp[i, j]:.2f})"
            else:
                txt = f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_score_distributions(
    errors_df: pd.DataFrame,
    *,
    title: str = "OOF score distributions (with errors)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    df = errors_df.copy()
    thr = float(df["threshold"].iloc[0]) if len(df) else 0.5

    pos = df[df["y_true"] == 1]["score"].to_numpy(dtype=float)
    neg = df[df["y_true"] == 0]["score"].to_numpy(dtype=float)

    fp = df[df["error_type"] == "FP"]["score"].to_numpy(dtype=float)
    fn = df[df["error_type"] == "FN"]["score"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(neg, bins=30, alpha=0.6, label="True 0 (not <30)")
    ax.hist(pos, bins=30, alpha=0.6, label="True 1 (<30)")
    ax.axvline(thr, linestyle="--", linewidth=2, label=f"threshold={thr:.2f}")

    # mark errors as rug/points near bottom
    if fp.size:
        ax.scatter(fp, np.full_like(fp, -0.01), marker="x", s=40, label="FP (0→1)")
    if fn.size:
        ax.scatter(fn, np.full_like(fn, -0.02), marker="x", s=40, label="FN (1→0)")

    ax.set_title(title)
    ax.set_xlabel("score (proba or margin)")
    ax.set_ylabel("count")
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_case_profile(
    df_raw: pd.DataFrame,
    case_row: pd.Series,
    errors_df: pd.DataFrame,
    *,
    numeric_priority: Sequence[str],
    categorical_priority: Sequence[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Case profile visualization:
      - numeric deltas vs median (within true class) scaled by IQR
      - categorical snapshot as text
    """
    sample_id = case_row["sample_id"]
    # find the row in df_raw (sample_id may be df index)
    if sample_id in df_raw.index:
        rec = df_raw.loc[sample_id]
    else:
        # fallback: assume sample_id is positional
        rec = df_raw.iloc[int(sample_id)]

    y_true = int(case_row["y_true"])
    y_pred = int(case_row["y_pred"])
    score = float(case_row["score"])
    etype = str(case_row["error_type"])
    fold_id = int(case_row["fold_id"])
    thr = float(case_row["threshold"])

    # choose numeric columns present
    num_cols = [c for c in numeric_priority if c in df_raw.columns]
    # keep those that are parseable
    num_cols_clean = []
    for c in num_cols:
        v = pd.to_numeric(pd.Series([rec[c]]), errors="coerce").iloc[0]
        if not pd.isna(v):
            num_cols_clean.append(c)
        if len(num_cols_clean) >= 8:
            break

    # compute medians + IQR within true class (OOF label grouping uses df_raw’s y)
    # if df_raw doesn't contain y_true column, we approximate using errors_df y_true vector alignment by position
    # Here we only need class masks by row order; safest: require df_raw has same row order as errors_df.
    # We assume df_raw is the same dataframe used to generate errors_df (same ordering).
    yt_all = errors_df["y_true"].to_numpy(dtype=int)
    mask = (yt_all == y_true)

    deltas = []
    labels = []
    for c in num_cols_clean:
        col_all = pd.to_numeric(df_raw[c], errors="coerce").to_numpy(dtype=float)
        med = np.nanmedian(col_all[mask])
        q1 = np.nanpercentile(col_all[mask], 25)
        q3 = np.nanpercentile(col_all[mask], 75)
        iqr = float(q3 - q1)
        denom = iqr if iqr > 1e-9 else float(np.nanstd(col_all[mask]) if np.nanstd(col_all[mask]) > 1e-9 else 1.0)
        v = float(pd.to_numeric(rec[c], errors="coerce"))
        deltas.append((v - float(med)) / denom)
        labels.append(c)

    # categorical snapshot
    cat_cols = [c for c in categorical_priority if c in df_raw.columns]
    cat_lines = []
    for c in cat_cols[:12]:
        v = rec[c]
        if isinstance(v, float) and np.isnan(v):
            vv = "<NA>"
        else:
            vv = str(v)
        cat_lines.append(f"{c}: {vv}")

    fig = plt.figure(figsize=(10, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # numeric delta bars
    if deltas:
        ax0.barh(range(len(deltas)), deltas)
        ax0.axvline(0.0, linestyle="--", linewidth=1)
        ax0.set_yticks(range(len(labels)))
        ax0.set_yticklabels(labels)
        ax0.set_xlabel("delta vs median (scaled by IQR/std) within true class")
        ax0.set_title("Numeric pattern (relative)")
    else:
        ax0.axis("off")
        ax0.text(0.0, 0.5, "No numeric features available for this record.", va="center")

    ax1.axis("off")
    header = (
        f"sample_id: {sample_id}\n"
        f"fold_id: {fold_id}\n"
        f"true={y_true}  pred={y_pred}\n"
        f"error_type={etype}\n"
        f"score={score:.4f}  thr={thr:.2f}\n"
    )
    ax1.text(0.0, 1.0, header, va="top")
    ax1.text(0.0, 0.68, "\n".join(cat_lines), va="top")

    fig.suptitle("Systematic failure case profile", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# -----------------------------
# End-to-end runner (Section 16)
# -----------------------------

@dataclass(frozen=True)
class Section16Report:
    split_fingerprint: str
    error_table: pd.DataFrame
    confusion_matrix: np.ndarray
    selected_cases_df: pd.DataFrame
    saved_plots: Dict[str, str]


def run_section16_error_analysis(
    df: pd.DataFrame,
    best_pipeline,
    *,
    cv_splits: Sequence[Split],
    target_col: str = "readmitted_30d",
    drop_cols: Sequence[str] = ("readmitted", "readmitted_30d", "readmitted_3class", "y"),
    n_cases: int = 3,
    threshold: Optional[float] = None,
    save_dir: Optional[str] = "section16_plots",
    numeric_priority: Sequence[str] = (
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
        "age_mid", "total_visits", "n_active_diabetes_meds", "n_changed_diabetes_meds",
        "stay_x_meds", "stay_x_diagnoses", "stay_x_diagnoses", "stay_x_meds",
    ),
    categorical_priority: Sequence[str] = (
        "race", "gender", "age", "admission_type", "discharge_disposition", "admission_source",
        "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3",
        "diag_1_group", "diag_2_group", "diag_3_group",
        "A1Cresult", "max_glu_serum", "insulin", "change", "diabetesMed",
    ),
) -> Section16Report:
    """
    Section 16 contract:
      - Uses identical CV splits (cv_splits) and out-of-fold predictions/scores.
      - Finds ≥3 high-confidence misclassifications (systematic failures under CV).
      - Saves required visualizations (confusion matrix + score distributions + per-case profiles).
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in df.")

    y = df[target_col].astype(int).to_numpy()
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No feature columns available after dropping targets.")

    # OOF predictions + scores
    res = s7.evaluate_binary_pipeline_cv(
        best_pipeline,
        X,
        y,
        cv_splits=cv_splits,
        metrics=("roc_auc", "accuracy", "precision", "recall", "f1", "balanced_accuracy"),
        return_oof=True,
    )
    if res.oof_pred is None or res.oof_score is None:
        raise RuntimeError("Section 16 requires out-of-fold predictions and scores; got None.")

    fp = res.split_fingerprint

    # error table (sample_id uses df.index so you can trace back to the original record)
    errors_df = build_error_table(
        y_true=y,
        y_pred=res.oof_pred,
        scores=res.oof_score,
        sample_ids=df.index.tolist(),
        cv_splits=cv_splits,
        threshold=threshold,
    )

    cm = confusion_matrix_from_oof(errors_df)
    selected = select_systematic_failures(errors_df, n_cases=n_cases, prefer_balance_fp_fn=True)

    saved: Dict[str, str] = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        p_cm = os.path.join(save_dir, "confusion_matrix_oof.png")
        plot_confusion_matrix(cm, normalize=False, save_path=p_cm)
        saved["confusion_matrix_oof"] = p_cm

        p_cm_norm = os.path.join(save_dir, "confusion_matrix_oof_normalized.png")
        plot_confusion_matrix(cm, normalize=True, title="Confusion Matrix (OOF, normalized)", save_path=p_cm_norm)
        saved["confusion_matrix_oof_normalized"] = p_cm_norm

        p_scores = os.path.join(save_dir, "score_distributions_oof.png")
        plot_score_distributions(errors_df, save_path=p_scores)
        saved["score_distributions_oof"] = p_scores

        # per-case profiles
        for i in range(len(selected)):
            row = selected.iloc[i]
            p_case = os.path.join(save_dir, f"case_{i+1}_{row['error_type']}_profile.png")
            plot_case_profile(
                df_raw=df,
                case_row=row,
                errors_df=errors_df,
                numeric_priority=numeric_priority,
                categorical_priority=categorical_priority,
                save_path=p_case,
            )
            saved[f"case_{i+1}_profile"] = p_case

    return Section16Report(
        split_fingerprint=fp,
        error_table=errors_df,
        confusion_matrix=cm,
        selected_cases_df=selected,
        saved_plots=saved,
    )
