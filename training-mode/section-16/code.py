from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

READMITTED_CLASSES_3: Tuple[str, str, str] = ("NO", "<30", ">30")


class TargetEngineeringError(ValueError):
    """Raised when the readmitted target contains unexpected/invalid values."""


def _normalize_readmitted_value(v: object) -> Optional[str]:
    """
    Normalize a single readmitted value:
    - None/NaN -> None
    - strings -> stripped, uppercased (except <30 and >30 remain as-is after upper)
    """
    if v is None:
        return None
    # pandas NA / numpy nan
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    s = str(v).strip()
    if s == "":
        return None
    return s.upper()


def validate_readmitted_values(
    values: Union[pd.Series, Iterable[object]],
    allowed: Sequence[str] = READMITTED_CLASSES_3,
    *,
    allow_na: bool = False,
) -> None:
    """
    Validate that all non-missing readmitted values are inside 'allowed'.

    Raises:
        TargetEngineeringError if an unexpected value is found.
    """
    allowed_set = {a.upper() for a in allowed}
    if isinstance(values, pd.Series):
        raw_iter = values.tolist()
    else:
        raw_iter = list(values)

    unexpected = set()
    for v in raw_iter:
        nv = _normalize_readmitted_value(v)
        if nv is None:
            if not allow_na:
                # Missing values are unexpected unless allow_na=True
                unexpected.add(None)
            continue
        if nv not in allowed_set:
            unexpected.add(nv)

    if unexpected:
        raise TargetEngineeringError(
            f"Unexpected readmitted values found: {sorted([x for x in unexpected if x is not None])}"
            + (" (and missing values)" if None in unexpected else "")
            + f". Allowed values: {list(allowed)}."
        )


def binarize_readmitted(
    readmitted: pd.Series,
    *,
    positive_value: str = "<30",
    negative_values: Sequence[str] = ("NO", ">30"),
    output_name: str = "readmitted_30d",
    unknown_policy: Literal["error", "nan"] = "error",
) -> pd.Series:
    """
    Convert {NO, <30, >30} into binary {1 if <30, 0 otherwise}.

    Args:
        readmitted: pandas Series with original readmitted values.
        positive_value: value mapped to 1.
        negative_values: values mapped to 0.
        output_name: name for the output series.
        unknown_policy:
            - "error": raise if any unexpected/NA values appear
            - "nan": map unexpected/NA to <NA> (nullable integer)

    Returns:
        pd.Series of dtype int8 (or nullable Int8 if unknown_policy="nan").
    """
    pos = positive_value.upper()
    neg = tuple(v.upper() for v in negative_values)

    # Normalize series
    norm = readmitted.map(_normalize_readmitted_value)

    mapping: Dict[str, int] = {pos: 1, **{v: 0 for v in neg}}

    if unknown_policy == "error":
        validate_readmitted_values(norm, allowed=(pos, *neg), allow_na=False)
        out = norm.map(mapping).astype(np.int8)
        out.name = output_name
        return out

    if unknown_policy == "nan":
        # allow NA/unknown -> <NA>
        out = norm.map(mapping)
        out = out.astype("Int8")  # nullable integer supports <NA>
        out.name = output_name
        return out

    raise ValueError(f"unknown_policy must be 'error' or 'nan', got: {unknown_policy}")


def keep_multiclass_readmitted(
    readmitted: pd.Series,
    *,
    output_name: str = "readmitted_3class",
    allowed: Sequence[str] = READMITTED_CLASSES_3,
    unknown_policy: Literal["error", "nan"] = "error",
) -> pd.Series:
    """
    Keep the original 3-class label, optionally validating values.

    Returns:
        pd.Series of dtype 'category' with categories in the allowed order,
        or with missing if unknown_policy="nan".
    """
    norm = readmitted.map(_normalize_readmitted_value)

    if unknown_policy == "error":
        validate_readmitted_values(norm, allowed=allowed, allow_na=False)
    elif unknown_policy == "nan":
        # allow NA/unknown; do not raise
        pass
    else:
        raise ValueError(f"unknown_policy must be 'error' or 'nan', got: {unknown_policy}")

    cat = pd.Categorical(norm, categories=[a.upper() for a in allowed], ordered=False)
    out = pd.Series(cat, index=readmitted.index, name=output_name)
    return out


def engineer_targets(
    df: pd.DataFrame,
    *,
    source_col: str = "readmitted",
    binary_col: str = "readmitted_30d",
    multiclass_col: str = "readmitted_3class",
    add_multiclass: bool = True,
    drop_source: bool = False,
    unknown_policy: Literal["error", "nan"] = "error",
) -> pd.DataFrame:
    """
    Add engineered target columns to a dataframe:
      - binary: 1 if <30 else 0
      - optional multiclass: categorical {NO, <30, >30}

    This keeps Section 2 self-contained and reproducible.

    Returns:
        A copy of df with new columns added (and optionally source_col dropped).
    """
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in df. Available: {list(df.columns)}")

    out = df.copy()
    out[binary_col] = binarize_readmitted(
        out[source_col],
        output_name=binary_col,
        unknown_policy=unknown_policy,
    )

    if add_multiclass:
        out[multiclass_col] = keep_multiclass_readmitted(
            out[source_col],
            output_name=multiclass_col,
            unknown_policy=unknown_policy,
        )

    if drop_source:
        out.drop(columns=[source_col], inplace=True)

    return out


@dataclass(frozen=True)
class BinaryConfusionTerms:
    tp: int
    fp: int
    fn: int
    tn: int


def confusion_terms_binary(
    y_true: Union[pd.Series, np.ndarray, Sequence[int]],
    y_pred: Union[pd.Series, np.ndarray, Sequence[int]],
    *,
    positive_label: int = 1,
) -> BinaryConfusionTerms:
    """
    Return TP/FP/FN/TN for a binary task once the positive class is defined.

    This directly supports the Section 2 narrative about TP/FP/FN/TN meaning.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm layout with labels [0,1] is:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    return BinaryConfusionTerms(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def metrics_binary(
    y_true: Union[pd.Series, np.ndarray, Sequence[int]],
    y_pred: Union[pd.Series, np.ndarray, Sequence[int]],
    y_score: Optional[Union[pd.Series, np.ndarray, Sequence[float]]] = None,
) -> Dict[str, Optional[float]]:
    """
    Convenience metric bundle for binary framing (Section 2 awareness):
      - accuracy
      - f1
      - balanced_accuracy
      - roc_auc (if y_score is provided)
    """
    res: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": None,
    }
    if y_score is not None:
        res["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return res


def metrics_multiclass(
    y_true: Union[pd.Series, np.ndarray, Sequence[str]],
    y_pred: Union[pd.Series, np.ndarray, Sequence[str]],
    y_proba: Optional[np.ndarray] = None,
    *,
    labels: Sequence[str] = READMITTED_CLASSES_3,
) -> Dict[str, Optional[float]]:
    """
    Metric bundle for the advanced extension (3-class framing):
      - macro_f1
      - weighted_f1
      - balanced_accuracy (macro recall)
      - ovr_auc_macro (if y_proba is provided with shape [n_samples, n_classes])

    Notes:
      - For AUC, we use one-vs-rest multi-class AUC (macro).
      - Requires y_proba columns correspond to 'labels' in the same order.
    """
    labels_up = [l.upper() for l in labels]

    yt = pd.Series(y_true).map(_normalize_readmitted_value)
    yp = pd.Series(y_pred).map(_normalize_readmitted_value)

    res: Dict[str, Optional[float]] = {
        "macro_f1": float(f1_score(yt, yp, labels=labels_up, average="macro")),
        "weighted_f1": float(f1_score(yt, yp, labels=labels_up, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "ovr_auc_macro": None,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim != 2 or y_proba.shape[1] != len(labels_up):
            raise ValueError(
                f"y_proba must have shape [n_samples, {len(labels_up)}] matching labels order {labels_up}."
            )
        res["ovr_auc_macro"] = float(
            roc_auc_score(yt, y_proba, multi_class="ovr", average="macro", labels=labels_up)
        )

    return res
