from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7


SelectionMethod = Literal[
    "filter_chi2",        
    "filter_mi",          
    "embedded_l1",        
    "wrapper_rfe_mi",     
]



def _ensure_binary_target(y: Iterable[int]) -> np.ndarray:
    y_arr = np.asarray(list(y), dtype=int).ravel()
    if y_arr.size == 0:
        raise ValueError("Empty target.")
    uniq = set(np.unique(y_arr).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"Target must be binary in {{0,1}}. Found: {sorted(list(uniq))}")
    return y_arr


def _default_X_y(
    df: pd.DataFrame,
    *,
    target_col: str,
    cfg: s5.DiabetesPreprocessConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in df.")

    y = _ensure_binary_target(df[target_col].astype(int).to_numpy())

    drop_cols = [c for c in cfg.target_cols if c in df.columns]
    if target_col not in drop_cols:
        drop_cols.append(target_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No feature columns left after dropping target columns.")
    return X, y


def _logreg_l2(*, random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=3000,
        random_state=int(random_state),
        n_jobs=-1,
    )


def _logreg_l1(*, C: float = 0.05, random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        penalty="l1",
        C=float(C),
        max_iter=4000,
        random_state=int(random_state),
        n_jobs=-1,
    )


def _nonneg_config(base: s5.DiabetesPreprocessConfig) -> s5.DiabetesPreprocessConfig:
    """
    Make preprocessing non-negative so chi2 works:
    - disable scaling (StandardScaler can create negatives)
    - remove ordinal encoding (age ordinal can create -1 for unknown); treat age as categorical one-hot instead
    """
    return replace(base, scale_numeric=False, ordinal_cols={})


def _ohe_feature_names(onehot, input_features: List[str]) -> np.ndarray:
    """Sklearn compatibility: get_feature_names_out (new) vs get_feature_names (old)."""
    if hasattr(onehot, "get_feature_names_out"):
        return np.asarray(onehot.get_feature_names_out(input_features), dtype=object)
    if hasattr(onehot, "get_feature_names"):
        return np.asarray(onehot.get_feature_names(input_features), dtype=object)
    raise AttributeError("OneHotEncoder has no get_feature_names_out/get_feature_names.")


def _cols_to_names(cols, feature_names_in: List[str]) -> List[str]:
    """
    Convert ColumnTransformer column selectors to a list of column names.
    Handles: list[str], list[int], numpy arrays, slices.
    """
    if cols is None:
        return []
    if isinstance(cols, slice):
        return [str(c) for c in feature_names_in[cols]]
    if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
        cols_list = list(cols)
        if len(cols_list) == 0:
            return []
        if isinstance(cols_list[0], (int, np.integer)):
            return [str(feature_names_in[int(i)]) for i in cols_list]
        return [str(c) for c in cols_list]
    return [str(cols)]


def _feature_names_after_preprocessing(pre_fitted: Pipeline) -> np.ndarray:
    """
    Extract output feature names in the EXACT order produced by the fitted preprocessor.
    This avoids mismatches with selector.get_support().
    """
    ct = pre_fitted.named_steps["features"]

    feature_names_in = list(getattr(ct, "feature_names_in_", []))

    names: List[str] = []

    for name, trans, cols in getattr(ct, "transformers_", []):
        if name == "remainder" or trans == "drop":
            continue

        cols_names = _cols_to_names(cols, feature_names_in)

        if name == "num":
            names.extend([f"num__{c}" for c in cols_names])

        elif name == "ord":
            names.extend([f"ord__{c}" for c in cols_names])

        elif name == "cat":
            onehot = getattr(trans, "named_steps", {}).get("onehot", None)
            if onehot is None:
                raise RuntimeError("Expected categorical transformer to be a Pipeline with a 'onehot' step.")
            ohe = _ohe_feature_names(onehot, cols_names)
            names.extend([f"cat__{n}" for n in ohe])

        else:
            names.extend([f"{name}__{c}" for c in cols_names])

    return np.asarray(names, dtype=object)

def _base_column_from_feature_name(feature_name: str, base_cols: Sequence[str]) -> str:
    """
    Map expanded one-hot feature names back to original column names.
    Works with names like:
      - num__time_in_hospital
      - ord__age
      - cat__medical_specialty_Cardiology
    """
    s = str(feature_name)
    if s.startswith("num__"):
        return s[len("num__"):]
    if s.startswith("ord__"):
        return s[len("ord__"):]
    if s.startswith("cat__"):
        rest = s[len("cat__"):]
        for col in sorted(base_cols, key=len, reverse=True):
            if rest == col or rest.startswith(col + "_"):
                return col
        return rest.split("_", 1)[0]
    return s

def _build_interpretation_tables(
    fitted_pipe: Pipeline,
    df: pd.DataFrame,
    cfg: s5.DiabetesPreprocessConfig,
    *,
    method: SelectionMethod,
    top_n_features: int = 40,
    top_n_groups: int = 25,
) -> Dict[str, pd.DataFrame]:
    """
    Fit is already done. Extract:
      - top features (expanded)
      - top groups (original columns)
    """
    pre = fitted_pipe.named_steps["pre"]
    ct = pre.named_steps["features"]

    feature_names = _feature_names_after_preprocessing(pre)

    support = None
    importance = None

    if method in {"filter_chi2", "filter_mi", "embedded_l1"}:
        sel = fitted_pipe.named_steps["select"]
        support = sel.get_support() if hasattr(sel, "get_support") else None

        if method in {"filter_chi2", "filter_mi"} and hasattr(sel, "scores_"):
            importance = np.asarray(sel.scores_, dtype=float)
        elif method == "embedded_l1":
            est = getattr(sel, "estimator_", None)
            if est is not None and hasattr(est, "coef_"):
                importance = np.abs(np.asarray(est.coef_, dtype=float)).ravel()

    elif method == "wrapper_rfe_mi":
        kb = fitted_pipe.named_steps["kbest"]
        rfe = fitted_pipe.named_steps["select"]

        kb_support = kb.get_support()
        kb_idx = np.where(kb_support)[0]

        rfe_support = rfe.get_support()
        picked_in_kb = np.where(rfe_support)[0]
        picked_idx = kb_idx[picked_in_kb]

        support = np.zeros(len(feature_names), dtype=bool)
        support[picked_idx] = True

        imp_full = np.zeros(len(feature_names), dtype=float)
        ranking = np.asarray(getattr(rfe, "ranking_", np.ones(len(kb_idx))), dtype=float)
        imp_kb = 1.0 / np.maximum(1.0, ranking)
        imp_full[kb_idx] = imp_kb
        importance = imp_full

    if support is None:
        support = np.ones(len(feature_names), dtype=bool)
    if importance is None:
        importance = np.zeros(len(feature_names), dtype=float)

    if len(feature_names) != len(support) or len(feature_names) != len(importance):
        raise ValueError(
            "Interpretation shape mismatch:\n"
            f"  n_feature_names={len(feature_names)}\n"
            f"  n_support={len(support)}\n"
            f"  n_importance={len(importance)}\n"
            "This usually means feature-name reconstruction is out of sync with the fitted preprocessor."
        )

    base_cols = list(getattr(ct, "feature_names_in_", []))
    if not base_cols:
        g = s5.infer_feature_groups(df, cfg)
        base_cols = list(g.numeric) + list(g.ordinal) + list(g.categorical)

    base_col = [_base_column_from_feature_name(fn, base_cols=base_cols) for fn in feature_names]

    feat_df = pd.DataFrame(
        {
            "feature": feature_names,
            "base_column": base_col,
            "selected": support.astype(bool),
            "importance": importance.astype(float),
        }
    )

    feat_df_sorted = feat_df.sort_values(["selected", "importance"], ascending=[False, False])
    top_features = feat_df_sorted.head(int(top_n_features)).reset_index(drop=True)

    grp = (
        feat_df.assign(importance_selected=feat_df["importance"] * feat_df["selected"].astype(int))
        .groupby("base_column", as_index=False)
        .agg(
            n_selected=("selected", "sum"),
            n_total=("selected", "size"),
            importance_sum=("importance_selected", "sum"),
        )
    )
    grp["selected_rate"] = grp["n_selected"] / grp["n_total"].replace(0, np.nan)
    grp = grp.sort_values(["importance_sum", "n_selected", "selected_rate"], ascending=False)
    top_groups = grp.head(int(top_n_groups)).reset_index(drop=True)

    return {"top_features": top_features, "top_groups": top_groups, "all_features_table": feat_df}


def build_selection_pipeline(
    df: pd.DataFrame,
    *,
    method: SelectionMethod,
    cfg: s5.DiabetesPreprocessConfig,
    random_state: int = 42,
    k_best: int = 800,
    chi2_k: int = 800,
    l1_C: float = 0.05,
    l1_threshold: str = "median",
    l1_max_features: Optional[int] = None,
    wrapper_k_pre: int = 1200,
    wrapper_n: int = 250,
    wrapper_step: float = 0.2,
) -> Tuple[Pipeline, s5.DiabetesPreprocessConfig]:
    """
    Returns (pipeline, cfg_used).
    """
    cfg_used = cfg
    if method == "filter_chi2":
        cfg_used = _nonneg_config(cfg_used)

    pre = s5.build_preprocessor(df, cfg_used)
    clf = _logreg_l2(random_state=random_state)

    if method == "filter_chi2":
        selector = SelectKBest(score_func=chi2, k=int(chi2_k))
        pipe = Pipeline([("pre", pre), ("select", selector), ("clf", clf)])
        return pipe, cfg_used

    if method == "filter_mi":
        mi = partial(mutual_info_classif, discrete_features="auto", random_state=int(random_state))
        selector = SelectKBest(score_func=mi, k=int(k_best))
        pipe = Pipeline([("pre", pre), ("select", selector), ("clf", clf)])
        return pipe, cfg_used

    if method == "embedded_l1":
        base_est = _logreg_l1(C=l1_C, random_state=random_state)
        selector = SelectFromModel(
            estimator=base_est,
            threshold=l1_threshold,   
            max_features=l1_max_features,
        )
        pipe = Pipeline([("pre", pre), ("select", selector), ("clf", clf)])
        return pipe, cfg_used

    if method == "wrapper_rfe_mi":
        mi = partial(mutual_info_classif, discrete_features="auto", random_state=int(random_state))
        kbest = SelectKBest(score_func=mi, k=int(wrapper_k_pre))

        rfe_est = _logreg_l2(random_state=random_state)
        rfe = RFE(estimator=rfe_est, n_features_to_select=int(wrapper_n), step=float(wrapper_step))

        pipe = Pipeline([("pre", pre), ("kbest", kbest), ("select", rfe), ("clf", clf)])
        return pipe, cfg_used

    raise ValueError(f"Unknown method: {method}")


@dataclass(frozen=True)
class Section12Result:
    split_fingerprint: str
    summary_table: pd.DataFrame
    interpretations: Dict[str, Dict[str, pd.DataFrame]]  


def run_section12_feature_selection(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    preprocess_config_base: Optional[s5.DiabetesPreprocessConfig] = None,
    cv_splits: Optional[Sequence[s7.Split]] = None,
    n_splits: int = 10,
    cv_random_state: int = 42,
    model_random_state: int = 42,
    methods: Sequence[SelectionMethod] = ("filter_chi2", "embedded_l1"),
    k_best: int = 800,
    chi2_k: int = 800,
    l1_C: float = 0.05,
    l1_threshold: str = "median",
    l1_max_features: Optional[int] = None,
    wrapper_k_pre: int = 1200,
    wrapper_n: int = 250,
    wrapper_step: float = 0.2,
    build_interpretation: bool = True,
    top_n_features: int = 40,
    top_n_groups: int = 25,
) -> Section12Result:
    """
    Produces:
      - CV comparison table: all-features vs selected-features (same splits)
      - Interpretation tables to support your “hypothesis about selected groups”
    """
    if preprocess_config_base is None:
        preprocess_config_base = s5.DiabetesPreprocessConfig()

    X_base, y = _default_X_y(df, target_col=target_col, cfg=preprocess_config_base)

    if cv_splits is None:
        cv_splits = s7.make_stratified_cv_splits(
            y, n_splits=int(n_splits), shuffle=True, random_state=int(cv_random_state)
        )
    else:
        s7.validate_cv_splits(cv_splits, n_samples=len(y), n_splits_expected=int(n_splits))

    fp = s7.cv_splits_fingerprint(cv_splits)

    rows: List[Dict[str, Any]] = []
    interpretations: Dict[str, Dict[str, pd.DataFrame]] = {}

    def _eval_and_row(method_name: str, variant: str, pipe: Pipeline) -> s7.CVEvaluationResult:
        res = s7.evaluate_binary_pipeline_cv(
            pipe, X_base, y,
            cv_splits=cv_splits,
            metrics=("roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"),
            return_oof=False,
        )
        row = {
            "method": method_name,
            "variant": variant,  
            "split_fingerprint": fp,
        }
        for m in res.mean_scores:
            row[f"{m}_mean"] = float(res.mean_scores[m])
            row[f"{m}_std"] = float(res.std_scores[m])
        rows.append(row)
        return res

    for method in methods:
        sel_pipe, cfg_used = build_selection_pipeline(
            df,
            method=method,
            cfg=preprocess_config_base,
            random_state=model_random_state,
            k_best=k_best,
            chi2_k=chi2_k,
            l1_C=l1_C,
            l1_threshold=l1_threshold,
            l1_max_features=l1_max_features,
            wrapper_k_pre=wrapper_k_pre,
            wrapper_n=wrapper_n,
            wrapper_step=wrapper_step,
        )

        pre_all = s5.build_preprocessor(df, cfg_used)
        all_pipe = Pipeline([("pre", pre_all), ("clf", _logreg_l2(random_state=model_random_state))])

        _eval_and_row(str(method), "all_features", all_pipe)
        _eval_and_row(str(method), "selected", sel_pipe)

        if build_interpretation:
            fitted = Pipeline(sel_pipe.steps)  
            fitted.fit(X_base, y)
            tables = _build_interpretation_tables(
                fitted, df, cfg_used,
                method=method,
                top_n_features=top_n_features,
                top_n_groups=top_n_groups,
            )
            interpretations[str(method)] = tables

    summary = pd.DataFrame(rows).sort_values(["method", "variant"]).reset_index(drop=True)
    return Section12Result(split_fingerprint=fp, summary_table=summary, interpretations=interpretations)
