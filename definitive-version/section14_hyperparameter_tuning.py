from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7

Split = s7.Split

class SparseToCSC(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to CSC (helps some tree models)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            from scipy import sparse
        except Exception as e:
            raise RuntimeError("scipy is required. Install with: pip install scipy") from e
        return X.tocsc() if sparse.issparse(X) else X


def _safe_index(X, idx: np.ndarray):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    return X[idx]


def _score_vector(estimator, X, *, positive_label: int = 1) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba must return [n_samples, 2+] for binary AUC.")
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            pos_idx = 1
        else:
            classes = np.asarray(classes)
            pos_idx = int(np.where(classes == positive_label)[0][0]) if positive_label in set(classes.tolist()) else 1
        return proba[:, pos_idx].astype(float)

    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float).ravel()

    raise ValueError("Estimator must expose predict_proba or decision_function for ROC-AUC.")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def default_X_y(
    df: pd.DataFrame,
    *,
    target_col: str,
    preprocess_config: s5.DiabetesPreprocessConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in df.")
    y = df[target_col].astype(int).to_numpy()
    drop_cols = [c for c in preprocess_config.target_cols if c in df.columns]
    if target_col not in drop_cols:
        drop_cols.append(target_col)
    X = df.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No features left after dropping target columns.")
    return X, y

def manual_sweep_logreg_C(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    cv_splits: Sequence[Split],
    C_values: Sequence[float] = (0.01, 0.1, 1.0, 3.0, 10.0),
    class_weight: Optional[str] = None,  
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate LogisticRegression for several C values on the SAME fixed outer folds.
    This is exactly the "manual hyperparameter exploration" objective.
    """
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig(scale_numeric=True, scaler="standard")

    X, y = default_X_y(df, target_col=target_col, preprocess_config=preprocess_config)
    s7.validate_cv_splits(cv_splits, n_samples=len(y))

    from sklearn.linear_model import LogisticRegression

    pre = s5.build_preprocessor(df, preprocess_config)

    rows: List[Dict[str, Any]] = []
    for C in C_values:
        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("clf", LogisticRegression(
                    solver="saga",
                    penalty="l2",
                    C=float(C),
                    class_weight=class_weight,
                    max_iter=4000,
                    n_jobs=-1,
                    random_state=int(random_state),
                )),
            ]
        )

        res = s7.evaluate_binary_pipeline_cv(
            pipe, X, y,
            cv_splits=cv_splits,
            metrics=("roc_auc","precision","recall","f1","accuracy","balanced_accuracy"),
            return_oof=False,
        )
        rows.append({
            "model": "logreg",
            "C": float(C),
            "class_weight": (class_weight if class_weight is not None else "None"),
            **{f"{m}_mean": float(res.mean_scores[m]) for m in res.mean_scores},
            **{f"{m}_std": float(res.std_scores[m]) for m in res.std_scores},
        })

    out = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    return out

def gridsearch_logreg(
    df: pd.DataFrame,
    *,
    target_col: str = "readmitted_30d",
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    inner_folds: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
):
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig(scale_numeric=True, scaler="standard")

    X, y = default_X_y(df, target_col=target_col, preprocess_config=preprocess_config)
    from sklearn.linear_model import LogisticRegression

    pre = s5.build_preprocessor(df, preprocess_config)

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(
                solver="saga",
                max_iter=5000,
                n_jobs=-1,
                random_state=int(random_state),
            )),
        ]
    )

    param_grid = [
        {"clf__penalty": ["l2"], "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0], "clf__class_weight": [None, "balanced"]},
        {"clf__penalty": ["l1"], "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0], "clf__class_weight": [None, "balanced"]},
    ]

    inner_cv = StratifiedKFold(n_splits=int(inner_folds), shuffle=True, random_state=int(random_state))

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        refit=True,
        cv=inner_cv,
        n_jobs=int(n_jobs),
        verbose=int(verbose),
        return_train_score=False,
    )
    gs.fit(X, y)
    return gs


# Nested CV
SearchKind = Literal["grid", "random"]


@dataclass(frozen=True)
class NestedCVResult:
    model_name: str
    outer_split_fingerprint: str
    fold_metrics: pd.DataFrame
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    best_params_per_fold: List[Dict[str, Any]]
    best_params_counts: List[Tuple[str, int]]


def nested_cv_tuning(
    df: pd.DataFrame,
    *,
    model_name: Literal["logreg","random_forest"] = "logreg",
    target_col: str = "readmitted_30d",
    preprocess_config: Optional[s5.DiabetesPreprocessConfig] = None,
    outer_splits: Sequence[Split],
    inner_folds: int = 3,
    search: SearchKind = "random",
    n_iter: int = 20,                
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
) -> NestedCVResult:
    if preprocess_config is None:
        preprocess_config = s5.DiabetesPreprocessConfig()

    X, y = default_X_y(df, target_col=target_col, preprocess_config=preprocess_config)
    s7.validate_cv_splits(outer_splits, n_samples=len(y))
    fp_outer = s7.cv_splits_fingerprint(outer_splits)

    inner_cv = StratifiedKFold(n_splits=int(inner_folds), shuffle=True, random_state=int(random_state))

    if model_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        cfg = preprocess_config
        if not cfg.scale_numeric:
            cfg = s5.DiabetesPreprocessConfig(**{**cfg.__dict__, "scale_numeric": True, "scaler": "standard"})  

        pre = s5.build_preprocessor(df, cfg)
        base = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(
                solver="saga",
                max_iter=5000,
                n_jobs=-1,
                random_state=int(random_state),
            )),
        ])
        param_grid = [
            {"clf__penalty": ["l2"], "clf__C": [0.001,0.01,0.1,1,3,10], "clf__class_weight": [None,"balanced"]},
            {"clf__penalty": ["l1"], "clf__C": [0.001,0.01,0.1,1,3,10], "clf__class_weight": [None,"balanced"]},
        ]
        param_dist = {
            "clf__penalty": ["l2","l1"],
            "clf__C": list(np.logspace(-3, 1, 20)),
            "clf__class_weight": [None, "balanced"],
        }

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        cfg = preprocess_config
        pre = s5.build_preprocessor(df, cfg)
        base = Pipeline([
            ("pre", pre),
            ("to_csc", SparseToCSC()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=int(random_state),
            )),
        ])
        param_grid = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 10],
            "clf__min_samples_leaf": [1, 5],
            "clf__max_features": ["sqrt", 0.3],
            "clf__class_weight": [None, "balanced", "balanced_subsample"],
        }
        param_dist = {
            "clf__n_estimators": [200, 300, 400, 600],
            "clf__max_depth": [None, 8, 12, 18, 25],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 5, 10],
            "clf__max_features": ["sqrt", 0.2, 0.3, 0.5],
            "clf__class_weight": [None, "balanced", "balanced_subsample"],
        }

    else:
        raise ValueError("model_name must be 'logreg' or 'random_forest'")

    fold_rows: List[Dict[str, Any]] = []
    best_params: List[Dict[str, Any]] = []

    for fold_id, (tr, te) in enumerate(outer_splits, start=1):
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)

        X_tr, y_tr = _safe_index(X, tr), y[tr]
        X_te, y_te = _safe_index(X, te), y[te]

        if search == "grid":
            searcher = GridSearchCV(
                estimator=base,
                param_grid=param_grid,
                scoring="roc_auc",
                refit=True,
                cv=inner_cv,
                n_jobs=int(n_jobs),
                verbose=int(verbose),
                return_train_score=False,
            )
        else:
            searcher = RandomizedSearchCV(
                estimator=base,
                param_distributions=param_dist,
                n_iter=int(n_iter),
                scoring="roc_auc",
                refit=True,
                cv=inner_cv,
                random_state=int(random_state),
                n_jobs=int(n_jobs),
                verbose=int(verbose),
                return_train_score=False,
            )

        searcher.fit(X_tr, y_tr)
        best_est = searcher.best_estimator_
        best_params.append(dict(searcher.best_params_))

        y_pred = np.asarray(best_est.predict(X_te), dtype=int)
        y_score = _score_vector(best_est, X_te, positive_label=1)
        m = _compute_metrics(y_te, y_pred, y_score)

        fold_rows.append({
            "fold": fold_id,
            "roc_auc": m["roc_auc"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
        })

    fold_df = pd.DataFrame(fold_rows)
    mean_scores = {c: float(fold_df[c].mean()) for c in fold_df.columns if c != "fold"}
    std_scores = {c: float(fold_df[c].std(ddof=1)) for c in fold_df.columns if c != "fold"}

    sigs = [str(sorted(p.items())) for p in best_params]
    counts = Counter(sigs).most_common(10)

    return NestedCVResult(
        model_name=model_name,
        outer_split_fingerprint=fp_outer,
        fold_metrics=fold_df,
        mean_scores=mean_scores,
        std_scores=std_scores,
        best_params_per_fold=best_params,
        best_params_counts=counts,
    )
