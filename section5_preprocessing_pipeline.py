
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

import math

# ---------------------------------------------------------------------
# Section 5 (PDF-aligned):
# - Numeric values: scaling/normalization (StandardScaler / MinMaxScaler)
# - Categorical values: binarization (dummies / one-hot)
# - Missing/unknown: remove variable/sample or impute (Simple/KNN/Iterative)
# - No leakage: all learned steps must be fit inside CV folds via Pipeline
# ---------------------------------------------------------------------

DEFAULT_MISSING_TOKENS: Tuple[str, ...] = (
    "",
    "?",
    "NA",
    "N/A",
    "NONE",
    "NULL",
    "NAN",
    "UNKNOWN",
    "UNKNOWN/INVALID",
)

DEFAULT_ID_COLS: Tuple[str, ...] = ("encounter_id", "patient_nbr")
DEFAULT_TARGET_COLS: Tuple[str, ...] = ("readmitted", "readmitted_30d", "readmitted_3class", "y")

# Ordinal default: only AGE is treated as ordinal (safe + clinically meaningful order).
AGE_ORDER: Tuple[str, ...] = (
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
)

# Integer-coded fields that are nominal categories (avoid imposing fake geometry)
DEFAULT_FORCE_CATEGORICAL: Tuple[str, ...] = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
)

# Count-like numeric fields (plus weight if present)
DEFAULT_NUMERIC_HINTS: Tuple[str, ...] = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "weight",
)


def _normalize_token(x: str) -> str:
    return x.strip().upper()


def _missing_token_set(tokens: Sequence[str]) -> set:
    return {_normalize_token(t) for t in tokens}

class MissingTokenCleaner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        missing_tokens=("?", "UNKNOWN/INVALID", ""),
        *,
        case_insensitive: bool = True,
        strip: bool = True,
        treat_empty_as_missing: bool = True,
    ):
        self.missing_tokens = tuple(missing_tokens)
        self.case_insensitive = case_insensitive
        self.strip = strip
        self.treat_empty_as_missing = treat_empty_as_missing

    def fit(self, X, y=None):
        toks = []
        for t in self.missing_tokens:
            s = str(t)
            s = s.strip() if self.strip else s
            s = s.upper() if self.case_insensitive else s
            toks.append(s)
        self._missing_set_ = set(toks)
        return self

    def transform(self, X):
        # Soporta DataFrame y arrays
        if isinstance(X, pd.DataFrame):
            out = X.copy()
            missing_set = getattr(self, "_missing_set_", None)
            if missing_set is None:
                self.fit(X)
                missing_set = self._missing_set_

            for c in out.columns:
                s = out[c]
                if not (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
                    continue

                def _clean(v):
                    if v is None:
                        return np.nan
                    try:
                        if pd.isna(v):
                            return np.nan
                    except Exception:
                        pass

                    if isinstance(v, str):
                        vv = v.strip() if self.strip else v
                        if self.treat_empty_as_missing and vv == "":
                            return np.nan
                        key = vv.upper() if self.case_insensitive else vv
                        return np.nan if key in missing_set else vv

                    vv = str(v)
                    vv = vv.strip() if self.strip else vv
                    key = vv.upper() if self.case_insensitive else vv
                    return np.nan if key in missing_set else v

                out[c] = s.map(_clean)

            return out

        # array-like (n_samples, n_features)
        arr = np.asarray(X, dtype=object)
        missing_set = getattr(self, "_missing_set_", None)
        if missing_set is None:
            # fit “perezoso”
            self.fit(pd.DataFrame(arr))
            missing_set = self._missing_set_

        def _clean_scalar(v):
            if v is None:
                return np.nan
            try:
                if pd.isna(v):
                    return np.nan
            except Exception:
                pass
            if isinstance(v, str):
                vv = v.strip() if self.strip else v
                if self.treat_empty_as_missing and vv == "":
                    return np.nan
                key = vv.upper() if self.case_insensitive else vv
                return np.nan if key in missing_set else vv
            vv = str(v)
            vv = vv.strip() if self.strip else vv
            key = vv.upper() if self.case_insensitive else vv
            return np.nan if key in missing_set else v

        vfunc = np.vectorize(_clean_scalar, otypes=[object])
        return vfunc(arr)



class NumericCoercer(BaseEstimator, TransformerMixin):
    """Coerce selected columns to numeric (float), invalid parsing -> NaN."""

    def __init__(self, numeric_cols: Sequence[str]):
        self.numeric_cols = list(numeric_cols)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X
        df = X.copy()
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """
    Cast categorical values to strings (keeping missing as np.nan).
    Prevents mixed types (e.g., int-coded categories + string missing filler).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = self._to_2d_array(X).astype("object", copy=True)
        out = np.empty_like(X_arr, dtype="object")

        for i in range(X_arr.shape[0]):
            for j in range(X_arr.shape[1]):
                v = X_arr[i, j]
                if v is None:
                    out[i, j] = np.nan
                    continue
                try:
                    if pd.isna(v):
                        out[i, j] = np.nan
                        continue
                except Exception:
                    pass
                out[i, j] = str(v)

        return out

    @staticmethod
    def _to_2d_array(X):
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype="object")
        if isinstance(X, pd.Series):
            return X.to_frame().to_numpy(dtype="object")
        X_arr = np.asarray(X, dtype="object")
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr
from sklearn.base import BaseEstimator, TransformerMixin

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        min_count: int = 10,
        *,
        min_frequency: float | None = None,
        other_label: str = "__OTHER__",
        missing_label: str = "__MISSING__",
    ):
        self.min_count = int(min_count)
        self.min_frequency = min_frequency
        self.other_label = other_label
        self.missing_label = missing_label

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n_rows = arr.shape[0]

        # If min_frequency is provided, convert it to an absolute threshold per fit()
        freq_count = 0
        if self.min_frequency is not None:
            if not (0.0 < float(self.min_frequency) <= 1.0):
                raise ValueError(f"min_frequency must be in (0,1], got {self.min_frequency}")
            freq_count = int(math.ceil(float(self.min_frequency) * n_rows))

        effective_min = max(1, self.min_count, freq_count)

        allowed = []
        for j in range(arr.shape[1]):
            col = arr[:, j]

            col2 = np.array(
                [
                    self.missing_label
                    if (v is None or pd.isna(v))
                    else v
                    for v in col
                ],
                dtype=object,
            )

            vc = pd.Series(col2).value_counts(dropna=False)
            keep = set(vc[vc >= effective_min].index.tolist())

            # Always allow missing bucket
            keep.add(self.missing_label)

            allowed.append(keep)

        self.allowed_categories_ = allowed
        self.effective_min_count_ = effective_min
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if not hasattr(self, "allowed_categories_"):
            self.fit(arr)

        out = arr.copy()
        for j in range(out.shape[1]):
            keep = self.allowed_categories_[j]

            def _map(v):
                if v is None or pd.isna(v):
                    return self.missing_label
                if v == self.missing_label:
                    return self.missing_label
                return v if v in keep else self.other_label

            out[:, j] = np.vectorize(_map, otypes=[object])(out[:, j])

        return out

class ToNumeric(BaseEstimator, TransformerMixin):
    """
    Convert input columns to numeric (float) using pandas.to_numeric(errors='coerce').
    This is mandatory for numeric-like string columns (e.g., 'weight' = "70") so that
    median imputation + scaling work reliably.
    """

    def __init__(self, dtype: str = "float64"):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # DataFrame path (best case)
        if isinstance(X, pd.DataFrame):
            return X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=self.dtype)

        # numpy / array-like path
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        out = np.empty(arr.shape, dtype=self.dtype)
        for j in range(arr.shape[1]):
            out[:, j] = pd.to_numeric(pd.Series(arr[:, j]), errors="coerce").to_numpy(dtype=self.dtype)
        return out

def make_onehot_encoder(*, sparse_output: bool):
    # sklearn >= 1.2 uses sparse_output; older uses sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse_output)

def _make_one_hot_encoder(*, sparse: bool, handle_unknown: str = "ignore") -> OneHotEncoder:
    # sklearn compatibility (sparse vs sparse_output)
    try:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=sparse)


def _make_ordinal_encoder(categories: List[List[str]]) -> OrdinalEncoder:
    # sklearn compatibility: handle_unknown might not exist in older versions
    try:
        return OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    except TypeError:
        return OrdinalEncoder(categories=categories)


@dataclass(frozen=True)
class DiabetesPreprocessConfig:
    # Missingness tokens and core columns
    missing_tokens: Tuple[str, ...] = DEFAULT_MISSING_TOKENS
    id_cols: Tuple[str, ...] = DEFAULT_ID_COLS
    target_cols: Tuple[str, ...] = DEFAULT_TARGET_COLS

    # Ordinal specs
    ordinal_cols: Mapping[str, Sequence[str]] = field(default_factory=lambda: {"age": AGE_ORDER})

    # Force integer-coded categoricals
    force_categorical_cols: Tuple[str, ...] = DEFAULT_FORCE_CATEGORICAL

    # Numeric imputation + scaling
    numeric_impute_strategy: str = "median"   # PDF: mean/median are standard
    scale_numeric: bool = True               # PDF: scaling/normalization recommended for numeric for many models
    scaler: str = "standard"                 # 'standard' or 'minmax' (PDF mentions both)

    # Categorical handling
    cat_impute_strategy: str = "constant"    # constant -> explicit "__MISSING__" bucket
    cat_impute_fill_value: str = "__MISSING__"
    rare_min_count: int = 50
    rare_min_frequency: Optional[float] = None
    rare_other_label: str = "__OTHER__"

    # Output format
    onehot_sparse: bool = True


@dataclass(frozen=True)
class FeatureGroups:
    numeric: List[str]
    ordinal: List[str]
    categorical: List[str]
    dropped: List[str]


def infer_feature_groups(df: pd.DataFrame, config: DiabetesPreprocessConfig) -> FeatureGroups:
    cols = list(df.columns)

    id_set = set(config.id_cols)
    target_set = set(config.target_cols)

    dropped = [c for c in cols if c in id_set]
    usable = [c for c in cols if c not in id_set and c not in target_set]

    ordinal = [c for c in usable if c in set(config.ordinal_cols.keys())]
    remaining = [c for c in usable if c not in set(ordinal)]

    force_cat = set(config.force_categorical_cols)

    # Numeric:
    # (1) known count-like fields present
    # (2) detected numeric dtypes, excluding forced categoricals
    numeric: List[str] = [c for c in remaining if c in set(DEFAULT_NUMERIC_HINTS)]
    numeric_set = set(numeric)

    for c in remaining:
        if c in numeric_set or c in force_cat:
            continue
        if pd.api.types.is_numeric_dtype(df[c].dtype) and not pd.api.types.is_bool_dtype(df[c].dtype):
            numeric.append(c)
            numeric_set.add(c)

    # Everything else -> categorical (includes forced categoricals)
    categorical = [c for c in remaining if c not in numeric_set]
    return FeatureGroups(
        numeric=list(numeric),
        ordinal=list(ordinal),
        categorical=list(categorical),
        dropped=list(dropped),
    )


def build_preprocessor(df: pd.DataFrame, config: Optional[DiabetesPreprocessConfig] = None) -> Pipeline:
    """
    Single-Source-of-Truth preprocessing pipeline.

    Guarantees:
    - Unified missingness: tokens -> NaN
    - Categorical: one-hot (handle_unknown='ignore') + optional rare-category grouping
    - Numeric: median imputation + (optional) scaling/normalization
    - Ordinal: explicit ordered encoding (age)
    - Leakage-safe when used inside CV: all learned steps are fit on training folds only.
    """
    if config is None:
        config = DiabetesPreprocessConfig()

    groups = infer_feature_groups(df, config)

    # Numeric pipeline
    num_steps = [
        ("to_numeric", ToNumeric()),  # <<< AQUÍ VA
        ("imputer", SimpleImputer(strategy=config.numeric_impute_strategy)),
    ]

    if config.scale_numeric:
        scaler_name = config.scaler.lower()
        if scaler_name == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaler_name == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        else:
            raise ValueError(f"Unknown scaler: {config.scaler}. Use 'standard' or 'minmax'.")

    num_pipe = Pipeline(steps=num_steps)

    # Ordinal pipeline (age)
    if groups.ordinal:
        categories: List[List[str]] = [list(config.ordinal_cols[c]) for c in groups.ordinal]
        ord_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=config.cat_impute_fill_value)),
                ("ordinal", _make_ordinal_encoder(categories=categories)),
            ]
        )
    else:
        ord_pipe = "drop"

    # Categorical pipeline
    if config.cat_impute_strategy == "constant":
        cat_imputer = SimpleImputer(strategy="constant", fill_value=config.cat_impute_fill_value)
    elif config.cat_impute_strategy == "most_frequent":
        cat_imputer = SimpleImputer(strategy="most_frequent")
    else:
        raise ValueError("cat_impute_strategy must be 'constant' or 'most_frequent'.")

    onehot = make_onehot_encoder(sparse_output=config.onehot_sparse)

    cat_pipe = Pipeline(
        steps=[
            ("cast_to_str", CategoricalCaster()),
            ("imputer", cat_imputer),
            ("rare", RareCategoryGrouper(
                min_count=config.rare_min_count,
                min_frequency=config.rare_min_frequency,
                other_label=config.rare_other_label,
                missing_label=config.cat_impute_fill_value,
            )),
            ("onehot", onehot),
        ]
    )

    transformers = []
    if groups.numeric:
        transformers.append(("num", num_pipe, groups.numeric))
    if groups.ordinal:
        transformers.append(("ord", ord_pipe, groups.ordinal))
    if groups.categorical:
        transformers.append(("cat", cat_pipe, groups.categorical))

    ct = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    sparse_threshold=0.0 if not config.onehot_sparse else 0.3,
)

    return Pipeline(
        steps=[
            ("clean_missing_tokens", MissingTokenCleaner(missing_tokens=config.missing_tokens)),
            ("coerce_numeric", NumericCoercer(numeric_cols=groups.numeric)),
            ("features", ct),
        ]
    )
