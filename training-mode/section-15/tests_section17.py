import unittest
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


import section17_xai_explainability as xai


def _make_stratified_splits(y, n_splits=5, random_state=42):
    """
    Local fallback in case your project Section 7 splitter is not importable.
    Returns list of (train_idx, test_idx).
    """
    try:
        import section7_validation_protocol as s7
        return s7.make_stratified_cv_splits(y, n_splits=n_splits, random_state=random_state)
    except Exception:
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y = np.asarray(y).ravel()
        return [(tr, te) for tr, te in skf.split(np.zeros_like(y), y)]


def _make_mixed_type_pipeline():
    """
    Pipeline that can handle numeric + categorical columns in the toy dataset.
    NOTE: step name is 'ct' (NOT 'pre'), on purpose: this should make
    xai.explain_instance_linear(...) fail with XAIError in that test,
    because Section17's linear explainer expects your Section5-style naming.
    """
    numeric_features = ["x1", "x2"]
    categorical_features = ["cat"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("ct", pre),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    return model


class TestSection17XAI(unittest.TestCase):
    def _make_toy_data(self, n=300, seed=0):
        rng = np.random.RandomState(seed)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        cat = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.4, 0.2])

        # label depends mostly on x1 + a bit on cat
        logit = 2.0 * x1 + 0.2 * x2 + (cat == "C") * 0.5 + rng.normal(scale=0.5, size=n)
        y = (logit > 0).astype(int)

        X = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat})
        return X, y

    def test_select_hard_errors_from_oof(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0])  # wrong at idx 2 (FP) and 3 (FN)
        y_score = np.array([0.1, 0.9, 0.99, 0.01, 0.8, 0.2])  # very confident wrong on both
        cases = xai.select_hard_errors_from_oof(y_true=y_true, y_pred=y_pred, y_score=y_score, top_n=2)
        self.assertEqual(len(cases), 2)
        self.assertIn(cases[0].error_type, ("FP", "FN"))

    def test_permutation_importance_cv_raw_runs(self):
        X, y = self._make_toy_data()
        model = _make_mixed_type_pipeline()

        splits = _make_stratified_splits(y, n_splits=5, random_state=42)

        res = xai.compute_global_importance_permutation_cv_raw(
            model,
            X,
            y,
            cv_splits=splits,
            n_repeats=2,
            random_state=42,
            max_features=5,
        )

        self.assertEqual(res.level, "raw")
        self.assertTrue({"feature", "importance_mean", "importance_std"}.issubset(set(res.importance.columns)))
        self.assertLessEqual(len(res.importance), 5)

    def test_stratified_sample_indices(self):
        _, y = self._make_toy_data()
        idx = xai.stratified_sample_indices(y, n=50, random_state=1)
        self.assertTrue(len(idx) <= 50)
        self.assertTrue(np.min(idx) >= 0)

    def test_explain_instance_linear_requires_coef(self):
        # This should raise XAIError because our test pipeline is NOT Section5-style,
        # so Section17's feature-name inference should fail (even though the model fits).
        X, y = self._make_toy_data()
        pipe = _make_mixed_type_pipeline().fit(X, y)
        one = X.iloc[[0]].copy()

        with self.assertRaises(xai.XAIError):
            _ = xai.explain_instance_linear(pipe, one)

    def test_shap_optional_dependency_error(self):
        X, y = self._make_toy_data()
        pipe = _make_mixed_type_pipeline().fit(X, y)
        one = X.iloc[[0]].copy()
        bg = X.iloc[:20].copy()

        try:
            import shap  # noqa: F401
            shap_installed = True
        except Exception:
            shap_installed = False

        if not shap_installed:
            with self.assertRaises(xai.XAIError):
                _ = xai.explain_instance_shap(pipe, bg, one)


if __name__ == "__main__":
    unittest.main(verbosity=2)
