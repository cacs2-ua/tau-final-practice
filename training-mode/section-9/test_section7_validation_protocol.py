import unittest
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import section7_validation_protocol as s7
import section5_preprocessing_pipeline as s5


def _splits_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    for (tr1, te1), (tr2, te2) in zip(a, b):
        if not (np.array_equal(tr1, tr2) and np.array_equal(te1, te2)):
            return False
    return True


class TestSection7ValidationProtocol(unittest.TestCase):
    def _toy_diabetes_like_df(self, n: int = 200) -> pd.DataFrame:
        rng = np.random.RandomState(0)

        # 20% positives, 80% negatives
        y = np.array([0] * int(n * 0.8) + [1] * (n - int(n * 0.8)), dtype=int)
        rng.shuffle(y)

        ages = list(s5.AGE_ORDER)
        df = pd.DataFrame(
            {
                "encounter_id": np.arange(n),
                "patient_nbr": np.arange(10_000, 10_000 + n),
                "age": rng.choice(ages, size=n),
                "weight": rng.choice(["?", "70", "80", "90", "Unknown/Invalid"], size=n),
                "num_lab_procedures": rng.poisson(lam=40, size=n),
                "admission_type_id": rng.choice([1, 2, 6], size=n),
                "medical_specialty": rng.choice(["Cardiology", "InternalMedicine", "?", "RareSpec"], size=n),
                "diag_1": rng.choice(["250.83", "276", "648", "8"], size=n),
                "readmitted_30d": y,
            }
        )
        return df

    def test_make_stratified_cv_splits_reproducible_and_valid(self):
        df = self._toy_diabetes_like_df(n=200)
        y = df["readmitted_30d"].astype(int).tolist()

        splits1 = s7.make_stratified_cv_splits(y, n_splits=10, random_state=42)
        splits2 = s7.make_stratified_cv_splits(y, n_splits=10, random_state=42)

        self.assertTrue(_splits_equal(splits1, splits2))
        # validate should not raise
        s7.validate_cv_splits(splits1, n_samples=len(y), n_splits_expected=10)

    def test_cv_fingerprint_changes_with_seed(self):
        df = self._toy_diabetes_like_df(n=200)
        y = df["readmitted_30d"].astype(int).tolist()

        s0 = s7.make_stratified_cv_splits(y, n_splits=10, random_state=0)
        s1 = s7.make_stratified_cv_splits(y, n_splits=10, random_state=1)

        fp0 = s7.cv_splits_fingerprint(s0)
        fp1 = s7.cv_splits_fingerprint(s1)

        self.assertNotEqual(fp0, fp1)

    def test_evaluate_binary_pipeline_cv_returns_fold_scores_and_oof(self):
        df = self._toy_diabetes_like_df(n=200)
        y = df["readmitted_30d"].astype(int)
        X = df.drop(columns=["readmitted_30d"])

        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=False, rare_min_count=1)
        pre = s5.build_preprocessor(df, cfg)

        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )

        splits = s7.make_stratified_cv_splits(y, n_splits=10, random_state=42)
        res = s7.evaluate_binary_pipeline_cv(pipe, X, y, cv_splits=splits, return_oof=True)

        self.assertEqual(res.n_splits, 10)
        self.assertEqual(len(res.fold_scores["roc_auc"]), 10)
        self.assertTrue(np.isfinite(res.fold_scores["roc_auc"]).all())

        # means in [0,1]
        for v in res.mean_scores.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        # OOF outputs cover all rows
        self.assertIsNotNone(res.oof_pred)
        self.assertIsNotNone(res.oof_score)
        self.assertEqual(len(res.oof_pred), len(df))
        self.assertEqual(len(res.oof_score), len(df))
        self.assertTrue(np.isfinite(res.oof_score).all())

    def test_two_models_share_identical_folds(self):
        df = self._toy_diabetes_like_df(n=200)
        y = df["readmitted_30d"].astype(int)
        X = df.drop(columns=["readmitted_30d"])

        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=False, rare_min_count=1)
        pre = s5.build_preprocessor(df, cfg)

        splits = s7.make_stratified_cv_splits(y, n_splits=10, random_state=42)

        pipe_lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
        pipe_dt = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(random_state=0))])

        r1 = s7.evaluate_binary_pipeline_cv(pipe_lr, X, y, cv_splits=splits, return_oof=False)
        r2 = s7.evaluate_binary_pipeline_cv(pipe_dt, X, y, cv_splits=splits, return_oof=False)

        self.assertEqual(r1.split_fingerprint, r2.split_fingerprint)
        self.assertTrue(_splits_equal(r1.fold_indices, r2.fold_indices))


if __name__ == "__main__":
    unittest.main(verbosity=2)
