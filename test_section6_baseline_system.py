import unittest
import numpy as np
import pandas as pd

import section6_baseline_system as s6
import section5_preprocessing_pipeline as s5


class TestSection6BaselineSystem(unittest.TestCase):
    def _toy_diabetes_like_df(self, n: int = 120) -> pd.DataFrame:
        rng = np.random.RandomState(0)

        # Fixed imbalance: 20% positives (early readmission)
        y = np.array([0] * int(n * 0.8) + [1] * (n - int(n * 0.8)), dtype=int)
        rng.shuffle(y)

        ages = list(s5.AGE_ORDER)
        df = pd.DataFrame(
            {
                "encounter_id": np.arange(n),
                "patient_nbr": np.arange(10_000, 10_000 + n),
                "age": rng.choice(ages, size=n),
                "weight": rng.choice(["?", "70", "80", "90"], size=n),
                "num_lab_procedures": rng.poisson(lam=40, size=n),
                "admission_type_id": rng.choice([1, 2, 6], size=n),
                "medical_specialty": rng.choice(["Cardiology", "InternalMedicine", "?", "RareSpec"], size=n),
                "diag_1": rng.choice(["250.83", "276", "648", "8"], size=n),
                "readmitted_30d": y,
            }
        )
        return df

    def test_evaluate_most_frequent_baseline_expected_behavior(self):
        df = self._toy_diabetes_like_df(n=120)

        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=False, rare_min_count=1)

        res = s6.evaluate_dummy_baseline_cv(
            df,
            target_col="readmitted_30d",
            strategy="most_frequent",
            n_splits=4,
            cv_random_state=0,
            dummy_random_state=0,
            preprocess_config=cfg,
        )

        self.assertEqual(res.strategy, "most_frequent")
        self.assertEqual(res.n_splits, 4)

        # Fold arrays exist and have correct length
        for k in ["roc_auc", "precision", "recall", "f1", "accuracy"]:
            self.assertIn(k, res.fold_scores)
            self.assertEqual(len(res.fold_scores[k]), 4)
            self.assertTrue(np.isfinite(res.fold_scores[k]).all())

        # Most-frequent baseline predicts only the majority class => recall/F1 for positives should be 0
        self.assertAlmostEqual(res.mean_scores["recall"], 0.0, places=12)
        self.assertAlmostEqual(res.mean_scores["f1"], 0.0, places=12)

        # Constant scoring => ROC-AUC should be chance-level (~0.5)
        self.assertAlmostEqual(res.mean_scores["roc_auc"], 0.5, places=12)

    def test_evaluate_stratified_baseline_runs_and_scores_are_valid(self):
        df = self._toy_diabetes_like_df(n=120)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=False, rare_min_count=1)

        res = s6.evaluate_dummy_baseline_cv(
            df,
            target_col="readmitted_30d",
            strategy="stratified",
            n_splits=4,
            cv_random_state=0,
            dummy_random_state=0,
            preprocess_config=cfg,
        )

        # Basic sanity: fold scores exist and are finite
        for k in ["roc_auc", "precision", "recall", "f1", "accuracy"]:
            self.assertIn(k, res.fold_scores)
            self.assertEqual(len(res.fold_scores[k]), 4)
            self.assertTrue(np.isfinite(res.fold_scores[k]).all())

        # Mean scores must be within [0,1]
        for k, v in res.mean_scores.items():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        # Chance-level discrimination is *approximately* 0.5 for a random baseline.
        # Do NOT assert exact 0.5 due to finite-sample variability.
        roc = res.mean_scores["roc_auc"]
        self.assertGreaterEqual(roc, 0.35)
        self.assertLessEqual(roc, 0.65)


    def test_run_dummy_baselines_returns_table_with_expected_columns(self):
        df = self._toy_diabetes_like_df(n=120)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=False, rare_min_count=1)

        table = s6.run_dummy_baselines(
            df,
            target_col="readmitted_30d",
            strategies=("most_frequent", "stratified"),
            n_splits=4,
            cv_random_state=0,
            dummy_random_state=0,
            preprocess_config=cfg,
        )

        self.assertEqual(table.shape[0], 2)
        self.assertIn("strategy", table.columns)
        self.assertIn("roc_auc_mean", table.columns)
        self.assertIn("f1_mean", table.columns)
        self.assertTrue(set(table["strategy"].tolist()) == {"most_frequent", "stratified"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
