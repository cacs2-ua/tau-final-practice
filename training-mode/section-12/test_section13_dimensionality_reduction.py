
import unittest
import numpy as np
import pandas as pd

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7
import section13_dimensionality_reduction as s13


class TestSection13DimensionalityReduction(unittest.TestCase):
    def _toy_diabetes_like_df(self, n: int = 240) -> pd.DataFrame:
        rng = np.random.RandomState(0)

        # 20% positives, 80% negatives (enough for stratified CV)
        y = np.array([0] * int(n * 0.8) + [1] * (n - int(n * 0.8)), dtype=int)
        rng.shuffle(y)

        ages = list(s5.AGE_ORDER)
        df = pd.DataFrame(
            {
                "encounter_id": np.arange(n),
                "patient_nbr": np.arange(10_000, 10_000 + n),
                "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "?"], size=n),
                "gender": rng.choice(["Male", "Female"], size=n),
                "age": rng.choice(ages + ["?"], size=n),
                "weight": rng.choice(["?", "70", "80", "90", "Unknown/Invalid"], size=n),
                "admission_type_id": rng.choice([1, 2, 6], size=n),
                "time_in_hospital": rng.choice([1, 2, 3, 4, 5, 6, 7], size=n),
                "payer_code": rng.choice(["MC", "MD", "?", "SP"], size=n),
                "medical_specialty": rng.choice(
                    ["Cardiology", "InternalMedicine", "?", "RareSpecA", "RareSpecB"], size=n
                ),
                "diag_1": rng.choice(["250.83", "276", "648", "8", "401"], size=n),
                "diag_2": rng.choice(["?", "250.01", "403", "V27"], size=n),
                "diag_3": rng.choice(["?", "7", "9", "6"], size=n),
                "A1Cresult": rng.choice(["None", "Norm", ">7", ">8", "?"], size=n),
                "insulin": rng.choice(["No", "Steady", "Up", "Down"], size=n),
                "change": rng.choice(["Ch", "No"], size=n),
                "diabetesMed": rng.choice(["Yes", "No"], size=n),
                "readmitted_30d": y,
            }
        )
        return df

    def test_build_section13_pipelines_contains_baseline_and_svd_variants(self):
        df = self._toy_diabetes_like_df(n=120)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        variants, pipes = s13.build_section13_pipelines(
            df, preprocess_config=cfg, model_random_state=0, svd_components=(5, 10)
        )

        names = [v.name for v in variants]
        self.assertIn("no_reduction", names)
        self.assertIn("svd_5", names)
        self.assertIn("svd_10", names)

        self.assertIn("no_reduction", pipes)
        self.assertIn("svd_5", pipes)
        self.assertIn("svd_10", pipes)

    def test_run_section13_produces_summary_and_identical_folds(self):
        df = self._toy_diabetes_like_df(n=240)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        y = df["readmitted_30d"].astype(int).to_numpy()
        splits = s7.make_stratified_cv_splits(y, n_splits=5, random_state=123)

        out = s13.run_section13_dimensionality_reduction(
            df,
            target_col="readmitted_30d",
            preprocess_config=cfg,
            svd_components=(5, 10),
            n_splits=5,
            cv_random_state=123,
            model_random_state=0,
            cv_splits=splits,
            return_oof=False,
        )

        # Expect: baseline + 2 svd variants => 3 rows
        self.assertEqual(out.summary_table.shape[0], 3)
        self.assertIn("roc_auc_mean", out.summary_table.columns)
        self.assertIn("f1_mean", out.summary_table.columns)

        # All variants must share the same split fingerprint (same cv_splits)
        fps = set()
        for name, res in out.results_by_variant.items():
            fps.add(res.split_fingerprint)
            self.assertEqual(len(res.fold_scores["roc_auc"]), 5)
            self.assertTrue(np.isfinite(res.fold_scores["roc_auc"]).all())
        self.assertEqual(len(fps), 1)
        self.assertEqual(out.split_fingerprint, list(fps)[0])

    def test_invalid_components_raises(self):
        df = self._toy_diabetes_like_df(n=120)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        with self.assertRaises(ValueError):
            _ = s13.build_section13_pipelines(df, preprocess_config=cfg, svd_components=(0, 10))

        with self.assertRaises(ValueError):
            _ = s13.build_section13_pipelines(df, preprocess_config=cfg, svd_components=(-5,))

    def test_pick_best_variant_by_auc_returns_valid_name(self):
        df = self._toy_diabetes_like_df(n=240)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        y = df["readmitted_30d"].astype(int).to_numpy()
        splits = s7.make_stratified_cv_splits(y, n_splits=4, random_state=7)

        out = s13.run_section13_dimensionality_reduction(
            df,
            target_col="readmitted_30d",
            preprocess_config=cfg,
            svd_components=(5,),
            n_splits=4,
            cv_random_state=7,
            model_random_state=0,
            cv_splits=splits,
            return_oof=False,
        )

        best = s13.pick_best_variant_by_auc(out)
        self.assertIn(best, set(out.summary_table["variant"].tolist()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
