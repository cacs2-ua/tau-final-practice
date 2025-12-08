import unittest
import numpy as np
import pandas as pd

import section5_preprocessing_pipeline as s5
import section8_core_modeling as s8
import section7_validation_protocol as s7


class TestSection8CoreModeling(unittest.TestCase):
    def _toy_diabetes_like_df(self, n: int = 300) -> pd.DataFrame:
        rng = np.random.RandomState(0)

        # 20% positives, 80% negatives
        y = np.array([0] * int(n * 0.8) + [1] * (n - int(n * 0.8)), dtype=int)
        rng.shuffle(y)

        ages = list(s5.AGE_ORDER)
        df = pd.DataFrame(
            {
                "encounter_id": np.arange(n),
                "patient_nbr": np.arange(10_000, 10_000 + n),
                "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "?"], size=n),
                "gender": rng.choice(["Male", "Female"], size=n),
                "age": rng.choice(ages, size=n),
                "weight": rng.choice(["?", "70", "80", "90", "Unknown/Invalid"], size=n),
                "admission_type_id": rng.choice([1, 2, 6], size=n),
                "discharge_disposition_id": rng.choice([1, 3, 25], size=n),
                "admission_source_id": rng.choice([1, 7], size=n),
                "time_in_hospital": rng.choice([1, 2, 3, 4, 5, 6, 7], size=n),
                "payer_code": rng.choice(["MC", "MD", "?", "SP"], size=n),
                "medical_specialty": rng.choice(
                    ["Cardiology", "InternalMedicine", "?", "RareSpecA", "RareSpecB"], size=n
                ),
                "num_lab_procedures": rng.poisson(lam=40, size=n),
                "num_procedures": rng.poisson(lam=1, size=n),
                "num_medications": rng.poisson(lam=10, size=n),
                "number_outpatient": rng.poisson(lam=1, size=n),
                "number_emergency": rng.poisson(lam=0.2, size=n),
                "number_inpatient": rng.poisson(lam=0.5, size=n),
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

    def test_build_pipelines_contains_two_models(self):
        df = self._toy_diabetes_like_df(n=50)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        pipes = s8.build_section8_pipelines(df, preprocess_config=cfg, model_random_state=42)
        self.assertIn("logistic_regression", pipes)
        self.assertIn("decision_tree", pipes)

        # Decision tree pipeline should include the CSC converter step
        self.assertTrue(any(step_name == "to_csc" for step_name, _ in pipes["decision_tree"].steps))

    def test_run_section8_core_models_returns_table_and_identical_folds(self):
        df = self._toy_diabetes_like_df(n=300)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        # Create splits once and reuse
        y = df["readmitted_30d"].astype(int).to_numpy()
        splits = s7.make_stratified_cv_splits(y, n_splits=5, random_state=123)

        out = s8.run_section8_core_models(
            df,
            target_col="readmitted_30d",
            preprocess_config=cfg,
            n_splits=5,
            cv_random_state=123,
            model_random_state=42,
            cv_splits=splits,
            return_oof=True,
        )

        # Table shape: 2 models
        self.assertEqual(out.summary_table.shape[0], 2)
        self.assertIn("roc_auc_mean", out.summary_table.columns)
        self.assertIn("f1_mean", out.summary_table.columns)

        # Both models must share the exact same folds fingerprint
        r_lr = out.results_by_model["logistic_regression"]
        r_dt = out.results_by_model["decision_tree"]
        self.assertEqual(r_lr.split_fingerprint, r_dt.split_fingerprint)
        self.assertEqual(r_lr.split_fingerprint, out.split_fingerprint)

        # Fold scores are finite and length = n_splits
        self.assertEqual(len(r_lr.fold_scores["roc_auc"]), 5)
        self.assertTrue(np.isfinite(r_lr.fold_scores["roc_auc"]).all())
        self.assertEqual(len(r_dt.fold_scores["roc_auc"]), 5)
        self.assertTrue(np.isfinite(r_dt.fold_scores["roc_auc"]).all())

        # OOF vectors exist and cover all rows
        self.assertEqual(len(r_lr.oof_pred), len(df))
        self.assertEqual(len(r_lr.oof_score), len(df))
        self.assertTrue(np.isfinite(r_lr.oof_score).all())

    def test_raises_if_target_missing(self):
        df = self._toy_diabetes_like_df(n=50).drop(columns=["readmitted_30d"])
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)
        with self.assertRaises(KeyError):
            _ = s8.run_section8_core_models(df, target_col="readmitted_30d", preprocess_config=cfg, n_splits=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
