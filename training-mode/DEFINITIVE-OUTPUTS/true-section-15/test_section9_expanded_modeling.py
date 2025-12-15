import unittest
import numpy as np
import pandas as pd

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7
import section9_expanded_modeling as s9


class TestSection9ExpandedModeling(unittest.TestCase):
    def _toy_diabetes_like_df(self, n: int = 260) -> pd.DataFrame:
        rng = np.random.RandomState(0)

        # 20% positives, 80% negatives (enough for stratified folds)
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

    def test_build_model_specs_contains_six_models(self):
        df = self._toy_diabetes_like_df(n=120)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        specs = s9.build_section9_model_specs(
            df,
            preprocess_config_base=cfg,
            model_random_state=0,
            svd_components=10,
            fast_mode=True,
        )
        names = [s.name for s in specs]
        self.assertEqual(len(names), 6)
        for must in [
            "decision_tree",
            "naive_bayes",
            "svm_linear",
            "mlp_svd",
            "random_forest",
            "gradient_boosting_svd",
        ]:
            self.assertIn(must, names)

    def test_run_section9_produces_table_and_identical_folds(self):
        df = self._toy_diabetes_like_df(n=260)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        y = df["readmitted_30d"].astype(int).to_numpy()
        splits = s7.make_stratified_cv_splits(y, n_splits=5, random_state=123)

        out = s9.run_section9_expanded_suite(
            df,
            target_col="readmitted_30d",
            preprocess_config_base=cfg,
            n_splits=5,
            cv_random_state=123,
            model_random_state=0,
            cv_splits=splits,
            svd_components=10,
            fast_mode=True,
            return_oof=False,
        )

        # 6 models in the summary table
        self.assertEqual(out.summary_table.shape[0], 6)
        self.assertIn("roc_auc_mean", out.summary_table.columns)
        self.assertIn("f1_mean", out.summary_table.columns)

        # All models share the same fold fingerprint
        fps = set()
        for name, res in out.results_by_model.items():
            fps.add(res.split_fingerprint)
            self.assertEqual(len(res.fold_scores["roc_auc"]), 5)
            self.assertTrue(np.isfinite(res.fold_scores["roc_auc"]).all())
        self.assertEqual(len(fps), 1)
        self.assertEqual(out.split_fingerprint, list(fps)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
