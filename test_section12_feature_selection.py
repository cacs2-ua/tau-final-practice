import unittest
import numpy as np
import pandas as pd

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7
import section12_feature_selection as s12


class TestSection12FeatureSelection(unittest.TestCase):
    def _toy_diabetes_like_df(self, n: int = 180) -> pd.DataFrame:
        rng = np.random.RandomState(0)

        # 20 % positives, 80 % negatives
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

    def test_inspect_preprocessor_feature_slices_returns_nonnegative_dims(self):
        df = self._toy_diabetes_like_df(n=50)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        info = s12.inspect_preprocessor_feature_slices(df, cfg)

        self.assertGreaterEqual(info.n_num_features, 1)
        self.assertGreater(info.total_dim, 0)
        self.assertLessEqual(info.num_slice.stop, info.total_dim)

    def test_build_logistic_pipeline_with_and_without_selection(self):
        df = self._toy_diabetes_like_df(n=80)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        # Full-features pipeline
        pipe_full = s12.build_logistic_pipeline_with_selection(
            df,
            preprocess_config=cfg,
            fs_config=s12.FeatureSelectionConfig(selector_type="none", k_numeric=5),
        )

        # Selected-features pipeline (numeric k-best)
        pipe_fs = s12.build_logistic_pipeline_with_selection(
            df,
            preprocess_config=cfg,
            fs_config=s12.FeatureSelectionConfig(selector_type="filter_numeric_kbest", k_numeric=3),
        )

        X = df.drop(columns=["readmitted_30d"])
        y = df["readmitted_30d"].astype(int)

        pipe_full.fit(X, y)
        pipe_fs.fit(X, y)

        # Both pipelines must produce valid probability scores
        proba_full = pipe_full.predict_proba(X)
        proba_fs = pipe_fs.predict_proba(X)
        self.assertEqual(proba_full.shape[0], len(df))
        self.assertEqual(proba_fs.shape[0], len(df))

    def test_run_section12_feature_selection_produces_table_and_uses_same_folds(self):
        df = self._toy_diabetes_like_df(n=180)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        y = df["readmitted_30d"].astype(int).to_numpy()
        splits = s7.make_stratified_cv_splits(y, n_splits=5, random_state=123)

        out = s12.run_section12_feature_selection(
            df,
            target_col="readmitted_30d",
            preprocess_config=cfg,
            n_splits=5,
            cv_random_state=123,
            model_random_state=0,
            k_numeric=5,
            cv_splits=splits,
        )

        # Two rows: logistic_full vs logistic_fs
        self.assertEqual(out.summary_table.shape[0], 2)
        self.assertIn("roc_auc_mean", out.summary_table.columns)
        self.assertIn("f1_mean", out.summary_table.columns)

        r_full = out.results_by_name["logistic_full"]
        r_fs = out.results_by_name["logistic_fs"]

        # Same fold fingerprint (identical folds)
        self.assertEqual(r_full.split_fingerprint, r_fs.split_fingerprint)
        self.assertEqual(r_full.split_fingerprint, out.split_fingerprint)

        # ROC-AUC values are finite in both cases
        self.assertEqual(len(r_full.fold_scores["roc_auc"]), 5)
        self.assertTrue(np.isfinite(r_full.fold_scores["roc_auc"]).all())
        self.assertEqual(len(r_fs.fold_scores["roc_auc"]), 5)
        self.assertTrue(np.isfinite(r_fs.fold_scores["roc_auc"]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
