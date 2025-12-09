import unittest
import warnings

import numpy as np
import pandas as pd

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7
import section14_hyperparameter_tuning as s14


def _make_synthetic_diabetes_like_df(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    ages = list(s5.AGE_ORDER)
    diag_vals = ["250.83", "401", "414", "486", "V27", "E879", "?", None]
    genders = ["Male", "Female", "?"]

    time_in_hospital = rng.randint(1, 15, size=n)
    num_medications = rng.randint(1, 35, size=n)
    number_diagnoses = rng.randint(1, 12, size=n)

    age = rng.choice(ages, size=n, replace=True)
    diag_1 = rng.choice(diag_vals, size=n, replace=True)
    gender = rng.choice(genders, size=n, replace=True)

    # Build a probabilistic target correlated with a couple of signals
    # (long stay + many meds + diabetes diag)
    is_diabetes = np.array([str(v).startswith("250") if v is not None and v != "?" else 0 for v in diag_1], dtype=float)
    lin = 0.15 * (time_in_hospital - 7) + 0.06 * (num_medications - 15) + 0.9 * is_diabetes + rng.normal(0, 0.7, size=n)
    p = 1.0 / (1.0 + np.exp(-lin))
    y = (rng.rand(n) < p).astype(int)

    df = pd.DataFrame(
        {
            "encounter_id": np.arange(n) + 1000,
            "patient_nbr": rng.randint(10_000, 99_999, size=n),
            "age": age,
            "gender": gender,
            "time_in_hospital": time_in_hospital,
            "num_medications": num_medications,
            "number_diagnoses": number_diagnoses,
            "diag_1": diag_1,
            "readmitted_30d": y,
        }
    )
    return df


class TestSection14HyperparameterTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")  # keep tests clean in Colab

    def test_manual_sweep_runs_and_selects_best(self):
        df = _make_synthetic_diabetes_like_df(n=220, seed=1)

        cfg = s5.DiabetesPreprocessConfig(
            onehot_sparse=True,
            rare_min_count=1,
            scale_numeric=True,
            scaler="standard",
        )

        # Fixed splits (Section 7 contract)
        y = df["readmitted_30d"].astype(int).to_numpy()
        splits = s7.make_stratified_cv_splits(y, n_splits=4, random_state=42)

        X = df.drop(columns=[c for c in cfg.target_cols if c in df.columns], errors="ignore")
        pipe = s14.build_tunable_pipeline(df, model="logistic_regression", preprocess_config_base=cfg, fast_mode=True)

        sweep = s14.manual_hyperparameter_sweep(
            pipe,
            X,
            y,
            cv_splits=splits,
            param_list=[{"clf__C": 0.1}, {"clf__C": 1.0}],
            refit_metric="roc_auc",
        )

        self.assertEqual(len(sweep.results_table), 2)
        self.assertIn("roc_auc_mean", sweep.results_table.columns)
        self.assertIn("clf__C", sweep.best_params)
        self.assertTrue(np.isfinite(sweep.best_score))

    def test_grid_search_fixed_splits_runs(self):
        df = _make_synthetic_diabetes_like_df(n=220, seed=2)

        cfg = s5.DiabetesPreprocessConfig(
            onehot_sparse=True,
            rare_min_count=1,
            scale_numeric=True,
            scaler="standard",
        )

        y = df["readmitted_30d"].astype(int).to_numpy()
        splits = s7.make_stratified_cv_splits(y, n_splits=4, random_state=42)

        X = df.drop(columns=[c for c in cfg.target_cols if c in df.columns], errors="ignore")
        pipe = s14.build_tunable_pipeline(df, model="logistic_regression", preprocess_config_base=cfg, fast_mode=True)

        res = s14.grid_search_fixed_splits(
            pipe,
            X,
            y,
            cv_splits=splits,
            param_grid={"clf__C": [0.1, 1.0]},
            scoring="roc_auc",
            n_jobs=1,
        )

        self.assertIn("mean_test_score", res.cv_results_table.columns)
        self.assertIn("rank_test_score", res.cv_results_table.columns)
        self.assertIn("clf__C", res.best_params)
        self.assertTrue(np.isfinite(res.best_score))

    def test_nested_cv_grid_search_runs(self):
        df = _make_synthetic_diabetes_like_df(n=240, seed=3)

        cfg = s5.DiabetesPreprocessConfig(
            onehot_sparse=True,
            rare_min_count=1,
            scale_numeric=True,
            scaler="standard",
        )

        y = df["readmitted_30d"].astype(int).to_numpy()
        outer_splits = s7.make_stratified_cv_splits(y, n_splits=4, random_state=42)

        X = df.drop(columns=[c for c in cfg.target_cols if c in df.columns], errors="ignore")
        pipe = s14.build_tunable_pipeline(df, model="logistic_regression", preprocess_config_base=cfg, fast_mode=True)

        nested = s14.nested_cv_grid_search(
            pipe,
            X,
            y,
            outer_splits=outer_splits,
            param_grid={"clf__C": [0.1, 1.0]},
            inner_n_splits=3,
            inner_random_state=123,
            n_jobs=1,
        )

        self.assertEqual(nested.outer_fold_scores.shape[0], 4)
        self.assertTrue(np.isfinite(nested.mean_score))
        self.assertIn("roc_auc", nested.outer_metrics_mean)
        self.assertEqual(len(nested.best_params_per_fold), 4)


if __name__ == "__main__":
    unittest.main()
