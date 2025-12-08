
import unittest
import numpy as np
import pandas as pd

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7
import section10_imbalance_handling as s10


class TestSection10ImbalanceHandling(unittest.TestCase):
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

    def test_random_over_and_under_sampling_balance_counts(self):
        X = np.arange(20).reshape(-1, 1)
        y = np.array([0] * 16 + [1] * 4, dtype=int)  # minority=1

        Xo, yo = s10.random_oversample(X, y, ratio=1.0, random_state=0)
        c0, c1 = int((yo == 0).sum()), int((yo == 1).sum())
        self.assertEqual(c0, c1)

        Xu, yu = s10.random_undersample(X, y, ratio=1.0, random_state=0)
        c0u, c1u = int((yu == 0).sum()), int((yu == 1).sum())
        self.assertEqual(c0u, c1u)

    def test_best_threshold_max_f1_matches_bruteforce_on_small_case(self):
        y = np.array([0, 0, 1, 1], dtype=int)
        scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=float)

        thr = s10.best_threshold_max_f1(y, scores, n_grid=50)

        # brute force over unique scores
        best = None
        best_f1 = -1
        for t in np.unique(scores):
            yp = (scores >= t).astype(int)
            f1 = float(s10.f1_score(y, yp, pos_label=1, zero_division=0))  # sklearn metric via module import
            if f1 > best_f1:
                best_f1 = f1
                best = float(t)

        self.assertAlmostEqual(float(thr), float(best), places=12)

    def test_run_section10_experiments_runs_and_returns_table(self):
        df = self._toy_diabetes_like_df(n=260)
        y = df["readmitted_30d"].astype(int).to_numpy()

        splits = s7.make_stratified_cv_splits(y, n_splits=5, random_state=123)
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)

        table, results_map = s10.run_section10_imbalance_experiments(
            df,
            target_col="readmitted_30d",
            preprocess_config=cfg,
            cv_splits=splits,
            models=("logistic_regression", "decision_tree"),
            methods=("none", "class_weight_balanced", "random_over", "threshold_f1"),
            sampling_ratio=1.0,
            random_state=42,
            fast_mode=True,   # keep unit test fast
        )

        # 2 models * 4 methods = 8 rows
        self.assertEqual(table.shape[0], 8)
        self.assertIn("roc_auc_mean", table.columns)
        self.assertIn("f1_mean", table.columns)
        self.assertIn("recall_mean", table.columns)

        # results_map contains per-fold vectors
        self.assertEqual(len(results_map), 8)

        for (model, method), res in results_map.items():
            self.assertEqual(res.n_splits, 5)
            self.assertEqual(len(res.fold_scores["roc_auc"]), 5)
            self.assertTrue(np.isfinite(res.fold_scores["roc_auc"]).all())
            # Means should be within [0,1] for these metrics
            for m in ["roc_auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy"]:
                self.assertGreaterEqual(res.mean_scores[m], 0.0)
                self.assertLessEqual(res.mean_scores[m], 1.0)

            if method == "threshold_f1":
                self.assertIsNotNone(res.fold_thresholds)
                self.assertEqual(len(res.fold_thresholds), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
