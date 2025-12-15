import unittest
import numpy as np
import pandas as pd

import section5_preprocessing_pipeline as s5
import section7_validation_protocol as s7
import section12_feature_selection as s12


class TestSection12FeatureSelection(unittest.TestCase):
    def _toy_df(self, n=220):
        rng = np.random.RandomState(0)
        y = np.array([0]*int(n*0.8) + [1]*(n-int(n*0.8)), dtype=int)
        rng.shuffle(y)

        df = pd.DataFrame({
            "encounter_id": np.arange(n),
            "patient_nbr": np.arange(10000, 10000+n),
            "age": rng.choice(list(s5.AGE_ORDER) + ["?"], size=n),
            "weight": rng.choice(["?", "70", "80", "90", "Unknown/Invalid"], size=n),
            "time_in_hospital": rng.randint(1, 8, size=n),
            "num_lab_procedures": rng.poisson(40, size=n),
            "medical_specialty": rng.choice(["Cardiology", "InternalMedicine", "?", "RareSpec"], size=n),
            "diag_1": rng.choice(["250.83", "276", "648", "8", "401"], size=n),
            "insulin": rng.choice(["No", "Steady", "Up", "Down"], size=n),
            "readmitted_30d": y,
        })
        return df

    def test_run_section12(self):
        df = self._toy_df()
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=True, rare_min_count=1)
        y = df["readmitted_30d"].astype(int).to_numpy()
        splits = s7.make_stratified_cv_splits(y, n_splits=5, random_state=42)

        out = s12.run_section12_feature_selection(
            df,
            target_col="readmitted_30d",
            preprocess_config_base=cfg,
            cv_splits=splits,
            n_splits=5,
            methods=("filter_chi2", "embedded_l1"),
            chi2_k=30,
            l1_C=0.1,
            top_n_features=10,
            top_n_groups=10,
        )

        self.assertIn("method", out.summary_table.columns)
        self.assertIn("roc_auc_mean", out.summary_table.columns)
        self.assertTrue(len(out.summary_table) == 4)  # 2 methods x (all vs selected)
        self.assertIn("filter_chi2", out.interpretations)
        self.assertIn("top_groups", out.interpretations["filter_chi2"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
