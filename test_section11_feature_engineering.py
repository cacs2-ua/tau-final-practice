
import unittest
import pandas as pd
import numpy as np

import section11_feature_engineering as s11


class TestSection11FeatureEngineering(unittest.TestCase):
    def _toy_df(self):
        return pd.DataFrame(
            {
                "age": ["[30-40)", "[50-60)", "?", None],
                "time_in_hospital": [1, 4, 7, 0],
                "num_lab_procedures": [10, 20, 70, "?"],
                "num_procedures": [0, 2, 1, 3],
                "num_medications": [3, 8, 12, 5],
                "number_outpatient": [0, 1, 0, 2],
                "number_emergency": [0, 0, 2, 1],
                "number_inpatient": [0, 2, 1, 0],
                "number_diagnoses": [1, 3, 5, 2],
                "diag_1": ["250.83", "401", "V27", "?"],
                "diag_2": ["276", "403", "250.01", None],
                "diag_3": ["8", "?", "E879", "250"],
                "A1Cresult": ["None", "Norm", ">7", "?"],
                "max_glu_serum": ["None", ">200", "Norm", ">300"],
                "metformin": ["No", "Up", "Steady", None],
                "insulin": ["No", "Steady", "Up", "Down"],
            }
        )

    def test_icd9_grouping_basic(self):
        self.assertEqual(s11.icd9_group("250.83"), "DIABETES")
        self.assertEqual(s11.icd9_group("401"), "CIRCULATORY")
        self.assertEqual(s11.icd9_group("V27"), "SUPPLEMENTARY_V")
        self.assertEqual(s11.icd9_group("E879"), "EXTERNAL_E")
        self.assertEqual(s11.icd9_group("?"), "__MISSING__")

    def test_add_engineered_features_adds_expected_columns(self):
        df = self._toy_df()
        out = s11.add_engineered_features(df)

        # Not in-place
        self.assertNotEqual(id(df), id(out))

        must_cols = [
            "age_midpoint",
            "time_in_hospital_log1p",
            "number_inpatient_gt0",
            "los_bin",
            "polypharmacy_bin",
            "labs_per_day",
            "total_prior_utilization",
            "diag_1_group",
            "diag_2_group",
            "diag_3_group",
            "n_distinct_diag_groups",
            "has_diabetes_diag",
            "n_meds_up",
            "n_meds_changed",
            "any_insulin",
            "A1C_measured",
            "A1C_abnormal",
            "glu_measured",
            "glu_high",
            "age_x_inpatient",
            "los_x_meds",
            "diagnoses_x_meds",
            "diabetes_diag_x_insulin",
        ]
        for c in must_cols:
            self.assertIn(c, out.columns)

    def test_age_midpoint_values(self):
        df = self._toy_df()
        out = s11.add_engineered_features(df)

        self.assertAlmostEqual(float(out.loc[0, "age_midpoint"]), 35.0)
        self.assertAlmostEqual(float(out.loc[1, "age_midpoint"]), 55.0)
        self.assertTrue(np.isnan(out.loc[2, "age_midpoint"]))
        self.assertTrue(np.isnan(out.loc[3, "age_midpoint"]))

    def test_bins(self):
        df = self._toy_df()
        out = s11.add_engineered_features(df)

        self.assertEqual(out.loc[0, "los_bin"], "short_1_2")
        self.assertEqual(out.loc[1, "los_bin"], "medium_3_5")
        self.assertEqual(out.loc[2, "los_bin"], "long_6_plus")

        self.assertEqual(out.loc[0, "polypharmacy_bin"], "low_0_5")
        self.assertEqual(out.loc[1, "polypharmacy_bin"], "medium_6_10")
        self.assertEqual(out.loc[2, "polypharmacy_bin"], "high_11_plus")

    def test_rates(self):
        df = self._toy_df()
        out = s11.add_engineered_features(df)

        # Row 0: labs_per_day = 10 / 1
        self.assertAlmostEqual(float(out.loc[0, "labs_per_day"]), 10.0)
        # Row 1: labs_per_day = 20 / 4 = 5
        self.assertAlmostEqual(float(out.loc[1, "labs_per_day"]), 5.0)

    def test_med_summaries(self):
        df = self._toy_df()
        out = s11.add_engineered_features(df)

        # Row 1: metformin=Up -> n_meds_up at least 1
        self.assertGreaterEqual(int(out.loc[1, "n_meds_up"]), 1)
        # insulin Steady -> any_insulin = 1
        self.assertEqual(int(out.loc[1, "any_insulin"]), 1)
        # Row 0 insulin No -> any_insulin = 0
        self.assertEqual(int(out.loc[0, "any_insulin"]), 0)

    def test_lab_decomposition(self):
        df = self._toy_df()
        out = s11.add_engineered_features(df)

        # A1C: None => not measured
        self.assertEqual(int(out.loc[0, "A1C_measured"]), 0)
        # A1C: Norm => measured but not abnormal
        self.assertEqual(int(out.loc[1, "A1C_measured"]), 1)
        self.assertEqual(int(out.loc[1, "A1C_abnormal"]), 0)
        # A1C: >7 => abnormal
        self.assertEqual(int(out.loc[2, "A1C_abnormal"]), 1)

        # Glu: >200 => measured and high
        self.assertEqual(int(out.loc[1, "glu_measured"]), 1)
        self.assertEqual(int(out.loc[1, "glu_high"]), 1)

    def test_diag_breadth_and_diabetes_flag(self):
        df = self._toy_df()
        out = s11.add_engineered_features(df)

        # Row 0 has diag_1=DIABETES
        self.assertEqual(int(out.loc[0, "has_diabetes_diag"]), 1)
        # Row 1 diag_1=401 CIRC + diag_2=403 CIRC + diag_3 missing => breadth should be 1
        self.assertEqual(int(out.loc[1, "n_distinct_diag_groups"]), 1)

    def test_interactions(self):
        df = self._toy_df()
        out = s11.add_engineered_features(df)

        # age_x_inpatient row 1: age_midpoint=55, inpatient=2 => 110
        self.assertAlmostEqual(float(out.loc[1, "age_x_inpatient"]), 110.0)
        # los_x_meds row 2: los=7, meds=12 => 84
        self.assertAlmostEqual(float(out.loc[2, "los_x_meds"]), 84.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
