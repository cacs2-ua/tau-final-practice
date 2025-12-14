import unittest
import numpy as np
import pandas as pd
import matplotlib
try:
    matplotlib.use("Agg")
except Exception:
    pass

import section16_error_analysis as s16


class TestSection16ErrorAnalysis(unittest.TestCase):
    def test_infer_decision_threshold_proba(self):
        scores = np.array([0.1, 0.3, 0.9])
        self.assertAlmostEqual(s16.infer_decision_threshold(scores), 0.5)

    def test_infer_decision_threshold_margin(self):
        scores = np.array([-2.0, 0.1, 3.4])
        self.assertAlmostEqual(s16.infer_decision_threshold(scores), 0.0)

    def test_build_error_table_types(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        scores = [0.1, 0.9, 0.2, 0.8]  # threshold inferred = 0.5
        df = s16.build_error_table(y_true, y_pred, scores)

        self.assertEqual(df.loc[0, "error_type"], "TN")
        self.assertEqual(df.loc[1, "error_type"], "FP")
        self.assertEqual(df.loc[2, "error_type"], "FN")
        self.assertEqual(df.loc[3, "error_type"], "TP")

        # confidence = |score - 0.5|
        self.assertAlmostEqual(float(df.loc[1, "confidence"]), abs(0.9 - 0.5))

    def test_select_systematic_failures_balanced(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [1, 0, 1, 0, 1, 0]  # FP at 0,2 and FN at 3,5
        scores = [0.99, 0.2, 0.95, 0.01, 0.8, 0.02]
        df = s16.build_error_table(y_true, y_pred, scores)
        sel = s16.select_systematic_failures(df, n_cases=3, prefer_balance_fp_fn=True)

        self.assertEqual(len(sel), 3)
        self.assertTrue(set(sel["error_type"]).issubset({"FP", "FN"}))
        # should include at least one FP and one FN if both exist
        self.assertIn("FP", set(sel["error_type"]))
        self.assertIn("FN", set(sel["error_type"]))

    def test_confusion_matrix_from_oof(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        scores = [0.1, 0.9, 0.2, 0.8]
        df = s16.build_error_table(y_true, y_pred, scores)
        cm = s16.confusion_matrix_from_oof(df)
        # layout [[TN, FP],[FN, TP]] => [[1,1],[1,1]]
        self.assertTrue((cm == np.array([[1, 1], [1, 1]])).all())

    def test_plot_functions_return_fig(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        scores = [0.1, 0.9, 0.2, 0.8]
        df = s16.build_error_table(y_true, y_pred, scores)
        cm = s16.confusion_matrix_from_oof(df)

        fig1 = s16.plot_confusion_matrix(cm)
        fig2 = s16.plot_score_distributions(df)

        self.assertIsNotNone(fig1)
        self.assertIsNotNone(fig2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
