
# test_section15_statistical_comparison.py
import unittest
import numpy as np
import pandas as pd
import warnings

import section15_statistical_comparison as s15


class DummyCVResult:
    def __init__(self, fold_scores, split_fingerprint=None):
        self.fold_scores = fold_scores
        self.split_fingerprint = split_fingerprint


class TestSection15StatisticalComparison(unittest.TestCase):

    # ---------- _as_1d_float ----------
    def test_as_1d_float_returns_1d_float_array(self):
        arr = s15._as_1d_float([1, 2, 3])
        self.assertEqual(arr.ndim, 1)
        self.assertEqual(arr.dtype, float)
        np.testing.assert_allclose(arr, np.array([1.0, 2.0, 3.0]))

    def test_as_1d_float_rejects_non_numeric(self):
        with self.assertRaises(ValueError):
            s15._as_1d_float(["a", "b"])

    # ---------- _rank_biserial_from_differences ----------
    def test_rank_biserial_all_positive_is_one(self):
        d = np.array([0.1, 0.2, 0.3, 0.4])
        r, n = s15._rank_biserial_from_differences(d)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(r, 1.0, places=12)

    def test_rank_biserial_all_negative_is_minus_one(self):
        d = np.array([-0.1, -0.2, -0.3, -0.4])
        r, n = s15._rank_biserial_from_differences(d)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(r, -1.0, places=12)

    def test_rank_biserial_ignores_zeros(self):
        d = np.array([0.0, 1.0, 0.0, -2.0])  # two non-zero diffs
        r, n = s15._rank_biserial_from_differences(d)
        self.assertEqual(n, 2)
        self.assertTrue(-1.0 <= r <= 1.0)

    def test_rank_biserial_all_zeros_returns_zero_and_n0(self):
        d = np.array([0.0, 0.0, 0.0])
        r, n = s15._rank_biserial_from_differences(d)
        self.assertEqual(n, 0)
        self.assertEqual(r, 0.0)

    # ---------- wilcoxon_compare_fold_scores ----------
    def test_wilcoxon_compare_mismatched_lengths_raises(self):
        a = [0.7, 0.8, 0.9]
        b = [0.7, 0.8]
        with self.assertRaises(ValueError):
            s15.wilcoxon_compare_fold_scores(a, b)

    def test_wilcoxon_identical_scores_default_wilcox_tie_and_not_reject(self):
        a = [0.7, 0.8, 0.9, 0.85]
        b = [0.7, 0.8, 0.9, 0.85]

        # Some SciPy versions emit RuntimeWarning and return p=nan instead of raising
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rep = s15.wilcoxon_compare_fold_scores(a, b, zero_method="wilcox")

        self.assertEqual(rep.winner, "tie")
        self.assertFalse(rep.reject_h0)
        self.assertEqual(rep.n_pairs, 0)  # effect size removes zero diffs

        # Accept either p=1.0 (if you add the guard) or p=nan (SciPy behavior)
        self.assertTrue(np.isnan(rep.p_value) or abs(rep.p_value - 1.0) < 1e-12)


    def test_wilcoxon_identical_scores_with_zsplit_returns_p1(self):
        a = [0.7, 0.8, 0.9, 0.85]
        b = [0.7, 0.8, 0.9, 0.85]
        rep = s15.wilcoxon_compare_fold_scores(a, b, zero_method="zsplit")
        self.assertAlmostEqual(rep.p_value, 1.0, places=12)
        self.assertFalse(rep.reject_h0)
        self.assertEqual(rep.winner, "tie")
        self.assertEqual(rep.n_pairs, 0)  # because effect size removes zero diffs

    def test_wilcoxon_clear_winner_sets_winner_and_effect_positive(self):
        # A consistently better than B
        a = np.array([0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
        b = np.array([0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79])
        rep = s15.wilcoxon_compare_fold_scores(a, b, model_a_name="A", model_b_name="B", alpha=0.05)
        self.assertEqual(rep.winner, "A")
        self.assertGreater(rep.median_diff, 0.0)
        self.assertGreater(rep.effect_r_rank_biserial, 0.0)
        self.assertLess(rep.p_value, 0.05)

    # ---------- compare_two_cv_results ----------
    def test_compare_two_cv_results_metric_missing_raises_keyerror(self):
        res_a = DummyCVResult(fold_scores={"roc_auc": np.array([0.7, 0.8])}, split_fingerprint="fp1")
        res_b = DummyCVResult(fold_scores={"f1": np.array([0.7, 0.8])}, split_fingerprint="fp1")
        with self.assertRaises(KeyError):
            s15.compare_two_cv_results(res_a, res_b, metric="roc_auc")

    def test_compare_two_cv_results_fingerprint_mismatch_raises(self):
        res_a = DummyCVResult(fold_scores={"roc_auc": np.array([0.7, 0.8, 0.9])}, split_fingerprint="fpA")
        res_b = DummyCVResult(fold_scores={"roc_auc": np.array([0.6, 0.7, 0.8])}, split_fingerprint="fpB")
        with self.assertRaises(ValueError):
            s15.compare_two_cv_results(res_a, res_b, metric="roc_auc")

    def test_compare_two_cv_results_works_when_fingerprint_matches(self):
        a = np.array([0.8, 0.82, 0.81, 0.83, 0.84])
        b = np.array([0.75, 0.76, 0.74, 0.77, 0.78])
        res_a = DummyCVResult(fold_scores={"roc_auc": a}, split_fingerprint="fp1")
        res_b = DummyCVResult(fold_scores={"roc_auc": b}, split_fingerprint="fp1")
        rep = s15.compare_two_cv_results(res_a, res_b, metric="roc_auc", model_a_name="A", model_b_name="B")
        self.assertEqual(rep.split_fingerprint, "fp1")
        self.assertEqual(rep.winner, "A")

    def test_compare_two_cv_results_uses_existing_fingerprint_if_other_none(self):
        a = np.array([0.8, 0.81, 0.82])
        b = np.array([0.79, 0.80, 0.81])
        res_a = DummyCVResult(fold_scores={"roc_auc": a}, split_fingerprint=None)
        res_b = DummyCVResult(fold_scores={"roc_auc": b}, split_fingerprint="fpX")
        rep = s15.compare_two_cv_results(res_a, res_b, metric="roc_auc")
        self.assertEqual(rep.split_fingerprint, "fpX")

    # ---------- pick_best_two ----------
    def test_pick_best_two_higher_is_better(self):
        df = pd.DataFrame({
            "model": ["m1", "m2", "m3"],
            "roc_auc_mean": [0.70, 0.90, 0.80],
        })
        best, second = s15.pick_best_two(df, id_col="model", metric_col="roc_auc_mean", higher_is_better=True)
        self.assertEqual(best, "m2")
        self.assertEqual(second, "m3")

    def test_pick_best_two_lower_is_better(self):
        df = pd.DataFrame({
            "model": ["m1", "m2", "m3"],
            "loss": [0.30, 0.10, 0.20],
        })
        best, second = s15.pick_best_two(df, id_col="model", metric_col="loss", higher_is_better=False)
        self.assertEqual(best, "m2")
        self.assertEqual(second, "m3")

    def test_pick_best_two_requires_two_rows(self):
        df = pd.DataFrame({"model": ["m1"], "roc_auc_mean": [0.7]})
        with self.assertRaises(ValueError):
            s15.pick_best_two(df, id_col="model", metric_col="roc_auc_mean")

    # ---------- report formatting ----------
    def test_report_to_text_contains_key_lines(self):
        a = [0.9, 0.91, 0.92]
        b = [0.8, 0.81, 0.82]
        rep = s15.wilcoxon_compare_fold_scores(a, b, model_a_name="A", model_b_name="B")
        txt = rep.to_text()
        self.assertIn("Wilcoxon signed-rank test", txt)
        self.assertIn("Models: A=A vs B=B", txt)
        self.assertIn("decision:", txt)
        self.assertIn("winner", txt)


if __name__ == "__main__":
    unittest.main()
