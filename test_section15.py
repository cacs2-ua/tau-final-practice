import unittest
import numpy as np

import section15_statistical_comparison as s15


class TestSection15Wilcoxon(unittest.TestCase):
    def test_rejects_when_a_consistently_better(self):
        a = np.linspace(0.75, 0.84, 10)
        b = a - 0.05
        res = s15.compare_two_models_wilcoxon("A", "B", a, b, metric="roc_auc", alpha=0.05, alternative="two-sided")
        self.assertTrue(res.p_value < 0.05)
        self.assertTrue(res.reject_null)
        self.assertIsNotNone(res.effect_size_rbc)
        self.assertGreater(res.effect_size_rbc, 0.8)  # should be strongly positive

    def test_mismatched_lengths_raise(self):
        a = np.ones(10)
        b = np.ones(9)
        with self.assertRaises(ValueError):
            s15.compare_two_models_wilcoxon("A", "B", a, b)

    def test_identical_scores_returns_p_one(self):
        a = np.ones(10) * 0.77
        b = np.ones(10) * 0.77
        res = s15.compare_two_models_wilcoxon("A", "B", a, b, alpha=0.05)
        self.assertEqual(res.p_value, 1.0)
        self.assertFalse(res.reject_null)
        self.assertEqual(res.n_nonzero, 0)
        self.assertIsNone(res.effect_size_rbc)

    def test_pick_top2_by_mean_metric(self):
        # fake result objects with fold_scores dict
        class R:
            def __init__(self, scores, fp="same"):
                self.fold_scores = {"roc_auc": np.asarray(scores, dtype=float)}
                self.split_fingerprint = fp

        results = {
            "m1": R([0.70] * 10),
            "m2": R([0.80] * 10),
            "m3": R([0.75] * 10),
        }
        best, second = s15.pick_top2_by_mean_metric(results, metric="roc_auc")
        self.assertEqual(best, "m2")
        self.assertEqual(second, "m3")

    def test_fingerprint_mismatch_raises_when_strict(self):
        class R:
            def __init__(self, scores, fp):
                self.fold_scores = {"roc_auc": np.asarray(scores, dtype=float)}
                self.split_fingerprint = fp

        a = R([0.80] * 10, fp="fp_a")
        b = R([0.78] * 10, fp="fp_b")

        with self.assertRaises(ValueError):
            s15.compare_results_objects_wilcoxon(
                "A", a, "B", b,
                metric="roc_auc",
                strict_same_folds=True,
            )


if __name__ == "__main__":
    unittest.main()
