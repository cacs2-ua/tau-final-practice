import unittest
import numpy as np
import pandas as pd

import section4_eda_imbalance as s4


class TestSection4EDAImbalance(unittest.TestCase):
    def setUp(self):
        # Pequeño dataframe con números, categóricas y target binario
        self.df = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4],
                "feat2": ["a", "?", "", "b"],       # '?' y '' se consideran missing
                "feat3": [np.nan, "x", "y", "z"],  # NaN se considera missing
                "y": [0, 0, 1, 0],
            }
        )

    # ---- 4.1: class imbalance -----------------------------------------

    def test_compute_class_imbalance_basic(self):
        summary = s4.compute_class_imbalance(self.df["y"])
        self.assertEqual(summary.N, 4)
        self.assertEqual(summary.n_pos, 1)
        self.assertEqual(summary.n_neg, 3)
        self.assertAlmostEqual(summary.pos_rate, 0.25)
        self.assertAlmostEqual(summary.neg_rate, 0.75)
        d = summary.as_dict()
        self.assertIn("n_pos", d)
        self.assertIn("pos_rate", d)

    # ---- 4.2: missingness predicates ----------------------------------

    def test_is_missing_handles_nan_and_tokens(self):
        self.assertTrue(s4.is_missing(None))
        self.assertTrue(s4.is_missing(np.nan))
        self.assertTrue(s4.is_missing("   "))          # vacío tras strip
        self.assertTrue(s4.is_missing("?"))
        self.assertTrue(s4.is_missing(" unknown "))    # token normalizado
        self.assertFalse(s4.is_missing("value"))

    def test_missingness_by_feature_rates_and_counts(self):
        mf = s4.missingness_by_feature(
            self.df,
            feature_cols=["feat1", "feat2", "feat3"],
        )
        # feat1 no tiene missing
        self.assertEqual(int(mf.loc["feat1", "missing_count"]), 0)
        self.assertAlmostEqual(float(mf.loc["feat1", "missing_rate"]), 0.0)

        # feat2: '?' y '' -> 2 de 4 => 0.5
        self.assertEqual(int(mf.loc["feat2", "missing_count"]), 2)
        self.assertAlmostEqual(float(mf.loc["feat2", "missing_rate"]), 0.5)

        # feat3: un NaN -> 1 de 4 => 0.25
        self.assertEqual(int(mf.loc["feat3", "missing_count"]), 1)
        self.assertAlmostEqual(float(mf.loc["feat3", "missing_rate"]), 0.25)

    def test_missingness_by_row_summary_and_high_missing_fraction(self):
        res = s4.missingness_by_row(
            self.df,
            feature_cols=["feat1", "feat2", "feat3"],
            high_missing_threshold=0.30,
        )

        # Comprobamos longitudes
        self.assertEqual(len(res.missing_per_row), 4)
        self.assertEqual(len(res.missing_rate_per_row), 4)

        # Cuentas esperadas por fila:
        # fila0: NaN en feat3 -> 1
        # fila1: '?' en feat2 -> 1
        # fila2: '' en feat2 -> 1
        # fila3: sin missing -> 0
        self.assertListEqual(res.missing_per_row.tolist(), [1, 1, 1, 0])

        # 3 de 4 filas con ratio >= 1/3 (~0.33) > 0.30
        self.assertAlmostEqual(res.high_missing_fraction, 0.75)

        d = res.as_dict()
        self.assertIn("missing_per_row_summary", d)
        self.assertIn("high_missing_fraction", d)

    # ---- 4.3: top-k tables for high-cardinality categoricals ----------

    def test_topk_table_collapses_missing_and_computes_coverage(self):
        top, nunique, coverage = s4.topk_table(self.df["feat2"], k=2)
        # categorías: 'a', 'b', '__MISSING__'
        self.assertEqual(nunique, 3)
        # top-2 cubre '__MISSING__'(2) + 'a'(1) = 3/4 => 0.75
        self.assertAlmostEqual(coverage, 0.75)

        # la categoría '__MISSING__' debe estar en la tabla
        self.assertIn("__MISSING__",
                      top.index.astype(str).tolist())

    # ---- 4.4: dummy baseline under imbalance --------------------------

    def test_run_dummy_most_frequent_baseline_behaviour(self):
        # Dataset más grande e imbalance más claro
        df_large = pd.DataFrame(
            {
                "f1": np.arange(50),
                "y": [0] * 40 + [1] * 10,  # 80% negativos
            }
        )

        result = s4.run_dummy_most_frequent_baseline(
            df_large,
            feature_cols=["f1"],
            target_col="y",
            test_size=0.2,
            random_state=0,
        )

        cm = result.confusion_terms
        metrics_dict = result.metrics

        # Estrategia 'most_frequent' debe predecir solo la clase mayoritaria (0)
        self.assertEqual(cm["tp"], 0)
        self.assertEqual(cm["fp"], 0)
        self.assertGreater(cm["tn"], 0)
        self.assertGreater(cm["fn"], 0)

        # Bajo desequilibrio: accuracy > 0, pero recall y F1 del positivo = 0
        self.assertGreater(metrics_dict["accuracy"], 0.0)
        self.assertAlmostEqual(metrics_dict["recall"], 0.0)
        self.assertAlmostEqual(metrics_dict["f1"], 0.0)

        d = result.as_dict()
        self.assertIn("confusion_terms", d)
        self.assertIn("metrics", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
