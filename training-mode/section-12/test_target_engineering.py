import os
import unittest
import importlib.util

import numpy as np
import pandas as pd

# Robust import of /content/.../code.py without colliding with the stdlib 'code' module
def import_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_PATH = os.path.join(PROJECT_DIR, "code.py")
mod = import_module_from_path("project_code", CODE_PATH)


class TestTargetEngineering(unittest.TestCase):
    def test_binarize_basic_mapping(self):
        s = pd.Series(["NO", "<30", ">30", "NO", ">30", "<30"])
        y = mod.binarize_readmitted(s, unknown_policy="error")
        self.assertEqual(list(y.astype(int)), [0, 1, 0, 0, 0, 1])
        self.assertEqual(y.name, "readmitted_30d")
        self.assertTrue(str(y.dtype).lower() in ("int8", "int64", "int32"))

    def test_binarize_handles_whitespace_and_case(self):
        s = pd.Series([" no ", " <30", ">30 ", "No", "<30", " >30"])
        y = mod.binarize_readmitted(s, unknown_policy="error")
        self.assertEqual(list(y.astype(int)), [0, 1, 0, 0, 1, 0])

    def test_binarize_unknown_raises(self):
        s = pd.Series(["NO", "MAYBE", "<30"])
        with self.assertRaises(mod.TargetEngineeringError):
            _ = mod.binarize_readmitted(s, unknown_policy="error")

    def test_binarize_unknown_to_nan(self):
        s = pd.Series(["NO", "MAYBE", "<30", None])
        y = mod.binarize_readmitted(s, unknown_policy="nan")
        # Expect: NO->0, MAYBE->NA, <30->1, None->NA
        self.assertEqual(int(y.iloc[0]), 0)
        self.assertTrue(pd.isna(y.iloc[1]))
        self.assertEqual(int(y.iloc[2]), 1)
        self.assertTrue(pd.isna(y.iloc[3]))
        self.assertEqual(str(y.dtype), "Int8")

    def test_engineer_targets_adds_columns(self):
        df = pd.DataFrame(
            {
                "readmitted": ["NO", "<30", ">30"],
                "age": ["[50-60)", "[60-70)", "[40-50)"],
            }
        )
        out = mod.engineer_targets(df, add_multiclass=True, drop_source=False, unknown_policy="error")
        self.assertIn("readmitted_30d", out.columns)
        self.assertIn("readmitted_3class", out.columns)
        self.assertIn("readmitted", out.columns)
        self.assertEqual(list(out["readmitted_30d"].astype(int)), [0, 1, 0])
        # multiclass should be categorical with expected categories
        self.assertTrue(pd.api.types.is_categorical_dtype(out["readmitted_3class"]))

    def test_confusion_terms_binary(self):
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        terms = mod.confusion_terms_binary(y_true, y_pred)
        # Manually:
        # TP: positions 0 and 4 => 2
        # FN: position 1 => 1
        # FP: position 3 => 1
        # TN: positions 2 and 5 => 2
        self.assertEqual((terms.tp, terms.fn, terms.fp, terms.tn), (2, 1, 1, 2))

    def test_metrics_multiclass_with_auc(self):
        y_true = ["NO", "<30", ">30", "NO", "<30", ">30"]
        y_pred = ["NO", "<30", "NO", "NO", "<30", ">30"]
        # Probabilities aligned with labels order ("NO", "<30", ">30")
        y_proba = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.4, 0.2, 0.4],
                [0.7, 0.2, 0.1],
                [0.1, 0.7, 0.2],
                [0.2, 0.1, 0.7],
            ],
            dtype=float,
        )
        res = mod.metrics_multiclass(y_true, y_pred, y_proba=y_proba)
        for key in ["macro_f1", "weighted_f1", "balanced_accuracy", "ovr_auc_macro"]:
            self.assertIn(key, res)
            self.assertIsInstance(res[key], float)

    def test_keep_multiclass_error_policy(self):
        s = pd.Series(["NO", "<30", "???"])
        with self.assertRaises(mod.TargetEngineeringError):
            _ = mod.keep_multiclass_readmitted(s, unknown_policy="error")

    def test_keep_multiclass_nan_policy(self):
        s = pd.Series(["NO", "<30", "???"])
        out = mod.keep_multiclass_readmitted(s, unknown_policy="nan")
        self.assertTrue(pd.isna(out.iloc[2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
