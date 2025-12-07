
import unittest

import math


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


import section5_preprocessing_pipeline as s5


class TestSection5PreprocessingPipeline(unittest.TestCase):
    def _toy_df(self):
        # Mini dataset that matches the real schema patterns:
        # - ID columns to drop
        # - ordinal age
        # - numeric counts + numeric-like strings (weight)
        # - int-coded categorical (admission_type_id)
        # - missing tokens ("?", "Unknown/Invalid")
        # - rare categorical values
        df = pd.DataFrame(
            {
                "encounter_id": [1, 2, 3, 4, 5, 6],
                "patient_nbr": [10, 11, 12, 13, 14, 15],
                "age": ["[0-10)", "[10-20)", "?", "[30-40)", "[40-50)", "[50-60)"],
                "weight": ["?", "80", "90", "?", "Unknown/Invalid", "70"],
                "num_lab_procedures": [41, 59, 11, 44, np.nan, 50],
                "admission_type_id": [6, 1, 1, 1, 6, 6],
                "medical_specialty": ["Pediatrics-Endocrinology", "?", "Cardiology", "Cardiology", "RareSpec", "Cardiology"],
                "diag_1": ["250.83", "276", "648", "8", "777", "8"],
                "readmitted_30d": [0, 1, 0, 0, 1, 0],
            }
        )
        return df

    def test_missing_token_cleaner_replaces_tokens_with_nan(self):
        df = self._toy_df()
        cleaner = s5.MissingTokenCleaner(missing_tokens=("?", "UNKNOWN/INVALID"))
        out = cleaner.transform(df[["medical_specialty", "weight", "age"]])

        # '?' and 'Unknown/Invalid' should become NaN
        self.assertTrue(pd.isna(out.loc[1, "medical_specialty"]))
        self.assertTrue(pd.isna(out.loc[0, "weight"]))
        self.assertTrue(pd.isna(out.loc[4, "weight"]))
        self.assertTrue(pd.isna(out.loc[2, "age"]))

    def test_infer_feature_groups_expected(self):
        df = self._toy_df()
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=False, rare_min_count=2)

        groups = s5.infer_feature_groups(df, cfg)

        # IDs dropped
        self.assertIn("encounter_id", groups.dropped)
        self.assertIn("patient_nbr", groups.dropped)

        # age is ordinal
        self.assertIn("age", groups.ordinal)

        # numeric includes weight + num_lab_procedures
        self.assertIn("weight", groups.numeric)
        self.assertIn("num_lab_procedures", groups.numeric)

        # admission_type_id forced categorical (not numeric)
        self.assertIn("admission_type_id", groups.categorical)
        self.assertNotIn("admission_type_id", groups.numeric)

    def test_preprocessor_fit_transform_no_crash_and_no_nan(self):
        df = self._toy_df()
        X = df.drop(columns=["readmitted_30d"])
        y = df["readmitted_30d"]

        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=False, rare_min_count=2)
        pre = s5.build_preprocessor(df, cfg)

        pre.fit(X, y)
        Xt = pre.transform(X)

        self.assertEqual(Xt.shape[0], len(df))
        # After imputation/encoding there should be no NaNs (Ordinal unknown becomes -1, not NaN)
        self.assertFalse(np.isnan(np.asarray(Xt)).any())

    def test_unseen_category_does_not_break_transform(self):
        df = self._toy_df()
        train = df.iloc[:4].copy()
        test = df.iloc[4:].copy()

        X_train = train.drop(columns=["readmitted_30d"])
        y_train = train["readmitted_30d"]
        X_test = test.drop(columns=["readmitted_30d"])

        # Force rare grouping to be aggressive, so unseen/rare go to __OTHER__
        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=False, rare_min_count=2)
        pre = s5.build_preprocessor(train, cfg)

        pre.fit(X_train, y_train)
        _ = pre.transform(X_test)  # should not raise

        self.assertTrue(True)

    def test_rare_category_grouper_maps_rare_to_other(self):
        X = np.array(
            [
                ["A"],
                ["A"],
                ["B"],  # rare if min_count=2
                ["__MISSING__"],
            ],
            dtype=object,
        )
        g = s5.RareCategoryGrouper(min_count=2, other_label="__OTHER__", missing_label="__MISSING__")
        g.fit(X)
        Xt = g.transform(np.array([["B"], ["A"], ["C"], [np.nan]], dtype=object))

        # B is rare -> __OTHER__; C unseen -> __OTHER__; nan -> __MISSING__
        self.assertEqual(Xt[0, 0], "__OTHER__")
        self.assertEqual(Xt[1, 0], "A")
        self.assertEqual(Xt[2, 0], "__OTHER__")
        self.assertEqual(Xt[3, 0], "__MISSING__")

    def test_pipeline_works_inside_cross_validation(self):
        df = self._toy_df()
        X = df.drop(columns=["readmitted_30d"])
        y = df["readmitted_30d"]

        cfg = s5.DiabetesPreprocessConfig(onehot_sparse=False, rare_min_count=2)
        pre = s5.build_preprocessor(df, cfg)

        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )

        # ROC-AUC requires BOTH classes in every test fold.
        # With tiny toy data, choose n_splits <= min(class_count) to avoid undefined AUC.
        counts = pd.Series(y).value_counts()
        min_class = int(counts.min())
        if min_class < 2:
            self.skipTest("Need at least 2 samples per class to compute ROC-AUC in CV.")

        from sklearn.model_selection import StratifiedKFold

        n_splits = min(3, min_class)  # ensures each fold has at least 1 sample of each class
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        self.assertEqual(len(scores), n_splits)
        self.assertTrue(np.isfinite(scores).all())

