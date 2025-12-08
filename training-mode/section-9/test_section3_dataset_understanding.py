
import unittest
import pandas as pd

import section3_dataset_understanding as s3


class TestSection3DatasetUnderstanding(unittest.TestCase):
    def _tiny_df(self):
        df = pd.DataFrame(
            {
                "title": ["Paper A", "Paper B", "Paper B"],
                "abstract": ["Text A", "Text B", "Text B"],
                "url": ["u1", "u2", "u2"],   # duplicate to test duplicate detection
                "venue": ["EMNLP", "EMNLP", "?"],  # token-like missing/unknown
                "year": [2016, 2017, 2017],
            }
        )
        df["text"] = df["title"] + " " + df["abstract"]  # derived
        return df

    def test_build_data_dictionary_infers_expected_roles_and_scales(self):
        df = self._tiny_df()
        dd = s3.build_data_dictionary(df)

        cols = set(dd["column"].tolist())
        self.assertTrue({"title", "abstract", "url", "venue", "year", "text"}.issubset(cols))

        year_row = dd.loc[dd["column"] == "year"].iloc[0]
        self.assertEqual(year_row["kind"], "numeric")
        self.assertEqual(year_row["scale"], "interval")  # year should be interval by default
        self.assertIn(year_row["role"], {"metadata_feature", "numeric_feature"})

        url_row = dd.loc[dd["column"] == "url"].iloc[0]
        self.assertEqual(url_row["role"], "identifier")

        text_row = dd.loc[dd["column"] == "text"].iloc[0]
        self.assertEqual(text_row["kind"], "text")
        self.assertEqual(text_row["role"], "derived_feature")

        title_row = dd.loc[dd["column"] == "title"].iloc[0]
        self.assertEqual(title_row["kind"], "text")
        self.assertEqual(title_row["role"], "raw_text_feature")

    def test_detect_missing_tokens_finds_question_mark(self):
        df = self._tiny_df()
        mt = s3.detect_missing_tokens(df, missing_tokens=("?", "N/A"))

        # We expect venue has "?" exactly once
        found = mt[(mt["column"] == "venue") & (mt["token"] == "?")]
        self.assertEqual(int(found["count"].iloc[0]), 1)

    def test_missing_summary_counts_empty_strings(self):
        df = self._tiny_df()
        df.loc[1, "abstract"] = "   "  # empty after strip
        ms = s3.summarize_missingness(df, treat_empty_as_missing=True)

        abs_row = ms.loc[ms["column"] == "abstract"].iloc[0]
        self.assertGreaterEqual(int(abs_row["empty_string_count"]), 1)
        self.assertGreaterEqual(int(abs_row["missing_total"]), 1)

    def test_duplicates_report_detects_key_duplicates(self):
        df = self._tiny_df()
        rep = s3.duplicates_report(df, key_columns=("url",), content_columns=("title", "abstract"))

        key_row = rep.loc[rep["scope"] == "key_duplicates"].iloc[0]
        self.assertEqual(key_row["available_subset"], ["url"])
        self.assertGreaterEqual(int(key_row["duplicate_rows"]), 2)  # u2 repeated => two rows duplicated

    def test_leakage_risk_report_flags_url_as_identifier_like(self):
        df = self._tiny_df()
        lr = s3.leakage_risk_report(df, id_columns=("url", "id"))

        self.assertTrue("column" in lr.columns)
        self.assertTrue(any(lr["column"] == "url"))

    def test_profile_dataset_accepts_records_list(self):
        records = [
            {"title": "T1", "abstract": "A1", "url": "u1", "venue": "EMNLP", "year": 2016},
            {"title": "T2", "abstract": "A2", "url": "u2", "venue": "EMNLP", "year": 2017},
        ]
        report = s3.profile_dataset(records)
        self.assertTrue(hasattr(report, "data_dictionary"))
        self.assertTrue(hasattr(report, "missing_summary"))
        self.assertTrue(hasattr(report, "duplicates_summary"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
