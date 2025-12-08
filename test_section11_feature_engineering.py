import numpy as np
import pandas as pd
import pytest

import section11_feature_engineering as fe


def _is_ordered_categorical(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) and bool(s.dtype.ordered)


# -------------------------
# Missing tokens + age
# -------------------------
def test_is_missing_token_default_behaviour():
    assert fe.is_missing_token(None) is True
    assert fe.is_missing_token(np.nan) is True
    assert fe.is_missing_token("") is True
    assert fe.is_missing_token("   ") is True
    assert fe.is_missing_token("?") is True
    assert fe.is_missing_token("unknown/invalid") is True
    assert fe.is_missing_token("NaN") is True
    assert fe.is_missing_token("NULL") is True

    # Non-missing examples
    assert fe.is_missing_token("0") is False
    assert fe.is_missing_token("No") is False
    assert fe.is_missing_token("Steady") is False


def test_age_midpoint_parsing_and_missing():
    # Your implementation accepts both bracketed and plain "a-b" formats (robust parsing).
    assert fe.age_midpoint("[50-60)") == 55.0
    assert fe.age_midpoint("(0-10]") == 5.0
    assert fe.age_midpoint("50-60") == 55.0

    # Invalid / missing -> np.nan
    assert np.isnan(fe.age_midpoint("50"))          # no hyphen / not a range
    assert np.isnan(fe.age_midpoint("abc-def"))     # non-numeric
    assert np.isnan(fe.age_midpoint("?"))
    assert np.isnan(fe.age_midpoint(None))


# -------------------------
# Fixed binning
# -------------------------
def test_bin_numeric_fixed_is_deterministic_and_ordered():
    s = pd.Series([0, 1, 2, 3, np.nan, "4"])
    b = fe.bin_numeric_fixed(s, bins=[-np.inf, 0, 2, np.inf], labels=["low", "mid", "high"])
    assert _is_ordered_categorical(b)
    # Expected bins (right=True by default): (-inf,0], (0,2], (2,inf]
    assert b.iloc[0] == "low"   # 0
    assert b.iloc[1] == "mid"   # 1
    assert b.iloc[2] == "mid"   # 2
    assert b.iloc[3] == "high"  # 3
    assert pd.isna(b.iloc[4])   # nan stays nan
    assert b.iloc[5] == "high"  # "4" coerces -> 4


# -------------------------
# ICD-9 grouping
# -------------------------
@pytest.mark.parametrize(
    "code,expected",
    [
        ("250.13", "diabetes"),
        ("250", "diabetes"),
        ("250.0", "diabetes"),
        ("V45", "supplementary_v"),
        ("E812", "external_e"),
        ("401", "circulatory"),
        ("530.81", "digestive"),
        ("786", "symptoms"),
        ("999", "injury"),
        ("abc", "other"),
        ("?", "__MISSING__"),
        (None, "__MISSING__"),
    ],
)
def test_icd9_grouping(code, expected):
    assert fe.icd9_group(code) == expected


# -------------------------
# Section functions
# -------------------------
def test_add_diag_group_features_adds_expected_columns_and_aggregates():
    df = pd.DataFrame(
        {
            "diag_1": ["250.13", "401", "?"],
            "diag_2": ["401", "V45", "250"],
            "diag_3": ["530.81", "250.0", None],
        }
    )
    out = fe.add_diag_group_features(df)

    # Group columns exist
    for c in ["diag_1_group", "diag_2_group", "diag_3_group"]:
        assert c in out.columns

    # Aggregates exist
    for c in ["n_diabetes_diags", "primary_diag_is_diabetes", "n_unique_diag_groups", "n_diags_same_as_primary_group"]:
        assert c in out.columns

    # Row-level checks
    # row0: diag groups -> diabetes, circulatory, digestive => n_diabetes_diags=1, primary is diabetes
    assert int(out.loc[0, "n_diabetes_diags"]) == 1
    assert int(out.loc[0, "primary_diag_is_diabetes"]) == 1
    assert int(out.loc[0, "n_unique_diag_groups"]) == 3
    assert int(out.loc[0, "n_diags_same_as_primary_group"]) == 1  # only diag_1 matches itself

    # row1: 401 circulatory, V45 supplementary_v, 250 diabetes => n_diabetes_diags=1, primary not diabetes
    assert int(out.loc[1, "n_diabetes_diags"]) == 1
    assert int(out.loc[1, "primary_diag_is_diabetes"]) == 0

    # row2: missing, diabetes, missing => n_diabetes_diags=1, primary missing -> 0
    assert int(out.loc[2, "n_diabetes_diags"]) == 1
    assert int(out.loc[2, "primary_diag_is_diabetes"]) == 0


def test_add_lab_features_measured_high_and_levels():
    df = pd.DataFrame(
        {
            "A1Cresult": ["None", "Norm", ">7", ">8", "?", None],
            "max_glu_serum": ["None", "Norm", ">200", ">300", "?", None],
            "num_lab_procedures": [10, 25, 45, 70, None, "60"],
        }
    )
    out = fe.add_lab_features(df)

    for c in ["a1c_measured", "a1c_high", "a1c_level", "glu_measured", "glu_high", "glu_level", "num_lab_procedures_bin"]:
        assert c in out.columns

    # a1c
    assert out["a1c_measured"].tolist() == [0, 1, 1, 1, 0, 0]
    assert out["a1c_high"].tolist() == [0, 0, 1, 1, 0, 0]
    assert out["a1c_level"].tolist() == [0, 1, 2, 3, 0, 0]

    # glucose
    assert out["glu_measured"].tolist() == [0, 1, 1, 1, 0, 0]
    assert out["glu_high"].tolist() == [0, 0, 1, 1, 0, 0]
    assert out["glu_level"].tolist() == [0, 1, 2, 3, 0, 0]

    assert _is_ordered_categorical(out["num_lab_procedures_bin"])


def test_add_utilization_features_creates_expected_features():
    df = pd.DataFrame(
        {
            "number_outpatient": [0, 1, 2, None],
            "number_emergency": [0, 0, 1, 3],
            "number_inpatient": [0, 2, 0, 1],
            "time_in_hospital": [2, 5, 10, "15"],
            "num_medications": [4, 10, 21, None],
            "number_diagnoses": [2, 4, 7, "6"],
        }
    )
    out = fe.add_utilization_features(df)

    for c in [
        "total_visits",
        "any_emergency",
        "any_inpatient",
        "any_outpatient",
        "any_prior_visit",
        "total_visits_bin",
        "stay_bin",
        "num_medications_bin",
        "polypharmacy",
        "number_diagnoses_bin",
    ]:
        assert c in out.columns

    # Row 0: no visits
    assert float(out.loc[0, "total_visits"]) == 0.0
    assert int(out.loc[0, "any_prior_visit"]) == 0

    # Row 1: outpatient 1 + inpatient 2 = 3 visits
    assert float(out.loc[1, "total_visits"]) == 3.0
    assert int(out.loc[1, "any_inpatient"]) == 1
    assert int(out.loc[1, "polypharmacy"]) == 1  # num_medications == 10

    assert _is_ordered_categorical(out["total_visits_bin"])
    assert _is_ordered_categorical(out["stay_bin"])
    assert _is_ordered_categorical(out["num_medications_bin"])
    assert _is_ordered_categorical(out["number_diagnoses_bin"])


def test_add_medication_burden_features_counts_active_and_changed_and_insulin_flags():
    df = pd.DataFrame(
        {
            "metformin": ["No", "Steady", "Up", "Down", "?", None],
            "insulin": ["No", "Steady", "Down", "Up", "?", None],
            # include another med from default list to test multi-col counting
            "glipizide": ["No", "No", "Steady", "Up", "Down", "?"],
        }
    )
    out = fe.add_medication_burden_features(df)

    for c in ["n_active_diabetes_meds", "n_changed_diabetes_meds", "any_active_diabetes_med", "any_changed_diabetes_med", "insulin_active", "insulin_changed"]:
        assert c in out.columns

    # Row-wise expectations:
    # row0: No/No/No -> active=0 changed=0 insulin_active=0 insulin_changed=0
    assert int(out.loc[0, "n_active_diabetes_meds"]) == 0
    assert int(out.loc[0, "n_changed_diabetes_meds"]) == 0
    assert int(out.loc[0, "insulin_active"]) == 0
    assert int(out.loc[0, "insulin_changed"]) == 0

    # row1: Steady insulin Steady -> active=2 (metformin+insulin), glipizide No -> total active=2, changed=0
    assert int(out.loc[1, "n_active_diabetes_meds"]) == 2
    assert int(out.loc[1, "n_changed_diabetes_meds"]) == 0
    assert int(out.loc[1, "insulin_active"]) == 1
    assert int(out.loc[1, "insulin_changed"]) == 0

    # row3: metformin Down + insulin Up + glipizide Up -> active=3, changed=3, insulin_changed=1
    assert int(out.loc[3, "n_active_diabetes_meds"]) == 3
    assert int(out.loc[3, "n_changed_diabetes_meds"]) == 3
    assert int(out.loc[3, "insulin_active"]) == 1
    assert int(out.loc[3, "insulin_changed"]) == 1


def test_add_interaction_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "age": ["[60-70)", "[30-40)", "?", None],
            "time_in_hospital": [2, 5, 10, 1],
            "num_medications": [5, 10, 2, None],
            "number_diagnoses": [2, 4, 7, 1],
            "total_visits": [0, 3, 1, 2],
        }
    )
    out = fe.add_interaction_features(df)

    for c in ["age_mid", "elderly", "stay_x_meds", "stay_x_diagnoses", "visits_x_stay", "age_x_meds", "age_x_visits"]:
        assert c in out.columns

    # elderly: age midpoint >= 65
    assert int(out.loc[0, "elderly"]) == 1
    assert int(out.loc[1, "elderly"]) == 0
    assert int(out.loc[2, "elderly"]) == 0  # missing treated as -1

    # interactions are numeric and non-negative with fillna(0)
    assert float(out.loc[0, "stay_x_meds"]) == 2 * 5
    assert float(out.loc[1, "stay_x_meds"]) == 5 * 10
    assert float(out.loc[3, "stay_x_meds"]) == 1 * 0  # num_medications missing -> 0


# -------------------------
# Main entry point
# -------------------------
def test_apply_feature_engineering_does_not_mutate_input_and_reports_added_columns():
    df = pd.DataFrame(
        {
            "age": ["[60-70)", "[30-40)"],
            "diag_1": ["250.13", "401"],
            "diag_2": ["401", "V45"],
            "diag_3": ["530.81", "250"],
            "A1Cresult": ["Norm", ">8"],
            "max_glu_serum": ["None", ">300"],
            "num_lab_procedures": [10, 70],
            "number_outpatient": [0, 1],
            "number_emergency": [0, 1],
            "number_inpatient": [0, 2],
            "time_in_hospital": [2, 5],
            "num_medications": [5, 10],
            "number_diagnoses": [2, 4],
            "metformin": ["No", "Steady"],
            "insulin": ["No", "Down"],
            "glipizide": ["No", "Up"],
        }
    )
    df_before = df.copy(deep=True)

    res = fe.apply_feature_engineering(df)

    # Original unchanged
    pd.testing.assert_frame_equal(df, df_before)

    # Returned df includes original columns + engineered ones
    assert set(df.columns).issubset(set(res.df.columns))
    assert isinstance(res.added_columns, list)

    # added_columns matches actual difference
    diff = sorted(list(set(res.df.columns) - set(df.columns)))
    assert res.added_columns == diff

    # Some key engineered columns exist
    expected_some = {
        "diag_1_group", "diag_2_group", "diag_3_group",
        "n_diabetes_diags", "primary_diag_is_diabetes",
        "a1c_measured", "a1c_level", "glu_measured", "glu_level",
        "total_visits", "total_visits_bin", "stay_bin",
        "n_active_diabetes_meds", "insulin_active",
        "age_mid", "elderly", "stay_x_meds",
    }
    assert expected_some.issubset(set(res.df.columns))


def test_apply_feature_engineering_handles_missing_columns_gracefully():
    # Minimal df should not crash; only age-related features should be added.
    df = pd.DataFrame({"age": ["[20-30)", "?"]})
    res = fe.apply_feature_engineering(df)
    assert "age_mid" in res.df.columns
    assert "elderly" in res.df.columns


def test_apply_feature_engineering_type_check():
    with pytest.raises(TypeError):
        fe.apply_feature_engineering(["not", "a", "dataframe"])
