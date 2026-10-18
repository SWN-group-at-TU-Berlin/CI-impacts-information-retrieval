import numpy as np
import pandas as pd

from src import postprocess as pp


def test_split_ci_entities_on_and_or():
    df = pd.DataFrame({"ci": ["roads and bridges", "trains OR buses"]})
    result = pp.split_ci_entities(df, ["ci"])
    assert result["ci"].tolist() == ["roads", "bridges", "trains", "buses"]


def test_split_ci_entities_shared_head_word():
    df = pd.DataFrame({"ci": ["water and gas pipelines", "power or water supply"]})
    result = pp.split_ci_entities(df, ["ci"])
    assert result["ci"].tolist() == [
        "water pipelines",
        "gas pipelines",
        "power supply",
        "water supply",
    ]


def test_split_ci_entities_multiword_parts_not_given_shared_head():
    df = pd.DataFrame({"ci": ["rail lines and local bus services"]})
    result = pp.split_ci_entities(df, ["ci"])
    assert result["ci"].tolist() == ["rail lines", "local bus services"]


def test_split_ci_entities_no_split_inside_words():
    df = pd.DataFrame({"ci": ["Anderson road", "  hospital  "]})
    result = pp.split_ci_entities(df, ["ci"])
    assert result["ci"].tolist() == ["Anderson road", "hospital"]


def test_split_ci_entities_keeps_other_columns():
    df = pd.DataFrame({"ci": ["roads and bridges"], "location": ["Berlin"]})
    result = pp.split_ci_entities(df, ["ci"])
    assert result.to_dict("records") == [
        {"ci": "roads", "location": "Berlin"},
        {"ci": "bridges", "location": "Berlin"},
    ]


def test_split_ci_entities_multiple_columns_cross_product():
    df = pd.DataFrame({"ci": ["roads and bridges"], "location": ["Berlin or Hamburg"]})
    result = pp.split_ci_entities(df, ["ci", "location"])
    assert list(zip(result["ci"], result["location"])) == [
        ("roads", "Berlin"),
        ("roads", "Hamburg"),
        ("bridges", "Berlin"),
        ("bridges", "Hamburg"),
    ]


def test_split_ci_entities_missing_and_empty_values_kept_as_nan():
    df = pd.DataFrame({"ci": [np.nan, "", "roads"]})
    result = pp.split_ci_entities(df, ["ci"])
    assert len(result) == 3
    assert result["ci"].iloc[:2].isna().all()
    assert result["ci"].iloc[2] == "roads"
