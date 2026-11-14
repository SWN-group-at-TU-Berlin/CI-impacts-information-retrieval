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


def test_map_damage_to_category():
    assert pp.map_damage_to_category("severely damaged") == "structural_damage"
    assert pp.map_damage_to_category("Outage(s)") == "service_loss"
    assert pp.map_damage_to_category("traffic jam(s)") == "disruption"
    assert pp.map_damage_to_category("repaired") == "restored_recovered"
    assert pp.map_damage_to_category("disruption") == "disruption"  # already a category
    # several keywords -> most severe category
    assert pp.map_damage_to_category("flooded or destroyed") == "structural_damage"
    assert pp.map_damage_to_category("closed, restored") == "disruption"
    assert pp.map_damage_to_category("unknown damage") is None
    assert pp.map_damage_to_category(np.nan) is None
    assert pp.map_damage_to_category(None) is None


def test_describe_damage_categories():
    text = pp.describe_damage_categories()
    assert len(text.split("\n")) == len(pp.DAMAGE_CATEGORIES)
    assert text.startswith("- structural_damage: physical damage")



def test_postprocess_response_keeps_only_ci_and_cleans_locations():
    from src.tools.ner import load_ci_patterns
    df = pd.DataFrame({"infrastructure_type": ["roads (A1)", "theatre"], "location": ["the Ahr valley, Germany", "Berlin"]})
    df_ci, df_not_grouped = pp.postprocess_response(df, load_ci_patterns())
    assert df_ci["infrastructure_type"].tolist() == ["roads A1"]
    assert df_ci["infrastructure_group"].notna().all()
    assert df_ci["location_orig"].tolist() == ["the Ahr valley, Germany"]
    assert df_ci["location"].tolist() == ["Ahr valley"]  # "the" and the text after the comma are removed
    assert df_not_grouped["infrastructure_type"].tolist() == ["theatre"]


def test_postprocess_locations_removes_the_in_and_passing_only_at_the_beginning():
    df = pd.DataFrame({"location": [
        "the Ahr valley, Germany", "in the Ahr valley", "passing Cologne", "near the A1", "Sinzig (in North Rhine-Westphalia)",
        "roads in Berlin", "Bridge passing the Elbe", "railway tracks between the A1 and B2", "Theale", "Berlin",
    ]})
    assert pp.postprocess_locations(df, "location").tolist() == [
        "Ahr valley", "Ahr valley", "Cologne", "A1", "Sinzig",
        "roads in Berlin", "Bridge passing the Elbe", "A1 and B2", "Theale", "Berlin",
    ]
