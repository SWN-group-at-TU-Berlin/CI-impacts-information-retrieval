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


ALL_OLD_DAMAGE_KEYWORDS = [
    "affected", "damaged", "severely damaged", "highly damaged", "disrupted", "blocked", "destroyed", "largely destroyed",
    "heavily destroyed", "partly closed", "closed", "outage(s)", "blackout", "loss of water/electricity", "no supply",
    "left without water/electricity", "contaminated", "polluted", "flooded", "inundated", "fallen tree(s)", "broken tree(s)",
    "overturned tree(s)", "uprooted tree(s)", "felled tree(s)", "debris", "debris removal", "debris on", "water on",
    "dam failure(s)", "dyke break(s)", "dike break(s)", "breached dike(s)", "broken embankment(s)", "closure(s)", "derailed",
    "impassable", "traffic jam(s)", "traffic disruption(s)", "traffic congestion(s)", "indirectly affected", "flight delay(s)",
    "train delay(s)", "delay(s)", "train failure(s)", "failure(s)", "cancellation(s)", "canceled", "re-established",
    "reestablished", "restored", "reconnected", "rebuild", "repaired",
]


def test_damage_categories_cover_each_old_keyword_once():
    keywords = [k for entry in pp.DAMAGE_CATEGORIES.values() for k in entry["keywords"]]
    assert sorted(keywords) == sorted(ALL_OLD_DAMAGE_KEYWORDS)
    assert len(keywords) == len(set(keywords))


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

