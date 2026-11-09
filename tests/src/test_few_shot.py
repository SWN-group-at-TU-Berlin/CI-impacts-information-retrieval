import re

import numpy as np
import pandas as pd
import pytest

from src import few_shot as fs


COUNTS = {"direct": (3, 0), "detect": (2, 2), "economic": (3, 1), "societal": (3, 1), "secondary": (3, 1), "environmental": (1, 1)}
COLUMNS = [
    "publication_id", "sentence_reference", "ci1_type", "ci1_amount", "ci1_name", "ci1_location", "ci1_damage_group", "ci1_damage_type",
    "secondary_infrastructure_asset", "economic_impact_origin", "economic_impact_type", "economic_impact_amount",
    "societal_impact_origin", "societal_impact_type", "societal_impact_amount",
    "environmental_impact_origin", "environmental_impact_type", "environmental_impact_amount",
]


def make_table(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def row(publication, sentence, **values):
    return {"publication_id": publication, "sentence_reference": sentence, "ci1_type": "roads", "ci1_location": "Trier", "ci1_damage_group": "inundation", **values}


def encode(texts):
    """Bag-of-words embeddings: similar words give a high cosine similarity"""
    vectors = np.zeros((len(texts), 512))
    for i, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            vectors[i, sum(ord(c) * (j + 1) for j, c in enumerate(word)) % 512] += 1
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make_store(records, min_sim=0.1):
    return fs.FewShotStore(records, encode([r["text"] for r in records]), encode, COUNTS, min_sim)


def record(identifier, publication, text, direct=True, **impacts):
    base = {
        "id": identifier, "publication_id": publication, "text": text, "ner_entities": ["roads"],
        "direct": [{"ci_asset": "roads", "ci_asset_amount": None, "ci_asset_name": None, "location": "Trier", "damage": "inundation"}] if direct else [],
        "has_cascading_impacts": bool(impacts), "dropped_targets": [],
        "economic": [], "societal": [], "secondary": [], "environmental": [],
    }
    return {**base, **impacts}


ECONOMIC = [{"source_asset": "roads", "source_location": "Trier", "impact_origin": "repair_recovery_cost", "impact_type": "repair cost", "amount": "5 million EUR"}]
SOCIETAL = [{"source_asset": "roads", "source_location": "Trier", "impact_origin": "disruption_of_mobility", "impact_type": "people", "amount": "100"}]


def test_groups_rows_per_reference_and_removes_repeated_cases():
    df = make_table([
        row("A 2020", "Flooded roads cost 5 million.", economic_impact_type="repair cost", economic_impact_amount="5 million"),
        row("A 2020", "Flooded roads cost 5 million.", societal_impact_type="people"),
    ])
    records, _ = fs.build_records(df)
    assert len(records) == 1
    assert len(records[0]["direct"]) == 1
    assert records[0]["economic"][0]["impact_type"] == "repair cost"
    assert records[0]["has_cascading_impacts"] is True


@pytest.mark.parametrize("missing,dropped", [
    ({"ci1_type": None, "ci1_location": None}, 1),
    ({"ci1_type": None, "ci1_damage_group": None}, 1),
    ({"ci1_type": None}, 0),
    ({"ci1_location": None, "ci1_damage_group": None}, 0),
])
def test_impact_row_is_dropped_only_without_source(missing, dropped):
    df = make_table([{**row("A 2020", "Text.", economic_impact_type="repair cost"), **missing}])
    _, report = fs.build_records(df)
    assert report["dropped_impact_without_source"] == dropped


def test_text_with_dropped_impact_is_no_negative_example():
    df = make_table([
        row("A 2020", "Roads were flooded.", societal_impact_type="people"),
        {**row("A 2020", "Roads were flooded.", economic_impact_type="cost"), "ci1_type": None, "ci1_location": None},
    ])
    records, _ = fs.build_records(df)
    assert records[0]["dropped_targets"] == ["economic"]
    assert not fs.is_negative(records[0], "economic")
    assert fs.is_negative(records[0], "secondary")


def test_undone_and_misaligned_rows_are_removed():
    df = make_table([
        row("A 2020", "Undone text.", ci1_amount="xx"),
        row("B 2020", "Misaligned text.", secondary_infrastructure_asset="damaged"),
        row("unknown source", "Other text."),
        row("C 2020", "Good text."),
    ])
    records, _ = fs.build_records(df)
    assert [r["publication_id"] for r in records] == ["C 2020"]


def test_split_keeps_publications_together_and_is_reproducible():
    records = [record(f"{p}#{i}", f"P{p}", f"text {p} {i}", economic=ECONOMIC if i % 2 else []) for p in range(12) for i in range(3)]
    split = fs.make_split(records, n_iter=2000)
    assert split == fs.make_split(records, n_iter=2000)
    assert set(split) == {f"P{p}" for p in range(12)}
    assert list(split.values()).count("pool") == 4


def test_shared_sentences_are_detected_also_within_longer_references():
    pool = [record("a", "A", "The roads were flooded.")]
    holdout = [record("b", "B", "Heavy rain. The roads were flooded. Costs were high.")]
    with pytest.raises(ValueError):
        fs.assert_no_shared_sentences(pool, holdout)
    fs.assert_no_shared_sentences(pool, [record("c", "C", "Bridges collapsed.")])


def test_windows_cover_single_and_consecutive_sentences():
    windows = fs.split_into_windows("One a. Two b. Three c. Four d.")
    assert "Two b." in windows
    assert "Two b. Three c. Four d." in windows
    assert fs.split_into_windows("Only one sentence") == ["Only one sentence"]
    assert fs.split_into_windows("One a. Two b.") == ["One a.", "Two b.", "One a. Two b."]


def test_example_spread_over_sentences_matches_three_sentence_window():
    example = "Roads were flooded. Trains stopped. Passengers were stranded."
    chunk = "Unrelated opening remark here. Roads were flooded. Trains stopped. Passengers were stranded. Another closing remark follows."
    store = make_store([record("a", "A", example)], min_sim=0.95)
    assert [r["id"] for r, _ in store.select(chunk)["direct"]] == ["a"]


def test_selection_uses_one_example_per_publication_and_min_sim():
    records = [
        record("a1", "A", "roads flooded in Trier"),
        record("a2", "A", "roads flooded in Trier today"),
        record("b1", "B", "roads flooded in Bonn"),
        record("c1", "C", "completely different words here"),
    ]
    ids = [r["id"] for r, _ in make_store(records, min_sim=0.3).select("roads flooded in Trier")["direct"]]
    assert ids == ["a1", "b1"]


def test_no_examples_below_min_sim_gives_empty_text():
    store = make_store([record("a", "A", "roads flooded")], min_sim=0.9)
    assert store.select_all("completely different words") == {task: "" for task in COUNTS}


def test_detect_selects_positive_and_negative_examples():
    records = [
        record("pos", "A", "roads flooded cost money", economic=ECONOMIC),
        record("neg", "B", "roads flooded nothing else"),
    ]
    assert [r["id"] for r, _ in make_store(records).select("roads flooded")["detect"]] == ["pos", "neg"]


def test_negative_of_extractor_is_a_cascade_of_another_kind():
    assert fs.is_negative(record("a", "A", "text", societal=SOCIETAL), "economic")
    assert not fs.is_negative(record("a", "A", "text", economic=ECONOMIC), "economic")
    assert not fs.is_negative(record("a", "A", "text"), "economic")


def test_render_formats_match_static_examples():
    example = record("a", "A", "The roads flooded.", economic=ECONOMIC)
    assert fs.render_example(example, "economic") == (
        'TEXT: The roads flooded.\nCI_DAMAGES: ci_asset="roads", location="Trier", damage="inundation"\n'
        "source_asset: roads\nsource_location: Trier\nimpact_origin: repair_recovery_cost\nimpact_type: repair cost\namount: 5 million EUR"
    )
    assert 'InfrastructureCase(ci_asset="roads", ci_asset_amount=None, ci_asset_name=None, location="Trier", damage="inundation")' in fs.render_example(example, "direct")
    assert fs.render_example(example, "detect") == "Text: The roads flooded.\n- roads → repair cost"
    assert "impacts = []" in fs.render_example(example, "societal")


def test_render_empty_selection_leaves_prompt_unchanged():
    assert fs.render({"direct": []}) == {"direct": ""}
