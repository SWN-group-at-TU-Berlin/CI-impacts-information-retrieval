import numpy as np
import pandas as pd

from src import evaluation as ev


THRESHOLDS = {"ci": 1.0, "damage": 0.7, "location": 0.7}


def exact_match(pred, valid):
    return float(pred == valid)


def make_data():
    # doc "A": paragraph 1 has 2 predictions, paragraph 2 has 1 prediction (no validation record -> FP)
    df_pred = pd.DataFrame({
        "citation_id": ["A", "A", "A"],
        "chunk_id": [1, 1, 2],
        "id_pred": [0, 1, 2],
        "infrastructure_group": ["roads", "power", "rail"],
        "damage": ["flooded", "outage", "destroyed"],
        "location": ["Berlin", "Hamburg", "Bonn"],
    })
    # 3 validation records: two belong to paragraph 1, one matches no paragraph (-> FN)
    df_valid = pd.DataFrame({
        "publication_id": ["A", "A", "A"],
        "id_valid": [0, 1, 2],
        "ci1_group": ["roads", "power", "water"],
        "ci1_damage": ["flooded", "damaged", "polluted"],
        "ci1_location": ["Berlin", np.nan, "Köln"],
    })
    # sentence of valid 0 also partially matches paragraph 2, but less well than paragraph 1
    pairs = [(0, 0, 1, 100), (1, 0, 1, 100), (0, 1, 1, 90), (1, 1, 1, 90), (2, 0, 2, 70)]
    rows = []
    for id_pred, id_valid, chunk, sim in pairs:
        p, v = df_pred.loc[id_pred], df_valid.loc[id_valid]
        rows.append({
            "citation_id": "A", "chunk_id_pred": chunk, "id_pred": id_pred, "id_valid": id_valid,
            "text_similarity": sim,
            "ci_group_pred": p.infrastructure_group, "damage_pred": p.damage, "location_pred": p.location,
            "ci_group_valid": v.ci1_group, "damage_valid": v.ci1_damage, "location_valid": v.ci1_location,
        })
    return df_pred, df_valid, pd.DataFrame(rows)


def run_pipeline():
    df_pred, df_valid, df_pred_valid_all = make_data()
    df_valid_paragraphs = ev.assign_valid_to_paragraphs(df_pred_valid_all)
    df_pairs = ev.select_paragraph_pairs(df_pred_valid_all, df_valid_paragraphs)
    df_pairs = ev.calc_pair_similarities(df_pairs, exact_match, exact_match)
    df_matched = ev.match_pairs_one_to_one(df_pairs)
    return df_valid_paragraphs, df_matched, ev.count_paragraph_scores(
        df_matched, df_pred, df_valid, df_valid_paragraphs, THRESHOLDS
    )


def test_assign_valid_to_best_paragraph():
    df_valid_paragraphs, _, _ = run_pipeline()
    assert df_valid_paragraphs.set_index("id_valid")["chunk_id"].to_dict() == {0: 1, 1: 1}


def test_match_pairs_one_to_one():
    _, df_matched, _ = run_pipeline()
    assert sorted(zip(df_matched["id_pred"], df_matched["id_valid"])) == [(0, 0), (1, 1)]


def test_count_paragraph_scores():
    _, _, df_scores = run_pipeline()
    scores = df_scores.set_index(["impact", "chunk_id"])

    # paragraph 1: both CIs correct; paragraph 2: one prediction without validation record
    assert scores.loc[("ci", 1), ["tps", "fps", "fns"]].tolist() == [2, 0, 0]
    assert scores.loc[("ci", 2), ["tps", "fps", "fns"]].tolist() == [0, 1, 0]
    assert np.isnan(scores.loc[("ci", 2), "recall"])

    # damage in paragraph 1: "outage" vs "damaged" is wrong -> 1 TP, 1 FP, 1 FN
    assert scores.loc[("damage", 1), ["tps", "fps", "fns"]].tolist() == [1, 1, 1]
    assert scores.loc[("damage", 1), "precision"] == 0.5

    # location in paragraph 1: valid 1 has no location -> prediction "Hamburg" is a FP
    assert scores.loc[("location", 1), ["tps", "fps", "fns"]].tolist() == [1, 1, 0]


def test_unmatched_valid_records_counted_as_fns():
    _, _, df_scores = run_pipeline()
    unmatched = df_scores[df_scores["chunk_id"].isna()].set_index("impact")
    assert unmatched.loc["ci", ["tps", "fps", "fns"]].tolist() == [0, 0, 1]
    assert unmatched.loc["ci", "recall"] == 0.0

    # every prediction and validation record is counted exactly once
    ci = df_scores[df_scores["impact"] == "ci"]
    assert ci["tps"].sum() + ci["fps"].sum() == 3
    assert ci["tps"].sum() + ci["fns"].sum() == 3


def test_fuzzy_match_counts_one_to_one():
    from fuzzywuzzy import fuzz

    pred = ["3 bridges", "about 200 km of roads", "5 hospitals"]
    valid = ["200 km of roads", "3 bridges", "3 bridges"]
    tps, fps, fns, matches = ev.fuzzy_match_counts(pred, valid, fuzz.partial_ratio, threshold=70)

    # each prediction matches at most one validation entry -> the second "3 bridges" stays a FN
    assert (tps, fps, fns) == (2, 1, 1)
    assert sorted(m[:2] for m in matches) == [("3 bridges", "3 bridges"), ("about 200 km of roads", "200 km of roads")]


def test_fuzzy_match_counts_below_threshold():
    from fuzzywuzzy import fuzz

    assert ev.fuzzy_match_counts(["5 hospitals"], ["3 bridges"], fuzz.partial_ratio, threshold=70)[:3] == (0, 1, 1)
    assert ev.fuzzy_match_counts([], ["3 bridges"], fuzz.partial_ratio, threshold=70)[:3] == (0, 0, 1)


def test_extract_numbers():
    assert ev.extract_numbers("1,000 km and 2.5 bridges") == [2.5, 1000.0]
    assert ev.extract_numbers("1.000 Brücken") == [1000.0]
    assert ev.extract_numbers("2,5 km") == [2.5]
    assert ev.extract_numbers("several bridges") == []


def test_numbers_must_match():
    from fuzzywuzzy import fuzz

    scorer = ev.numbers_must_match(fuzz.partial_ratio)
    assert scorer("2", "12") == 0
    assert scorer("3", "300") == 0
    assert scorer("3 bridges", "30 bridges") == 0
    assert scorer("several bridges", "3 bridges") == 0
    assert scorer("about 200 km of roads", "200 km of roads") == 100
    assert scorer("1,000 houses", "1000 houses") > 70
    assert scorer("several bridges", "several bridges") == 100

    # with the rule, "3" is no TP for "300" anymore
    assert ev.fuzzy_match_counts(["3", "200 km"], ["300", "about 200 km"], scorer, threshold=70)[:3] == (1, 1, 1)


def test_normalize_numbers_thousands_separators():
    for text in ["20 000 houses", "20 000 houses", "20,000 houses", "20.000 houses", "20'000 houses", "20000 houses"]:
        assert ev.normalize_numbers(text) == "20000 houses"
    assert ev.normalize_numbers("1 000 000") == "1000000"
    assert ev.normalize_numbers("1,000,000") == "1000000"


def test_normalize_numbers_decimals():
    assert ev.normalize_numbers("2,5 km") == "2.5 km"
    assert ev.normalize_numbers("2.50 km") == "2.5 km"
    assert ev.normalize_numbers("1.000,5 km") == "1000.5 km"
    assert ev.normalize_numbers("1,000.5 km") == "1000.5 km"
    assert ev.normalize_numbers("3.0 bridges") == "3 bridges"


def test_normalize_numbers_keeps_separate_numbers():
    assert ev.normalize_numbers("in 2021 300 houses") == "in 2021 300 houses"
    assert ev.normalize_numbers("3, 5 and 7 bridges") == "3, 5 and 7 bridges"
    assert ev.normalize_numbers("1.2.3") == "1.2.3"
    assert ev.normalize_numbers("several bridges") == "several bridges"


def test_numbers_must_match_with_different_writings():
    from fuzzywuzzy import fuzz

    scorer = ev.numbers_must_match(fuzz.partial_ratio)
    assert scorer("20 000 houses", "20,000 houses") == 100
    assert scorer("about 20.000 houses", "20000 houses") == 100
    assert scorer("20 000 houses", "2000 houses") == 0


def test_calc_scores():
    assert ev.calc_scores(tps=2, fps=2, fns=0) == (0.5, 1.0, 2 / 3)
    # no predictions / no validation entries / nothing at all -> no ZeroDivisionError
    assert ev.calc_scores(tps=0, fps=0, fns=3) == (0.0, 0.0, 0.0)
    assert ev.calc_scores(tps=0, fps=3, fns=0) == (0.0, 0.0, 0.0)
    assert np.isnan(ev.calc_scores(0, 0, 0, zero_division=np.nan)[0])


def test_numbers_must_match_ignores_repeated_numbers():
    from fuzzywuzzy import fuzz

    scorer = ev.numbers_must_match(fuzz.partial_ratio)
    assert scorer("100 to 200, between 100 to 200 households", "100 to 200") == 100
    assert ev.fuzzy_match_counts(
        ["100 to 200, between 100 to 200 households"], ["100 to 200"], scorer, threshold=70
    )[:3] == (1, 0, 0)
    # different numbers still do not match
    assert scorer("100 to 300 households", "100 to 200") == 0
    assert scorer("100 to 200 households, 5 schools", "100 to 200") == 0


def test_is_missing_impact():
    for value in [None, np.nan, "", " ", "nan", "NaN", "None", "none"]:
        assert ev.is_missing_impact(value)
    # combined impact pairs are missing when one part is missing
    assert ev.is_missing_impact("roads - nan")
    assert ev.is_missing_impact("nan - Berlin")
    assert not ev.is_missing_impact("roads - Berlin")
    assert not ev.is_missing_impact("Emilia-Romagna")
    assert not ev.is_missing_impact(0)


def test_drop_missing_impacts():
    assert ev.drop_missing_impacts(["roads", "nan", None, "rail - None", "rail - Bonn"]) == ["roads", "rail - Bonn"]
