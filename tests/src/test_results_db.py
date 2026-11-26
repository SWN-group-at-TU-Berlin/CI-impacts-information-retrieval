import json
import sqlite3

import pandas as pd
import pytest

import src.results_db as rdb


@pytest.fixture
def records():
    """Flat records of the pipeline (one row per infrastructure case), two documents"""
    return pd.DataFrame(
        [
            {
                "citation_id": "Koks 2022",
                "chunk_id": 0,
                "chunk_text": "The A1 highway was flooded. 2,000 households were without electricity.",
                "infrastructure_type": "A1 highway",
                "infrastructure_group": "road_network",
                "location": "Cologne",
                "damage": "inundation",
                "has_cascading_impacts": "True",  # as read from the csv
                "societal_impact_origin": "loss_of_critical_service_or_inaccessibility, disruption_of_mobility",
                "societal_impact_type": "households, passengers",
                "societal_impact_amount": "2,000",  # only one amount for two impacts -> kept joined
                "economic_impact_type": "",
                "secondary_infrastructure_asset": "electricity grid, drinking water supply",
                "judge_verdict": "adapted",
                "judge_adapted_fields": json.dumps({"location": {"old": "near Cologne", "new": "Cologne"}}),
                "judge_evidence_in_text": True,
            },
            {
                "citation_id": "Koks 2022",
                "chunk_id": 3,
                "chunk_text": "Repair costs of the bridge were €5 million and of the road €2 million.",
                "infrastructure_type": "bridge",
                "infrastructure_group": "road_network",
                "damage": "structural_damage",
                "economic_impact_origin": "repair_recovery_cost, repair_recovery_cost",
                "economic_impact_type": "repair cost, repair cost",
                "economic_impact_amount": "€5 million, €2 million",
                "judge_verdict": "correct",
            },
            {
                "citation_id": "Text 2026",
                "chunk_id": 0,
                "chunk_text": "The railway line was closed.",
                "infrastructure_type": "railway line",
                "infrastructure_group": "rail_general",
                "damage": "disruption",
            },
        ]
    )


def query(path, sql):
    with sqlite3.connect(path) as conn:
        return conn.execute(sql).fetchall()


def test_split_impacts_splits_only_equal_counts():
    row = pd.Series({"x_type": "a, b", "x_amount": "1, 2", "x_unit": None})
    assert rdb.split_impacts(row, "x", ["type", "amount", "unit"]) == [
        {"type": "a", "amount": "1", "unit": None},
        {"type": "b", "amount": "2", "unit": None},
    ]
    row = pd.Series({"x_type": "a, b", "x_amount": "10,000"})
    assert rdb.split_impacts(row, "x", ["type", "amount"]) == [{"type": "a, b", "amount": "10,000"}]
    assert rdb.split_impacts(pd.Series({"x_type": ""}), "x", ["type"]) == []


def test_write_results_db_tables_and_relations(tmp_path, records):
    path = rdb.write_results_db(
        records,
        tmp_path / "results.sqlite",
        documents=[{"citation_id": "Text 2026", "title": "pasted text", "source_type": "text", "language": "en"}],
        chunks=[{"citation_id": "Text 2026", "chunk_id": 1, "text": "No infrastructure here.", "passed_ner": False}],
        run={"extractor_model": "openai/gpt-oss-20b", "n_errors": 0},
    )
    assert set(rdb.list_tables(path)) >= {"documents", "chunks", "infrastructure_cases", "v_cases_flat"}
    assert query(path, "SELECT count(*) FROM documents") == [(2,)]
    # 3 chunks with records + 1 chunk filtered by NER
    assert query(path, "SELECT count(*), sum(passed_ner) FROM chunks") == [(4, 3)]
    assert query(path, "SELECT count(*) FROM infrastructure_cases") == [(3,)]
    assert query(path, "SELECT infrastructure_sector FROM infrastructure_cases WHERE infrastructure_type = 'railway line'") == [("Transportation",)]
    # equal counts -> split; unequal counts -> one joined impact
    assert query(path, "SELECT amount FROM economic_impacts ORDER BY id") == [("€5 million",), ("€2 million",)]
    assert query(path, "SELECT type, amount FROM societal_impacts") == [("households, passengers", "2,000")]
    assert query(path, "SELECT target_asset FROM secondary_infrastructure_impacts ORDER BY id") == [("electricity grid",), ("drinking water supply",)]
    assert query(path, "SELECT field, old_value, new_value FROM judge_adaptations") == [("location", "near Cologne", "Cologne")]
    assert query(path, "SELECT has_cascading_impacts FROM chunks JOIN documents USING (doc_id) WHERE chunk_no = 0 AND citation_id = 'Koks 2022'") == [(1,)]
    assert query(path, "SELECT extractor_model FROM runs") == [("openai/gpt-oss-20b",)]
    # no orphans
    assert query(path, "SELECT count(*) FROM economic_impacts e LEFT JOIN infrastructure_cases c USING (case_id) WHERE c.case_id IS NULL") == [(0,)]


def test_flat_view_matches_records(tmp_path, records):
    path = rdb.write_results_db(records, tmp_path / "results.sqlite")
    df_flat = rdb.read_table(path, "v_cases_flat")
    assert len(df_flat) == len(records)
    assert df_flat["infrastructure_type"].tolist() == records["infrastructure_type"].tolist()
    assert df_flat.loc[1, "economic_impact_amount"] == "€5 million, €2 million"


def test_write_results_db_empty(tmp_path):
    path = rdb.write_results_db(pd.DataFrame(), tmp_path / "empty.sqlite", documents=[{"citation_id": "Text 2026"}])
    assert query(path, "SELECT count(*) FROM infrastructure_cases") == [(0,)]
    assert query(path, "SELECT count(*) FROM documents") == [(1,)]
