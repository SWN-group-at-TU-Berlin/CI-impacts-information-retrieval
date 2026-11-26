#!/usr/bin/env python
"""Relational results database (SQLite) of the CI impact extraction.

The pipeline returns one flat table, with one row per infrastructure case and the cascading impacts joined in cells
(see docs/extracted_features.md). This module splits that table into related tables:

    documents 1─n chunks 1─n infrastructure_cases 1─n economic_impacts
                                                  1─n societal_impacts
                                                  1─n environmental_impacts
                                                  1─n secondary_infrastructure_impacts
                                                  1─n judge_adaptations
    runs (models and duration of the run)
    v_cases_flat (view, same columns as the flat csv of the pipeline)

usage:
    df_records = pipeline.run(documents)
    write_results_db(df_records, "results.sqlite", documents=[{"citation_id": "Koks 2022", "title": ..., "source_type": "pdf"}])
"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

NER_PATTERNS_FILE = Path(__file__).resolve().parents[1] / "ner_patterns.jsonl" / "patterns"

# impact table -> column prefix in the flat records (e.g. economic_impact_type -> economic_impacts.type)
IMPACT_TABLES = {
    "economic_impacts": "economic_impact",
    "societal_impacts": "societal_impact",
    "environmental_impacts": "environmental_impact",
}
IMPACT_FIELDS = ["origin", "type", "amount", "unit"]
SEPARATOR = ", "  # several impacts of the same kind are joined with this separator, see src/models/llm_extract.py

SCHEMA = """
CREATE TABLE documents (
    doc_id INTEGER PRIMARY KEY,
    citation_id TEXT NOT NULL UNIQUE,
    title TEXT,
    source_type TEXT,             -- pdf | text | html
    language TEXT,                -- ISO 639-1 code of the source document
    n_chunks INTEGER,
    created_at TEXT
);
CREATE TABLE chunks (
    chunk_pk INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(doc_id),
    chunk_no INTEGER NOT NULL,    -- chunk_id of the pipeline
    text TEXT,
    ner_entities TEXT,            -- JSON list of the infrastructure entities of the NER pre-filter
    passed_ner INTEGER,           -- 1: passed to the LLMs
    is_infrastructure_case INTEGER,
    has_cascading_impacts INTEGER,
    UNIQUE (doc_id, chunk_no)
);
CREATE TABLE infrastructure_cases (
    case_id INTEGER PRIMARY KEY,
    chunk_pk INTEGER NOT NULL REFERENCES chunks(chunk_pk),
    infrastructure_type TEXT,
    infrastructure_group TEXT,
    infrastructure_sector TEXT,
    infrastructure_amount TEXT,
    infrastructure_name TEXT,
    location TEXT,
    location_orig TEXT,
    damage TEXT,
    judge_verdict TEXT,           -- correct | adapted | false | not_judged, NULL: judge not run
    judge_reason TEXT,
    judge_evidence TEXT,
    judge_evidence_in_text INTEGER,
    judge_adaptation_in_text INTEGER
);
CREATE TABLE economic_impacts (
    id INTEGER PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES infrastructure_cases(case_id),
    origin TEXT, type TEXT, amount TEXT, unit TEXT
);
CREATE TABLE societal_impacts (
    id INTEGER PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES infrastructure_cases(case_id),
    origin TEXT, type TEXT, amount TEXT, unit TEXT
);
CREATE TABLE environmental_impacts (
    id INTEGER PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES infrastructure_cases(case_id),
    origin TEXT, type TEXT, amount TEXT, unit TEXT
);
CREATE TABLE secondary_infrastructure_impacts (
    id INTEGER PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES infrastructure_cases(case_id),
    target_asset TEXT,
    target_group TEXT,
    target_sector TEXT
);
CREATE TABLE judge_adaptations (
    id INTEGER PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES infrastructure_cases(case_id),
    field TEXT, old_value TEXT, new_value TEXT
);
CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY,
    extractor_model TEXT,
    judge_model TEXT,
    started_at TEXT,
    duration_s REAL,
    n_errors INTEGER,
    note TEXT
);
CREATE VIEW v_cases_flat AS
SELECT d.citation_id, ch.chunk_no AS chunk_id, c.infrastructure_type, c.infrastructure_group, c.infrastructure_sector,
       c.infrastructure_amount, c.infrastructure_name, c.location, c.damage,
       (SELECT group_concat(type, ', ') FROM economic_impacts e WHERE e.case_id = c.case_id) AS economic_impact_type,
       (SELECT group_concat(amount, ', ') FROM economic_impacts e WHERE e.case_id = c.case_id) AS economic_impact_amount,
       (SELECT group_concat(type, ', ') FROM societal_impacts s WHERE s.case_id = c.case_id) AS societal_impact_type,
       (SELECT group_concat(amount, ', ') FROM societal_impacts s WHERE s.case_id = c.case_id) AS societal_impact_amount,
       (SELECT group_concat(type, ', ') FROM environmental_impacts v WHERE v.case_id = c.case_id) AS environmental_impact_type,
       (SELECT group_concat(target_asset, ', ') FROM secondary_infrastructure_impacts x WHERE x.case_id = c.case_id) AS secondary_infrastructure_asset,
       c.judge_verdict, c.judge_reason
FROM infrastructure_cases c
JOIN chunks ch ON ch.chunk_pk = c.chunk_pk
JOIN documents d ON d.doc_id = ch.doc_id;
"""


def _value(value):
    """None for missing values (NaN, None, empty string), else the value as str"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value).strip()
    return value or None


def _bool(value):
    """1/0 for truthy values (also 'True'/'False' strings of a csv), None for missing values"""
    value = _value(value)
    if value is None:
        return None
    return int(value.lower() in ("true", "1", "yes"))


def split_impacts(row: pd.Series, prefix: str, fields: list[str]) -> list[dict]:
    """Single impacts of one kind (e.g. economic) of a flat record.
    The values of several impacts are joined with ", " per column. They are split back into single impacts only
    if all non-empty columns have the same number of values; otherwise one impact with the joined values is kept
    (e.g. an amount "10, 000" or a missing unit of one of the impacts would else shift the values).
    """
    values = {f: _value(row.get(f"{prefix}_{f}")) for f in fields}
    if not any(values.values()):
        return []
    parts = {f: v.split(SEPARATOR) for f, v in values.items() if v}
    n_parts = {len(p) for p in parts.values()}
    if len(n_parts) == 1 and (n := n_parts.pop()) > 1:
        return [{f: (parts[f][i].strip() if f in parts else None) for f in fields} for i in range(n)]
    return [values]


def load_group_sectors(patterns_file: Path = NER_PATTERNS_FILE) -> dict:
    """infrastructure_group (subgroup_name of the NER patterns) -> sector"""
    patterns = pd.read_json(patterns_file, lines=True).dropna(subset=["subgroup_name"])
    return dict(zip(patterns["subgroup_name"], patterns["sector"]))


def group_secondary_asset(asset: str, ci_patterns: pd.DataFrame | None) -> str | None:
    """infrastructure_group of a secondary infrastructure asset, matched with the same CI patterns as infrastructure_type"""
    if ci_patterns is None or not asset:
        return None
    import src.postprocess as pp

    df = pp.group_ci_types(pd.DataFrame({"type": [asset]}), "type", "group", ci_patterns)
    return _value(df["group"].iloc[0]) if df["group"].iloc[0] != "None" else None


def write_results_db(
    df_records: pd.DataFrame,
    path: str | Path,
    documents: list[dict] | None = None,
    chunks: list[dict] | None = None,
    run: dict | None = None,
    ci_patterns: pd.DataFrame | None = None,
) -> Path:
    """Write the records of CiImpactPipeline (flat table, one row per infrastructure case) to a new SQLite database.

    df_records: records of the pipeline, incl. citation_id, chunk_id, chunk_text and (optional) the judge columns
    documents: metadata per document {citation_id, title, source_type, language, n_chunks}, default: from df_records
    chunks: all chunks {citation_id, chunk_id, text, ner_entities, passed_ner, is_infrastructure_case (optional)}, also those
        without records, default: from df_records
    run: {extractor_model, judge_model, started_at, duration_s, n_errors, note}
    ci_patterns: NER patterns (src.tools.ner.load_ci_patterns()) to group the secondary infrastructure assets, None: not grouped
    """
    path = Path(path)
    path.unlink(missing_ok=True)
    df = df_records.copy() if df_records is not None else pd.DataFrame()
    group_sectors = load_group_sectors()
    now = datetime.now(UTC).isoformat(timespec="seconds")

    # documents and chunks, also documents/chunks without any record (e.g. filtered by NER)
    documents = list(documents or [])
    known = {d["citation_id"] for d in documents}
    if not df.empty:
        documents += [{"citation_id": c} for c in df["citation_id"].unique() if c not in known]
    chunks = list(chunks or [])
    known_chunks = {(c["citation_id"], int(c["chunk_id"])) for c in chunks}
    if not df.empty:
        for (citation, chunk_no), df_chunk in df.groupby(["citation_id", "chunk_id"]):
            if (citation, int(chunk_no)) not in known_chunks:
                chunks.append({"citation_id": citation, "chunk_id": int(chunk_no), "text": df_chunk["chunk_text"].iloc[0], "passed_ner": True})

    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        doc_ids = {}
        for doc in documents:
            n_chunks = doc.get("n_chunks", sum(c["citation_id"] == doc["citation_id"] for c in chunks))
            cur = conn.execute(
                "INSERT INTO documents (citation_id, title, source_type, language, n_chunks, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (doc["citation_id"], doc.get("title"), doc.get("source_type"), doc.get("language"), n_chunks, now),
            )
            doc_ids[doc["citation_id"]] = cur.lastrowid

        chunk_pks = {}
        for chunk in chunks:
            key = (chunk["citation_id"], int(chunk["chunk_id"]))
            df_chunk = df[(df["citation_id"] == key[0]) & (df["chunk_id"] == key[1])] if not df.empty else df
            ner = chunk.get("ner_entities")
            cur = conn.execute(
                "INSERT INTO chunks (doc_id, chunk_no, text, ner_entities, passed_ner, is_infrastructure_case, has_cascading_impacts) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_ids[key[0]],
                    key[1],
                    chunk.get("text"),
                    json.dumps(ner) if ner is not None else None,
                    _bool(chunk.get("passed_ner")),
                    _bool(chunk["is_infrastructure_case"]) if "is_infrastructure_case" in chunk else int(not df_chunk.empty),
                    _bool(df_chunk["has_cascading_impacts"].iloc[0]) if "has_cascading_impacts" in df_chunk and not df_chunk.empty else None,
                ),
            )
            chunk_pks[key] = cur.lastrowid

        for _, row in df.iterrows():
            group = _value(row.get("infrastructure_group"))
            cur = conn.execute(
                """INSERT INTO infrastructure_cases (chunk_pk, infrastructure_type, infrastructure_group, infrastructure_sector, infrastructure_amount,
                   infrastructure_name, location, location_orig, damage, judge_verdict, judge_reason, judge_evidence, judge_evidence_in_text,
                   judge_adaptation_in_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_pks[(row["citation_id"], int(row["chunk_id"]))],
                    _value(row.get("infrastructure_type")),
                    group,
                    group_sectors.get(group),
                    _value(row.get("infrastructure_amount")),
                    _value(row.get("infrastructure_name")),
                    _value(row.get("location")),
                    _value(row.get("location_orig")),
                    _value(row.get("damage")),
                    _value(row.get("judge_verdict")),
                    _value(row.get("judge_reason")),
                    _value(row.get("judge_evidence")),
                    _bool(row.get("judge_evidence_in_text")),
                    _bool(row.get("judge_adaptation_in_text")),
                ),
            )
            case_id = cur.lastrowid

            for table, prefix in IMPACT_TABLES.items():
                for impact in split_impacts(row, prefix, IMPACT_FIELDS):
                    conn.execute(
                        f"INSERT INTO {table} (case_id, origin, type, amount, unit) VALUES (?, ?, ?, ?, ?)",
                        (case_id, *(impact[f] for f in IMPACT_FIELDS)),
                    )

            assets = _value(row.get("secondary_infrastructure_asset"))
            for asset in assets.split(SEPARATOR) if assets else []:
                target_group = group_secondary_asset(asset.strip(), ci_patterns)
                conn.execute(
                    "INSERT INTO secondary_infrastructure_impacts (case_id, target_asset, target_group, target_sector) VALUES (?, ?, ?, ?)",
                    (case_id, asset.strip(), target_group, group_sectors.get(target_group)),
                )

            adapted = _value(row.get("judge_adapted_fields"))
            for field, change in (json.loads(adapted) if adapted else {}).items():
                conn.execute(
                    "INSERT INTO judge_adaptations (case_id, field, old_value, new_value) VALUES (?, ?, ?, ?)",
                    (case_id, field, _value(change.get("old")), _value(change.get("new"))),
                )

        run = run or {}
        conn.execute(
            "INSERT INTO runs (extractor_model, judge_model, started_at, duration_s, n_errors, note) VALUES (?, ?, ?, ?, ?, ?)",
            (run.get("extractor_model"), run.get("judge_model"), run.get("started_at", now), run.get("duration_s"), run.get("n_errors"), run.get("note")),
        )
        conn.commit()
    finally:
        conn.close()
    return path


def read_table(path: str | Path, table: str) -> pd.DataFrame:
    """Table or view of the results database as DataFrame"""
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def list_tables(path: str | Path) -> list[str]:
    """Names of the tables and views of the results database"""
    with sqlite3.connect(path) as conn:
        return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type, rowid")]
