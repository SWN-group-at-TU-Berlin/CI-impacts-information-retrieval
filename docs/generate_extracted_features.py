#!/usr/bin/env python
"""Generate the feature tables of docs/extracted_features.md from the extraction schemas (src/models/schemas.py),
the damage categories (src/postprocess.py) and the CI patterns (ner_patterns.jsonl/patterns).

Only the block between the BEGIN/END GENERATED markers is rewritten, the rest of the file is kept.
Run from the project root: uv run python docs/generate_extracted_features.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.postprocess as pp  # noqa: E402
from src.models.llm_judge import literal_values  # noqa: E402
from src.models.schemas import (  # noqa: E402
    JUDGE_FIELD_SCHEMA,
    SOCIETAL_TYPE_KEYWORDS,
    EconomicImpact,
    EnvironmentalImpact,
    InfrastructureCase,
    SecondaryInfrastructureImpact,
    SocietalImpact,
)

DOC = PROJECT_ROOT / "docs" / "extracted_features.md"
BEGIN = "<!-- BEGIN GENERATED (docs/generate_extracted_features.py), do not edit by hand -->"
END = "<!-- END GENERATED -->"

SECTIONS = {
    InfrastructureCase: "Infrastructure case",
    EconomicImpact: "Economic impacts",
    SocietalImpact: "Societal impacts",
    EnvironmentalImpact: "Environmental impacts",
    SecondaryInfrastructureImpact: "Secondary (cascading) infrastructure impacts",
}

# free-text fields which the field_validator maps to a fixed set of values
NORMALISED_VALUES = {(SocietalImpact, "impact_type"): list(SOCIETAL_TYPE_KEYWORDS)}


def cell(text) -> str:
    """Single-line markdown table cell"""
    return " ".join(str(text).split()).replace("|", "\\|")


def codes(values) -> str:
    return ", ".join(f"`{v}`" for v in values)


def feature_row(column: str, model, name: str) -> str:
    field = model.model_fields[name]
    # the damage descriptions contain the full category list (see damage table), keep only the first line
    description = (field.description or "").split("\n")[0]
    categories = literal_values(field.annotation)
    normalised = NORMALISED_VALUES.get((model, name))
    if categories:
        kind, allowed = "categorical", codes(categories)
        if set(categories) == set(pp.DAMAGE_CATEGORIES):
            description = description.split(" If several categories apply")[0]  # the rule is explained in the damage table
            allowed = "see [damage categories](#damage-categories)"
    elif normalised:
        kind, allowed = "categorical (mapped from free text)", codes(normalised)
    elif name == "amount" or name.endswith("_amount"):
        kind, allowed = "free text (original wording, not converted to a number)", "–"
    else:
        kind, allowed = "free text", "–"
    return f"| `{column}` | {cell(description)} | {kind} | {allowed} |"


def feature_tables() -> list[str]:
    lines = []
    for model, title in SECTIONS.items():
        rows = [feature_row(column, m, name) for column, (m, name) in JUDGE_FIELD_SCHEMA.items() if m is model]
        if not rows:
            continue
        lines += [
            f"### {title}",
            "",
            f"Schema: `{model.__name__}` in [`src/models/schemas.py`](../src/models/schemas.py)",
            "",
            "| Column | Description | Type | Allowed values |",
            "|---|---|---|---|",
            *rows,
            "",
        ]
    return lines


def damage_table() -> list[str]:
    rows = [f"| `{category}` | {cell(entry['definition'])} | {cell(', '.join(entry['keywords']))} |" for category, entry in pp.DAMAGE_CATEGORIES.items()]
    return [
        "### Damage categories",
        "",
        "Allowed values of `damage`, ordered from most to least severe. If several categories apply, the most severe one is used. "
        "Free-text damages of the LLM (e.g. 'severely damaged') are mapped to a category via the keywords, "
        "see `DAMAGE_CATEGORIES` in [`src/postprocess.py`](../src/postprocess.py).",
        "",
        "| Category | Definition | Keywords mapped to this category |",
        "|---|---|---|",
        *rows,
        "",
    ]


def infrastructure_group_table() -> list[str]:
    patterns = pd.read_json(PROJECT_ROOT / "ner_patterns.jsonl" / "patterns", lines=True)
    groups = (
        patterns[["subgroup_name", "group_name", "sector"]]
        .dropna(subset=["subgroup_name"])  # e.g. location patterns, which are no CI type
        .drop_duplicates()
        .sort_values(["sector", "group_name", "subgroup_name"])
    )
    rows = [f"| `{r.subgroup_name}` | {r.group_name} | {r.sector} |" for r in groups.itertuples()]
    return [
        "### Infrastructure groups",
        "",
        "Allowed values of `infrastructure_group` (added in postprocessing): `infrastructure_type` is matched against the CI patterns in "
        "[`ner_patterns.jsonl/patterns`](../ner_patterns.jsonl/patterns) and gets the `subgroup_name` of the matching pattern. "
        "Records whose type matches no pattern (e.g. theatres, stadiums) are removed.",
        "",
        f"<details><summary>Show all {len(rows)} groups</summary>",
        "",
        "| `infrastructure_group` | Group | Sector |",
        "|---|---|---|",
        *rows,
        "",
        "</details>",
        "",
    ]


def generate() -> str:
    return "\n".join([BEGIN, "", *feature_tables(), *damage_table(), *infrastructure_group_table(), END])


def main() -> int:
    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"Markers not found in {DOC}", file=sys.stderr)
        return 1
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = head + generate() + tail
    if new != text:
        DOC.write_text(new, encoding="utf-8")
        print(f"Updated {DOC.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
