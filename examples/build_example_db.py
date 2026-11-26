# # Example results database
#
# Builds examples/example_results.sqlite, the example database shown in the web demo (hf_space/app.py).
#
# The records are ILLUSTRATIVE: written by hand from examples/example_input.txt (a fictional flood report)
# in the format of LLM-extractor and the LLM-as-a-judge, so that the demo shows the output format without any
# LLM call. They then go through the real postprocessing (CI grouping, location cleaning, src/postprocess.py)
# and the real database writer (src/results_db.py).
#
# To replace them with real pipeline results: run the pipeline on the text in the web demo and download the .sqlite.
#
# run from the project root:  uv run python examples/build_example_db.py

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.postprocess as pp  # noqa: E402
import src.results_db as rdb  # noqa: E402
from src.tools.ner import load_ci_patterns  # noqa: E402

CITATION = "Example 2021"
OUTPUT = PROJECT_ROOT / "examples" / "example_results.sqlite"

chunk_texts = [p.strip() for p in (PROJECT_ROOT / "examples" / "example_input.txt").read_text().split("\n\n") if p.strip()]

# one row per infrastructure case, as returned by src.models.llm_extract.result_to_dataframe (impacts joined with ", ")
records = [
    {
        "chunk_id": 0,
        "infrastructure_type": "B267 federal road",
        "infrastructure_name": "B267",
        "location": "Altenahr",
        "damage": "structural_damage",
        "has_cascading_impacts": False,
        "judge_verdict": "correct",
        "judge_reason": "Road, location and damage are stated in the text.",
        "judge_evidence": "In Altenahr, the B267 federal road was severely damaged over a length of two kilometres and had to be closed.",
    },
    {
        "chunk_id": 0,
        "infrastructure_type": "electricity grid",
        "location": "near Bad Neuenahr",
        "damage": "inundation",
        "has_cascading_impacts": True,
        "societal_impact_origin": "loss_of_critical_service_or_inaccessibility",
        "societal_impact_type": "households",
        "societal_impact_amount": "Around 3,000",
        "societal_impact_unit": "households",
        "secondary_infrastructure_asset": "pumping stations of the drinking water supply",
        "judge_verdict": "adapted",
        "judge_adapted_fields": json.dumps({"infrastructure_type": {"old": "parts of the electricity grid", "new": "electricity grid"}}),
        "judge_reason": "The asset is the electricity grid; 'parts of' is not part of the asset name.",
        "judge_evidence": "Around 3,000 households in the district were without electricity after parts of the electricity grid near Bad Neuenahr were flooded",
    },
    {
        "chunk_id": 1,
        "infrastructure_type": "railway line",
        "location": "between Remagen and Ahrbrück",
        "damage": "structural_damage",
        "has_cascading_impacts": True,
        "economic_impact_origin": "repair_recovery_cost",
        "economic_impact_type": "repair cost",
        "economic_impact_amount": "about €1.3 billion",
        "economic_impact_unit": "€",
        "societal_impact_origin": "disruption_of_mobility",
        "societal_impact_type": "passengers",
        "societal_impact_amount": "Thousands",
        "societal_impact_unit": "commuters",
        "judge_verdict": "correct",
        "judge_reason": "Damage, repair costs and affected commuters are stated in the text.",
        "judge_evidence": "The railway line between Remagen and Ahrbrück was destroyed at several points. The operator estimated the repair costs at about €1.3 billion.",
    },
    {
        "chunk_id": 2,
        "infrastructure_type": "sewage treatment plant",
        "location": "Sinzig",
        "damage": "inundation",
        "has_cascading_impacts": True,
        # two environmental impacts of the same asset, joined in one cell -> two rows in environmental_impacts
        "environmental_impact_origin": "environmental_impact, environmental_impact",
        "environmental_impact_type": "pollution of environment, waste fills and debris",
        "judge_verdict": "false",
        "judge_reason": "The debris on farmland is not attributed to the sewage treatment plant in the text.",
        "judge_evidence": "Debris and waste were deposited on around 100 hectares of farmland along the river banks.",
    },
]

df = pd.DataFrame(records)
df["citation_id"] = CITATION
df["chunk_text"] = df["chunk_id"].map(dict(enumerate(chunk_texts)))
df["is_infrastructure_case"] = True
df["judge_evidence_in_text"] = [ev in txt for ev, txt in zip(df["judge_evidence"], df["chunk_text"])]
df["judge_adaptation_in_text"] = [True if v == "adapted" else None for v in df["judge_verdict"]]

# real postprocessing of the pipeline: CI grouping and location cleaning
ci_patterns = load_ci_patterns()
df, not_grouped = pp.postprocess_response(df, ci_patterns)
assert not_grouped.empty, f"records without CI group: {not_grouped['infrastructure_type'].tolist()}"

rdb.write_results_db(
    df,
    OUTPUT,
    documents=[{"citation_id": CITATION, "title": "Illustrative flood report (fictional example)", "source_type": "text", "language": "en"}],
    chunks=[{"citation_id": CITATION, "chunk_id": i, "text": t, "passed_ner": True, "is_infrastructure_case": True} for i, t in enumerate(chunk_texts)],
    run={"extractor_model": "illustrative (hand-written records)", "judge_model": "illustrative (hand-written records)", "note": "built by examples/build_example_db.py"},
    ci_patterns=ci_patterns,
)
print(f"{OUTPUT.relative_to(PROJECT_ROOT)}: {rdb.list_tables(OUTPUT)}")
