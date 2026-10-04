# %% [markdown]
# # AIM:
# create evaluation workflow. Taking the manually extracted Ci impacts (validation set) and compare it with the CI impacts (llm_geolocations.ipynb) extrracted by the first LLM 1. 
# As a first step the evaluation should be done only for the direct CI impacts - CI type, damage and geolocation
# 
# Issue:
# * What is needed an approach that recognizes when an direct impact case is not detected by the model
# 
# Idea: 
# * Split the original texts passed to the model on the exact chunks as again
# * Then chunkwise check if the CI impacts from the validation set correspond in number and their textual similarity to the CI impacts infered by the LLM 1 and Entity Linking 

# %%
import os
import sys
from pathlib import Path
import io
import gc
import time
import warnings
import subprocess
import importlib
import glob

from unidecode import unidecode
import langdetect
from fuzzywuzzy import fuzz
import torch
from huggingface_hub import login
import numpy as np
import spacy
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib import pyplot as plt


sys.path.append('../')
from src.settings import settings as s
import src.document_cleaning as dc
import src.translation_model as tm
import src.utils as u
import src.datahandler as dh
import src.postprocess as pp
import src.visualization as v

# login to HF
# NOTE raises exception when env.variable does not exist (compared to os.envrion.get and its shortcut os.getenv)
os.getenv("HUGGINGFACE_TOKEN")

#  automatic linebreaks and multi-line cells.
pd.set_option("display.colheader_justify", "left")
pd.set_option('display.max_colwidth', 5000)


step = "step2"


# %%


# %% [markdown]
# ### Direct CI impacts: LLM 1 vs domain-expertise 

# %%
os.getcwd()

# %%
#  Suppress future warnings from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)


#  Define data dir where tags.csv and domain-expertise derived tag lists are found 
# VALID_DATA_FILENAME = s.VALID_DATA_FILENAME
VALID_DATA_FILENAME = "table_ci_impacts_sm_hpc.csv"
PATH_VALID_DATA = s.PATH_VALID_DATA
PATH_EVAL_RESULT = s.PATH_EVAL_RESULT

# s.LLM_DATA_FILENAME = "llm1_geollm_step2_Koks 2022.csv" #f"df_responses_{step}_ner_geollm.csv"
# LLM_DATA_FILEPATH = Path("interim_results" / s.LLM_DATA_FILENAME)
# LLM_DATA_FILEPATH = Path(s.PATH_LLM_DATA / s.LLM_DATA_FILENAME)
SIMILARITY_LLM_FILENAME = s.SIMILARITY_LLM_FILENAME

df_valid_org = pd.read_csv(
    "../" + str(PATH_VALID_DATA / VALID_DATA_FILENAME),
    # usecols=["publication_id", "ci1_type", "ci1_damage", "ci1_location", "sentence_reference"],
)
print(len(df_valid_org))
## pre-process: 
# remove undone entries
df_valid_org = df_valid_org[~df_valid_org.astype(str).apply(lambda x: x.str.contains("xx")).any(axis=1)]
# remove further location info (e.g. that entry is a town, Landkreis, Bavaria etc.)
# df_valid_org["ci1_location"] = df_valid_org["ci1_location"].replace(r"\s*\(.*\)", "", regex=True).str.strip()
df_valid_org = df_valid_org.dropna(subset=["publication_id"], how="all") # drop rows where citation info is missing
print(len(df_valid_org))
print(df_valid_org.publication_id.unique())

print(f"Collecting LLM responses from {step}")
# ## prediction data
# df_pred = pd.read_csv(
#     LLM_DATA_FILEPATH,
#    # usecols=["citation_id", "chunk_id", "infrastructure_type", "damage", "location", "chunk_text"]
# )
df_pred = pd.DataFrame()
for i, file in enumerate(glob.glob(os.path.join("../interim_results", f'*{step}*.csv'))):
#for i, file in enumerate(glob.glob(os.path.join(s.PATH_DATA, "llm_outputs/responses_single_docs", f'*{step}*.csv'))):
    print(i, file)
    df_pred = pd.concat([df_pred, pd.read_csv(file)], ignore_index=True)



# %%
df_pred["citation_id"].unique()

# %%


# %% [markdown]
# ### cleanup entries in prediction set

# %%
## remove entry in df_pred when no "infrastructure_type_org" exists, then it is likely a hallucinated case
df_pred = df_pred.dropna(subset=["infrastructure_type_org"], how="all")


# %% [markdown]
# #### Remove records which have a faulty record (FAC or other non-CI entity) in CI_entity column ->
# -> due that respective LLM response is often faulty (hallucinated)
# 

# %%

## remove entry in df_pred when a "non-CI" record occurs in the "ci_entity" column 
# NOTE this should solves the issue from fix-commit "9fb76a6" - where some ci_entires contained LOCs or non-CI (e.g. EFE 2024: "the Monastery of La Cartuja", "the Advanced Command Post")
 
print(len(df_pred))

ci_patterns = pd.read_json("../ner_patterns.jsonl/patterns", lines=True)

# treat removal only on records which have something in "ci_entity" column
tt = df_pred
tt = tt[tt["ci_entity"].notna()]
# get where actual Ci entities are present in "ci_entity" column
tt["ci_entity_grouped"] = None
tt = pp.group_ci_types(tt, col_type="ci_entity", col_grouped="ci_entity_grouped", ci_patterns=ci_patterns)
## get cases of non-CI 
non_ci_records = tt[tt["ci_entity_grouped"].isna()]
# write back - keep only records which are in "ci_entity" either np.nan or a CI type
print(f"Remove {len(non_ci_records)} from {len(df_pred)} records which are not CI")
df_pred = df_pred.drop(index=non_ci_records.index)

print(len(df_pred))

# %%


# %% [markdown]
# #### AS FUNC: Evaluate on same documents that were passed to LLM
# 
# 

# %% [markdown]
# 

# %%
print("Use only citations which are in both")


df_valid = df_valid_org[df_valid_org["publication_id"].isin(df_pred.citation_id.unique())]
# df_valid_org[df_valid_org["publication_id"].isin(citation_list)]
print("Valid citations:")
print(df_valid.publication_id.unique())

df_pred = df_pred[df_pred["citation_id"].isin(df_valid.publication_id.unique())]
print("Predicted citations:")
print(df_pred.citation_id.unique())

# %%
print("Use only citations which are in both")


df_valid = df_valid_org[df_valid_org["publication_id"].isin(df_pred.citation_id.unique())]
# df_valid_org[df_valid_org["publication_id"].isin(citation_list)]
print("Valid citations:")
print(df_valid.publication_id.unique())

df_pred = df_pred[df_pred["citation_id"].isin(df_valid.publication_id.unique())]
print("Predicted citations:")
print(df_pred.citation_id.unique())

# %%
print(df_valid.info())
print(df_pred.info())

# %%
print(df_valid.info())
print(df_pred.info())

# %%
# df_pred.loc[df_pred["citation_id"]== "Krausmann 2014"] # Krausmann -> large hallucinations when chunk-text is title or contact info (i.e when not about CI /impacts)

# %% [markdown]
# ## MV to postprocess.fuc() drop dublicated predictions + upd (encod-utf-8)saving_llm_reuslts in loop (rm fix saving) + pp of NAN strings in LLm response
# 

# %% [markdown]
# #### MV to postprocess: cases with NANs 

# %%

def convert_nan(series: pd.Series) -> pd.Series:
    """ convert representations of "NAN" to np.nan """
    # TODO use regex instead of ["NAN", "NaN", "nan"] by setting all possible representations of nan (e.g. "Nan") to lowercase 
    series = series.replace(["NAN", "NaN", "nan"], np.nan)

    return series


print(df_pred.info())
df_pred["infrastructure_type"] = convert_nan(df_pred["infrastructure_type"])
df_pred["damage"] = convert_nan(df_pred["damage"])
df_pred["location"] = convert_nan(df_pred["location"])

print(df_pred.info(), len(df_pred))



# %% [markdown]
# #### group CI into subgroups (MAKE AS FUNC)
# 
# make CI_groups after clean-up LOC and CI-columns

# %%
ci_patterns = pd.read_json("../ner_patterns.jsonl/patterns", lines=True)


## group Ci types into subgroups,
if not "infrastructure_group" in df_pred.columns or df_pred["infrastructure_group"].isna().any():
    #print("Remove all potential brackets for plural forms in infrastructure types [(s)]")
    #df_pred["infrastructure_type"] = df_pred["infrastructure_type"].str.replace(r"\(s\)", "", regex=True).str.strip()
    df_pred = pp.group_ci_types(df_pred, "infrastructure_type", "infrastructure_group", ci_patterns)
    ## keep only records which are actually about CI (e.g., not theatre, stadion ..)
    df_pred.dropna(subset=["infrastructure_group"], inplace=True)

if not "ci1_group" in df_valid.columns or df_valid["ci1_group"].isna().any():
    df_valid["ci1_group"] = None
    df_valid = pp.group_ci_types(df_valid, "ci1_type", "ci1_group", ci_patterns)
    ## keep only records which are actually about CI (e.g., not theatre, stadion ..)
    df_valid.dropna(subset=["ci1_group"], inplace=True)


print(df_pred.infrastructure_group.isna().sum())  # mostly cases which are not CI (theater, stadion..)
print(df_pred.infrastructure_group.value_counts())
# df_pred.infrastructure_group.unique()


print(df_valid.ci1_group.isna().sum())
print(df_valid.ci1_group.value_counts())
# df_pred.infrastructure_group.unique()


# %%

print("Workaround for changing ci_group for drinking water ")

df_valid["ci1_group"] = df_valid["ci1_group"].replace(["drinking_water"], "water_supply")
df_pred["infrastructure_group"] = df_pred["infrastructure_group"].replace(["drinking_water"], "water_supply")
df_pred.infrastructure_group.value_counts()

# %%
print("Cases of CI which could not be grouped")

print(df_pred[df_pred.infrastructure_group.isna()].shape[0])
print(df_valid[df_valid.ci1_group.isna()].shape[0])

# %% [markdown]
# #### drop cases in valid and pred where Ci or LOC is empty
# 

# %%
print("Removing all records which have erroneous CI or missing LOC entry")

df_pred = df_pred[~df_pred.infrastructure_group.isna()]
df_valid = df_valid[~df_valid.ci1_group.isna()]

df_pred = df_pred[~df_pred.location.isna()]
df_valid = df_valid[~df_valid.ci1_location.isna()]


# %%
## set "affected" to NAN in damage columns (pred, valid)
## TODO check if affected is in general decreasing recall or precision score for "dam" class if yes then set to NAN otherwise keep unchanged


# %%


# %%
print(len(df_pred))
unique_ci_geo_pairs = df_pred.drop_duplicates()
print("number of duplicates to remove:", len(df_pred) - len(unique_ci_geo_pairs))

df_pred = df_pred.drop_duplicates( )# .reset_index(drop=True, inplace=True)
print(len(df_pred))


# %% [markdown]
# #### cleanup multiple locs/Cis in one entry (MV FUNC. to pp )

# %%
def split_text_into_multiple_rows(df: pd.DataFrame, column: str, split_at = " and ") -> pd.DataFrame:
    """ split text at splitting_point into multiple rows """
    # split CIs and LOCs with "and" into multiple rows
    df[column] = df[column].str.split(split_at)   
    # NOTE: Removes info from CI - drops info if CI is singular o plural (e.g, road and railway infrastrcutre --> "road", "railway infrastructure")
    df = df.explode(column=column)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

# disentangle rows which contain multiple locations or CIs
df_pred = split_text_into_multiple_rows(df_pred, "location",  split_at = " or ")
df_valid = split_text_into_multiple_rows(df_valid, "ci1_location",  split_at = " or ")
df_pred = split_text_into_multiple_rows(df_pred, "location",  split_at = ", ")  # Germany, Neterlands, Belgium
df_valid = split_text_into_multiple_rows(df_valid, "ci1_location",  split_at = ", ")

# multiple location splitting - "and"
df_pred = split_text_into_multiple_rows(df_pred, "location",  split_at = " and ")
df_pred = split_text_into_multiple_rows(df_pred, "infrastructure_type",  split_at = " and ").reset_index(drop=True)
df_valid = split_text_into_multiple_rows(df_valid, "ci1_location",  split_at = " and ")
df_valid = split_text_into_multiple_rows(df_valid, "ci1_type",  split_at = " and ").reset_index(drop=True) # make sure that same valid_sentences can occur multiple times, eg. in "Valencia and Sagunto port" (=2 rows)


# %%


# %%


# %% [markdown]
# ### remove records which are not about Europe 
# 

# %%
df_pred = df_pred[~df_pred["location"].isin(["New York", "New Jersey"])]

# %% [markdown]
# ### remove all records which are on country-level or "Europe"
# 

# %%
# import geonamescache

# def get_countries():
#     geolocs_cache = geonamescache.GeonamesCache()
#     countries = geolocs_cache.get_countries()
#     ci_geo_countries = [*u.gen_dict_extract(countries, 'name')] 
#     # add further country names with abbrev. or "the" , incl. als regions which have the same name as their country (eg. Luxembourg- Provinz in Belgium)
#     ci_geo_countries = ci_geo_countries + ["the Netherlands", "Netherlands", "UK", "US", "U.S.", "USA"]
#     return ci_geo_countries


# # remove all records which are on country-level
# ci_geo_countries = get_countries()

# print(f"Removing {len(df_pred[df_pred['location'].isin(ci_geo_countries)])} records which are on country-level in prediction set")
# print(f"Removing {len(df_valid[df_valid['ci1_location'].isin(ci_geo_countries)])} records which are on country-level in validation set")

# df_pred = df_pred[~df_pred["location"].isin(ci_geo_countries)]
# df_valid = df_valid[~df_valid["ci1_location"].isin(ci_geo_countries)]

## remove all records which mention Europe
df_pred = df_pred[~df_pred["location"].str.contains(r"Europe.*|European.*", case=False, na=False)]
df_valid = df_valid[~df_valid["ci1_location"].str.contains(r"Europe.*|European.*", case=False, na=False)]



# # df_pred[col].apply(lambda x: unidecode(x) if isinstance(x, str) else x)

# %% [markdown]
# #### FIXME country-wise removal can be loss of info for certain CI sectors or monetary impacts

# %%
# df_pred[df_pred["location"].isin(ci_geo_countries)].drop(["chunk_text","coord_potential_locations"], axis=1)
# df_valid[df_valid["ci1_location"].isin(ci_geo_countries)].drop(["sentence_reference"], axis=1)

## FIXME country-wise removal can be loss of info for certain CI sectors or monetary impacts: 
#        pred: it/telecommunication, waste_*, wastewater; Valid: also gas_supply, electricity infrastructure	

# %% [markdown]
# ### drop dublicated cases which differ only in Tier 2 or Tier 3 impacts
# > e.g. valid ABC 2024: - identical c1_type, ci1_damage, ci1_loc (but diff. ci2_damages -which are not used in this eval) 
# 

# %%
print(f"Dropping {df_valid.duplicated().sum()} duplicates in valid data")
df_valid = df_valid.drop_duplicates()

print(f"Dropping {df_pred[['citation_id', 'chunk_id', 'infrastructure_type','damage', 'location', 'chunk_text']].duplicated().sum()} duplicates in pred data")
df_pred = df_pred[df_pred[['citation_id', 'chunk_id', 'infrastructure_type', 'damage', "damage_value", 'location', 'chunk_text']].duplicated() == False]


# %%
df_pred

# %%


# %% [markdown]
# #### AS FUNC: clean-up locations

# %%

print(df_pred.location.unique())
print(df_valid.ci1_location.unique())

## TODO clean-up locs --> MV this postprocessing to LLm extraction before passing to LLm Step2
# "( "  eg "Sinzig (in North Rhine-Westphalia)""
df_pred["location"] = df_pred["location"].str.split(r"\(", regex=True).str[0].str.strip()
df_valid["ci1_location"] = df_valid["ci1_location"].str.split(r"\(", regex=True).str[0].str.strip()

# rm text after comma, e.g. "Ahr valley, Germany" --> "Ahr valley"
df_pred["location"] = df_pred["location"].str.split(", ").str[0].str.strip()  
df_valid["ci1_location"] = df_valid["ci1_location"].str.split(", ").str[0].str.strip()  
# remove all remaining brackets and commas
df_pred["location"] = df_pred["location"].replace(r"[\(\),]", "", regex=True)
df_valid["ci1_location"] = df_valid["ci1_location"].replace(r"[\(\),]", "", regex=True)

## handling loc with "railway tracks between "
df_pred["location"] = df_pred["location"].str.split("between").str[-1].str.strip()
df_valid["ci1_location"] = df_valid["ci1_location"].str.split("between").str[-1].str.strip()
## handling loc with "railway tracks in"  , set this after cleaning up "( " and ", " as it otherwise would take the later location
df_pred["location"] = df_pred["location"].str.split("in ").str[-1].str.strip()
df_valid["ci1_location"] = df_valid["ci1_location"].str.split("in ").str[-1].str.strip()

#  try to remove "the"
#  already before passing to GeoLLM in extraction-WF
df_pred["location"] = df_pred["location"].replace("the ", " ", regex=True).str.strip()   # regex= True needed to identify string part (e.g "the ") 
df_valid["ci1_location"] = df_valid["ci1_location"].replace("the ", " ", regex=True).str.strip()
## handling loc with "passing "
df_pred["location"] = df_pred["location"].replace("passing ", " ", regex=True).str.strip()   
df_valid["ci1_location"] = df_valid["ci1_location"].replace("passing ", " ", regex=True).str.strip()


print("After cleanup: drop records which are not about CI anymore")
ci_patterns = pd.read_json("../ner_patterns.jsonl/patterns", lines=True)

df_pred = pp.group_ci_types(df_pred, "infrastructure_type", "infrastructure_group", ci_patterns)
## keep only records which are actually about CI (e.g., not remainings from cleanup)
df_pred.dropna(subset=["infrastructure_group"], inplace=True)


print(df_pred.location.unique())
print(df_valid.ci1_location.unique())

# %% [markdown]
# 

# %%


# %% [markdown]
# ### locations_2_coordinates() matching, MAKE AS FUNC

# %%
def get_bbox(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

# TODO mv to utils.py or geolocalization.py

# %%
# extract dict from geollm-string
df_pred["coord_potential_locations"] = df_pred["coord_potential_locations"].apply(lambda x: eval(str(x))) 


df_pred["coords"] = None
df_pred["coords_coarse"] = None

# reindex to avoid issues in iloc
df_pred.reset_index(drop=True, inplace=True)

## get most likely coordinates for locations
counter_errors = 0
for entry in range(len(df_pred)):

    loc = df_pred["location"].iloc[entry]
    loc_potential_coords = df_pred["coord_potential_locations"].iloc[entry]
    try:
        # go to next entity when it is NAN
        if loc == np.nan or loc == "nan" or loc == "NAN" or loc == "NaN" or loc == "":
            continue
        # write coordinates to df
        df_pred.at[entry,  "coords"] = loc_potential_coords[loc][0:2]

    except Exception as e:
        try:
            for coords in loc_potential_coords.keys(): 

                ## WORKAROUND handling "Ahr river valley" <-> "Ahrtal"
                if coords == "Ahrtal":  # geollama response
                    df_pred.at[entry, "location"] = df_pred.at[entry,"location"].replace("Ahr River valley", "Ahrtal")
                    loc = df_pred["location"].iloc[entry]

                smlrty = fuzz.partial_ratio(loc, coords)
                if smlrty > 90: 
                    print("\nUsing partial ratio for matching:", loc, "<->", coords, f"{smlrty}")
                    # NOTE: handles "Malaga airport", "the Ahr valley", "Erft region"
                    # FIXME needs imrovement in the future, to get more concrete spat. info (if it is region, a river etc.)
                    df_pred.at[entry,  "coords"] = loc_potential_coords[coords][0:2]
                else: pass
       
        except Exception as e:
            print(f"\nEntry {entry}: {df_pred['location'].iloc[entry]}")
            print(f"No geolocalization possible for row {entry}: {e} \n{df_pred['coord_potential_locations'].iloc[entry]}")
            # e.g. "flood region", "A76 in both directions"
            print("Creating BBox of potential location based on locations mentioned in respective chunk text")
            coords_list = [[float(v[0]), float(v[1])] for v in loc_potential_coords.values()]
            df_pred.at[entry,  "coords_coarse"] =  get_bbox(coords_list)
            counter_errors = counter_errors + 1


print(f"{counter_errors} Cases where only coarse loc could be extracted ( based on loc in entire chunk):")
## drop cases where no geolocalization could be done 


# # TODO when loc= "A76 in both directions" --> make new column with "eigenname" new column "potentially_location_in" with list of geollm returns and bbox based on these geollm_locs
# # TODO measure location new based on centroid of loc_red or centroid of "potentially_in"

# %%
len(df_pred)

# %%
df_pred[[
    "citation_id","chunk_id", "infrastructure_type", "infrastructure_group", "damage", "damage_value", "location", "ci_entity",	"geo_entity", "infrastructure_type_org",	"damage_org","damage_value_org","locations_org","coords","coords_coarse"
    ]][4:50]  
# check for duplicates


# %% [markdown]
# ##### Loc_2_coords mathcing for validation set

# %%
# extract dict from geollm-string
df_pred["coord_potential_locations"] = df_pred["coord_potential_locations"].apply(lambda x: eval(str(x))) 


df_pred["coords"] = None
df_pred["coords_coarse"] = None

# reindex to avoid issues in iloc
df_pred.reset_index(drop=True, inplace=True)

## get most likely coordinates for locations
counter_errors = 0
for entry in range(len(df_pred)):

    loc = df_pred["location"].iloc[entry]
    loc_potential_coords = df_pred["coord_potential_locations"].iloc[entry]
    try:
        # go to next entity when it is NAN
        if loc == np.nan or loc == "nan" or loc == "NAN" or loc == "NaN" or loc == "":
            continue
        # write coordinates to df
        df_pred.at[entry,  "coords"] = loc_potential_coords[loc][0:2]
        if loc_potential_coords[loc][2] == False:
            print("\n   Not in RAG database (exact matching):", loc, loc_potential_coords[loc])

    except KeyError as e:
        try:
            for potential_coords in loc_potential_coords.keys(): 

                # TODO mv as pre-step in LLM-WF betw STEP1 & 2 (as geollm potetially can identify these words already as LOCs )
                ## WORKAROUND handling "Ahr river valley" <-> "Ahrtal"
                if potential_coords == "Ahrtal":  # geollama response  
                    df_pred.at[entry, "location"] = df_pred.at[entry,"location"].replace("Ahr River valley", "Ahrtal")
                    loc = df_pred["location"].iloc[entry]
                if potential_coords == "Málaga":  # geollama response  
                    df_pred.at[entry, "location"] = df_pred.at[entry,"location"].replace("Malaga", "Málaga")
                    loc = df_pred["location"].iloc[entry]
                # Overwrite loc and coords when it is Rhineland-Palatinate as its variations are often not geolocalized by the GeoLLM
                if loc in ["Rhineland-Palatinate", "Rheinland-Pfalz", "Rheinland Pfalz"]:
                    df_pred.at[entry, "location"] = "Rhineland-Palatinate"
                    df_pred.at[entry,  "coords"] = tuple(["50.4728741", "6.9477258"])
                    continue  
                # TODO same for typos+coordinates in "Saxony-Anhalt", "SachsenAnhalt"

                smlrty = fuzz.partial_ratio(loc, potential_coords)
                if smlrty > 90: 
                    print("\nUsing partial ratio for matching:", loc, "<->", potential_coords, f"{smlrty}")
                    # NOTE: handles "Malaga airport", "the Ahr valley", "Erft region"
                    # FIXME needs imrovement in the future, to get more concrete spat. info (if it is region, a river etc.)
                    df_pred.at[entry,  "coords"] = loc_potential_coords[potential_coords][0:2]
                    if loc_potential_coords[potential_coords][2] == False:
                        print("  Not in RAG database (partial_ratio matching):", loc, loc_potential_coords[potential_coords])
                else: pass
       
        except Exception as e:
            print(f"\nEntry {entry}: {df_pred['location'].iloc[entry]}")
            print(f"No geolocalization possible for row {entry}: {e} \n{df_pred['coord_potential_locations'].iloc[entry]}")
            # e.g. "flood region", "A76 in both directions"
            print("Creating BBox of potential location based on locations mentioned in respective chunk text")
            coords_list = [[float(v[0]), float(v[1])] for v in loc_potential_coords.values()]
            df_pred.at[entry,  "coords_coarse"] =  get_bbox(coords_list)
            counter_errors = counter_errors + 1


print(f"\n\n {counter_errors} Cases from {len(df_pred)} where only coarse loc could be extracted ( based on loc in entire chunk):")
## drop cases where no geolocalization could be done 


# # TODO when loc= "A76 in both directions" --> make new column with "eigenname" new column "potentially_location_in" with list of geollm returns and bbox based on these geollm_locs
# # TODO measure location new based on centroid of loc_red or centroid of "potentially_in"

# %%


# %% [markdown]
# #### Load spaCy language model
# 

# %%
## load english model with contextual vectors included


## RELOAD spacy pipeline
nlp = spacy.load("../spacy_model_pipeline")


# %%


# %% [markdown]
# #### Select records which have text references

# %%
df_valid.info()

# %%
print(len(df_pred), len(df_valid))
df_pred = df_pred[~df_pred["chunk_text"].isna()].reset_index(drop=True)
df_valid = df_valid[~df_valid["sentence_reference"].isna()].reset_index(drop=True)
print(len(df_pred), len(df_valid))


# %% [markdown]
# #### Translation of validation sentences

# %%

for entry in df_valid.itertuples():
    
    src_language = langdetect.detect(str(entry.sentence_reference))
    
    if src_language != "en":
        supported_languages = ["fr", "de", "es", "it", "itc", "nl"]
        if src_language not in supported_languages:
            print(f"Unsupported source language: {src_language}. Continue with original version of the sentence in validation set ")
            continue 

        print(f"\n ######## -------- Translating {entry.publication_id}: {src_language} --> en -------- ######## \n")

        # # clean up before applying translator
        # gc.collect()
        # torch.cuda.empty_cache()  # mainly after training needed, small effect when LLM applied only for inference
        # torch.no_grad()
        
        # overwrite original sentence(s) with translated versions
        translated_sentence = tm.translate_2_english(src_language, str(entry.sentence_reference))
        df_valid.loc[df_valid.index[df_valid["sentence_reference"] == entry.sentence_reference], "sentence_reference"] = translated_sentence


# %%
df_valid.info()

# %%
# # unicode to ascii representation
# print("Apply unicode on CI and damages, but not on Locations (with ä, ü and other special chars) as it removes them potentially from the DFs" )
# try:
#     for col in ["infrastructure_group", "infrastructure_type", "damage", "location", "ci_entity", "geo_entity"]:
#         df_pred[col] = df_pred[col].apply(lambda x: unidecode(x) if isinstance(x, str) else x) # handle potential np.nan
# except KeyError as e:
#     for col in ["infrastructure_group", "infrastructure_type", "damage", "location"]:
#         df_pred[col] = df_pred[col].apply(lambda x: unidecode(x) if isinstance(x, str) else x) # handle potential np.nan

# for col in ["ci1_group", "ci1_type", "ci1_damage", "ci1_location"]:
#     df_valid[col] = df_valid[col].apply(lambda x: unidecode(x) if isinstance(x, str) else x) # handle potential np.nan


# %%


# %% [markdown]
# ## add unique identifiers
# helps in calculating FPs and FNs
# 

# %%
# df_pred[["citation_id",	"chunk_id",	"infrastructure_type",	"infrastructure_group",	"damage",	"location"	]]

df_pred.loc[:,"id_pred"] = df_pred.reset_index().index
df_valid.loc[:,"id_valid"] = df_valid.reset_index().index

# %%


# %% [markdown]
# ## Merge prediction entries with potential validation entries (nth:1 pairs)

# %%
import re


print("Match chunk text of each prediction entry with related validation entries (nth:1 pairs)")
print("Align texts from valid and pred set by removing potential whitespaces")
# NOTE Solves issue: of having doubled whitespace or whitespaces due to linebreaks. eg. "Bad Münstereifel" where valid senence differed to pred_chunk due to "- " (instead of "-") in fresh-water

df_pred_valid_all = pd.DataFrame()
threshold = 65  # keep low due to differences in the tranlsation and when linebreaks where used

# find for each prediction entry all validation entries for respective chunk 
# these validation entries are candidates from which the most similar one to the pred. entry is taken to calc. model performance 
# including also entries where pred_info or valid_info is missing (e.g FNs, FPs)
for _, pred_entry in df_pred.iterrows():
    for _, valid_entry in df_valid.iterrows():  # all validation entries of all docs

        if valid_entry.sentence_reference is np.nan:
            continue

        valid_entry.sentence_reference = re.sub(r"([^\s-])\n([^\s-])", r"\1 \2", valid_entry.sentence_reference) # replace linebreak symbols when they occur just once, with whitespace (two linebreaks - probably new subsection)
        valid_entry.sentence_reference = valid_entry.sentence_reference.replace("/\n{2,}/g", "\n")  # remove linebreaks only when they occurred just once, but not for multiple linebreaks (e.g. before subsection)
        valid_entry.sentence_reference = re.sub(r"\s+", " ", valid_entry.sentence_reference)  # replace >1 whitespaces with single whitespace
        valid_entry.sentence_reference = re.sub(r"([^\s-])- ([^\s-])", r"\1-\2", valid_entry.sentence_reference)  # remove hypens in the middle of lines
        
        pred_entry.chunk_text = re.sub(r"([^\s-])\n([^\s-])", r"\1 \2", pred_entry.chunk_text) # replace linebreak symbols when they occur just once, with whitespace (two linebreaks - probably new subsection)
        pred_entry.chunk_text = pred_entry.chunk_text.replace("/\n{2,}/g", "\n")  # remove linebreaks only when they occurred just once, but not for multiple linebreaks (e.g. before subsection)
        pred_entry.chunk_text = re.sub(r"\s+", " ", pred_entry.chunk_text)  # replace >1 whitespaces with single whitespace
        pred_entry.chunk_text = re.sub(r"([^\s-])- ([^\s-])", r"\1-\2", pred_entry.chunk_text)  # remove hypens in the middle of lines

        # Calculate match score by accounting for partial string matches. 
        # In detail, it calculates the similarity ratio using the shortest string (length n, here: "sentence_reference") against all n-length substrings of the larger string and returns the highest score 
        score = fuzz.partial_ratio(valid_entry['sentence_reference'].replace(" ", ""), pred_entry['chunk_text'].replace(" ", ""))



        if score >= threshold:
            entry_pred_valid = {
                "citation_id": pred_entry["citation_id"],
                "ci_pred": pred_entry["infrastructure_type"],
                "ci_group_pred": pred_entry["infrastructure_group"],
                "damage_pred": pred_entry["damage"],
                "location_pred": pred_entry["location"],
                "coords_pred": pred_entry["coords"],
                "chunk_id_pred": pred_entry["chunk_id"],
                "chunk_text_pred": pred_entry["chunk_text"],
                "ci_valid": valid_entry["ci1_type"],
                "ci_group_valid": valid_entry["ci1_group"],
                "damage_valid": valid_entry["ci1_damage"],
                "location_valid": valid_entry["ci1_location"],
                # "coords_valid": valid_entry["coords"],
                "sentence_text_valid": valid_entry["sentence_reference"],
                "text_similarity": score,
                "id_pred": pred_entry["id_pred"],
                "id_valid": valid_entry["id_valid"]
            }
            df_pred_valid_all = pd.concat([df_pred_valid_all, pd.DataFrame([entry_pred_valid])], ignore_index=True)  # n:1 relationship DF
        
print(len(df_pred_valid_all))

# 85 threshold - 778 entries
# 75 threshold - 778 entries
# 75 threshold + CI subgrou - 513 entries




# %%
df_pred_valid_all.info() # 143 -190 entries  # 656 entries (bei simil 65)

df_pred_valid_all[df_pred_valid_all.text_similarity < 100][["chunk_text_pred", "sentence_text_valid", "text_similarity"]].sort_values(by="text_similarity", ascending=True)[:5]

## --> FPs are more common compared to FNs, especially for predicting locations, 
# as it is easier to get a prep-valid match when pred.info is actually missing due to larger chunk-text (pred set) compared to sentence-text (valid set)


# %%
## entries with lowest similarity
df_pred_valid_all.text_similarity.describe() # 413  (no regex, step2) , 284 (no c regex, step1)
# df_pred_valid_all.iloc[df_pred_valid_all.text_similarity.sort_values(ascending=True).index] [["sentence_text_valid", "chunk_text_pred","text_similarity"]]

# %% [markdown]
# ## Document-wise evaluation
# 
# Measures simply if the predicted CI-LOC case also occurs in the validation set\
# It does not measure the frequency - just if the CI-LOC case exists in the validation set. In this way, the approach is similar to a spatial evaluation which also just captures the occurrence and location, not the frequency with which the impact (ie CI-LOC case) was reported in the document 
# 

# %%
## WORKAORUND - to check how spat. eval could perform eventually (or when variations of location names are aligned)
df_pred["location"] = df_pred["location"].str.replace(r"Málaga", "Malaga", regex=True) # regex=True needed to recognize relevant part of longer string (here "Malaga ")
df_pred["location"] = df_pred["location"].str.replace(r"Malaga airport", "Malaga", regex=True)
df_pred["location"] = df_pred["location"].str.replace(r"Port of Valencia", "Valencia port", regex=True)
df_valid["ci1_location"] = df_valid["ci1_location"].str.replace(r"Málaga", "Malaga", regex=True)
df_valid["ci1_location"] = df_valid["ci1_location"].str.replace(r"Malaga airport", "Malaga", regex=True)
df_valid["ci1_location"] = df_valid["ci1_location"].str.replace(r"Port of Valencia", "Valencia port", regex=True)


# %%
# ## TEST rm records where no geolocalization was possible (incl. coarse coords)
print(f"Would remove {len(df_pred[df_pred['coords'].isna()])} from {len(df_pred)} records where no geolocalization was possible (incl. coarse coords)")

## --> currently incl. non-geolocalized entries in the EVAL process ->  leads better performance scores
df_pred_geolocalized = df_pred[~df_pred["coords"].isna()]

# df_pred[df_pred['coords'].isna()][
#     ["citation_id", "chunk_id", "infrastructure_type", "infrastructure_group", "location", "damage", "coord_potential_locations"]
#     ][:60]

# %%
fns_list = []
fps_list = []   
tps_list = []



for publication in df_valid.publication_id.unique():

    print(f"\n\nDocument: {publication}")
    docs_valid = df_valid[df_valid["publication_id"] == publication]
    docs_pred = df_pred_geolocalized[df_pred_geolocalized["citation_id"] == publication]

    # get all valid and predicted CI-LOC pairs for each publication
    docs_valid_pairs = docs_valid[["ci1_group", "ci1_location"]].drop_duplicates().values.tolist()
    docs_pred_pairs = docs_pred[["infrastructure_group", "location"]].drop_duplicates().values.tolist()
    print(docs_valid_pairs)
    print(docs_pred_pairs)

    # calc TPs, FPs, FNs
    tps = len([t for t in docs_pred_pairs if t in docs_valid_pairs]) 
    fps = len([t for t in docs_pred_pairs if t not in docs_valid_pairs]) 
    fns = len([t for t in docs_valid_pairs if t not in docs_pred_pairs]) # geo-llama.trsting_on_news2024.ipynb
    print(f"TPs: {tps}, FPs: {fps}, FNs: {fns}")
    tps_list.append(tps)
    fps_list.append(fps)
    fns_list.append(fns)
    print(f"  FPs: { [t for t in docs_pred_pairs if t not in docs_valid_pairs]}")
    print(f"  FNs: { [t for t in docs_valid_pairs if t not in docs_pred_pairs]}")

recall_score = u.calc_recall(tps_no=sum(tps_list), fns_no=sum(fns_list)) 
precision_score = u.calc_precision(tps_no=sum(tps_list), fps_no=sum(fps_list))
try:
    f1_score = u.calc_f1(precision=precision_score, recall=recall_score)
except ZeroDivisionError:
    f1_score = 0.0

print(f"\n\n ---------- Evaluation statistics (document-wise)-----------")
print(f"Precision: {precision_score}, Recall: {recall_score}, F1-score: {f1_score}")


# only gelocolized:
# Precision: 0.275, Recall: 0.3728813559322034,  F1-score: 0.3165467625899281
# Precision: 0.3, Recall: 0.3305084745762712, F1-score: 0.31451612903225806
# Precision: 0.4230769230769231, Recall: 0.3618421052631579, F1-score: 0.3900709219858156
# Precision: 0.38636363636363635, Recall: 0.288135593220339, F1-score: 0.3300970873786408 # noner , Koks: TPs: 16, FPs: 12, FNs: 34



# incl non gelocalized:
# Precision: 0.2660098522167488, Recall: 0.4576271186440678, F1-score: 0.33644859813084116
# Precision: 0.27647058823529413, Recall: 0.3983050847457627, F1-score: 0.3263888888888889
# Precision: 0.3941176470588235, Recall: 0.4407894736842105, F1-score: 0.4161490683229813



# %% [markdown]
# 

# %%
print(len(df_valid), len(df_pred), len(df_pred_geolocalized))
# df_pred = df_pred_geolocalized

# TPs: 16, FPs: 12, FNs: 34 noner


# %% [markdown]
# ## spatial mapping

# %%
df_pred_geolocalized#[["citation_id", "chunk_id", "infrastructure_type", "infrastructure_group", "damage", "location", "coords"]][:50]

# %%


# %%


# %% [markdown]
# ## Calc similarities 
# * TPs (for all cases where text info in pred and valid set exists)
# * FNs  (model missed actual cases)
# * FPs  (model hallucinated cases)

# %%
list_entity_valid = ["ci_group_valid", "damage_valid", "location_valid"]
list_entity_pred = ["ci_group_pred", "damage_pred", "location_pred"]



#  Set similarity threshold (self-defined) when CI case is valid or not FN/FP
cos_smlrty_thresh = 0.7
norm_pr_smlrty_thresh = 0.7


print(" --- For each unique valid case (unique combi: [ci_valid, damage_valid, location_valid, sentence_text]) calculate similarity ---")
print("Using 100% match for CI types based on subgroups")
print("Using cosine similarity threshold for damages", cos_smlrty_thresh)
print("Using normalized partial ratio similarity threshold for locations", norm_pr_smlrty_thresh)

## AIM of evaluation loop below: 
# remove all cases in df_pred_valid_all where pred_entities were wrongly assigned to a valid_entity
## ie keep only pre-valid pairs with highest similarity per unique valid case

## call embedding model for semantic similarity calculation
embedding_model = u.EmbeddingModel()


# store evaluation results
df_eval_records = pd.DataFrame()


# For each impact case (rows) 
for record_no, impact_record in df_pred_valid_all.iterrows():

    print(f"Record: {record_no } / {len(df_pred_valid_all)}")

    # init dict to store results for each records (row=)
    df_eval = {
        "citation": impact_record.citation_id,
        "chunk_text_pred": impact_record.chunk_text_pred,
        "sentence_text_valid": impact_record.sentence_text_valid,
        "id_pred": impact_record.id_pred,
        "id_valid": impact_record.id_valid
    }
    
    # iterate over the three entity classes (ci, damage, location) to assess LLM performance
    for entity_valid, entity_pred in zip(list_entity_valid, list_entity_pred):

        # Calculate similarities for entries in column pair: entity_pred - entity_valid

        ## calc similarity when both valid_info exist (not NAN) 
        # NOTE df_pred model can put out NAN when case exists but it couldnt find suitable value (e.g. damage_pred="NaN", damage_valid="polluted")
        if impact_record[entity_valid] is not np.nan:
        # if impact_record[entity_pred] and impact_record[entity_valid] is not np.nan:
            pred_impact = impact_record[entity_pred]
            valid_impact = impact_record[entity_valid]

            if entity_pred == "ci_group_pred": # for CI group, only partial ratio similarity is calculated as it is more important to get the correct group than the exact match (e.g. "port infrastructure" <-> "port")
                
                # similarity on idential match 
                if pred_impact == valid_impact:
                    ci_smlrty = 1
                else:
                    ci_smlrty = 0

                # store result for ci entity 
                df_eval["ci_pred"] = pred_impact  # CI subgroup
                df_eval["ci_valid"] = valid_impact # CI subgroup
                df_eval["ci_smlrty"] = ci_smlrty

            if entity_pred == "damage_pred": 

                if isinstance(pred_impact, str) & isinstance(valid_impact, str): # check that pred or valid are not NAN
                    # contextual vectors (transformer-based)
                    embedded_list = embedding_model.vector_calculation(pred_impact, valid_impact)
                    # calculate cosine similarity for each pred-valid damage pair
                    similarity_score_cos = embedding_model.cosine_similarity(embedded_list[0], embedded_list[1])  # 0-1 value, the higher the more similar
                
                # if nan -> then it is either FP or FN
                elif isinstance(pred_impact, float) | isinstance(np.nan, float):
                    similarity_score_cos = 0.0     

                # store result for DAM and LOC entity 
                df_eval["dam_pred"] = pred_impact  
                df_eval["dam_valid"] = valid_impact 
                df_eval["dam_smlrty"] = similarity_score_cos


            if entity_pred == "location_pred": 
                ## Cosine similarity calc.
                if isinstance(pred_impact, str):
                    # contextual vectors (transformer-based)
                    embedded_list = embedding_model.vector_calculation(pred_impact, valid_impact)
                    # calculate cosine similarity for each pred-valid pair
                    similarity_score_cos = embedding_model.cosine_similarity(embedded_list[0], embedded_list[1])  # 0-1 value, the higher the more similar
                elif np.isnan(pred_impact):
                    similarity_score_cos = 0.0   # NOTE it is FNs
                ## Partial ratio similarity calc. (especially for locations and CI-type  "port infrastructure" <-> "port")
                similarity_score_pr = fuzz.partial_ratio(pred_impact, valid_impact)  

                # store result for DAM and LOC entity 
                df_eval["loc_pred"] = pred_impact  
                df_eval["loc_valid"] = valid_impact 
                df_eval["loc_smlrty"] = similarity_score_cos
                df_eval["loc_smlrty_norm_pr"] = similarity_score_pr / 100

    # collect all single records (row) with similarity scores
    df_eval_records = pd.concat([df_eval_records, pd.DataFrame([df_eval])], ignore_index=True)


## NOTE Description: How 1:1 pairs for pred-valid are extracted f
## 1. group by single records from df_valid (via id_valid indices), 
##    Column "id_valid": index represents single records from df_valid (when validation_sentence contains 2 cases: -> id-valid:0, id_valid:1,  sentence w 1 case: id-valid:2) 
## 2. then collect from each group the one with highest similarity to predictions
##    --> binary "mask" indicates where we have matches -e.g. correct predictions (true: TP, false: FN or FP)  is our match (1:1 pred-valid pair) - from which TPs can be calculated

## 1. + 2.
# select for each single valid record (ie rows in df_valid) the 1:1 match (pred-valid pair, "head(1)") with highest similarities across all three classes 
# NOTE need to sort based on all three smlrty cols to do correct Tp calc 
#      (if sort_values by on similartiy column would result in too many TPs- as then 1:1 pairs would contain also random matches where randomly CI_red is identical with CI_valid)
df_smltry_selmax = df_eval_records.groupby("id_valid").apply(lambda s: s.sort_values(["ci_smlrty","dam_smlrty","loc_smlrty_norm_pr"], ascending=False).head(1))
# # OLD  (makes too many 1:1 pairs as described in NOTE)
# mask = df_eval_records.groupby("id_valid").apply(lambda x: x==x["ci_smlrty"].max()).droplevel(0)
# df_smltry_selmax2 = df_eval_records.where(mask.ci_smlrty==mask.ci_smlrty.max()).dropna(how="all") # keep cases only which have highest similarity scores
# df_smltry_selmax2.reset_index(drop=True, inplace=True)


print("for each unique valid record keep only pred-valid pairs of highest similarity")


# iterate over the three entity classes (ci, damage, location) to assess LLM performance
for _, column_pred in zip(list_entity_valid, list_entity_pred):

    if column_pred == "ci_group_pred":

        # remove cases where no CI could be found (for Ci unlikelky, but more common for location or damage)
        df_valid_ci = df_valid[df_valid["ci1_group"].notnull()]
        df_pred_ci = df_pred[df_pred["infrastructure_group"].notnull()]

        # when no similarity could be calculated
        # ## FIXME move outside of loop
        # entries_with_no_similarity = df_eval_records.loc[df_eval_records["impact_sim_identical"].isna()]
        # print(f" --- Pred-valid pairs where no identical similarity score could be calculated: {len(entries_with_no_similarity)} ----")
        # print(entries_with_no_similarity[["impact_valid", "impact_pred", "impact_sim_identical", "impact_sim_cos", "impact_sim_pr", "citation"]])

        
        # TPs 
        tps = df_smltry_selmax.loc[df_smltry_selmax["ci_smlrty"] == 1]
        print(len(tps), len(df_smltry_selmax ), len(df_smltry_selmax[~df_smltry_selmax[["ci_pred", "ci_valid"]].isna().any(axis=1)]))
        
        
        # # FPs
        # --> make mask where records in df-eval record are identical to df_pred.columns (must be 1:1), 
        #     aplly mask on df_pred and substract from output all cases which are in TPs 
        # assert len(output) == fps_len
        
        # FNs
        ## missed docs
        df_valid_ci_pred_missed_docs = df_valid_ci[df_valid_ci["publication_id"].isin(df_pred["citation_id"]) == False]
        ## missed entries
        # extracts all duplicates (except first occurrence eg. id_Pred==537 occurs in df_smltry_selmax_p three times (1st case: TP or FP, 2nd and 3rd are FNs)
        df_valid_cases_missed_by_model = df_smltry_selmax[df_smltry_selmax.duplicated(subset="id_pred", keep="first")]
        ## OLD APPROACH: CI cases in valid set (for docs existing in both sets) - number of corectly predicted CI cases (Tps)
        ## no_valid_cases_missed_by_model  = len(df_valid_pred_same_docs["ci1_group"])  - len(df_smltry_selmax[df_smltry_selmax["impact_sim_identical"]==1])

        # ## TODO FIXME not sure if approach for df_valid_cases_missed_by_model based on df_smltry_selmax is correct
        # ##            as df_smltry_selmax contains only the cases of highest similarity for each case in df_valid (ie unique id_valid)
        # ##            can i then calc the number of missed cases by 
        
        # fns = pd.concat([df_valid_pred_missed_docs, df_valid_cases_missed_by_model], ignore_index=True, axis=0)

        # FIXME WORKAROUND for FP and FN calculation, but not extract respective cases (only numbers of FPs and FNs)
        fps_len = len(df_pred_ci["infrastructure_group"]) - tps.shape[0]
        fns_len = len(df_valid_ci["ci1_group"]) - tps.shape[0]

        print("tps", len(tps), " fps:", fps_len, " fns:", fns_len)
                    
        # performance scores
        recall_score = u.calc_recall(tps_no=len(tps), fns_no=fns_len) 
        precision_score = u.calc_precision(tps_no=len(tps), fps_no=fps_len)
        try:
            f1_score = u.calc_f1(precision=precision_score, recall=recall_score)
        except ZeroDivisionError:
            f1_score = 0.0

        print(f" ---------- Evaluation statistics: {column_pred}-----------")
        print(f"Recall: {recall_score}, Precision: {precision_score}, F1-score: {f1_score}")

        # saving
        dh.DataHandler().save_evaluation_results(column_pred, df_smltry_selmax, recall_score, precision_score, f1_score)


    if column_pred == "damage_pred":

        # remove cases where no CI could be found (for CI unlikelky, but more common for location or damage)
        df_valid_dam = df_valid[df_valid["ci1_damage"].notnull()]
        df_pred_dam = df_pred[df_pred["damage"].notnull()]  # when model gave NaN (then actually also corresponding df_valid record would be there NaN)

        # TPs 
        tps = df_smltry_selmax.loc[df_smltry_selmax["dam_smlrty"] >= cos_smlrty_thresh]

        # ## check if TPs calc correct
        # # tps_validmergedpred = df_valid.merge(df_pred, left_on=["ci1_damage"], right_on=["damage"], how="inner") ## ERROR as gives > 4000 entries
        # # assert len(tps) == len(tps_validmergedpred)

        # # FPs - model predicts condition wrongly (ie. predict condition when it is actually absent)
        # # get all valid. documents which were also used for LLM inference
        # df_valid_pred_same_docs = df_valid[df_valid["publication_id"].isin(df_pred["citation_id"])]
        # print(f"Doing evaluation based on {df_valid_pred_same_docs.publication_id.unique().__len__()} documents existing in both (valid.+pred. set)")
        # # get records where model predicted presence of impacts but they actually does not exist
        # fps = df_smltry_selmax.loc[df_smltry_selmax["impact_sim_identical"] < cos_smlrty_thresh]
        # # here as definition, that when simi=0 (or below threshold) then model predicted presences as false alarm
        # # TODO
        # # add also as Fps were model_pred case exist but no fitting_valid case could be found (during df_valid_pred pair generation in loop at begin of NB)
        # # df_pred selction needed

        # # FNs - CI impact cases not detected by model 
        # # NOTE: maybe FNs number is biased as wrong matches more likely as chunk-text (pred set) is longer than sentence text (valid set)
        # # WRONG? get all entries from df_valid_pred_same_docs where corresponding pred_record (in FPs) is missing

        # # get all documents in valid_set which does not occur in pred_set or where similarity is too low
        # ## missed docs
        # df_valid_pred_missed_docs = df_valid[df_valid["publication_id"].isin(df_pred["citation_id"]) == False]
        # print("Number of documents where model did not extract anything", df_valid_pred_missed_docs.shape)
        # ## missed entries
        # # extracts all duplicates (except first occurrence eg. id_Pred==537 occurs in df_smltry_selmax_p three times (1st case: TP or FP, 2nd and 3rd are FNs)
        # df_valid_cases_missed_by_model = df_smltry_selmax[df_smltry_selmax.duplicated(subset="id_pred", keep="first")]
        # ## OLD APPROACH: CI cases in valid set (for docs existing in both sets) - number of correctly predicted CI cases (Tps)
        # # no_valid_cases_missed_by_model  = len(df_valid_pred_same_docs["ci1_damage"])  - len(df_smltry_selmax["impact_valid"])
        # # no_valid_cases_missed_by_model  = len(df_valid_pred_same_docs["ci1_location"])  - len(df_smltry_selmax["impact_valid"])
        # fns = pd.concat([df_valid_pred_missed_docs, df_valid_cases_missed_by_model], ignore_index=True, axis=0)


        # FIXME WORKAROUND for FP and FN calculation, but not extract respective cases (only numbers of FPs and FNs)
        fps_len = len(df_pred_dam["damage"]) - tps.shape[0]  # that
        fns_len = len(df_valid_dam["ci1_damage"]) - tps.shape[0]

        print("tps", len(tps), " fps:", fps_len, " fns:", fns_len)

        # performance scores
        recall_score = u.calc_recall(tps_no=len(tps), fns_no=fns_len) 
        precision_score = u.calc_precision(tps_no=len(tps), fps_no=fps_len)
        try:
            f1_score = u.calc_f1(precision=precision_score, recall=recall_score)
        except ZeroDivisionError:
            f1_score = 0.0

        print(f" ---------- Evaluation statistics: {column_pred}-----------")
        print(f"Recall: {recall_score}, Precision: {precision_score}, F1-score: {f1_score}")
        
        ## saving
        dh.DataHandler().save_evaluation_results(column_pred, df_smltry_selmax, recall_score, precision_score, f1_score)
        

    if column_pred == "location_pred":

        # remove cases where no CI could be found (for CI unlikelky, but more common for location or damage)
        df_valid_loc = df_valid[df_valid["ci1_location"].notnull()]
        df_pred_loc = df_pred[df_pred["location"].notnull()]  # when model gave NaN (then actually also corresponding df_valid record would be there NaN)

        # TPs 
        tps = df_smltry_selmax.loc[df_smltry_selmax["loc_smlrty_norm_pr"] >= norm_pr_smlrty_thresh]

        # ## check if TPs calc correct
        # # tps_validmergedpred = df_valid.merge(df_pred, left_on=["ci1_location"], right_on=["location"], how="inner") ## ERROR as gives > 4000 entries
        # # assert len(tps) == len(tps_validmergedpred)

        # # FPs - model predicts condition wrongly (ie. predict condition when it is actually absent)
        # # get all valid. documents which were also used for LLM inference
        # df_valid_pred_same_docs = df_valid[df_valid["publication_id"].isin(df_pred["citation_id"])]
        # print(f"Doing evaluation based on {df_valid_pred_same_docs.publication_id.unique().__len__()} documents existing in both (valid.+pred. set)")
        # # get records where model predicted presence of impacts but they actually does not exist
        # fps = df_smltry_selmax.loc[df_smltry_selmax["impact_sim_identical"] < cos_smlrty_thresh]
        # # here as definition, that when simi=0 (or below threshold) then model predicted presences as false alarm
        # # TODO
        # # add also as Fps were model_pred case exist but no fitting_valid case could be found (during df_valid_pred pair generation in loop at begin of NB)
        # # df_pred selction needed

        # # FNs - CI impact cases not detected by model 
        # # NOTE: maybe FNs number is biased as wrong matches more likely as chunk-text (pred set) is longer than sentence text (valid set)
        # # WRONG? get all entries from df_valid_pred_same_docs where corresponding pred_record (in FPs) is missing

        # # get all documents in valid_set which does not occur in pred_set or where similarity is too low
        # ## missed docs
        # df_valid_pred_missed_docs = df_valid[df_valid["publication_id"].isin(df_pred["citation_id"]) == False]
        # print("Number of documents where model did not extract anything", df_valid_pred_missed_docs.shape)
        # ## missed entries
        # # extracts all duplicates (except first occurrence eg. id_Pred==537 occurs in df_smltry_selmax_p three times (1st case: TP or FP, 2nd and 3rd are FNs)
        # df_valid_cases_missed_by_model = df_smltry_selmax[df_smltry_selmax.duplicated(subset="id_pred", keep="first")]
        # ## OLD APPROACH: CI cases in valid set (for docs existing in both sets) - number of correctly predicted CI cases (Tps)
        # # no_valid_cases_missed_by_model  = len(df_valid_pred_same_docs["ci1_damage"])  - len(df_smltry_selmax["impact_valid"])
        # # no_valid_cases_missed_by_model  = len(df_valid_pred_same_docs["ci1_location"])  - len(df_smltry_selmax["impact_valid"])
        # fns = pd.concat([df_valid_pred_missed_docs, df_valid_cases_missed_by_model], ignore_index=True, axis=0)
        
        # FIXME WORKAROUND for FP and FN calculation, but not extract respective cases (only numbers of FPs and FNs)
        fps_len = len(df_pred_loc["location"]) - tps.shape[0]
        fns_len = len(df_valid_loc["ci1_location"]) - tps.shape[0]
        
        print("tps", len(tps), " fps:", fps_len, " fns:", fns_len)

        # performance scores
        recall_score = u.calc_recall(tps_no=len(tps),fns_no=fns_len) 
        precision_score = u.calc_precision(tps_no=len(tps), fps_no=fps_len)
        try:
            f1_score = u.calc_f1(precision=precision_score, recall=recall_score)
        except ZeroDivisionError:
            f1_score = 0.0
        
        print(f" ---------- Evaluation statistics: {column_pred}-----------")
        print(f"Recall: {recall_score}, Precision: {precision_score}, F1-score: {f1_score}")

        ## saving
        dh.DataHandler().save_evaluation_results(column_pred, df_smltry_selmax, recall_score, precision_score, f1_score)



## only geolocalzed + improved valid set
# 98 117 117
# tps 98  fps: 65  fns: 84
#  ---------- Evaluation statistics: ci_group_pred-----------
# Recall: 0.5384615384615384, Precision: 0.6012269938650306, F1-score: 0.5681159420289855
# Saving evaluation statistics, distribution plots, and scores to  ci_group_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 34  fps: 119  fns: 136
#  ---------- Evaluation statistics: damage_pred-----------
# Recall: 0.2, Precision: 0.2222222222222222, F1-score: 0.2105263157894737
# Saving evaluation statistics, distribution plots, and scores to  damage_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 74  fps: 89  fns: 108
#  ---------- Evaluation statistics: location_pred-----------
# Recall: 0.4065934065934066, Precision: 0.4539877300613497, F1-score: 0.42898550724637685
# Saving evaluation statistics, distribution plots, and scores to  location_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]


## only geolocalized --> better in PRECISION when df_valid_pred matching only w. geolocalized cases  comp. non-geolocaiized incl.)
# 70 87 87
# tps 70  fps: 122  fns: 76
#  ---------- Evaluation statistics: ci_group_pred-----------
# Recall: 0.4794520547945205, Precision: 0.3645833333333333, F1-score: 0.41420118343195267
# Saving evaluation statistics, distribution plots, and scores to  ci_group_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 32  fps: 149  fns: 107
#  ---------- Evaluation statistics: damage_pred-----------
# Recall: 0.2302158273381295, Precision: 0.17679558011049723, F1-score: 0.2
# Saving evaluation statistics, distribution plots, and scores to  damage_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 60  fps: 132  fns: 86
#  ---------- Evaluation statistics: location_pred-----------
# Recall: 0.410958904109589, Precision: 0.3125, F1-score: 0.35502958579881655
# Saving evaluation statistics, distribution plots, and scores to  location_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]


##  geolocalized + non-gelocalized
# for each unique valid record keep only pred-valid pairs of highest similarity
# 77 94 94
# tps 77  fps: 172  fns: 69
#  ---------- Evaluation statistics: ci_group_pred-----------
# Recall: 0.5273972602739726, Precision: 0.3092369477911647, F1-score: 0.389873417721519
# Saving evaluation statistics, distribution plots, and scores to  ci_group_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 33  fps: 205  fns: 106
#  ---------- Evaluation statistics: damage_pred-----------
# Recall: 0.23741007194244604, Precision: 0.13865546218487396, F1-score: 0.17506631299734746
# Saving evaluation statistics, distribution plots, and scores to  damage_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 65  fps: 184  fns: 81
#  ---------- Evaluation statistics: location_pred-----------
# Recall: 0.4452054794520548, Precision: 0.26104417670682734, F1-score: 0.32911392405063294
# Saving evaluation statistics, distribution plots, and scores to  location_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]

### LAMA 3 + NER + GeoLLM (step 2)

# 43 43 22  - 24 docs
# tps 43  fps: 104  fns: 38
#  ---------- Evaluation statistics: ci_group_pred-----------
# Recall: 0.5308641975308642, Precision: 0.2925170068027211, F1-score: 0.37719298245614036
# Saving evaluation statistics, distribution plots, and scores to  ci_group_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 8  fps: 91  fns: 67
#  ---------- Evaluation statistics: damage_pred-----------
# Recall: 0.10666666666666667, Precision: 0.08080808080808081, F1-score: 0.09195402298850575
# Saving evaluation statistics, distribution plots, and scores to  damage_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 32  fps: 115  fns: 49
#  ---------- Evaluation statistics: location_pred-----------
# Recall: 0.3950617283950617, Precision: 0.21768707482993196, F1-score: 0.2807017543859649
# Saving evaluation statistics, distribution plots, and scores to  location_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]


### LAMA 3 + NER + GeoLLM (step 1) - no countries (only regions, cities, etc)
## not better than with countries (slight increase in recall+precision for LOC )

### LAMA 3 + NER + GeoLLM (step 1)  - 24 docs
# or each unique valid record keep only pred-valid pairs of highest similarity
# 39 46 46
# tps 39  fps: 186  fns: 49
#  ---------- Evaluation statistics: ci_group_pred-----------
# Recall: 0.4431818181818182, Precision: 0.17333333333333334, F1-score: 0.24920127795527158
# Saving evaluation statistics, distribution plots, and scores to  ci_group_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 16  fps: 176  fns: 66
#  ---------- Evaluation statistics: damage_pred-----------
# Recall: 0.1951219512195122, Precision: 0.08333333333333333, F1-score: 0.11678832116788321
# Saving evaluation statistics, distribution plots, and scores to  damage_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 26  fps: 199  fns: 62
# ---------- Evaluation statistics: location_pred-----------
# Recall: 0.29545454545454547, Precision: 0.11555555555555555, F1-score: 0.16613418530351434
# Saving evaluation statistics, distribution plots, and scores to  location_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json] 

### LAMA 3 + NER
# 15 33
# tps 15  fps: 27  fns: 18
#  ---------- Evaluation statistics: ci_group_pred-----------
# Recall: 0.45454545454545453, Precision: 0.35714285714285715, F1-score: 0.4
# Saving evaluation statistics, distribution plots, and scores to  ci_group_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 11  fps: 31  fns: 19
#  ---------- Evaluation statistics: damage_pred-----------
# Recall: 0.36666666666666664, Precision: 0.2619047619047619, F1-score: 0.3055555555555555
# Saving evaluation statistics, distribution plots, and scores to  damage_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 10  fps: 32  fns: 23
#  ---------- Evaluation statistics: location_pred-----------
# Recall: 0.30303030303030304, Precision: 0.23809523809523808, F1-score: 0.26666666666666666

## Lama 3 # 1 doc Koks 2022
# 20 33
# tps 20  fps: 30  fns: 13
#  ---------- Evaluation statistics: ci_group_pred-----------
# Recall: 0.6060606060606061, Precision: 0.4, F1-score: 0.4819277108433735
# Saving evaluation statistics, distribution plots, and scores to  ci_group_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# ps 12  fps: 38  fns: 18
#  ---------- Evaluation statistics: damage_pred-----------
# Recall: 0.4, Precision: 0.24, F1-score: 0.3
# Saving evaluation statistics, distribution plots, and scores to  damage_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# ps 12  fps: 38  fns: 21
#  ---------- Evaluation statistics: location_pred-----------
# Recall: 0.36363636363636365, Precision: 0.24, F1-score: 0.2891566265060241
# Saving evaluation statistics, distribution plots, and scores to  location_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]



# llm_1_updprompt_dNER.csv
# 45 67
# tps 45  fps: 1115  fns: 22
#  ---------- Evaluation statistics: ci_group_pred-----------
# Recall: 0.6716417910447762, Precision: 0.03879310344827586, F1-score: 0.07334963325183375
# tps 14  fps: 1146  fns: 47
#  ---------- Evaluation statistics: damage_pred-----------
# Recall: 0.22950819672131148, Precision: 0.01206896551724138, F1-score: 0.022932022932022934
# tps 11  fps: 1149  fns: 43
#  ---------- Evaluation statistics: location_pred-----------
# Recall: 0.2037037037037037, Precision: 0.009482758620689655, F1-score: 0.018121911037891267


# %%


# %%
# # with FAC entity based incl: 94 241 178

# for each unique valid record keep only pred-valid pairs of highest similarity
# 44 53 53
# tps 44  fps: 134  fns: 50
#  ---------- Evaluation statistics: ci_group_pred-----------
# Recall: 0.46808510638297873, Precision: 0.24719101123595505, F1-score: 0.3235294117647059
# Saving evaluation statistics, distribution plots, and scores to  ci_group_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 19  fps: 148  fns: 68
#  ---------- Evaluation statistics: damage_pred-----------
# Recall: 0.21839080459770116, Precision: 0.11377245508982035, F1-score: 0.1496062992125984
# Saving evaluation statistics, distribution plots, and scores to  damage_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 29  fps: 149  fns: 65
#  ---------- Evaluation statistics: location_pred-----------
# Recall: 0.30851063829787234, Precision: 0.16292134831460675, F1-score: 0.21323529411764708
# Saving evaluation statistics, distribution plots, and scores to  location_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]

 

## without FAC-based records
## valid, pred, pred_local: 94 209 151

# for each unique valid record keep only pred-valid pairs of highest similarity
# 44 53 53
# tps 44  fps: 107  fns: 50
#  ---------- Evaluation statistics: ci_group_pred-----------
# Recall: 0.46808510638297873, Precision: 0.2913907284768212, F1-score: 0.3591836734693878
# Saving evaluation statistics, distribution plots, and scores to  ci_group_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 19  fps: 121  fns: 68
#  ---------- Evaluation statistics: damage_pred-----------
# Recall: 0.21839080459770116, Precision: 0.1357142857142857, F1-score: 0.16740088105726872
# Saving evaluation statistics, distribution plots, and scores to  damage_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]
# tps 25  fps: 126  fns: 69
#  ---------- Evaluation statistics: location_pred-----------
# Recall: 0.26595744680851063, Precision: 0.16556291390728478, F1-score: 0.20408163265306123
# Saving evaluation statistics, distribution plots, and scores to  location_smlrty_llm_1_dNER_fixNER [.parquet, _stats.json]

# %% [markdown]
# #### ISSUE: some pred-valid matches are wrongly matched in Text-Simil.based eval
# --> (probably bc no better pred_entry for corresponding valid_entry exists) e,g, 
# * pred: "in ahr valley some roads were destroyed. the railways in Germany were damaged" - roads,destroyed, ahr valley
# * valid: "the railways in Germany were damaged" - railways, damaged, Germany
# 
# IDEA. do document-wise comparison or based on spatial agg. --> later maybe better da 

# %%
df_pred.to_csv(f"df_pred_{step}_why_ci_dam_loc_bad.csv")
df_smltry_selmax.to_csv(f"df_smlrty_max_{step}_why_ci_dam_loc_bad.csv")

# %%


# %% [markdown]
# ## EVAL why performance is not good LLAMA 3
# 

# %%
df_smltry_selmax[["citation", "sentence_text_valid", "id_pred", "id_valid", "ci_pred", "ci_valid", "ci_smlrty", "dam_pred", "dam_valid", "dam_smlrty", "loc_pred", "loc_valid", "loc_smlrty", "loc_smlrty_norm_pr"]].head(5)

# %%


# %% [markdown]
# #### FIXME: find out which cases model predicted existence, but not in valid DS - maybe due that valid DS is incomppete?

# %%
df_pred.info()

# %%
## fix FPs 

## get all pred cases which 
# rows in df_valid where sentence_reference appears as substring in at least one df_pred.chunk_text
chunk_texts = df_pred["chunk_text"].dropna().astype(str)

df_valid_2 = df_valid[
    df_valid["sentence_reference"].fillna("").astype(str).apply(
        lambda s: any(s and s in chunk for chunk in chunk_texts)
    )
]

df_valid_2 # .shape (28, 7)

# %%
# cases where pred-case exist but no fitting valid case could be found based on sentence_reference
df_pred_not_in_valid = df_pred[df_pred['chunk_text'].str.contains('|'.join(df_valid["sentence_reference"]), regex=True)]
print(df_pred_not_in_valid.id_pred.value_counts())
df_pred_not_in_valid.head(10)

# TODO TODO
## documents where model found many CI cases as false-alarms:
# maybe i need to recheck those docs and make df_valid more complete
# print(df_pred_not_in_valid.groupby("citation_id").count())
# citation_id                                                            
# ABC 2024                        4
# Containerlift 2024             24
# European Investment Bank 2025  21
# Ferlita 2023                   22
# Koks 2022                      10


# %%
df_valid

# %% [markdown]
# #### FIXME: FPs and FNs

# %%
## TPs + FNs should be == len(df_valid.ci) == 67
        
# TPs 
tps = df_smltry_selmax.loc[df_smltry_selmax["impact_sim_identical"] == 1]

# FNs
df_valid_pred_missed_docs = df_valid[df_valid["publication_id"].isin(df_pred["citation_id"]) == False]
df_valid_cases_missed_by_model = df_smltry_selmax[df_smltry_selmax.duplicated(subset="id_pred", keep="first")]
fns = pd.concat([df_valid_pred_missed_docs, df_valid_cases_missed_by_model], ignore_index=True, axis=0)

print(tps.shape[0], fns.shape[0])
print(tps.shape[0] + fns.shape[0])

# --> 8 cases in FNs are too definitly too much --> fix FN calculation



## FPs should be == len(df_pred.ci) - TPs

## FPs
fps = df_smltry_selmax.loc[df_smltry_selmax["impact_sim_identical"] == 0]

print(len(df_pred.infrastructure_type), tps.shape[0], fps.shape[0])
print(len(df_pred.infrastructure_type) - tps.shape[0])




# %%
# TODO fix FPs
# get records where model predicted presence of impacts but they actually does not exist
# here as definition, that when simi=0 (or below threshold) then model predicted wrongly
df_smltry_not_sim = df_smltry_selmax.loc[df_smltry_selmax["impact_sim_identical"] == 0]

# TODO
# add also as Fps were model_pred case exist but no fitting_vlaid case could be found

# idea: 
# get all df_pred cases where chunk text not occurs in valid.sentece_text

df_pred_not_in_valid = df_pred[df_pred["chunk_text"].isin(df_valid["sentence_reference"])== False]
print(df_pred.shape, df_pred_not_in_valid.shape)
# df_pred_not_in_valid

# %%
# df_valid__pred_no_thresh.id_pred.nunique()
df_pred_valid_no_thresh.id_pred.nunique()

# %%
# df_valid__pred_no_thresh.info()
# df_pred_valid_no_thresh.info()  # 67 valid * 921 pred = 61707
# df_pred_valid_no_thresh.drop("chunk_text_pred", axis=1).sort_values("id_pred").iloc[0:100]
# df_pred_valid_no_thresh.groupby("id_pred").first().sort_values("text_similarity", ascending=False).iloc[0:100]
# df_valid__pred_no_thresh.groupby("id_valid").first().sort_values("text_similarity", ascending=False).iloc[0:100]
#df_valid__pred_no_thresh.sort_values("id_valid", ascending=False).sort_values("text_similarity", ascending=False).iloc[0:100]
#df_valid__pred_no_thresh.groupby("id_valid").first().sort_values("text_similarity", ascending=False).iloc[0:100]
df_valid__pred_no_thresh.groupby("id_valid").apply(lambda x: x.loc[x["text_similarity"].idxmax()])


# %%


# %%
# fns

# %%
## FNs 
df_smltry_selmax.loc[df_smltry_selmax.duplicated("id_pred")]
## --> ISSUE: this df (cases of highest sim) should NOT have duplicated cases of predictions -> maybe have to group based on id_pred and not id_valid

## try to fix issue
## --> currently i think this should group based on valid cases to measure were model predicted the same or missed info (ie FNs)
# df_smltry_selmax_p = df_smltry_selmax
# mask of rows with highest similarity score for each set of preds with unique valid case (droplevel(0) remove multiindex)
mask = df_smltry_all.loc[df_smltry_all.id_valid==37].groupby("id_valid").apply(lambda x: x==x["impact_sim_identical"].max()).droplevel(0)
df_smltry_selmax_p = df_smltry_all.where(mask.impact_sim_identical==mask.impact_sim_identical.max()).dropna(how="all") # drop cases which have not highest similairty score
# df_smltry_selmax_p = df_smltry_all.groupby("id_valid").apply(lambda x: x.loc[x["impact_sim_identical"].idxmax()]) # 52 cases
# df_smltry_selmax_p = df_smltry_all.groupby("id_pred").apply(lambda x: x.loc[x["impact_sim_identical"].idxmax()]) # 175 cases
df_smltry_selmax_p.reset_index(drop=True, inplace=True)

## FIXME  df_smltry_all.groupby("id_valid"): should it has duplicated cases of id_pred ? - i dont think so! 
#  bc it means that there model missed cases in valid_set
## --> so all duplicated cases (except one-this is TP or FP) are actual FNs
print(df_smltry_selmax_p.info())
print(df_smltry_selmax_p.id_pred.nunique()  )  # should be len of df
print(df_smltry_selmax_p.duplicated().sum())


# FNs: extracts all duplicates (except first occurrence eg. id_Pred==537 occurs in df_smltry_selmax_p three times (1st case: TP or FP, 2nd and 3rd are FNs)
fns = df_smltry_selmax_p[df_smltry_selmax_p.duplicated(subset="id_pred", keep="first")]

print(fns.info())
fns.id_pred.value_counts()


# %%
# return all cases which has max sim also when max score is shared by multiple rows 
# df_smltry_all.loc[df_smltry_all.groupby("id_valid").transform(lambda x: x==x.max()).astype('bool')].shape
mask = df_smltry_all.loc[df_smltry_all.id_valid==37].groupby("id_valid").apply(lambda x: x==x["impact_sim_identical"].max())
mask = mask.droplevel(0)
#.transform(lambda x: x==x.max())
tt = df_smltry_all.loc[df_smltry_all.id_valid==37]#
tt = tt.where(mask.impact_sim_identical==mask.impact_sim_identical.max()).dropna(how="all") # drop cases which have not highest similairty score

# tt[mask]

# would return only first case of max sim:
#df_smltry_all.loc[df_smltry_all.id_valid==37].groupby("id_valid").apply(lambda x: x.loc[x["impact_sim_identical"].idxmax()]) #


# %%


# %%
# FPs. 
print("False alarms (where model predicted ci but no corresponding valid case exists)", 
      len(df_pred["infrastructure_type"])  - len(df_smltry_selmax[df_smltry_selmax["impact_sim_identical"]== 1, "ci_group_pred"])
    )
# Get FPs - cases where model predicted presence of CI (but actually it is absent in valid set)
tt = df_pred.merge(
    # FIXME issue that df_pred_valid_all contains some duplicates where id_pred identical but not valid_entries
    df_smltry_selmax.drop_duplicates(), # safety: make sure that merging is done on 1:1 match
    left_on="id_pred",#["citation_id", "chunk_id","infrastructure_type", "damage", "location"], 
    right_on="id_pred",#["citation_id", "chunk_id_pred", "ci_pred", "damage_pred", "location_pred"],
    how="left",
    indicator=True    # return an extra column indicating which table the row was from.
)
tt = tt.loc[tt["_merge"] == "left_only"].drop(columns=["_merge"])
print("False positives (model predicted CI but no corresponding valid case exists):", len(tt))

# %%
print(df_pred.shape[0])
# print(df_pred_valid_all.info())
print(tt.info())

# %%


# %%
df_pred#["infrastructure_type"]

# %%
df_smltry_selmax.info()

# %%
# df_smltry_selmax["impact_sim_identical"] < cos_smlrty_thresh

# %%
# len(df_valid_pred_same_docs["ci1_group"]) 

# %%
# TODO fix FNs
print(df_valid_pred_same_docs.info())
print(df_smltry_selmax[df_smltry_selmax["impact_sim_identical"]==1].info())
# --> FNS should be  23
len(df_valid_pred_same_docs["ci1_group"])  - len(df_smltry_selmax[df_smltry_selmax["impact_sim_identical"]==1]["impact_valid"])

# %%
# #df_valid_pred_same_docs["id_valid"] = df_valid_pred_same_docs.apply(lambda x: f"{x['ci1_group']}_{x['ci1_damage']}_{x['ci1_location']}_{x['sentence_text_valid'][:50]}", axis=1)
# print(df_valid_pred_same_docs["id_valid"].unique().__len__())
# print(df_valid_pred_same_docs.shape[0])
# ## --> check why electricity_others_outages_nan = 3  - (seems correct as sentences_ref are diff). airports_affected_Malaga area=2 are not unique
# df_valid_pred_same_docs[df_valid_pred_same_docs["id_valid"] == "airports_affected_Malaga area"]

# %%


# %% [markdown]
# # Improve similarity
# As all similarity measures - no matter which emebdding model or kind of cosine similarity measure) were not sufficient eg. port ~ power to similar to port~harbor
# 
# Thus, it might be better to first group ci impacts into subgroups e.g .based on HARCI-EU categories,as some kind of postprocessing step before applying the similarity measurements
# 
# 

# %%


# %%
# s = "dyke" to s2 = "levee", s3 = "dam"
# bge-m3: 0.48  0.54
# all-MIniLM-L6-v2: 0.34 , 0.36  (similar all-mpnet-base-v2)
# gensim word2vec: 0.39 0.40


# s1 = "aviation" s2 = "air traffic"
# word vector spacy: 0.45
# contextual vector spacy: 0.68
# bge-m3: 0.76
# all-MIniLM-L6-v2: xx  (all-mpnet-base-v2: 0.79)
# gensim word2vec: 


# s1 = "power" s2 = "electricity"
# word vector spacy: 0.61
# contextual vector spacy: 0.66
# bge-m3: 
# all-MIniLM-L6-v2: xx   (all-mpnet-base-v2: 0.43)
# gensim word2vec: 0.58


# s1 = "electricity infrastructure" s2 = "electricity"
# word vector spacy: 0.87
# contextual vector spacy: 0.71
# bge-m3: 
# all-MIniLM-L6-v2:   xx  (all-mpnet-base-v2: 0.63)
# gensim word2vec: 


# s1 = "transportation" s2 = "transport infrastructure"
# word vector spacy: 
# contextual vector spacy: 
# bge-m3: 
# all-MIniLM-L6-v2:   xx  (all-mpnet-base-v2: 0.84)
# gensim word2vec: 

# s1 = "port" s2 = "power"  s3= harbour
# bge-m3: 0.58, 0.50
# all-MIniLM-L6-v2: 0.33 , 0.56  (similar all-mpnet-base-v2)
# gensim word2vec: 0.14 0.59

# s1 = "electricity" s2 = "transportation" 
# bge-m3:  0.64
# all-MIniLM-L6-v2:   (all-mpnet-base-v2: 0.47)
# gensim word2vec: 0.33

# %%
# # print(cos_sim(model_scs["transportation"], model_scs["transport infrastructure"]))
# # print(cos_sim(model_scs["electricity infrastructure"], model_scs["electricity"]))
# # print(cos_sim(model_scs["power plant"], model_scs["electricity"]))
# print(cos_sim(model_scs["power"], model_scs["electricity"]))
# print(cos_sim(model_scs["aviation"], model_scs["air traffic"]))
# # identical to model_scs.similarity("port", "power"))


# # similarity_score = 1-distance.cosine(model.encode([s1])[0], model.encode([s2])[0])

# %%


# %% [markdown]
# ### Analyse evaluation results 
# 

# %%
df_smltry_selmax#.info()

# %%
## find out for which docs model performed bad (or good)
## based on this info try to improve model 

df_smltry_selmax.dropna(subset=["impact_sim_cos"]).groupby("citation").apply(lambda x: x.loc[x["impact_sim_cos"].idxmax()]).sort_values(by="impact_sim_cos", ascending=True)
## check EFE, Wilson, European Investment Bank, Containerlift, Lloyds List, Gilbody Dickerson


# %%
## check entries of worst performace docs for damage
df_smltry_selmax.loc[df_smltry_selmax["citation"].isin(["Khazai 2023", "ABC 2024", "Containerlift 2024", "Lloyds List 2024", "Ferlita 2023"])]

# %%
## check entries of worst performance docs for Ci tyes
df_smltry_selmax.loc[df_smltry_selmax["citation"].isin(["EFE 2024", "Containerlift 2024", "Lloyds List 2024", "Wilson 2024", "Gilbody Dickerson 2024", "European Investment Bank 2025"])].head(50)

# %%


# %%


# %% [markdown]
# 
# ## For each validation entry, search for all prediction cases of the same chunk 

# %%
## get same impact entries
list_entity_valid = ["ci1_type", "ci1_damage", "ci1_location"]
list_entity_pred = ["infrastructure_type", "damage", "location"]


for entity_valid, entity_pred in zip(list_entity_valid, list_entity_pred):

    print(f" --------- Processing column pair: {entity_valid} - {entity_pred} ------------")
    
    df_valid_pred_all = pd.DataFrame()
    citations_list = []

    ## for each validation record
    for i in range(len(df_valid)):
        
        highest_similarity_score = 0.00
        
        ## needed to traceback info when entry is missing in pred. DS
        # chunk_id_value_valid = df_valid.chunk_id[i]

        # select nth validation record and check that it has value
        df_valid_entry = df_valid.iloc[i]
        if df_valid_entry[entity_valid] is np.nan:
            continue
        
        citation_str = df_valid_entry.publication_id
        citations_list.append(citation_str)


        # get all corresponding prediction records
        df_pred_entries = df_pred[df_pred["citation_id"].isin([citation_str])]

        #  handle on NANs
        df_pred_entries[entity_pred] = np.where(df_pred_entries[entity_pred].isna(), "nan", df_pred_entries[entity_pred])
        # df_pred_entries[entity_pred] = df_pred_entries[entity_pred].astype(str)
        # remove double whitespaces
        # df_pred_doc[entity_pred] = df_pred_doc[entity_pred].replace("  ", " ")
        # df_valid_entries[entity_valid] = df_valid_entries[entity_valid].replace("  ", " ")


        # vector of validiation entry 
        valid_impact = df_valid_entry[entity_valid]
        valid_vec = nlp(valid_impact).vector

        # print(" ------- Searching for citation:", citation_str, " in predictions ------- ")

        # Compute similarity between each validation CI impact case and all potential predicted CI impact cases (cross-product)
        for j in range(len(df_pred_entries[entity_pred])):

            if df_pred_entries[entity_pred].iloc[j] == "nan":
                continue

            pred_impact = df_pred_entries[entity_pred].iloc[j]

            pred_vec = nlp(pred_impact).vector
            similarity_score = embedding_model.cosine_similarity(valid_vec, pred_vec)  # 0-1 value, the higher the more similar
            # print(f"Similarity {i}-{j}: {similarity_score}")

            # print(f"Searching for highest similarity ... ")
            ## get only pair with highest similarity
            if similarity_score > highest_similarity_score:
                
                highest_similarity_score = similarity_score
                
                dict_pair = {
                    "impact_valid": valid_impact, 
                    "impact_pred": pred_impact, 
                    "similarity": highest_similarity_score,
                    "citation": citation_str,
                    "chunk_id_pred": (df_pred.chunk_id[i],  df_pred.chunk_id[j])
                }
            else:
                continue

        df_valid_pred_all = pd.concat([df_valid_pred_all, pd.DataFrame([dict_pair])], ignore_index=True)


    print(f" ---------- Evaluation summary statistics - {entity_pred}: -----------")
    print(df_valid_pred_all.similarity.describe())

    

    SIMILARITY_FILENAME = f'{entity_pred}_{SIMILARITY_LLM_FILENAME}'
    SIMILARITY_FILEPATH = Path(PATH_EVAL_RESULT / SIMILARITY_FILENAME)

    print("Saving evaluation statistics, distribution plots, and scores to ", SIMILARITY_FILEPATH.stem, "[.parquet, _stats.json]")
    with open(SIMILARITY_FILEPATH, 'w') as f:
        # results
        df_valid_pred_all.to_csv(SIMILARITY_FILEPATH.with_suffix('.csv'), index=False)
        df_valid_pred_all_pyarrow = pa.Table.from_pandas(df_valid_pred_all)
        pq.write_table(df_valid_pred_all_pyarrow, SIMILARITY_FILEPATH)   
        #   summary statistics
        df_valid_pred_all_stats = df_valid_pred_all.describe()
        f = SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_stats.json"  
        df_valid_pred_all_stats.to_json(f, indent=4)
        # distribution plots
        df_valid_pred_all.similarity.hist(bins=100).to_file(SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_hist.png")



    # cos_smlrty_thresh = 0.75
    # df_valid_pred_all['is_similar'] = df_valid_pred_all['similarity'] <= cos_smlrty_thresh
    # print(f"Number of similar impact cases (similarity >= {cos_smlrty_thresh}): {df_valid_pred_all['is_similar'].sum()} out of {len(df_valid_pred_all)}\n")

    # df_valid_pred_all =  df_valid_pred_all[df_valid_pred_all['similarity'] <= cos_smlrty_thresh]

    # SIMILARITY_FILENAME = f'{entity_pred}_lower75_{SIMILARITY_LLM_FILENAME}'
    # SIMILARITY_FILEPATH = Path(PATH_EVAL_RESULT / SIMILARITY_FILENAME)

    # with open(SIMILARITY_FILEPATH, 'w') as f:
    #     # results
    #     df_valid_pred_all.to_csv(SIMILARITY_FILEPATH.with_suffix('.csv'), index=False)
    #     df_valid_pred_all_pyarrow = pa.Table.from_pandas(df_valid_pred_all)
    #     pq.write_table(df_valid_pred_all_pyarrow, SIMILARITY_FILEPATH)  
    #     #   summary statistics
    #     df_valid_pred_all_stats = df_valid_pred_all.describe()
    #     f = SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_stats.json"  
    #     df_valid_pred_all_stats.to_json(f, indent=4)


# %%
# df_valid_pred_all[df_valid_pred_all['similarity'] <= 0.75]

# df_valid_pred_all.similarity.hist(bins=100)

# %%
list_entity_pred

# %%
LLM_DATA_FILEPATH

# %% [markdown]
# ### Load parquet file

# %%

list_entity_pred = ["infrastructure_type", "damage", "location"]

# %%
entity_pred = "infrastructure_type"
SIMILARITY_FILENAME = f'llm1_similarity_{entity_pred}_75.parquet'
SIMILARITY_FILEPATH = Path(PATH_EVAL_RESULT / SIMILARITY_FILENAME)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    df = pd.read_parquet(SIMILARITY_FILEPATH, engine='pyarrow')
    display(df)

# %%


# %% [markdown]
# ## Archive

# %%
## Aim 
## for all identical valid entries ie. with same [ci_valid	damage_valid	location_valid	sentence_text_valid]
## get the match to pred_entity with highest similarity

# %%
    # ## calc for each entry with the same chunk_text the similarity between valid_impact and pred_impact
    # ## means we calc also the False Negatives (ie. where valid entry exists but no prediction)


    # # iterate over groups of entities which refer to the same valid case (i.e. which are identical in valid_columns)
    # # TODO iterate over unqiue cases in df_valid (instead of using grouper)
    # grouper = df_pred_valid_all[["ci_valid", "damage_valid", "location_valid", "sentence_text_valid"]].drop_duplicates()
    # for group in grouper.itertuples():
    #     df_pred_valid_group = df_pred_valid_all[df_pred_valid_all[["ci_valid", "damage_valid", "location_valid", "sentence_text_valid"]] == group[["ci_valid", "damage_valid", "location_valid", "sentence_text_valid"]]]

    #     # calc. similarities to pred_entities
    #     for i, entry in df_pred_valid_group.iterrows():

    #         highest_similarity_score = 0 

    #         if entry[entity_pred].iloc[i] == "nan":
    #             continue
            
    #         # calc embeddings
    #         pred_impact = entry[entity_pred].iloc[i]
    #         pred_vec = nlp(pred_impact).vector

    #         valid_impact = entry[entity_valid].iloc[i] # is unique for each group
    #         valid_vec = nlp(valid_impact).vector
    #         print(valid_impact, "valid_impact")
            
    #         similarity_score = u.cosine_similarity(valid_vec, pred_vec)  # 0-1 value, the higher the more similar
    #         # print(f"Similarity {i}-{j}: {similarity_score}")

    #         ## return only pred-valid-pair with highest similarity
    #         if similarity_score > highest_similarity_score:
                
    #             highest_similarity_score = similarity_score
                
    #             entry["impact_similarity"] = highest_similarity_score

    # ## FNs
    # # # calc FN when valid_info exists but not corresponding pred_info
    # ## number of FNs is small due that wrong matching with any chunk-text is more likely due to its text size comapred sentence-level (valid set) 
    # elif entry[entity_pred] is np.nan:
    #     similarity_score = 0
    #     dict_pair = {
    #         "impact_valid": valid_impact, 
    #         "impact_pred": pred_impact, 
    #         "impact_similarity": similarity_score,
    #         "tp_tn_fp_fn": "fn",
    #         "citation": entry.citation_id,
    #         "chunk_text_pred": entry.chunk_text_pred,
    #         "sentence_text_valid": entry.sentence_text_valid,
    #         }
    #     df_smltry_selmax = pd.concat([df_smltry_selmax, pd.DataFrame([dict_pair])], ignore_index=True)

    # ## FPs
    # elif entry[entity_valid] is np.nan:
    #     similarity_score = 0
    #     dict_pair = {
    #         "impact_valid": valid_impact, 
    #         "impact_pred": pred_impact, 
    #         "impact_similarity": similarity_score,
    #         "tp_tn_fp_fn": "fp",
    #         "citation": entry.citation_id,
    #         "chunk_text_pred": entry.chunk_text_pred,
    #         "sentence_text_valid": entry.sentence_text_valid,
    #         }
    #     df_smltry_selmax = pd.concat([df_smltry_selmax, pd.DataFrame([dict_pair])], ignore_index=True)




# %%


# %%
# ## get same impact entries
# list_entity_valid = ["ci1_type", "ci1_damage", "ci1_location"]
# list_entity_pred = ["infrastructure_type", "damage", "location"]



## iterate over predictions and search for each prediction reocrds for corresponding valid cases 

# for entity_valid, entity_pred in zip(list_entity_valid, list_entity_pred):

#     print(f" --------- Processing column pair: {entity_valid} - {entity_pred} ------------")
    
#     df_valid_pred_all = pd.DataFrame()
#     citations_list = []

#     ## for each validation record
#     for i in range(len(df_valid)):
        
#         highest_similarity_score = 0.00
        
#         ## needed to traceback info when entry is missing in pred. DS
#         # chunk_id_value_valid = df_valid.chunk_id[i]

#         # select nth validation record
#         df_valid_entry = df_valid.iloc[i]
#         citation_str = df_valid_entry.publication_id
#         citations_list.append(citation_str)
#         print(" ------- Searching for citation:", citation_str, " in predictions ------- ")


#         # get all corresponding prediction records
#         df_pred_entries = df_pred[df_pred["citation_id"].isin([citation_str])]
#         #  handle on NANs
#         df_pred_entries[entity_pred] = np.where(df_pred_entries[entity_pred].isna(), "nan", df_pred_entries[entity_pred])
#         # df_pred_entries[entity_pred] = df_pred_entries[entity_pred].astype(str)
#         # remove double whitespaces
#         # df_pred_doc[entity_pred] = df_pred_doc[entity_pred].replace("  ", " ")
#         # df_valid_entries[entity_valid] = df_valid_entries[entity_valid].replace("  ", " ")

#         # skip when validation entry ha no value
#         if df_valid_entry[entity_valid] is np.nan:
#             continue

#         # vector of validiation entry 
#         valid_impact = df_valid_entry[entity_valid]
#         valid_vec = nlp(valid_impact).vector


#         # Compute similarity between each predicted impact case and all potential validation impact cases (cross-product)
#         # print(f"Searching for highest similarity of`{pred_impact}` in validation set ... ")
#         for j in range(len(df_pred_entries[entity_pred])):

#             if df_pred_entries[entity_pred].iloc[j] == "nan":
#                 continue

#             pred_impact = df_pred_entries[entity_pred].iloc[j]

#             pred_vec = nlp(pred_impact).vector
#             similarity_score = u.cosine_similarity(valid_vec, pred_vec)  # 0-1 value, the higher the more similar
#             # print(f"Similarity {i}-{j}: {similarity_score}")

#             ## get only pair with highest similarity
#             if similarity_score > highest_similarity_score:
                
#                 highest_similarity_score = similarity_score
                
#                 dict_pair = {
#                     "impact_valid": valid_impact, 
#                     "impact_pred": pred_impact, 
#                     "similarity": highest_similarity_score,
#                     "citation": citation_str,
#                     "chunk_id_pred": df_pred.chunk_id[i]
#                 }
#             else:
#                 continue

#         df_valid_pred_all = pd.concat([df_valid_pred_all, pd.DataFrame([dict_pair])], ignore_index=True)


#     print(" ---------- Evaluation summary statistics: -----------")
#     print(df_valid_pred_all.similarity.describe())



#     SIMILARITY_FILENAME = f'{SIMILARITY_LLM_FILENAME}_{entity_pred}.parquet'
#     SIMILARITY_FILEPATH = Path(PATH_EVAL_RESULT / SIMILARITY_FILENAME)

#     print("Saving evaluation statistics and scores to ", SIMILARITY_FILEPATH.stem, "[.parquet, _stats.json]")
#     with open(SIMILARITY_FILEPATH, 'w') as f:
#         # results
#         df_valid_pred_all_pyarrow = pa.Table.from_pandas(df_valid_pred_all)
#         pq.write_table(df_valid_pred_all_pyarrow, SIMILARITY_FILEPATH)   
#         #   summary statistics
#         df_valid_pred_all_stats = df_valid_pred_all.describe()
#         f = SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_stats.json"  
#         df_valid_pred_all_stats.to_json(f, indent=4)



#     cos_smlrty_thresh = 0.75
#     df_valid_pred_all['is_similar'] = df_valid_pred_all['similarity'] >= cos_smlrty_thresh
#     print(f"Number of similar impact cases (similarity >= {cos_smlrty_thresh}): {df_valid_pred_all['is_similar'].sum()} out of {len(df_valid_pred_all)}")

#     SIMILARITY_FILENAME = f'{SIMILARITY_LLM_FILENAME}_{entity_pred}_75.parquet'
#     SIMILARITY_FILEPATH = Path(PATH_EVAL_RESULT / SIMILARITY_FILENAME)

#     with open(SIMILARITY_FILEPATH, 'w') as f:
#         # results
#         df_valid_pred_all_pyarrow = pa.Table.from_pandas(df_valid_pred_all)
#         pq.write_table(df_valid_pred_all_pyarrow, SIMILARITY_FILEPATH)  
#         #   summary statistics
#         df_valid_pred_all_stats = df_valid_pred_all.describe()
#         f = SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_stats.json"  
#         df_valid_pred_all_stats.to_json(f, indent=4)


# %%

# #  Define folder for handling and writing outputs
# def write_to_file(data, out_folder, filename):
#     """Convert output to DataFrame and write to file"""
#     df = pd.DataFrame(list(data), columns=['tag', 'sts_score'])
#     #  Sort the DataFrame by similarity (explicitly)
#     df = df.sort_values(by='sts_score', ascending=False)
#     #  Assign integers to ranking
#     df['rank'] = df['sts_score'].rank(method='first', ascending=False).astype(int)
#     #  Only keep the first 20 resulting tags
#     df = df.head(50)
#     #  Save to file
#     df.to_csv(out_folder / f'{filename}_output.csv', index=False)

# #  Fill run metrics to dictionary
# def handle_metrics(metrics, model_name, length, end_time, start_time):
#     print(f'-> Took {end_time - start_time:.2f} seconds. Number of tags: {length}.')
#     metrics.append({
#         'modelname': model_name,
#         'runtime': round(end_time - start_time, 2),
#         'tagcount': length
#     })
#     return metrics

# class CPU_Unpickler(pickle.Unpickler):
#     """Fix for having issues with loading models on CPU"""
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else: return super().find_class(module, name)


# %%


# %%



