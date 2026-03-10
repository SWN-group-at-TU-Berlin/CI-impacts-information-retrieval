 
# # AIM:
# create evaluation workflow. Taking the manually extracted Ci impacts (validation set) and compare it with the CI impacts (llm_geolocations.ipynb) extrracted by the first LLM 1. 
# As a first step the evaluation should be done only for the direct CI impacts - CI type, damage and geolocation
# 
# Issue:
# * What is needed an approach that recognizes when an direct impact case is not detected by the model
# Idea: 
# * Split the original texts passed to the model on the exact chunks as again
# * Then chunkwise check if the CI impacts from the validation set correspond in number and their textual similarity to the CI impacts infered by the LLM 1 and Entity Linking 




 
# ## Semantic Textual Similarity (STS)
# 
# Calculating the STS for both model configurations (chain of prompts, orchestration of models)
# The outputs are cosine similarity scores for similar model outputs per chunk. They are ranked by score for each model, restricted to the top 20 results.  
# 


import os
import sys
import io
import numpy as np

from pathlib import Path
import pickle
import time
import warnings
import subprocess
import importlib

import spacy
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append("./")
from src.settings import settings as s
import src.document_cleaning as dc

 
# ### Direct CI impacts: LLM 1 vs domain-expertise 


#  Suppress future warnings from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)

#  Define data dir where tags.csv and domain-expertise derived tag lists are found 
PATH_MANUAL_DATA = Path(s.PATH_EVALUATION + 'manual_extracted')
PATH_EVAL_RESULT = s.PATH_EVAL_RESULT
os.makedirs(PATH_EVAL_RESULT, exist_ok=True)
LX_DATA_FILEPATH = Path(s.PATH_LX_DATA / s.LX_DATA_FILENAME)

df_valid = pd.read_csv(
    PATH_MANUAL_DATA / 'table_ci_impacts_sm.csv',
    usecols=["publication_id", "ci1_type", "ci1_damage", "ci1_location"]
)

print(len(df_valid))
## pre-process: 
# remove undone entries
# df_valid = df_valid[~df_valid["publication_id"].str.contains("xx")]



df_pred = pd.read_csv(
    LX_DATA_FILEPATH,
    usecols=["citation_id", "infrastructure_type", "damage", "geolocation"]
)

# #### Create vectors for evaluation and LLm output


## load english model with word vectors included
s.SPACY_MODEL = "en_core_web_lg"

print(f"Try loading spaCy language model ({s.SPACY_MODEL}) for remote instance (e.g., cluster)")
try: 
    nlp = spacy.load(s.SPACY_MODEL)
except (OSError, ValueError):
    print(f"spaCy language model '{s.SPACY_MODEL}' not found. Downloading ...")
    subprocess.check_call(["uv", "pip", "install", "spacy-transformers"])
    subprocess.check_call(["uv", "run", "python", "-m", "spacy", "download", s.SPACY_MODEL])
    nlp = spacy.load(s.SPACY_MODEL)
        



def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    return dot_product / (magnitude_a * magnitude_b)

# Access the vector for specific word(s)
vec_valid = nlp(df_valid.ci1_damage[1]).vector
vec_estimated = nlp(df_pred.damage[1]).vector
ci_impact_similarity = cosine_similarity(vec_estimated, vec_valid)  # 0-1 value, the higher the more similar
ci_impact_similarity


# import re
# def extract_citation_info(citation_entry: str) -> str:
#     """Extract citation information from the document filename."""
    
#     citation_pattern = r"(.*?)(\d{4})(.*)" # split at first occurrence of year
#     citation_entry = citation_entry.replace("et al ", "")
#     try:
#         authors, year, _ = re.findall(citation_pattern, citation_entry)[0]
#         citation = f"{authors} {year}"
#     except AttributeError as e:
#         print(f"Could not extract citation from title: {e}")
#         citation = citation_entry
#     return citation


## citation alignment
df_pred["citation_id"] = df_pred["citation_id"].map(dc.extract_citation_info)
print(df_pred["citation_id"])





## get same impact entries
columns_valid = ["ci1_type", "ci1_damage", "ci1_location"]
columns_pred = ["infrastructure_type", "damage", "geolocation"]

for column_valid, column_pred in zip(columns_valid, columns_pred):

    print(f" --------- Processing column pair: {column_valid} - {column_pred} ------------")
    
    df_valid_pred_all = pd.DataFrame()
    citations_list = []

    ## for each entry in df_pred
    for i in range(len(df_pred)):
        
        highest_similarity_score = 0.00
        
        ## needed to traceback info when entry is missing in valid. DS (eeg buildin impact fo EFE 2024)

        # get first entry of LLM prediction
        df_pred_doc = df_pred.iloc[i]
        citation_str = df_pred_doc.citation_id
        citations_list.append(citation_str)
        print("Searching for citation:", citation_str)


        # get all corresponding validation documents
        df_valid_entries = df_valid[df_valid["publication_id"].isin([citation_str])]

        ### # preprocessing:
        #  handle on NANs
        df_valid_entries.loc[:, column_valid] = df_valid_entries[column_valid].astype(str)

        # remove double whitespaces
        # df_pred_doc[column_pred] = df_pred_doc[column_pred].replace("  ", " ")
        # df_valid_entries[column_valid] = df_valid_entries[column_valid].replace("  ", " ")


        # compute similarity between each predicted impact case and all validation impact cases (cross-product)
        impact_pred = df_pred_doc[column_pred]
        vec_pred = nlp(impact_pred).vector


        # print(f"Searching for highest similarity to `{impact_pred}` in validation set ... ")
        for j in range(len(df_valid_entries[column_valid])):

            if df_valid_entries[column_valid].iloc[j] == "nan":
                continue

            impact_valid = df_valid_entries[column_valid].iloc[j]

            vec_valid = nlp(impact_valid).vector
            similarity_score = cosine_similarity(vec_valid, vec_pred)  # 0-1 value, the higher the more similar
            # print(f"Similarity {i}-{j}: {similarity_score}")

            ## get only  pair with highest similarity
            if similarity_score > highest_similarity_score:
                # print("New score, old score", similarity_score, highest_similarity_score)
                highest_similarity_score = similarity_score
                dict_pair = {
                    "impact_pred": impact_pred, 
                    "impact_valid": impact_valid, 
                    "similarity": highest_similarity_score,
                    "citation": citation_str,
                }
            else:
                continue

        df_valid_pred_all = pd.concat([df_valid_pred_all, pd.DataFrame([dict_pair])], ignore_index=True)

    print(" ---------- Evaluation summary statistics: -----------")
    print(df_valid_pred_all.similarity.describe())



    SIMILARITY_FILENAME = f'{s.SIMILARITY_FILENAME}_{column_pred}.parquet'
    SIMILARITY_FILEPATH = Path(PATH_EVAL_RESULT / SIMILARITY_FILENAME)

    print("Saving evaluation statistics and scores to ", SIMILARITY_FILEPATH.stem, "[.parquet, _stats.json]")
    with open(SIMILARITY_FILEPATH, 'w') as f:
        # results
        df_valid_pred_all_pyarrow = pa.Table.from_pandas(df_valid_pred_all)
        pq.write_table(df_valid_pred_all_pyarrow, SIMILARITY_FILEPATH)   
        #   summary statistics
        df_valid_pred_all_stats = df_valid_pred_all.describe()
        f = SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_stats.json"  
        df_valid_pred_all_stats.to_json(f, indent=4)
        


    similarity_threshold = 0.75
    df_valid_pred_all['is_similar'] = df_valid_pred_all['similarity'] >= similarity_threshold
    print(f"Number of similar impact cases (similarity >= {similarity_threshold}): {df_valid_pred_all['is_similar'].sum()} out of {len(df_valid_pred_all)}")

    SIMILARITY_FILENAME = f'{SIMILARITY_FILENAME}_{column_pred}_75.parquet'
    SIMILARITY_FILEPATH = Path(PATH_EVAL_RESULT / SIMILARITY_FILENAME)

    with open(SIMILARITY_FILEPATH, 'w') as f:
        # results
        df_valid_pred_all_pyarrow = pa.Table.from_pandas(df_valid_pred_all)
        pq.write_table(df_valid_pred_all_pyarrow, SIMILARITY_FILEPATH)  
        #   summary statistics
        df_valid_pred_all_stats = df_valid_pred_all.describe()
        f = SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_stats.json"  
        df_valid_pred_all_stats.to_json(f, indent=4)
    # iterate over documents


# df_valid_pred_all[:40]
df_valid_pred_all.describe()

# ### Load parquet file
columns_pred = ["infrastructure_type", "damage", "geolocation"]


column_pred = "infrastructure_type"
SIMILARITY_FILENAME = f'llm1_similarity_{column_pred}_75.parquet'
SIMILARITY_FILEPATH = Path(PATH_EVAL_RESULT / SIMILARITY_FILENAME)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    df = pd.read_parquet(SIMILARITY_FILEPATH, engine='pyarrow')
    # display(df)
    print(df.describe())
    print(df.head(20))

