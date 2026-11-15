#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NER model (spaCy pipeline + CI patterns) and the NER tools of LLM 1 and the LLM-as-a-judge"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import spacy
from langchain_core.tools import tool

# spaCy pipeline and CI patterns are stored in the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/tools/ner.py -> project root
SPACY_MODEL_PIPELINE_DIR = PROJECT_ROOT / "spacy_model_pipeline"
NER_PATTERNS_DIR = PROJECT_ROOT / "ner_patterns.jsonl"


def load_ci_patterns() -> pd.DataFrame:
    """NER patterns of the CI types and their subgroups (needed for grouping the CI types of the LLM response)"""
    return pd.read_json(NER_PATTERNS_DIR / "patterns", lines=True)




# 

# ## NER model

class NERModel():

    def __init__(self):
        self.nlp_model = self.initialize()
    

    def initialize(self):
        """ Return initialized NER model"""
  
        dir_model_pipeline = SPACY_MODEL_PIPELINE_DIR
        dir_ci_patterns = NER_PATTERNS_DIR

        # LOAD spacy pipeline
        nlp = spacy.load(dir_model_pipeline)

        # add CI_TYPE patterns to spacy nlp model pipeline
        config = {"spans_key": None, "annotate_ents": True, "overwrite": False}
        ## see for more info: https://spacy.io/usage/rule-based-matching#entityruler
        ## NOTE EntityRuler is hidden inside .add_pipe()
        try:
            ruler = nlp.add_pipe("span_ruler", config=config)
            ruler.from_disk(dir_ci_patterns)
        except ValueError:
            print("SpanRuler already exists in pipeline.")
            ruler = nlp.get_pipe("span_ruler")
            ruler.from_disk(dir_ci_patterns)

        return nlp


_nlp_model = None


def get_nlp_model():
    """NER model, loaded once on first use"""
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = NERModel().nlp_model
    return _nlp_model


@tool
def get_infrastructure_entities(text:str) -> List[str]| None:
    """This tool detects infrastructure assets and services from text by using a spaCy model to conduct Named Entity Recognition (NER)""" 

    # remove common misleading CI keywords 
    # ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
    # but keep when rest of entity is CI, eg, "damaged rails", 
    regex = re.compile(r"\b[Dd]amage\w*|\b[Tt]raining\w*|\b[Cc]ommunication\w*")
    text_clean = re.sub(regex, "", text).replace("  ", " ")
    

    # get ci entities
    nlp_chunk = get_nlp_model()(text_clean)
    ci_entities = [ent for ent in nlp_chunk.ents if ent.label_ in ["CI_TYPE", "FAC"]]
    
    ## cleaning:
    # remove common errorneous CI keywords ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
    regex = re.compile(r".*damage.*|.*training.*|\b[Cc]ommunication")
    ci_entities = [str(i) for i in ci_entities if not regex.match(str(i))]
    
    if ci_entities:
        return ci_entities
        
    return None
    # return f"Following entities about critical infrastructure were found in the text: {ci_entities}."


# @tool
def get_infrastructure_location_pairs(text:str) -> List[str] | None:
    """This tool detects infrastructure assets and services from text by using a spaCy model to conduct Named Entity Recognition (NER)""" 
    
    
    # remove common misleading CI keywords 
    # ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
    # but keep when rest of entity is CI, eg, "damaged rails", 
    regex = re.compile(r"\b[Dd]amage\w*|\b[Tt]raining\w*|\b[Cc]ommunication\w*")
    text_clean = re.sub(regex, "", text).replace("  ", " ")
    
    nlp_chunk = get_nlp_model()(text_clean)
    
 
    # GET CI and related LOCs entities
    all_ents = [ent for ent in nlp_chunk.ents]

    ## cleaning:
    # remove common errorneous CI keywords ("dam" - "damage.*", "train"- "training") as NER patterns not recognize stop-char "\b" for any reason
    regex = re.compile(r".*damage.*|.*training.*")
    all_ents = [i for i in all_ents if not (regex.match(str(i)) and i.label_ in ["CI_TYPE", "FAC"])]
    
    # remove all FAC which are not roads, airport or port names
    print("FAC before removal", [i for i in all_ents if i.label_ in ["FAC"]])
    regex_roadnames = re.compile(r".*[A-Z]{1,3}[0-9]{1,3}.*|.*[A-Z]{1,3}-[0-9]{1,3}.*")  # matches: A-75, BE105
    regex_others = re.compile(r".*airport.*|.*port.*|")
    all_ents = [i for i in all_ents if ((regex_roadnames.match(str(i)) or regex_others.match(str(i))) and i.label_ in ["FAC"])]
    
    print("FAC after removal", [i for i in all_ents if i.label_ in ["FAC"]])

    # init dfs for interim results for each chunk
    # TODO make as pydantic class with fixed attributes
    df_ci_geo_chunk = pd.DataFrame(
        columns=[
            "chunk_text",
            "ci_entity",
            "ci_entity_label",
            "geo_entity",
            "geo_2_entity",
            "geo_entity_label",
            "token_distance",
            "token_2_distance",
        ]
    )    

    # iterate over all entities within text
    for ent_idx in range(len(all_ents)):
        # when entity is CI_TYPE
        if all_ents[ent_idx].label_ in ["CI_TYPE", "FAC"]: 
            ci_idx = ent_idx

            ## .. calculate distances between CI_TYPE entity and all GEO entities based on index position
            distance_list = []
            idx_in_chunk = []
            try:
                for ent_idx in range(len(all_ents)):

                    # TODO calc distances between CI_TYPE ~ GEO entities based on word numbers and not entities (ie tokens)
                    if all_ents[ent_idx].label_ in ["GPE", "LOC"]:

                        ## check that GPE,LOC are not countries (too coarse info CI-GEO pair)
                        # of GPE/LOC is country -> proceed with next GPE/LOC 
                        # if all_ents[ent_idx].text in ci_geo_countries:
                        #     continue

                        geo_idx = ent_idx
                        dist_ent_pair = np.abs(ci_idx - geo_idx)
                        distance_list.append(dist_ent_pair)
                        idx_in_chunk.append((ent_idx))

                # for each calc of the distance of a ci-loc pair, sort distance list always in ascending order 
                closest_pair_idx = np.argsort(distance_list)[0]#np.argmin(distance_list)  # idx of closest GEO entity
                
                # handle case when only 1 location exists in text
                closest_2nd_pair_idx = None
                if len(distance_list) > 1: 
                    closest_2nd_pair_idx = np.argsort(distance_list)[1]                    
                
                distance_closest_pair = distance_list[closest_pair_idx]
                # print("If token dist > 5 then probably no link. See how often ths is th case")

                # # consideer only close CI-LOC pairs (based on number of all_ents)
                # # NOTE: distance is based on number of entities (all_ents) (( - which are in turn tokens but not comprise all existing in a chunk!))
                threshold = 4
                if distance_closest_pair > threshold:
                    # print(
                    #     f""" Token distance is too large between CI_TYPE/FAR and next GEO entity which is of {distance_closest_pair} [token distance] > {threshold} [max. token distance] """
                    # )
                    continue

                ## write as dict entry incl chunk_id, ci_entity, geo_entity, distance
                result_dict = {
                    "chunk_text": text,
                    "ci_entity": all_ents[ci_idx].text,
                    "ci_entity_label": all_ents[ci_idx].label_,
                    "geo_entity": all_ents[idx_in_chunk[closest_pair_idx]].text,
                    "geo_2_entity": all_ents[idx_in_chunk[closest_2nd_pair_idx]].text if (len(distance_list)>1 and distance_list[closest_2nd_pair_idx]<threshold) else None,
                    "geo_entity_label": all_ents[idx_in_chunk[closest_pair_idx]].label_,
                    "token_distance": distance_closest_pair,
                    "token_2_distance": distance_list[closest_2nd_pair_idx], # np.sort(distance_list)],
                }                        
                df_ci_geo_chunk = pd.concat(
                    [  df_ci_geo_chunk, pd.DataFrame([result_dict])], ignore_index=True
                )
            except (IndexError, NameError) as e:
                # print(e)
                continue
    
            # ## post-process of DF CI-GEO pairs for each chunk
            # df_ci_geo_chunk = df_ci_geo_chunk.drop_duplicates(
            #     subset=["ci_entity", "geo_entity","chunk_text"])
            # ## return only closest pairs
    
    return df_ci_geo_chunk[["ci_entity","geo_entity","geo_2_entity"]]# ,"token_distance", "token_2_distance"]]
