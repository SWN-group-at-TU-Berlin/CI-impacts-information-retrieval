#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Postprocess LLM responses"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"

import re
from io import StringIO
import pandas as pd

import src.document_cleaning as dc


def group_ci_types(df: pd.DataFrame, col_type, col_grouped, ci_patterns: pd.DataFrame) -> pd.DataFrame:

    # load regular expressions and subgroups from NER patterns as dict
    # # for general cases
    # try:
    #     regexes_3 = [
    #         {ci_patterns["pattern"][i][0]["TEXT"] : ci_patterns["subgroup_name"][i]}
    #         for i in range(len(ci_patterns)) 
    #             if len(ci_patterns["pattern"][i])==1 
    #     ]
    # except Exception:
    regexes_1 = [
            {ci_patterns["pattern"][i][0]["TEXT"]["REGEX"] : ci_patterns["subgroup_name"][i]}
            for i in range(len(ci_patterns)) 
                if len(ci_patterns["pattern"][i])==1 
    ]
    # for all special cases with "LOWER"-pattern
    regexes_2 = [
        {ci_patterns["pattern"][i][0]["TEXT"]["REGEX"] + " " + ci_patterns["pattern"][i][1]["LOWER"] : ci_patterns["subgroup_name"][i] }
        for i in range(len(ci_patterns))
            if len(ci_patterns["pattern"][i])==2
    ]
    # for speial cases where regex not worked due to reducnandcy to other regex Ci word,eg access roads <-> roads 

    regexes = regexes_1 + regexes_2 #+ regexes_3

    for i, r in enumerate(regexes):
        # get regex pattern for CI type (key) and its subgroup (value)
        # NOTE, nice shortcut: get key containing regex by unpacking each dict into list, then get key
        pattern = [*r][0]
        subgroup = r[pattern]
    
        # assign subgroups to CI records, na=False to remove all records which not match patterns
        mask = df[col_type].str.match(pattern)
        # mask = df[col_type].str.contains(pattern, regex=True, na=False)
        df.loc[mask, col_grouped] = subgroup
    
    return df
    

def postprocess_response(resp: str) -> pd.DataFrame:

    # Remove linebreak symbols and case numbers in the keys ("damage_1" -> "damage")
    resp = resp.replace("\n", "")
    resp = re.sub(pattern=r"_\d+", repl="", string=resp)

    # fix missing bracket at beginning and end of response by splitting at first/last complete entry
    if "[{" not in resp:
        resp = "[{" + resp.split("{", 1)[1]
    if "}]" or "},]" not in resp:
        resp = resp.rpartition('}')[-3] + "}]"

    # remove potential text outside - no matter if it exists or not
    resp = ("[" + resp.split("[")[1]) 
    resp = (resp.split("]", 1)[0] + "]") 

    df_resp = pd.read_json(StringIO(resp))
    
    return df_resp
        