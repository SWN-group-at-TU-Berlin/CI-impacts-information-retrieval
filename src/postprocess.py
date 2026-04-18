#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Postprocess LLM responses"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"


from io import StringIO
import pandas as pd

import src.document_cleaning as dc


def group_ci_types(series_ci: pd.Series, ci_patterns: pd.DataFrame) -> pd.Series:

    # load regular expressions and subgroups from NER patterns as dict
    # for general cases
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
    regexes = regexes_1 + regexes_2

    for i, r in enumerate(regexes):
        # get regex pattern for CI type (key) and its subgroup (value)
        # NOTE, nice shortcut: get key containing regex by unpacking each dict into list, then get key
        pattern = [*r][0]
        subgroup = r[pattern]
    
        # assign subgroups to CI records, na=False to remove all records which not match patterns
        mask = series_ci.str.contains(pattern, regex=True, na=False)
        series_ci.iloc[mask] = subgroup
    
        return series_ci
    

def postprocess_response(resp: str) -> pd.DataFrame:

    resp = resp.replace("\n", "")
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
        