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
        df[[col_type, col_grouped]] = df[[col_type, col_grouped]].astype(str)
        mask = is_ci_entity(df[col_type], pattern)
        # mask = df[col_type].str.contains(pattern, regex=True, na=False)
        df.loc[mask, col_grouped] = subgroup
    
    return df


def is_ci_entity(ci_entity: pd.Series, regex_pattern: str) -> pd.Series:
    """returns boolean mask where records in pd.Series are a certain CI type based on regex pattern"""
    
    return ci_entity.str.contains(regex_pattern, regex=True, na=False)




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
    final_resp = ("[" + resp.split("[")[1]) 
    final_resp = (final_resp.split("]", 1)[0] + "]") 

    # check for empty response due to removal of text outside of brackets
    if final_resp == "[]" or final_resp == "[{}]":
        # remove potential double [[ and ]]
        final_resp = resp.replace("[[{", "[{").replace("}]]", "}]")
        # remove potential text outside - no matter if it exists or not
        final_resp = ("[{" + final_resp.split("[{")[1]) 
        final_resp = (final_resp.split("}]", 1)[0] + "}]") 

    df_final_resp = pd.read_json(StringIO(final_resp))

    return df_final_resp
        