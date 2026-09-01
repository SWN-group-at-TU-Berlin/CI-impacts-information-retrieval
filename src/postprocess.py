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
    
        # remove words considered wrongly as Ci keywords, 
        # "the damaged rails"-> "the rails"
        regex = re.compile(r"\b[Dd]amage\w*|\b[Tt]raining\w*")
        df[col_type].str.replace(regex, "", regex=True).str.replace("  ", " ")
            
        # assign subgroups to CI records, na=False to remove all records which not match patterns
        df[[col_type, col_grouped]] = df[[col_type, col_grouped]].astype(str)
        mask = is_ci_entity(df[col_type], pattern)
        # mask = df[col_type].str.contains(pattern, regex=True, na=False)
        df.loc[mask, col_grouped] = subgroup
    
    return df


def is_ci_entity(ci_entity: pd.Series, regex_pattern: str) -> pd.Series:
    """returns boolean mask where records in pd.Series are a certain CI type based on regex pattern"""
    
    # return ci_entity.str.contains(regex_pattern, regex=True, na=False)
    return ci_entity.str.match(regex_pattern)


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

    try:
        df_final_resp = pd.read_json(StringIO(final_resp))

    except Exception as e:
    # special handling for some GPT oss responses where value is a list item
        final_resp = ("[" + resp.split("[")[2]) 
        final_resp = (final_resp.split("]", 1)[0] + "]") 
        df_final_resp = pd.read_json(StringIO(final_resp))

   
    return df_final_resp
        
def postprocess_locations(df:pd.Series, location_col:str) -> pd.Series:
    """remove surrounding text for locations"""
    # TODO condense function

    cleanup_patterns = [r"\(", ", "]
    for i in cleanup_patterns:
        # "( "  eg "Sinzig (in North Rhine-Westphalia)""
        # rm text after comma, e.g. "Ahr valley, Germany" --> "Ahr valley"
        df[location_col] = df[location_col].str.split(i, regex=True).str[0].str.strip()

    cleanup_patterns = [r"[\(\),]", "\bthe ",  "\bin ", "\bpassing ", "city of"]
    for i in cleanup_patterns:
        # remove all remaining brackets and commas
        # remove "the" when it is not contained in other word
        # remove "in"  when it is not contained in other word
            # set this after cleaning up "( " and ", " as it otherwise would take the later location
        df[location_col] = df[location_col].replace(rf"{i}", " ", regex=True).str.strip()   

    cleanup_patterns = ["between",]
    for i in cleanup_patterns:
        # handling loc with "railway tracks between "
        df[location_col] = df[location_col].str.split(i).str[-1].str.strip()

    ## remove "near ", "close to", "passing " from location names, and final double whitespace
    df[location_col] =  df[location_col].str.replace(r"^(\bnear |close to |parts of |direction of |passing |along )", " ", regex=True).str.strip()
    df[location_col] =  df[location_col].str.replace("  ", " ").str.strip()

    return df[location_col]

    

def split_text_into_multiple_rows(df: pd.DataFrame, column: str, split_at = " and ") -> pd.DataFrame:
    """ split text at splitting_point into multiple rows """
    # split CIs and LOCs with "and" into multiple rows
    df[column] = df[column].str.split(split_at)   
    # NOTE: Removes info from CI - drops info if CI is singular o plural (e.g, road and railway infrastrcutre --> "road", "railway infrastructure")
    df = df.explode(column=column)
    df = df.drop_duplicates().reset_index(drop=True)
    return df