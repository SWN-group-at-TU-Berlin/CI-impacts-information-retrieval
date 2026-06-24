#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluation utilities"""

import os

import numpy as np
import pandas as pd



def true_positives(df_similarity: pd.DataFrame, df_valid, df_valid_col, df_pred, df_pred_col, similarity_threshold: float) -> int:
    tps = df_similarity.loc[df_similarity["impact_sim_identical"] >= similarity_threshold]
    tps_validmergedpred = df_valid.merge(df_pred, left_on=[df_valid_col], right_on=[df_pred_col], how="inner")
    assert len(tps) == len(tps_validmergedpred)
    
    return len(tps)


def false_negatives(df_valid, df_pred, df_valid_col, df_pred_col) -> int:
    tps = df_valid.merge(df_pred, left_on=[df_valid_col], right_on=[df_pred_col], how="inner")
    fps_len = len(df_pred[df_pred_col]) - len(tps)
    fns_len = len(df_valid[df_valid_col]) - len(tps)
    return fns_len


def false_positives(df_valid, df_pred, df_valid_col, df_pred_col) -> int:
    tps = df_valid.merge(df_pred, left_on=[df_valid_col], right_on=[df_pred_col], how="inner")
    fps_len = len(df_pred[df_pred_col]) - len(tps)
    return fps_len