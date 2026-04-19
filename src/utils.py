import gc


import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.settings import settings as s


def vector_calculation(token_pred:str, token_valid:str) -> np.empty(shape=(2,)):

    ## Comparison different methods to calc cosine similarity
    # NOTE when using en_core_web_trf --> better use sentence_transformer which also accounts for context within multi-word LLM responses
    # as proposed by spaCy developers: https://github.com/explosion/spaCy/discussions/10361
    # while en_core_web_lg can only be used for word-level similarity,

    ## Example
    ## see the differences in the cos. similarity scores depending on how the vectors were calculated:
    # cos. similarity:  ["aviation","air traffic"] # _trf: tensor([[0.6789]])  (<-> compared to 0.4534 with spacy en_core_web_lg)
    # cos. similarity: ["transportation","transport infrastructure"]  # _trf: tensor([[0.7618]]) (<-> compared to 0.7564 with spacy en_core_web_lg)
    
    # infer device and clean up GPU memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gc.collect()
    torch.cuda.empty_cache() 
    torch.no_grad()
    
    # get contextual vectors
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    sentences = [token_pred, token_valid]
    embedded_list = model.encode(sentences, device=device, batch_size=5)

    return embedded_list


def cosine_similarity(vec_a:np.array, vec_b:np.array) -> float:
    """Calculate cosine similarity for contextual vectors"""
    similarity = cos_sim(vec_a, vec_b)
    
    return similarity.item()




def calc_recall(tps_no: int, fns_no: int):
    return tps_no / (tps_no + fns_no) 



def calc_precision(tps_no: int, fps_no: int):
    return tps_no / (tps_no + fps_no) 


def calc_f1(recall: int, precision: int):
    return 2 * (precision * recall) / (precision + recall)