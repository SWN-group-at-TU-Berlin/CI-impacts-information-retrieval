import gc

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.settings import settings as s



def gen_dict_extract(var, key):
    """Used for Ci-GEO pairs detection: Extract country names from geonamescache"""
    # See, https://stackoverflow.com/questions/59444065/differentiate-between-countries-and-cities-in-spacy-ner
    if isinstance(var, dict):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, (dict, list)):
                yield from gen_dict_extract(v, key)
    elif isinstance(var, list):
        for d in var:
            yield from gen_dict_extract(d, key)


class EmbeddingModel:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                
    def vector_calculation(self, token_pred:str, token_valid:str) -> np.empty(shape=(2,)):

        ## Comparison different methods to calc cosine similarity
        # NOTE when using en_core_web_trf --> better use sentence_transformer which also accounts for context within multi-word LLM responses
        # as proposed by spaCy developers: https://github.com/explosion/spaCy/discussions/10361
        # while en_core_web_lg can only be used for word-level similarity,

        ## Example
        ## see the differences in the cos. similarity scores depending on how the vectors were calculated:
        # cos. similarity:  ["aviation","air traffic"] # _trf: tensor([[0.6789]])  (<-> compared to 0.4534 with spacy en_core_web_lg)
        # cos. similarity: ["transportation","transport infrastructure"]  # _trf: tensor([[0.7618]]) (<-> compared to 0.7564 with spacy en_core_web_lg)
        
        # clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache() 
        torch.no_grad()
        
        # get contextual vectors
        sentences = [token_pred, token_valid]
        embedded_list = self.model.encode(sentences, device=self.model.device, batch_size=5)

        return embedded_list
    

    def cosine_similarity(self, vec_a:np.array, vec_b:np.array) -> float:
        """Calculate cosine similarity for contextual vectors"""
        similarity = cos_sim(vec_a, vec_b)
        return similarity.item()



def calc_recall(tps_no: int, fns_no: int):
    return tps_no / (tps_no + fns_no) 



def calc_precision(tps_no: int, fps_no: int):
    return tps_no / (tps_no + fps_no) 


def calc_f1(recall: int, precision: int):
    return 2 * (precision * recall) / (precision + recall)


def supports_flash_attention(device_id):
    """
    Check if GPU supports FlashAttention
    See, GPU checks for flash.attn, https://github.com/Dao-AILab/flash-attention/blob/197f2083a2f0953af9319cf4ce32d0bf2aae4bd8/csrc/flash_attn/flash_api.cpp#L303:
    """
    major, minor = torch.cuda.get_device_capability(device_id)
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90




def calc_recall(tps_no: int, fns_no: int):
    return tps_no / (tps_no + fns_no) 



def calc_precision(tps_no: int, fps_no: int):
    return tps_no / (tps_no + fps_no) 


def calc_f1(recall: int, precision: int):
    return 2 * (precision * recall) / (precision + recall)