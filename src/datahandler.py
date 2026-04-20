
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from src.settings import settings as s

class DataHandler:

    def __init__(self):
        self.PATH_EVAL_RESULT = s.PATH_EVAL_RESULT
        self.SIMILARITY_LLM_FILENAME = s.SIMILARITY_LLM_FILENAME


    def save_evaluation_results(self, entity_pred, df_smltry_entityclass_all, df_smltry_selmax, recall_score, precision_score, f1_score):


        SIMILARITY_FILENAME = f"{entity_pred.replace('_pred', '')}_{self.SIMILARITY_LLM_FILENAME}"
        SIMILARITY_FILEPATH = Path(self.PATH_EVAL_RESULT / SIMILARITY_FILENAME)
        
        print("Saving evaluation statistics, distribution plots, and scores to ", SIMILARITY_FILEPATH.stem, "[.parquet, _stats.json]")
        
        with open(SIMILARITY_FILEPATH, "w") as f:
        
            # results similarity df (with highest similarity per each valid_identifier) [csv, parquet]
            df_smltry_selmax.to_csv(SIMILARITY_FILEPATH.with_suffix('.csv'), index=False)
            df_smltry_selmax_pyarrow = pa.Table.from_pandas(df_smltry_selmax.astype( dtype="string[pyarrow]"))
            pq.write_table(df_smltry_selmax_pyarrow, SIMILARITY_FILEPATH.with_suffix(".parquet"))
        
            # results similarity all (all cases where valid_records was used multiple times to calc similairty to preds
            df_smltry_entityclass_all.to_csv(SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_allmatches.csv", index=False) 
        
            # summary statistics
            df_smltry_selmax_stats = pd.DataFrame(
                [{"recall": np.round(recall_score, 2), "precision": np.round(precision_score, 2), "f1_score": np.round(f1_score, 2)}]
            )
            f = SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_stats.json"  
            df_smltry_selmax_stats.to_json(f, indent=4)
            
            # distribution plots
            df_smltry_selmax["impact_sim_identical"].hist(bins=100)
            plt.savefig(SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_hist.png")
            plt.close()

            df_smltry_entityclass_all["impact_sim_identical"].hist(bins=100)
            plt.savefig(SIMILARITY_FILEPATH.parent / f"{SIMILARITY_FILEPATH.stem}_allmatches_hist.png")
            plt.close()  