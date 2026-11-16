from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.settings import init_logger
from src.settings import settings as s

logger = init_logger(__name__)


class DataHandler:
    def __init__(self):
        self.PATH_EVAL_RESULT = s.PATH_EVAL_RESULT
        self.SIMILARITY_LLM_FILENAME = s.SIMILARITY_LLM_FILENAME

    def save_evaluation_results(self, entity_pred, df_smltry_selmax, recall_score, precision_score, f1_score):
        similarity_filename = f"{entity_pred.replace('_pred', '')}_{self.SIMILARITY_LLM_FILENAME}"
        similarity_filepath = Path(self.PATH_EVAL_RESULT / similarity_filename)

        logger.info(f"Saving evaluation statistics, distribution plots, and scores to {similarity_filepath.stem} [.parquet, _stats.json]")

        with open(similarity_filepath, "w") as f:
            # results similarity df (with highest similarity per each valid_identifier) [csv, parquet]
            df_smltry_selmax.to_csv(similarity_filepath.with_suffix(".csv"), index=False)
            df_smltry_selmax_pyarrow = pa.Table.from_pandas(df_smltry_selmax.astype(dtype="string[pyarrow]"))
            pq.write_table(df_smltry_selmax_pyarrow, similarity_filepath.with_suffix(".parquet"))

            # summary statistics
            df_smltry_selmax_stats = pd.DataFrame(
                [
                    {
                        "recall": np.round(recall_score, 2),
                        "precision": np.round(precision_score, 2),
                        "f1_score": np.round(f1_score, 2),
                    }
                ]
            )
            f = similarity_filepath.parent / f"{similarity_filepath.stem}_stats.json"
            df_smltry_selmax_stats.to_json(f, indent=4)

            # distribution plots
            df_smltry_selmax["impact_sim_identical"].hist(bins=100)
            plt.savefig(similarity_filepath.parent / f"{similarity_filepath.stem}_hist.png")
            plt.close()
