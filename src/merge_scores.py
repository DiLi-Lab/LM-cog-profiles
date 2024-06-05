import argparse
import glob
import pickle

from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple, List

from helpers import create_list_defaultdict, split_into_groups

DATA_PATH = "./data/indico"
parser = argparse.ArgumentParser(description="Run population biases experiment")
parser.add_argument("--dataset", dest="data_set", default="ET")
args = parser.parse_args()

# column names without attention values
COL_NAMES = ["word", "accumulated_surprisal", "first_surprisal", "added_entropy", "first_entropy"]
SELECT_FEATURES = ["accumulated_surprisal", "accumulated_surprisal_prev_word", "accumulated_surprisal_prevprev_word", "added_entropy",
                   "added_entropy_prev_word", "added_entropy_prevprev_word"]

class Dataset:
    """Dataset class that loads the different eye-tracking datasets."""
    def __init__(self, data_path, data_type, **kwargs): 
        """Initializes the dataset object with the path to the data."""
        self.data_path = data_path
        self.col_names = COL_NAMES
        self.metrics_path = f"{self.data_path}/metrics/"
        self.data = self.load_data(data_type)
        self.scores = self.load_psychometric_scores()
        self.lm_probabilities = self.load_metrics()
        self.features = self.get_features_df()
        self.grouped_data = None

    def load_data(self, data_type: str): 
        return NotImplementedError("Subclasses must implement this method")
    
    def load_metrics(self):
        return  NotImplementedError("Subclasses must implement this method")
    
    def load_psychometric_scores(self):
        return NotImplementedError("Subclasses must implement this method")
    
    def get_features_df(self):
        return NotImplementedError("Subclasses must implement this method")


class IndicoDataset(Dataset):
    """Class for the Indico dataset that inherits from the Dataset class."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def load_data(self, data_type: str) -> pd.DataFrame:
        """Loads the reading data."""
        if data_type == "ET":
            self.data_type = "ET"
            return pd.read_csv(f"{self.data_path}/indico_et.csv", header="infer")
        elif data_type == "SPR":
            self.data_type = "SPR"
            return pd.read_csv(f"{self.data_path}/indico_spr.csv", header="infer")
        else:
            raise ValueError("Data type must be either 'ET' or 'SPR'")

    def load_metrics(self) -> List[Dict[str, Dict[Tuple[int, int], List[Tuple[str, float]]]]]:
        all_metrics = []
        metrics_files = sorted(glob.glob(f"{self.metrics_path}*.pickle"))
        for file in metrics_files:
            with open(file, "rb") as f:
                metrics = pickle.load(f)
                all_metrics.append(metrics)
        return all_metrics
    
    def load_psychometric_scores(self):
        scores = pd.read_csv(f"{self.data_path}/psychometric_scores.csv", header="infer")
        scores.rename(columns={"subj": "subj_id"}, inplace=True)
        score_names = [
            "SLRTWord", "SLRTPseudo", "MWTPR", 
            "RIASVixPR", "RIASNixPR", "RIASGixPR",
            "FAIRKPR", "StrRTEffect", "SimRTEffect",
            "MUmean", "OSmean", "SSmean", "SSTMRelScore",
        ]
        scores = scores[["subj_id"] + score_names]
        return scores

    def get_features_df(self) -> pd.DataFrame:
        """Create a dataframe with the features for the different language models."""
        all_df = []
        for model_dict in self.lm_probabilities:
            model_df = []
            model_name = list(model_dict.keys())[0]
            # remove dash
            model_column = model_name.replace("-", "_")
            for text_id, screen_id in tqdm(model_dict[model_name]):
                transposed_tuples = zip(*model_dict[model_name][(text_id, screen_id)])
                concatenated_entries = [tuple(entry) for entry in transposed_tuples]
                # there's 3 ways to pool the attention: multiplied, average and max therefore /3
                att_layers = int((len(concatenated_entries)-5)/3)
                col_names = self.col_names + [f"multiplied_att_l{i}" for i in range(1, att_layers+1)] + [f"average_att_l{i}" for i in range(1, att_layers+1)] + [f"max_att_l{i}" for i in range(1, att_layers+1)]
                df = pd.DataFrame(concatenated_entries).transpose()
                df.columns = col_names
                # create new column at first position with word index
                df.insert(0, "word_id", range(1, len(df) + 1))
                # use textId, screenId and wordId as index
                df["text_id"] = text_id
                df["screen_id"] = screen_id
                df.set_index(["text_id", "screen_id", "word_id"], inplace=True)
                df = df.drop(columns=[col for col in col_names if col not in SELECT_FEATURES])
                # shift surprisal column to get previous and n-2 word surprisal
                df["accumulated_surprisal_prev_word"] = df["accumulated_surprisal"].shift(1)
                df["accumulated_surprisal_prevprev_word"] = df["accumulated_surprisal"].shift(2)        
                # shift entropy column to get previous and n-2 word entropy
                df["added_entropy_prev_word"] = df["added_entropy"].shift(1)
                df["added_entropy_prevprev_word"] = df["added_entropy"].shift(2)
                # rename the columns that are among SELECT_FEATURES to "model_name + name
                renamed = {col: f"{model_column}__{col}" for col in df.columns if col in SELECT_FEATURES}
                df = df.rename(columns=renamed)
                model_df.append(df)
            all_df.append(pd.concat(model_df))
        final_df = pd.concat(all_df, axis=1, join="inner")
        return final_df
                
    def merge_with_features(self):
        # set index of self.data
        self.data.set_index(["text_id", "screen_id", "word_id"], inplace=True)
        # merge the dataframes
        self.data = pd.merge(self.data, self.features, how="left",
                             on=['text_id', 'screen_id', 'word_id'])

        self.data.reset_index(inplace=True)
        self.data = self.data[["subj_id"] + [col for col in self.data.columns if col != "subj_id"]]
    
    def group_and_merge(self):
        scores_grouped = split_into_groups(self.scores)
        self.grouped_data = pd.merge(self.data, scores_grouped.filter(regex=("group|subj_id")), on=["subj_id"])
        self.grouped_data.to_csv(f"{self.data_path}/indico_{self.data_type}_with_groups.csv", index=False, na_rep='NA')

if __name__ == "__main__":
    Indico = IndicoDataset(data_path=DATA_PATH, data_type=args.data_set)
    Indico.merge_with_features()
    Indico.group_and_merge()