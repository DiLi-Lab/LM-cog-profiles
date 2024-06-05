from collections import defaultdict
import glob
import pickle
from tqdm import tqdm
import os
import pdb
import torch
import gc
from typing import Dict, Tuple
import numpy as np
import argparse

from ModelExtractor import get_model
import helpers

parser = argparse.ArgumentParser(description="Run population biases experiment")
parser.add_argument("--lm", dest="lm")
args = parser.parse_args()

DATA_PATH = "./data/indico"
#create DATA_PATH / metrics if it does not exist
if not os.path.exists(f"{DATA_PATH}/metrics"):
    os.makedirs(f"{DATA_PATH}/metrics")


def create_list_defaultdict():
    return defaultdict(list)


class Dataset:
    """Create a Dataset class that loads the different eye-tracking datasets.
    For each dataset, there is a different class that inherits from this one.
    """
    def __init__(self, data_path, **kwargs): 
        """Initializes the dataset object with the path to the data."""
        self.data_path = data_path
        self.lm_probabilities = defaultdict(create_list_defaultdict)
        self.scorer = kwargs.get("Scorer", None)
        self.add_BOS = True if self.scorer.BOS is False else False

    def load_data(self): 
        return NotImplementedError("Subclasses must implement this method")
    
    def load_stimulus_texts(self): 
        return NotImplementedError("Subclasses must implement this method")
    
    def get_lm_probabilities(self):
        return NotImplementedError("Subclasses must implement this method")


class IndicoDataset(Dataset):
    """Class for the Indico dataset that inherits from the Dataset class."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stimulus_text_path = f"{self.data_path}/stimuli/daf_sentences_screens/text*DAF*"

    def load_stimulus_texts(self):
        return sorted(glob.glob(self.stimulus_text_path))

    def get_lm_probabilities(self):
        """Returns the language model probabilities."""
        # for text_id, screen_id, sent_file in tqdm(self.load_stimulus_texts(), desc="Processing screens"):
        for text_file in tqdm(self.load_stimulus_texts(), desc="Processing screens"):
            text_id = int(text_file[-12:-10])
            screen_id = int(text_file[-9])
            with open(text_file) as f:
                sents = f.readlines()
                sents = [sent.strip() for sent in sents]
                model_name = self.scorer.name
                for sent in sents:
                    sent = sent.strip().lstrip()
                    words = sent.split(" ")
                    probs, entropies, offset, attentions_list_layers = self.scorer.score(sentence=sent)
                    # averaging of probabilities
                    multiplied_probs = helpers.multiply_subword_metrics(offset, probs, sent, words)
                    # compute negative log of each entry in multiplied_probs
                    accumulated_surprisal = -np.log(multiplied_probs)
                    first_probs = helpers.get_first_subword_metric(offset, probs, sent, words)
                    first_surprisal = -np.log(first_probs)
                    added_entropies = helpers.add_subword_metrics(offset, entropies, sent, words)
                    first_entropies = helpers.get_first_subword_metric(offset, entropies, sent, words)
                    multiplied_atts_layers = [
                        helpers.multiply_subword_metrics(
                            offset, layer, sent, words
                        ) for layer in attentions_list_layers
                    ]
                    average_atts_layers = [
                        helpers.get_average_subword_metric(
                            offset, layer, sent, words
                        ) for layer in attentions_list_layers
                    ]
                    max_atts_layers = [
                        helpers.get_max_subword_metric(
                            offset, layer, sent, words
                        ) for layer in attentions_list_layers
                    ]
                    self.lm_probabilities[model_name][(text_id, screen_id)].extend(
                        list(
                            zip(
                                words, accumulated_surprisal, first_surprisal, added_entropies, first_entropies,
                                *multiplied_atts_layers, *average_atts_layers, *max_atts_layers
                            )
                        )
                    )
        torch.cuda.empty_cache()
        gc.collect()
        with open(f'{self.data_path}/metrics/indico_{args.lm}.pickle', 'wb') as surprisal_file:
            pickle.dump(self.lm_probabilities, surprisal_file)

    def get_lm_probabilities_no_att(self):
        """Returns the language model probabilities."""
        # for text_id, screen_id, sent_file in tqdm(self.load_stimulus_texts(), desc="Processing screens"):
        for text_file in tqdm(self.load_stimulus_texts(), desc="Processing screens"):
            text_id = int(text_file[-12:-10])
            screen_id = int(text_file[-9])
            with open(text_file) as f:
                sents = f.readlines()
                sents = [sent.strip() for sent in sents]
                model_name = self.scorer.name
                for sent in sents:
                    sent = sent.strip().lstrip()
                    words = sent.split(" ")
                    probs, entropies, offset, attentions_list_layers = self.scorer.score(sentence=sent)
                    # averaging of probabilities
                    multiplied_probs = helpers.multiply_subword_metrics(offset, probs, sent, words)
                    # compute negative log of each entry in multiplied_probs
                    accumulated_surprisal = -np.log(multiplied_probs)
                    first_probs = helpers.get_first_subword_metric(offset, probs, sent, words)
                    first_surprisal = -np.log(first_probs)
                    added_entropies = helpers.add_subword_metrics(offset, entropies, sent, words)
                    first_entropies = helpers.get_first_subword_metric(offset, entropies, sent, words)
                    self.lm_probabilities[model_name][(text_id, screen_id)].extend(
                        list(
                            zip(
                                words, accumulated_surprisal, first_surprisal, added_entropies, first_entropies
                            )
                        )
                    )
        torch.cuda.empty_cache()
        gc.collect()
        with open(f'{self.data_path}/metrics/indico_{args.lm}.pickle', 'wb') as surprisal_file:
            pickle.dump(self.lm_probabilities, surprisal_file)


if __name__ == "__main__":
    Indico = IndicoDataset(data_path=DATA_PATH, Scorer=get_model(args.lm))
    Indico.get_lm_probabilities()