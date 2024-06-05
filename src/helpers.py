import numpy as np
from collections import defaultdict
STOP_CHARS_SURP = []
import pdb

def create_list_defaultdict():
    return defaultdict(list)

def multiply_subword_metrics(offset, subtoken_probabilities, sent, words):
    word_probabilities = []
    # j is the index for the offset list (subword tokens)
    j = 0
    for i in range(0, len(words)):  # i index for reference word list
        # case 1: tokenized word = white-space separated word
        if words[i] == sent[offset[j][0]: offset[j][1]].strip().lstrip():
            word_probabilities += [subtoken_probabilities[j]]  # add probability of word to list
            j += 1
        # case 2: tokenizer split subword tokens: merge subwords and add up probabilities until the same
        else:
            concat_token = sent[offset[j][0]: offset[j][1]].strip().lstrip()
            concat_prob = subtoken_probabilities[j]
            while concat_token.strip() != words[i]:
                j += 1
                concat_token += sent[offset[j][0]: offset[j][1]].strip()
                if (
                        sent[offset[j][0]: offset[j][1]].strip().lstrip()
                        not in STOP_CHARS_SURP
                ):
                    concat_prob *= subtoken_probabilities[j]  # multiply
            word_probabilities += [concat_prob]
            j += 1
    assert len(word_probabilities) == len(words), f"Length of probabilities ({len(word_probabilities)}) does not match length of words ({len(words)}) for sentence {sent.split()}. Offsets: {offset}"
    return word_probabilities

def add_subword_metrics(offset, subtoken_probabilities, sent, words):
    word_probabilities = []
    # j is the index for the offset list (subword tokens)
    j = 0
    for i in range(0, len(words)):  # i index for reference word list
        # case 1: tokenized word = white-space separated word
        if words[i] == sent[offset[j][0]: offset[j][1]].strip().lstrip():
            word_probabilities += [subtoken_probabilities[j]]  # add probability of word to list
            j += 1
        # case 2: tokenizer split subword tokens: merge subwords and add up probabilities until the same
        else:
            concat_token = sent[offset[j][0]: offset[j][1]].strip().lstrip()
            concat_prob = subtoken_probabilities[j]
            while concat_token.strip() != words[i]:
                j += 1
                concat_token += sent[offset[j][0]: offset[j][1]].strip()
                if (
                        sent[offset[j][0]: offset[j][1]].strip().lstrip()
                        not in STOP_CHARS_SURP
                ):
                    concat_prob += subtoken_probabilities[j]  # add
            word_probabilities += [concat_prob]
            j += 1
    assert len(word_probabilities) == len(words), f"Length of probabilities ({len(word_probabilities)}) does not match length of words ({len(words)}) for sentence {sent.split()}. Offsets: {offset}"
    return word_probabilities

def get_average_subword_metric(offset, subtoken_probabilities, sent, words):
    word_probabilities = []
    # j is the index for the offset list (subword tokens)
    j = 0
    for i in range(0, len(words)):  # i index for reference word list
        # case 1: tokenized word = white-space separated word
        if words[i] == sent[offset[j][0]: offset[j][1]].strip().lstrip():
            word_probabilities += [subtoken_probabilities[j]]  # add probability of word to list
            j += 1
        # case 2: tokenizer split subword tokens: merge subwords and add up probabilities until the same
        else:
            concat_token = sent[offset[j][0]: offset[j][1]].strip().lstrip()
            concat_prob = np.array([subtoken_probabilities[j]])
            while concat_token.strip() != words[i]:
                j += 1
                concat_token += sent[offset[j][0]: offset[j][1]].strip()
                if (
                        sent[offset[j][0]: offset[j][1]].strip().lstrip()
                        not in STOP_CHARS_SURP
                ):
                    concat_prob = np.append(concat_prob, subtoken_probabilities[j])
            average_prob = np.mean(concat_prob)
            word_probabilities += [average_prob]
            j += 1
    assert len(word_probabilities) == len(words), f"Length of probabilities ({len(word_probabilities)}) does not match length of words ({len(words)}) for sentence {sent.split()}. Offsets: {offset}"
    return word_probabilities

def get_max_subword_metric(offset, subtoken_probabilities, sent, words):
    word_probabilities = []
    # j is the index for the offset list (subword tokens)
    j = 0
    for i in range(0, len(words)):  # i index for reference word list
        # case 1: tokenized word = white-space separated word
        if words[i] == sent[offset[j][0]: offset[j][1]].strip().lstrip():
            word_probabilities += [subtoken_probabilities[j]]  # add probability of word to list
            j += 1
        # case 2: tokenizer split subword tokens: merge subwords and add up probabilities until the same
        else:
            concat_token = sent[offset[j][0]: offset[j][1]].strip().lstrip()
            max_prob = subtoken_probabilities[j]
            while concat_token.strip() != words[i]:
                j += 1
                concat_token += sent[offset[j][0]: offset[j][1]].strip()
                if (
                        sent[offset[j][0]: offset[j][1]].strip().lstrip()
                        not in STOP_CHARS_SURP
                ):
                    if subtoken_probabilities[j] > max_prob:
                        max_prob = subtoken_probabilities[j]
            word_probabilities += [max_prob]
            j += 1
    assert len(word_probabilities) == len(words), f"Length of probabilities ({len(word_probabilities)}) does not match length of words ({len(words)}) for sentence {sent.split()}. Offsets: {offset}"
    return word_probabilities

def get_first_subword_metric(offset, subtoken_probabilities, sent, words):
    word_probabilities = []
    # j is the index for the offset list (subword tokens)
    j = 0
    for i in range(0, len(words)):  # i index for reference word list
        # case 1: tokenized word = white-space separated word
        if words[i] == sent[offset[j][0]: offset[j][1]].strip().lstrip():
            word_probabilities += [subtoken_probabilities[j]]  # add probability of word to list
            j += 1
        # case 2: tokenizer split subword tokens: merge subwords and add up probabilities until the same
        else:
            concat_token = sent[offset[j][0]: offset[j][1]].strip().lstrip()
            first_prob = subtoken_probabilities[j]
            while concat_token.strip() != words[i]:
                j += 1
                concat_token += sent[offset[j][0]: offset[j][1]].strip()
            word_probabilities += [first_prob]
            j += 1
    assert len(word_probabilities) == len(words), f"Length of probabilities ({len(word_probabilities)}) does not match length of words ({len(words)}) for sentence {sent.split()}. Offsets: {offset}"
    return word_probabilities

def split_into_groups(score_data):
    # repeat for each column except subjId
    score_names = [score_name for score_name in score_data.columns if score_name != "subj_id"]
    for score in score_names:
        # compute median
        median = score_data[score].median()
        # compute tertiles
        tertiles = score_data[score].quantile([0.33, 0.66])
        # assign group based on median and tertiles
        score_data[score + "_median_group"] = [np.nan if np.isnan(score) else "low" if score < median else "high" for score in score_data[score]]
        score_data[score + "_tertile_group"] = [np.nan if np.isnan(score) else "low" if score < tertiles[0.33] else "medium" if tertiles[0.33] <= score < tertiles[0.66] else "high" for score in score_data[score]]
    return score_data