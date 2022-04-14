from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils_contributions import *
import torch.nn.functional as F
from src.contributions import ModelWrapper, ClassificationModelWrapperCaptum, interpret_sentence_sst2
#import contributions
import json
import statistics
import random
random.seed(10)

import argparse

from collections import defaultdict
import json

import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
import sys


def compute_correlation(relevancies, methods_list, reference_method):
    """Spearman's rank correlation coefficient between methods in methods_list and reference_method."""

    corrs_method = defaultdict(list)
    for method in methods_list:
        results_corr = []
        for i in np.arange(len(relevancies[method])):
            sp = spearmanr(relevancies[method][i],relevancies[reference_method][i])
            results_corr.append(sp[0])

        corrs_method[method].append(np.mean(results_corr))
        corrs_method[method].append(np.std(results_corr))

    return corrs_method

def main(args):
    model_name = args.model
    dataset_name = args.dataset

    if dataset_name == 'sva':
        attributions_file = f'./data/{model_name}_{dataset_name}_attributions.npy'
        relevancies = np.load(attributions_file,allow_pickle=True)[()]

        methods_list = ['raw','rollout','norm','ours']
        reference_method = 'blankout'
        corrs_method = compute_correlation(relevancies, methods_list, reference_method)

        outfile = f'./data/{model_name}_{dataset_name}_correlations.json'
        with open(outfile, 'w') as f:
            json.dump(corrs_method, f)

    elif dataset_name == 'sst2':
        attributions_file = f'./data/{model_name}_{dataset_name}_attributions.npy'
        relevancies = np.load(attributions_file,allow_pickle=True)[()]

         # Compute mean and avg correlation between methods
        methods_list = methods_list = ['raw','rollout','norm','ours','grad']
        reference_method = 'grad'
        corrs_method = compute_correlation(relevancies, methods_list, reference_method)

        # Compute mean and avg rank of special tokens
        special_tok_attributions_file = f'./data/{model_name}_{dataset_name}_special_tok_attributions.npy'
        special_tokens_relevancies = np.load(special_tok_attributions_file,allow_pickle=True)[()]

        special_tokens_method = defaultdict(list)
        for method in special_tokens_relevancies.keys():
            special_tokens_method[method].append(statistics.mean(special_tokens_relevancies[method]))
            special_tokens_method[method].append(statistics.stdev(special_tokens_relevancies[method]))

        outfile = f'./data/{model_name}_{dataset_name}_rank_special_tokens.json'
        with open(outfile, 'w') as f:
            json.dump(special_tokens_method, f)

        outfile = f'./data/{model_name}_{dataset_name}_correlations.json'
        with open(outfile, 'w') as f:
            json.dump(corrs_method, f)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help="model used", type= str)
    parser.add_argument('-dataset', help="sst2/sva", type=str)
    args=parser.parse_args()

    main(args)