from scipy.stats import rankdata
from src.utils_contributions import *
import torch.nn.functional as F
from src.contributions import ModelWrapper
import pandas as pd
import seaborn as sns
import json
import random
random.seed(10)

import argparse
import math

import os

from collections import defaultdict

import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    model_name = args.model
    dataset_name = args.dataset
    print(model_name)
    print(dataset_name)
    model, tokenizer, dataset_partition = load_model_data(model_name,dataset_name)

    num_samples = args.samples

    attributions_file = f'./data/{model_name}_{dataset_name}_attributions.npy'
    model_attributions = np.load(attributions_file,allow_pickle=True)[()]

    # if args.dataset == 'sva':
    #     methods_list = ['ours', 'ours2', 'norm', 'norm2', 'blankout'] # ig_abs, grad_input_mean, grad_input_l2
    # else:
    methods_list = ['ours', 'rollout', 'norm', 'norm2', 'grad', 'ig_l2', 'ig_abs_mean', 'grad_input_l2', 'grad_input_mean_abs_mean', 'ours_l2']#, 'ours_l2'] # ig_abs, grad_input_mean, grad_input_l2

    # random.seed(10)
    # random_samples_list = random.sample(range(len(dataset_partition)), num_samples) 

    pert_curve_results = defaultdict(list,{ k:[] for k in methods_list})

    model_wrapped = ModelWrapper(model)

    i = 0
    #for sentence_i in random_samples_list:
    for sentence_i in range(num_samples):
        if sentence_i%50==0:
            print(sentence_i)
        if 'bin' in model_name:
            pt_batch, tokenized_text, original_target_idx = prepare_sva_sentence(sentence_i, dataset_partition, tokenizer)
        else:        
            sentence = dataset_partition[sentence_i]
            
            text = sentence[list(dataset_partition.features.keys())[0]]
            pt_batch = tokenizer(text, return_tensors="pt", return_token_type_ids= False).to(device)
            tokenized_text = tokenizer.convert_ids_to_tokens(pt_batch["input_ids"][0])

        len_tok_text = len(tokenized_text)
        if len(tokenized_text) > args.max_len:
            print('SKIP')
            continue

        # Forward-pass to get class prediction
        prediction_scores = model_wrapped.get_prediction(pt_batch['input_ids'])
        # If sva task, get scores of masked position classifier
        if 'bin' in model_name:
            prediction_scores = torch.squeeze(prediction_scores)
            prob_pred_class = torch.sigmoid(prediction_scores[original_target_idx])
            if prob_pred_class >= 0.5:
                pred_class = 1
            else:
                pred_class = 0
                prob_pred_class = 1 - prob_pred_class
        # If SC task, get scores of CLS classifier
        else:
            probs = torch.nn.functional.softmax(prediction_scores, dim=-1)
            pred_class = torch.argmax(probs)
            prob_pred_class = probs[:,pred_class]

        for method in methods_list:
            min_prob = 1
            max_prob = 0
            probs_list = []
            
            # Load attributions for sample from [CLS] (sentiment classification) or [MASK] (sva)
            attributions_cls = model_attributions[method][i]

            # Make sure the sentence we are dealing with matches the
            # one from the extracted attributions_cls
            assert len_tok_text == attributions_cls.shape[0]
            # Rank the attributions from [CLS] or [MASK]
            rank_attributions_cls = list(rankdata(attributions_cls, method='dense', axis = -1))

            # List of the index of the original attributions sorted from top to bottom
            # [3, 2, 1, 4] means top token (most influencial) is at position 3, the least
            # is at position 4
            order_index_rank = sorted(range(len(rank_attributions_cls)), key=lambda k: rank_attributions_cls[k], reverse=True)
            # Remove special tokens [CLS] and [SEP]
            # We manually add them afterwards to each perturbed sentence
            order_index_rank.remove(0)
            order_index_rank.remove(len_tok_text-1)

            if 'bin' in model_name:
                if args.fidelity_type == 'comp':
                    # Move [MASK] token to the end of the list (no deletion)
                    order_index_rank.remove(original_target_idx)
                    #order_index_rank.append(original_target_idx)
                elif args.fidelity_type == 'suff':
                    # Move [MASK] token to the beggining of the list (no deletion)
                    order_index_rank.remove(original_target_idx)
                    #order_index_rank.insert(0,original_target_idx)

            if args.bins == False:
                # One-by-one token deletion
                range_tok_delete = range(0,len(order_index_rank)+1)
            else:
                # Deletion following bins of top-k% tokens
                range_tok_delete = [0]
                for bin in [0.01,0.05,0.1,0.2,0.5]: # following DeYoung et al., 2019
                    range_tok_delete.append(math.ceil(len(order_index_rank)*bin))
            for j in range_tok_delete:
                # Break when top max_skip subwords have been deleted
                # Not in bins analysis
                if args.bins == False:
                    if j >= args.max_skip:
                        break
                if 'bin' in model_name:
                    # Only [MASK] left
                    if j == len(order_index_rank):
                        break
                if args.fidelity_type == 'comp':
                    # We keep least important tokens
                    kept_tokens = order_index_rank[j:]
                else:
                    # We keep most important tokens
                    kept_tokens = order_index_rank[:j]
                if 'bin' in model_name:
                    # Add [MASK] token
                    kept_tokens.append(original_target_idx)
                new_tokenized_text = []
                for idx, _ in enumerate(tokenized_text):
                    if idx in kept_tokens:
                        new_tokenized_text.append(tokenized_text[idx])
                input_ids = tokenizer.convert_tokens_to_ids(new_tokenized_text)
                input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
                input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64,device=device)

                ## Prediction
                prediction_scores = model_wrapped.get_prediction(input_ids_tensor.unsqueeze(0))
                if 'bin' in model_name:
                    # Recompute masked verb position
                    # +1 for including [CLS]
                    target_idx = new_tokenized_text.index(tokenizer.mask_token)+1 
                    prediction_scores = torch.squeeze(prediction_scores)
                    prob_pred_class = torch.sigmoid(prediction_scores[target_idx]).squeeze()
                    if pred_class == 0:
                        prob_pred_class = 1 - prob_pred_class
                else:
                    probs = torch.nn.functional.softmax(prediction_scores, dim=-1)
                    prob_pred_class = probs[:,pred_class]

                if args.fidelity_type == 'comp':
                    if prob_pred_class <= min_prob:
                        appended_prob = prob_pred_class
                        min_prob = appended_prob
                else:
                    if prob_pred_class > max_prob:
                        appended_prob = prob_pred_class
                        max_prob = appended_prob
                probs_list.append(appended_prob.item())
            pert_curve_results[method].append(probs_list)
                
        i += 1
    if not os.path.exists('./data/AUPC/'):
        os.makedirs('./data/AUPC/')
    if args.test== False:
        if args.bins == False:
            outfile = f'./data/AUPC/{model_name}_{dataset_name}_{args.fidelity_type}.json'
        else:
            outfile = f'./data/AUPC/{model_name}_{dataset_name}_{args.fidelity_type}_bins.json'
    else:
        if args.bins == False:
            outfile = f'./data/AUPC/{model_name}_{dataset_name}_{args.fidelity_type}_test.json'
        else:
            outfile = f'./data/AUPC/{model_name}_{dataset_name}_{args.fidelity_type}_bins_test.json'
    with open(outfile, 'w') as f:
            json.dump(pert_curve_results, f)
    print('FINISH')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="model used", type= str)
    parser.add_argument('--dataset', help="sst2/sva/yelp/imdb", type=str)
    parser.add_argument('--samples', help="number of samples for computing attributions", type=int)
    parser.add_argument('--fidelity-type', help="comprehesiveness (comp) or sufficiency (suff)", type=str)
    parser.add_argument('--bins', help="use bins (1%,5%,10%,20%,50%) or one-by-one token deletion", action='store_true')
    parser.add_argument('--max-len', default=200, help="maximum number of tokens", type=int)
    parser.add_argument('--max-skip', default=30, help="maximum number of removed tokens", type=int)
    parser.add_argument('--test', help="save in other file", action='store_true')
    args=parser.parse_args()

    main(args)