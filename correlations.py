from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils_contributions import *
import torch.nn.functional as F
from src.contributions import ModelWrapper, ClassificationModelWrapperCaptum, interpret_sentence_sst2
#import contributions
import pandas as pd
import h5py
import json
import statistics
import random
random.seed(10)

import argparse

from collections import defaultdict
import codecs, json

import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
import sys

def contribution_relevancies_sva(model, model_name, tokenizer, dataset_partition, random_samples_list):
    """Contribution rollout-based relevancies and blank-out relevancies in SVA."""

    relevancies = defaultdict(list)
    layer = -1

    for i in random_samples_list:
        na,_,masked,good,bad = dataset_partition[i].strip().split("\t")
        # Add Ġ to word if using RoBERTa
        if model_name == 'roberta':
            good = '\u0120' + good
            bad = '\u0120' + bad
        text = masked.replace('***mask***',tokenizer.mask_token)
        pt_batch = tokenizer(text, return_tensors="pt").to(device)
        target_idx = (pt_batch['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        target_idx = target_idx[0].item()
        good_verb_id = tokenizer.convert_tokens_to_ids(good)
        wrong_verb_id = tokenizer.convert_tokens_to_ids(bad)
        
        model_wrapped = ModelWrapper(model)
        prediction_scores, hidden_states, attentions, contributions_data = model_wrapped(pt_batch)

        probs = torch.nn.functional.softmax(prediction_scores, dim=-1)
        actual_verb_score = probs[0][target_idx][good_verb_id]
        inflected_verb_score = probs[0][target_idx][wrong_verb_id]

        main_diff_score = actual_verb_score - inflected_verb_score

        # Raw attentions relevances    
        _attentions = [att.detach().cpu().numpy() for att in attentions]
        attentions_mat = np.asarray(_attentions)[:,0]
        raw_attn_relevances = get_raw_att_relevance(attentions_mat,token_pos=target_idx)
        # raw_attn_relevances = np.delete(raw_attn_relevances,target_idx)
        # raw_attn_relevances = raw_attn_relevances[1:-1]
        relevancies['raw'].append(np.asarray(raw_attn_relevances))

        # Rollout attentions relevances
        att_mat_sum_heads = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
        joint_attentions = compute_rollout(att_mat_sum_heads)
        rollout_relevance_relevances = joint_attentions[layer][target_idx]
        # rollout_relevance_relevances = torch.cat((rollout_relevance_relevances[:target_idx], rollout_relevance_relevances[target_idx+1:]))
        # rollout_relevance_relevances = rollout_relevance_relevances[1:-1]
        relevancies['rollout'].append(np.asarray(rollout_relevance_relevances))

        # Norms + roll out relevances
        normalized_model_norms = normalize_contributions(contributions_data['transformed_vectors_norm'],scaling='sum_one')
        norms_mix = compute_joint_attention(normalized_model_norms)
        norms_mix_relevances = norms_mix[layer][target_idx]
        # norms_mix_relevances = torch.cat((norms_mix_relevances[:target_idx], norms_mix_relevances[target_idx+1:]))
        # norms_mix_relevances = norms_mix_relevances[1:-1]
        relevancies['norm'].append(np.asarray(norms_mix_relevances))

        # Our method relevances
        resultant_norm = resultants_norm = torch.norm(torch.squeeze(contributions_data['resultants']),p=1,dim=-1)
        normalized_contributions = normalize_contributions(contributions_data['contributions'],scaling='min_sum',resultant_norm=resultant_norm)#min_sum
        contributions_mix = compute_joint_attention(normalized_contributions)
        contributions_mix_relevances = contributions_mix[layer][target_idx]
        # contributions_mix_relevances = torch.cat((contributions_mix_relevances[:target_idx], contributions_mix_relevances[target_idx+1:]))
        # contributions_mix_relevances = contributions_mix_relevances[1:-1]
        relevancies['ours'].append(np.asarray(contributions_mix_relevances))

        # Blank-out method ####
        main_diff_score = actual_verb_score - inflected_verb_score

        # Repeating examples and replacing one token at a time with unk
        batch_size = 1
        max_len = pt_batch['input_ids'][0].size(0)

        # Repeat each example 'max_len' times
        x = pt_batch['input_ids'].cpu().detach().numpy()
        extended_x = np.reshape(np.tile(x[:,None,...], (1, max_len, 1)),(-1,x.shape[-1]))

        # Create unk sequences and unk mask
        unktoken = tokenizer.encode([tokenizer.mask_token])[1]
        unks = unktoken * np.eye(max_len)
        unks =  np.tile(unks, (batch_size, 1))

        unk_mask =  (unktoken - unks)/unktoken

        # Replace one token in each repeatition with unk
        extended_x = extended_x * unk_mask + unks

        # Get the new output
        extended_logits = model_wrapped.get_prediction(torch.tensor(extended_x, dtype=torch.int64, device=device))
        # extended_logits = extended_predictions[0]
        extended_probs = torch.nn.Softmax(dim=-1)(extended_logits)
        extended_correct_probs = extended_probs[:,target_idx,good_verb_id]
        # extended_wrong_probs =  extended_probs[:,target_idx,wrong_verb_id]
        # extended_diff_scores = extended_correct_probs - extended_wrong_probs

        # # Save the difference in the probability predicted for the correct class
        #diffs = abs(main_diff_score - extended_diff_scores)
        diffs = abs(actual_verb_score - extended_correct_probs)

        relevancies['blankout'].append(diffs.cpu().detach().numpy())

    return relevancies

def contribution_relevancies_sst2(model, model_name, tokenizer, dataset_partition, random_samples_list):
    """Contribution rollout-based relevancies in SST2."""

    special_tokens_relevancies = defaultdict(list)
    relevancies = defaultdict(list)

    special_tokens = ['[CLS]','[SEP]','.',',']
    special_tokens_roberta = ['<s>','</s>','\u0120'+'.','\u0120' + ',']
    layer = -1

    all_examples_x = []

    for i in random_samples_list:
        model_wrapped = ModelWrapper(model)
        sentence = dataset_partition[i]
        text = sentence['sentence']
        pt_batch = tokenizer(text, return_tensors="pt").to(device)
        tokenized_text = tokenizer.convert_ids_to_tokens(pt_batch["input_ids"][0])
        relevancies['examples'].append(tokenized_text)
        prediction_scores, hidden_states, attentions, contributions_data = model_wrapped(pt_batch)
        all_examples_x.append(pt_batch["input_ids"][0])

        # Raw attentions relevances    
        _attentions = [att.detach().cpu().numpy() for att in attentions]
        attentions_mat = np.asarray(_attentions)[:,0]
        raw_attn_relevances = get_raw_att_relevance(attentions_mat)
        relevancies['raw'].append(np.asarray(raw_attn_relevances))
        raw_rank_normalized = get_normalized_rank(np.asarray(raw_attn_relevances))  

        # Rollout attentions relevances
        # Sum over heads
        att_mat_sum_heads = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
        joint_attentions = compute_rollout(att_mat_sum_heads)
        rollout_relevance_relevances = joint_attentions[layer][0]
        relevancies['rollout'].append(np.asarray(rollout_relevance_relevances))
        rollout_rank_normalized = get_normalized_rank(np.asarray(rollout_relevance_relevances))

        # Norms + roll out relevances
        normalized_model_norms = normalize_contributions(contributions_data['transformed_vectors_norm'],scaling='sum_one')
        norms_mix = compute_joint_attention(normalized_model_norms)
        norms_mix_relevances = norms_mix[layer][0]
        relevancies['norm'].append(np.asarray(norms_mix_relevances))
        rollout_norm_rank_normalized = get_normalized_rank(np.asarray(norms_mix_relevances))

        # Our method relevances
        resultant_norm = torch.norm(torch.squeeze(contributions_data['resultants']),p=1,dim=-1)
        normalized_contributions = normalize_contributions(contributions_data['contributions'],scaling='min_sum',resultant_norm=resultant_norm)
        contributions_mix = compute_joint_attention(normalized_contributions)
        contributions_mix_relevances = contributions_mix[layer][0]
        relevancies['ours'].append(np.asarray(contributions_mix_relevances))
        our_rank_normalized = get_normalized_rank(np.asarray(contributions_mix_relevances))
       
        if model_name == 'roberta':
            special_tokens = special_tokens_roberta
        for x in special_tokens:
            if x in tokenized_text:
                special_tokens_relevancies['raw'].append(raw_rank_normalized[tokenized_text.index(x)])
                special_tokens_relevancies['rollout'].append(rollout_rank_normalized[tokenized_text.index(x)])
                special_tokens_relevancies['rollout_norm'].append(rollout_norm_rank_normalized[tokenized_text.index(x)])
                special_tokens_relevancies['our'].append(our_rank_normalized[tokenized_text.index(x)])

    return relevancies, special_tokens_relevancies

def gradient_relevancies(model, model_name, tokenizer, dataset_partition, random_samples_list):
    """Contribution gradient-based relevancies in SST2."""

    special_tokens_relevancies = defaultdict(list)
    relevancies = defaultdict(list)

    special_tokens = ['[CLS]','[SEP]','.',',']
    special_tokens_roberta = ['<s>','</s>','\u0120'+'.','\u0120' + ',']
    layer = -1

    bert_model_wrapper = ClassificationModelWrapperCaptum(model)

    for i in random_samples_list:
        sentence = dataset_partition[i]
        text = sentence['sentence']
        pt_batch = tokenizer(text, return_tensors="pt").to(device)
        tokenized_text = tokenizer.convert_ids_to_tokens(pt_batch["input_ids"][0])

        # Grad, grad input and ig relevances
        grad_relevance, pred_class= interpret_sentence_sst2(bert_model_wrapper, tokenizer, sentence=text, method='grad')
        relevancies['pred_class'].append(pred_class.item())
        grad_relevance = grad_relevance.cpu().detach().numpy()
        relevancies['grad'].append(grad_relevance)
        # Normalized grad relevancies for special tokens analysis
        grad_rank_normalized = get_normalized_rank(grad_relevance)

        grad_x_input_relevance, _ = interpret_sentence_sst2(bert_model_wrapper, tokenizer, sentence=text, method='grad_input')
        grad_x_input_relevance = grad_x_input_relevance.cpu().detach().numpy()
        relevancies['grad_input'].append(grad_x_input_relevance)
        #grad_x_input_rank_normalized = get_normalized_rank(grad_x_input_relevance)

        ig_relevance, _ = interpret_sentence_sst2(bert_model_wrapper, tokenizer, sentence=text, method='ig')
        ig_relevance = ig_relevance.cpu().detach().numpy()
        relevancies['ig'].append(ig_relevance)
        #ig_relevance_rank_normalized = get_normalized_rank(ig_relevance)

        if model_name == 'roberta':
            special_tokens = special_tokens_roberta
        for x in special_tokens:
            if x in tokenized_text:
                special_tokens_relevancies['grad'].append(grad_rank_normalized[tokenized_text.index(x)])

    return relevancies, special_tokens_relevancies

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
    model_name = args[0]
    dataset_name = args[1]
    model, tokenizer, dataset_partition = load_model_data(model_name,dataset_name)

    num_samples = int(args[2])

    # Random samples from test set (no duplicates)
    random.seed(10)
    random_samples_list = random.sample(range(len(dataset_partition)), num_samples)
    if dataset_name == 'sva':
        relevancies = contribution_relevancies_sva(model, model_name, tokenizer, dataset_partition, random_samples_list)
        methods_list = ['raw','rollout','norm','ours']
        reference_method = 'blankout'
        corrs_method = compute_correlation(relevancies, methods_list, reference_method)
    elif dataset_name == 'sst2:':
        relevancies, special_tokens_relevancies = contribution_relevancies_sst2(model, model_name, tokenizer, dataset_partition, random_samples_list)
        grad_relevancies, grad_special_tokens_relevancies = gradient_relevancies(model, model_name, tokenizer, dataset_partition, random_samples_list)

        # Merge both dictionaries (rollout-based relevancies and gradient relevancies)
        relevancies.update(grad_relevancies)
        special_tokens_relevancies.update(grad_special_tokens_relevancies)

         # Compute mean and avg correlation between methods
        methods_list = methods_list = ['raw','rollout','norm','ours','grad']
        reference_method = 'grad'
        corrs_method = compute_correlation(relevancies, methods_list, reference_method)

        # Compute mean and avg rank of special tokens
        special_tokens_method = defaultdict(list)
        for method in special_tokens_relevancies.keys():
            special_tokens_method[method].append(statistics.mean(special_tokens_relevancies[method]))
            special_tokens_method[method].append(statistics.stdev(special_tokens_relevancies[method]))

        outfile = f'./data/{model_name}_{dataset_name}_rank_special_tokens.json'
        with open(outfile, 'w') as f:
            json.dump(special_tokens_method, f)

        # Save sst2 rank relevancies to compute correlations between BERT and DistilBERT
        outfile = f'./data/sentences_{model_name}_{dataset_name}_rank_relevancies.npy'
        np.save(outfile,relevancies)

    outfile = f'./data/{model_name}_{dataset_name}_correlations.json'
    with open(outfile, 'w') as f:
        json.dump(corrs_method, f)



if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)