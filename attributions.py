from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils_contributions import *
import torch.nn.functional as F
from src.contributions import ModelWrapper, ClassificationModelWrapperCaptum, LMModelWrapperCaptum, interpret_sentence, occlusion
#import contributions
import json
import statistics
import random
random.seed(10)

import argparse
from types import SimpleNamespace

from collections import defaultdict

import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
import sys

def contribution_relevancies_sva(model, args_rel, binary=True):
    """Contribution rollout-based relevancies and blank-out relevancies in SVA."""

    relevancies = defaultdict(list)
    layer = -1

    #for i in args_rel.random_samples_list:
    for i in range(args_rel.num_samples):
        if i%50==0:
            print(i)
        pt_batch, tokenized_text, target_idx = prepare_sva_sentence(i,args_rel.dataset_partition,args_rel.tokenizer)
        
        model_wrapped = ModelWrapper(model)
        prediction_scores, hidden_states, attentions, contributions_data = model_wrapped(pt_batch)
        if binary:
            prediction_scores = torch.squeeze(prediction_scores)
            actual_verb_score = torch.sigmoid(prediction_scores[target_idx])

        # Raw attentions relevances
        _attentions = [att.detach().cpu().numpy() for att in attentions]
        attentions_mat = np.asarray(_attentions)[:,0]
        raw_attn_relevances = get_raw_att_relevance(attentions_mat,token_pos=target_idx)

        relevancies['raw'].append(np.asarray(raw_attn_relevances))

        # Rollout attentions relevances
        att_mat_sum_heads = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
        joint_attentions = compute_rollout(att_mat_sum_heads)
        rollout_relevance_relevances = joint_attentions[layer][target_idx]
        relevancies['rollout'].append(np.asarray(rollout_relevance_relevances))

        # Norms + roll out relevances
        normalized_model_norms = normalize_contributions(contributions_data['transformed_vectors_norm'],scaling='sum_one')
        norms_mix = compute_joint_attention(normalized_model_norms)
        norms_mix_relevances = norms_mix[layer][target_idx]
        relevancies['norm'].append(np.asarray(norms_mix_relevances))

        # Our method relevances
        resultant_norm = resultants_norm = torch.norm(torch.squeeze(contributions_data['resultants']),p=1,dim=-1)
        normalized_contributions = normalize_contributions(contributions_data['contributions'],scaling='min_sum',resultant_norm=resultant_norm)#min_sum
        contributions_mix = compute_joint_attention(normalized_contributions)
        contributions_mix_relevances = contributions_mix[layer][target_idx]
        relevancies['ours'].append(np.asarray(contributions_mix_relevances))

        # LayerNorm 2
        # Norms + roll out relevances
        normalized_model_norms = contributions_data['transformed_vectors_norm2']
        norms_mix = compute_joint_attention(normalized_model_norms)
        norms_mix_relevances = norms_mix[layer][target_idx]
        relevancies['norm2'].append(np.asarray(norms_mix_relevances))

        # Our method relevances
        resultant_norm = torch.norm(torch.squeeze(contributions_data['resultants2']),p=1,dim=-1)
        normalized_contributions = normalize_contributions(contributions_data['contributions2'],scaling='min_sum',resultant_norm=resultant_norm)
        contributions_mix = compute_joint_attention(normalized_contributions)
        contributions_mix_relevances = contributions_mix[layer][target_idx]
        relevancies['ours2'].append(np.asarray(contributions_mix_relevances))

        # Our method relevances with L2
        resultant_norm = torch.norm(torch.squeeze(contributions_data['resultants']),p=2,dim=-1)
        normalized_contributions = normalize_contributions(contributions_data['contributions_l2'],scaling='min_sum',resultant_norm=resultant_norm)
        contributions_mix = compute_joint_attention(normalized_contributions)
        contributions_mix_relevances = contributions_mix[layer][target_idx]
        relevancies['ours_l2'].append(np.asarray(contributions_mix_relevances))

        diffs = occlusion(model_wrapped, args_rel.tokenizer, target_idx, actual_verb_score, pt_batch)

        relevancies['blankout'].append(diffs.cpu().detach().numpy())

    return relevancies

def contribution_relevancies_sst2(model, args_rel):
    """Contribution rollout-based relevancies in SST2."""

    special_tokens_relevancies = defaultdict(list)
    relevancies = defaultdict(list)

    special_tokens = ['[CLS]','[SEP]','.',',']
    special_tokens_roberta = ['<s>','</s>','\u0120'+'.','\u0120' + ',']
    layer = -1

    model_wrapped = ModelWrapper(model)

    skip = 0

    #for i in args_rel.random_samples_list:
    for i in range(args_rel.num_samples):
        if i%50==0:
            print(i)
        sentence = args_rel.dataset_partition[i]
        text = sentence[list(args_rel.dataset_partition.features.keys())[0]]
        pt_batch = args_rel.tokenizer(text, return_tensors="pt", return_token_type_ids= False).to(device)
        tokenized_text = args_rel.tokenizer.convert_ids_to_tokens(pt_batch["input_ids"][0])
        if len(tokenized_text) > 200:
            skip += 1
            continue
        #relevancies['examples'].append(tokenized_text)
        prediction_scores, hidden_states, attentions, contributions_data = model_wrapped(pt_batch)
        probs = torch.nn.functional.softmax(prediction_scores, dim=-1)
        pred_class_ind = torch.argmax(probs)
        pred = torch.max(probs)#[pred_ind]
        prob_pred_class = probs[0][pred_class_ind].detach().cpu().item()
        relevancies['pred_class'].append(pred_class_ind.cpu().detach().item())

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

        # Norms + roll out relevances (Kobayashi 2021 + Rollout)
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

        # Our method relevances with L2
        resultant_norm = torch.norm(torch.squeeze(contributions_data['resultants']),p=2,dim=-1)
        normalized_contributions = normalize_contributions(contributions_data['contributions_l2'],scaling='min_sum',resultant_norm=resultant_norm)
        contributions_mix = compute_joint_attention(normalized_contributions)
        contributions_mix_relevances = contributions_mix[layer][0]
        relevancies['ours_l2'].append(np.asarray(contributions_mix_relevances))
        our_rank_normalized = get_normalized_rank(np.asarray(contributions_mix_relevances))

        # LayerNorm 2
        # Norms + roll out relevances (Globenc) / No normalization
        #normalized_model_norms = normalize_contributions(contributions_data['transformed_vectors_norm2'],scaling='sum_one')
        normalized_model_norms = contributions_data['transformed_vectors_norm2']
        norms_mix = compute_joint_attention(normalized_model_norms)
        norms_mix_relevances = norms_mix[layer][0]
        relevancies['norm2'].append(np.asarray(norms_mix_relevances))
        norm2_rank_normalized = get_normalized_rank(np.asarray(norms_mix_relevances))

        # Our method relevances
        resultant_norm = torch.norm(torch.squeeze(contributions_data['resultants2']),p=1,dim=-1)
        normalized_contributions = normalize_contributions(contributions_data['contributions2'],scaling='min_sum',resultant_norm=resultant_norm)
        contributions_mix = compute_joint_attention(normalized_contributions)
        contributions_mix_relevances = contributions_mix[layer][0]
        relevancies['ours2'].append(np.asarray(contributions_mix_relevances))

        # Blank-out (Occlusion)
        diffs = occlusion(model_wrapped, args_rel.tokenizer, pred_class_ind, prob_pred_class, pt_batch)
        relevancies['blankout'].append(np.asarray(diffs.cpu().detach().numpy()))
       
        if args_rel.special_tokens:
            if args_rel.model_name == 'roberta':
                special_tokens = special_tokens_roberta
            for x in special_tokens:
                if x in tokenized_text:
                    special_tokens_relevancies['raw'].append(raw_rank_normalized[tokenized_text.index(x)])
                    special_tokens_relevancies['rollout'].append(rollout_rank_normalized[tokenized_text.index(x)])
                    special_tokens_relevancies['rollout_norm'].append(rollout_norm_rank_normalized[tokenized_text.index(x)])
                    special_tokens_relevancies['ours'].append(our_rank_normalized[tokenized_text.index(x)])
                    special_tokens_relevancies['norm2'].append(norm2_rank_normalized[tokenized_text.index(x)])

    print('Skipped', skip)
    
    return relevancies, special_tokens_relevancies

def gradient_relevancies(model, args_rel):
    """Contribution gradient-based relevancies."""

    special_tokens_relevancies = defaultdict(list)
    relevancies = defaultdict(list)

    special_tokens = ['[CLS]','[SEP]','.',',']
    special_tokens_roberta = ['<s>','</s>','\u0120'+'.','\u0120' + ',']
    layer = -1

    if args_rel.dataset_name == 'sva':
        bert_model_wrapper = LMModelWrapperCaptum(model)
    else:
        bert_model_wrapper = ClassificationModelWrapperCaptum(model)

    #for i in args_rel.random_samples_list:
    for i in range(args_rel.num_samples):
        if i%50==0:
            print(i)
        if args_rel.dataset_name == 'sva':
            pt_batch, tokenized_text, target_idx = prepare_sva_sentence(i,args_rel.dataset_partition,args_rel.tokenizer)
            input_ids = pt_batch["input_ids"]
        else:
            sentence = args_rel.dataset_partition[i]
            text = sentence[list(args_rel.dataset_partition.features.keys())[0]]
            pt_batch = args_rel.tokenizer(text, return_tensors="pt", return_token_type_ids= False).to(device)
            input_ids = pt_batch["input_ids"]
            tokenized_text = args_rel.tokenizer.convert_ids_to_tokens(pt_batch["input_ids"][0])
        if len(tokenized_text) > 200:
            continue
        if args_rel.dataset_name == 'sva':
            pred_class_ind, prob_pred_class = bert_model_wrapper.get_prediction(input_ids, target_idx)
        else:
            pred_class_ind, prob_pred_class = bert_model_wrapper.get_prediction(input_ids)

        relevancies['pred_class'].append(pred_class_ind)

        # Grad, grad input and ig relevances
        grad_relevance = interpret_sentence(bert_model_wrapper, args_rel.tokenizer, input_ids, 'grad', pred_class_ind)
        grad_relevance = grad_relevance.cpu().detach().numpy()
        relevancies['grad'].append(grad_relevance)
        # Normalized grad relevancies for special tokens analysis
        grad_rank_normalized = get_normalized_rank(grad_relevance)

        grad_x_input_relevance = interpret_sentence(bert_model_wrapper, args_rel.tokenizer, input_ids, 'grad_input', pred_class_ind)
        grad_x_input_relevance_abs = grad_x_input_relevance['abs'].cpu().detach().numpy()
        relevancies['grad_input_abs'].append(grad_x_input_relevance_abs)
        grad_x_input_relevance_clip = grad_x_input_relevance['clip'].cpu().detach().numpy()
        relevancies['grad_input_clip'].append(grad_x_input_relevance_clip)
        grad_x_input_relevance_l2 = grad_x_input_relevance['l2'].cpu().detach().numpy()
        relevancies['grad_input_l2'].append(grad_x_input_relevance_l2)
        grad_x_input_l2_rank_normalized = get_normalized_rank(grad_x_input_relevance_l2)
        grad_x_input_relevance_mean = grad_x_input_relevance['mean'].cpu().detach().numpy()
        relevancies['grad_input_mean'].append(grad_x_input_relevance_mean)
        grad_x_input_relevance_abs_mean = grad_x_input_relevance['abs_mean'].cpu().detach().numpy()
        relevancies['grad_input_mean_abs_mean'].append(grad_x_input_relevance_abs_mean)
        grad_x_input_mean_abs_rank_normalized = get_normalized_rank(grad_x_input_relevance_abs_mean)

        ig_relevance = interpret_sentence(bert_model_wrapper, args_rel.tokenizer, input_ids, 'ig', pred_class_ind)
        ig_relevance_abs = ig_relevance['abs'].cpu().detach().numpy()
        relevancies['ig_abs'].append(ig_relevance_abs)
        ig_relevance_clip = ig_relevance['clip'].cpu().detach().numpy()
        relevancies['ig_clip'].append(ig_relevance_clip)
        ig_relevance_l2 = ig_relevance['l2'].cpu().detach().numpy()
        relevancies['ig_l2'].append(ig_relevance_l2)
        ig_relevance_mean = ig_relevance['mean'].cpu().detach().numpy()
        relevancies['ig_mean'].append(ig_relevance_mean)
        ig_relevance_abs_mean = ig_relevance['abs_mean'].cpu().detach().numpy()
        relevancies['ig_abs_mean'].append(ig_relevance_abs_mean)
        ig_abs_mean_relevance_rank_normalized = get_normalized_rank(ig_relevance_abs_mean)
        if args_rel.special_tokens:
            if args_rel.model_name == 'roberta':
                special_tokens = special_tokens_roberta
            for x in special_tokens:
                if x in tokenized_text:
                    special_tokens_relevancies['grad'].append(grad_rank_normalized[tokenized_text.index(x)])
                    special_tokens_relevancies['grad_input_l2'].append(grad_x_input_l2_rank_normalized[tokenized_text.index(x)])
                    special_tokens_relevancies['grad_input_mean_abs_mean'].append(grad_x_input_mean_abs_rank_normalized[tokenized_text.index(x)])
                    special_tokens_relevancies['ig_l2'].append(ig_relevance_l2[tokenized_text.index(x)])
                    special_tokens_relevancies['ig_abs_mean'].append(ig_abs_mean_relevance_rank_normalized[tokenized_text.index(x)])


    return relevancies, special_tokens_relevancies

def main(args):
    model_name = args.model
    dataset_name = args.dataset
    print(model_name)
    print(dataset_name)
    model, tokenizer, dataset_partition = load_model_data(model_name,dataset_name)

    num_samples = args.samples

    # Random samples from test set (no duplicates)
    random.seed(10)
    #random_samples_list = random.sample(range(len(dataset_partition)), num_samples)

    args_rel = SimpleNamespace(
        model_name = model_name,
        dataset_name = args.dataset,
        tokenizer = tokenizer,
        dataset_partition = dataset_partition,
        #random_samples_list = random_samples_list
        num_samples = num_samples,
        special_tokens = args.special_tokens 
        )

    if dataset_name == 'sva':
        if 'bin' in model_name:
            binary = True
        else:
            binary = False
        relevancies = contribution_relevancies_sva(model, args_rel, binary)

        # Save sva rank relevancies to compute correlations between BERT and DistilBERT
        outfile = f'./data/{model_name}_{dataset_name}_attributions.npy'
        np.save(outfile,relevancies)
        
    elif (dataset_name == 'sst2' or dataset_name == 'yelp' or dataset_name == 'imdb'):
        relevancies, special_tokens_relevancies = contribution_relevancies_sst2(model, args_rel)

    torch.cuda.empty_cache()
    grad_relevancies, grad_special_tokens_relevancies = gradient_relevancies(model, args_rel)
    # Merge both dictionaries (rollout-based relevancies and gradient relevancies)
    relevancies.update(grad_relevancies)
    # Save sst2 rank relevancies to compute correlations between BERT and DistilBERT
    outfile = f'./data/{args_rel.model_name}_{args_rel.dataset_name}_attributions.npy'
    np.save(outfile,relevancies)
    if args.special_tokens:
        special_tokens_relevancies.update(grad_special_tokens_relevancies)
        # Save special tokens attributions
        outfile = f'./data/{args_rel.model_name}_{args_rel.dataset_name}_special_tok_attributions.npy'
        np.save(outfile,special_tokens_relevancies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="model used", type= str)
    parser.add_argument('--dataset', help="sst2/yelp/imdb/sva", choices=['sst2', 'sva', 'imdb', 'yelp'], type=str)
    parser.add_argument('--samples', help="number of samples for computing attributions", type=int)
    parser.add_argument('--special-tokens', help="stores special tokens attributions", action='store_true')
    args=parser.parse_args()

    main(args)