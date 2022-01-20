import torch
import torch.nn.functional as F
from functools import partial
import collections
import torch.nn as nn

import json
import h5py
import random
import numpy as np

from captum.attr import IntegratedGradients, Saliency, InputXGradient, InterpretableEmbeddingBase, TokenReferenceBase

device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()

        self.model = model

        self.handles = {}
        for name, module in self.model.named_modules():
            self.handles[name] = module.register_forward_hook(partial(self.save_activation, name))

        self.func_outputs = collections.defaultdict(list)
        self.func_inputs = collections.defaultdict(list)

        self.num_attention_heads = self.model.config.num_attention_heads
        self.attention_head_size = int(self.model.config.hidden_size / self.model.config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def save_activation(self,name, mod, inp, out):
        self.func_inputs[name].append(inp)
        self.func_outputs[name].append(out)

    def clean_hooks(self):
        for k, v in self.handles.items():
            self.handles[k].remove()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_contributions(self, hidden_states_model, attentions, func_inputs, func_outputs):
        #   hidden_states: Representations from previous layer and inputs to self-attention. (batch, seq_length, all_head_size)
        #   attention_probs: Attention weights calculated in self-attention. (batch, num_heads, seq_length, seq_length)
        #   value_layer: Value vectors calculated in self-attention. (batch, num_heads, seq_length, head_size)
        #   dense: Dense layer in self-attention. nn.Linear(all_head_size, all_head_size)
        #   LayerNorm: nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #   pre_ln_states: Vectors just before LayerNorm (batch, seq_length, all_head_size)

        model_importance_list = []
        transformed_vectors_norm_list = []
        transformed_vectors_list = []

        try:
            num_layers = self.model.config.n_layers
        except:
            num_layers = self.model.config.num_hidden_layers

        for layer in range(num_layers):
            hidden_states = hidden_states_model[layer]
            attention_probs = attentions[layer]
            
            if self.model.config.model_type == 'bert':
                value_layer = self.transpose_for_scores(func_outputs[self.model.config.model_type + '.encoder.layer.' + str(layer) + '.attention.self.value'][0])
                dense = self.model.bert.encoder.layer[layer].attention.output.dense
                LayerNorm = self.model.bert.encoder.layer[layer].attention.output.LayerNorm
                pre_ln_states = func_inputs[self.model.config.model_type +'.encoder.layer.' + str(layer) + '.attention.output.LayerNorm'][0][0]
            elif self.model.config.model_type == 'distilbert':
                value_layer = self.transpose_for_scores(func_outputs[self.model.config.model_type + '.transformer.layer.' + str(layer) + '.attention.v_lin'][0])
                dense = self.model.distilbert.transformer.layer[layer].attention.out_lin
                LayerNorm = self.model.distilbert.transformer.layer[layer].sa_layer_norm
                pre_ln_states = func_inputs[self.model.config.model_type +'.transformer.layer.' + str(layer) + '.sa_layer_norm'][0][0]
            elif self.model.config.model_type == 'roberta':
                value_layer = self.transpose_for_scores(func_outputs[self.model.config.model_type + '.encoder.layer.' + str(layer) + '.attention.self.value'][0])
                dense = self.model.roberta.encoder.layer[layer].attention.output.dense
                LayerNorm = self.model.roberta.encoder.layer[layer].attention.output.LayerNorm
                pre_ln_states = func_inputs[self.model.config.model_type +'.encoder.layer.' + str(layer) + '.attention.output.LayerNorm'][0][0]      
            
            # Make transformed vectors T(x) from Value vectors (value_layer) and weight matrix (dense).
            dense_bias = dense.bias#.view(self.attention_head_size, self.num_attention_heads, 1)
            dense = dense.weight.view(self.all_head_size, self.num_attention_heads, self.attention_head_size)
            transformed_layer = torch.einsum('bhsv,dhv->bhsd', value_layer, dense) #(batch, num_heads, seq_length, all_head_size)

            # Make weighted vectors αf(x) from transformed vectors (transformed_layer) 
            # and attention weights (attentions):
            # (batch, num_heads, seq_length, seq_length, all_head_size)
            weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_probs, transformed_layer)
            
            # Sum each weighted vectors αf(x) over all heads:
            # (batch, seq_length, seq_length, all_head_size)
            summed_weighted_layer = weighted_layer.sum(dim=1) # sum over heads

            # Make residual matrix (batch, seq_length, seq_length, all_head_size)
            hidden_shape = hidden_states.size()  #(batch, seq_length, all_head_size)
            device = hidden_states.device
            residual = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1]).to(device), hidden_states)

            # Summed weighted vector + residual vectors -> (batch,seq_len,seq_len,embed_dim)
            residual_weighted_layer = summed_weighted_layer + residual

            # consider layernorm
            ln_weight = LayerNorm.weight.data
            ln_eps = LayerNorm.eps
            ln_bias = LayerNorm.bias

            ## norm_matrix computes mean and performs hadamard product with ln_bias as a linear transformation
            mask = torch.diag(torch.ones_like(ln_weight)).to(device)
            norm_matrix_ln_weight = mask*torch.diag(ln_weight)
            norm_matrix_mean = torch.zeros(self.all_head_size,self.all_head_size).to(device)+(-1/self.all_head_size)
            norm_matrix_mean = norm_matrix_mean.fill_diagonal_((self.all_head_size-1)/self.all_head_size)
            norm_matrix = torch.mul(norm_matrix_ln_weight,norm_matrix_mean) # (all_head_size,all_head_size)

            each_mean = residual_weighted_layer.mean(-1, keepdim=True) # (batch, seq_len, seq_len, 1)

            normalized_layer = residual_weighted_layer - each_mean

            # W^O bias term N(b^O)
            dense_bias_term = torch.einsum('df,f->d', norm_matrix, dense_bias)
            mean_pre_ln = pre_ln_states.mean(-1, keepdim=True) # (batch, seq_len, 1)
            var_pre_ln = (pre_ln_states - mean_pre_ln).pow(2).mean(-1, keepdim=True)#.unsqueeze(dim=2) # (batch, seq_len, 1)

            # Original
            transformed_vectors = torch.einsum('bskd,d->bskd', normalized_layer, ln_weight) # (batch, seq_len, seq_len, all_head_size)
            # V2 (add)
            # print(transformed_vectors.size())
            # print('var_pre_ln',var_pre_ln.size())
            transformed_vectors = transformed_vectors/((var_pre_ln.unsqueeze(-1) + ln_eps)**(1/2))

            transformed_vectors_norm = torch.norm(transformed_vectors, dim=-1) # (batch, seq_len, seq_len)

            # Output vectors 1 per source token
            attn_output = transformed_vectors.sum(dim=2) #(batch,seq_len,all_head_size)

            # Original
            #resultant = (attn_output + dense_bias_term)/((var_pre_ln + ln_eps)**(1/2)) + ln_bias

            # V2
            dense_bias_term = dense_bias_term/((var_pre_ln + ln_eps)**(1/2))
            # print('attn_output',attn_output.size())
            # print('dense_bias_term',dense_bias_term.size())
            # print('ln_bias',ln_bias.size())
            resultant = attn_output + dense_bias_term + ln_bias
            #print('resultant',resultant)

            #print('actual output', func_outputs[self.model.config.model_type +'.encoder.layer.' + str(layer) + '.attention.output.LayerNorm'][0])

            ## FINAL
            importance_matrix = -F.pairwise_distance(transformed_vectors, resultant.unsqueeze(2),p=1)
            #importance_matrix = 1/(F.pairwise_distance(transformed_vectors, resultant.unsqueeze(2),p=1))
            
            model_importance_list.append(torch.squeeze(importance_matrix))
            transformed_vectors_norm_list.append(torch.squeeze(transformed_vectors_norm))
            transformed_vectors_list.append(torch.squeeze(transformed_vectors))
        contributions_model = torch.stack(model_importance_list)
        transformed_vectors_norm_model = torch.stack(transformed_vectors_norm_list)
        transformed_vectors_model = torch.stack(transformed_vectors_list)

        return contributions_model, transformed_vectors_norm_model, transformed_vectors_model

    def get_prediction(self, input_model):
        with torch.no_grad():
            prediction_scores, hidden_states, attentions = self.model(input_model, output_hidden_states=True, output_attentions=True)
            return prediction_scores


    def __call__(self,input_model):
        with torch.no_grad():
            prediction_scores, hidden_states, attentions = self.model(**input_model, output_hidden_states=True, output_attentions=True)
            
            contributions_model, transformed_vectors_norm_model, transformed_vectors_model = self.get_contributions(hidden_states, attentions, self.func_inputs, self.func_outputs)
            # Clean forward_hooks dictionaries
            self.clean_hooks()
            return prediction_scores, hidden_states, attentions, transformed_vectors_norm_model, contributions_model, transformed_vectors_model


class ClassificationModelWrapperCaptum(nn.Module):
    def __init__(self, model):
        super(ClassificationModelWrapperCaptum, self).__init__()
        self.model = model
        
    
    def compute_pooled_outputs(self, embedding_output, attention_mask=None, head_mask=None):

        if attention_mask is None:
            attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

        encoder_outputs = self.model(inputs_embeds=embedding_output, attention_mask = attention_mask)
        return encoder_outputs[0]

    def get_prediction(self, embedding_output):
        logits = self.compute_pooled_outputs(embedding_output)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_ind = torch.argmax(probs[0])
        pred_value = torch.max(probs[0])
        
        return pred_ind, pred_value

    def forward(self, input_embedding):      
        logits = self.compute_pooled_outputs(input_embedding)

        return torch.softmax(logits, dim=-1)


class LMModelWrapperCaptum(nn.Module):
    def __init__(self, model):
        super(LMModelWrapperCaptum, self).__init__()
        self.model = model
        
    
    def compute_pooled_outputs(self, embedding_output, attention_mask=None, head_mask=None):

        if attention_mask is None:
            attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

        encoder_outputs = self.model(inputs_embeds=embedding_output, attention_mask = attention_mask)
        return encoder_outputs[0]

    def forward(self, input_embedding, mask_pos):      
        logits = self.compute_pooled_outputs(input_embedding)

        return torch.softmax(logits[:,mask_pos], dim=-1)

def interpret_sentence_sst2(model_wrapper, tokenizer, sentence, method):

    ig = IntegratedGradients(model_wrapper)
    saliency = Saliency(model_wrapper)
    input_x_gradient = InputXGradient(model_wrapper)

    model_wrapper.eval()
    model_wrapper.zero_grad()
    model_wrapper.to(device)

    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)]).to(device)
    input_embedding = get_embedding(model_wrapper,input_ids)
    pred_class, pred_value = model_wrapper.get_prediction(input_embedding)


    if method == 'ig':
        ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
        sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
        cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

        ref_input_ids = [cls_token_id] + [ref_token_id] * (input_embedding.size(1)-2) + [sep_token_id]
        ref_input_ids = torch.tensor([ref_input_ids], device=device)
        ref_input_ids = get_embedding(model_wrapper,ref_input_ids)

        attribution, _delta = ig.attribute(input_embedding, baselines=ref_input_ids, target = pred_class, n_steps=100, return_convergence_delta=True)
        attribution = torch.squeeze(torch.sum(attribution,dim=-1))
        # Clip negative scores
        # attribution = attribution.clip(min=0)
        # attribution = attribution / attribution.sum()[...,None]
        
        # Absolute value of attributions (Abnar and Zuidema, 2020)
        attribution = torch.abs(attribution)
        attribution = attribution / attribution.sum()[...,None]

    elif method == 'grad':
        attribution = saliency.attribute(input_embedding, target=pred_class, abs=False)
        attribution = torch.norm(attribution.squeeze(),dim=-1)
        attribution = attribution / attribution.sum()[...,None]
        
        
    elif method == 'grad_input':
        attribution = input_x_gradient.attribute(input_embedding, target=pred_class)
        attribution = torch.squeeze(torch.sum(attribution,dim=-1))
        
        # Clip negative scores
        # attribution = attribution.clip(min=0)
        # attribution = attribution / attribution.sum()[...,None]
        
        # Absolute value of attributions (Abnar and Zuidema, 2020)
        attribution = torch.abs(attribution)
        attribution = attribution / attribution.sum()[...,None]

    return attribution, pred_class

def interpret_sentence_sv(model_wrapper, tokenizer, sentence, method, mask_pos, label):

    ig = IntegratedGradients(model_wrapper)
    saliency = Saliency(model_wrapper)
    input_x_gradient = InputXGradient(model_wrapper)

    model_wrapper.eval()
    model_wrapper.zero_grad()
    model_wrapper.to(device)

    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)]).to(device)
    input_embedding = get_embedding(model_wrapper,input_ids)
    #pred_class, pred_value = model_wrapper.get_prediction(input_embedding)

    pred_class = label

    if method == 'ig':
        ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
        sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
        cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

        ref_input_ids = [cls_token_id] + [ref_token_id] * (input_embedding.size(1)-2) + [sep_token_id]
        ref_input_ids = torch.tensor([ref_input_ids], device=device)
        ref_input_ids = get_embedding(model_wrapper,ref_input_ids)

        attribution, _delta = ig.attribute(input_embedding, baselines=ref_input_ids,
                                            target = pred_class, n_steps=100,
                                            return_convergence_delta=True,
                                            additional_forward_args=(mask_pos))

        attribution = torch.squeeze(torch.sum(attribution,dim=-1))
        
        # Clip negative scores
        # attribution = attribution.clip(min=0)
        # attribution = attribution / attribution.sum()[...,None]
        
        # Absolute value of attributions (Abnar and Zuidema, 2020)
        attribution = torch.abs(attribution)
        attribution = attribution / attribution.sum()[...,None]

    elif method == 'grad':
        attribution = saliency.attribute(input_embedding,
                                        target=pred_class,
                                        abs=False,
                                        additional_forward_args=(mask_pos))
        attribution = torch.norm(attribution.squeeze(),dim=-1)
        attribution = attribution / attribution.sum()[...,None]
        
        
    elif method == 'grad_input':
        attribution = input_x_gradient.attribute(input_embedding, target=pred_class, additional_forward_args=(mask_pos))
        attribution = torch.squeeze(torch.sum(attribution,dim=-1))
        ## grad x input (dot product between grad and embedding), sum components
        # Clip negative scores
        # attribution = attribution.clip(min=0)
        # attribution = attribution / attribution.sum()[...,None]
        
        # Absolute value of attributions (Abnar and Zuidema, 2020)
        attribution = torch.abs(attribution)
        attribution = attribution / attribution.sum()[...,None]

    return attribution

def get_embedding(model_wrapper, input_ids):
    if model_wrapper.model.config.model_type == 'bert':
        input_embedding = model_wrapper.model.bert.embeddings(input_ids)
    elif model_wrapper.model.config.model_type == 'distilbert':
        input_embedding = model_wrapper.model.distilbert.embeddings(input_ids)
    elif model_wrapper.model.config.model_type == 'roberta':
        input_embedding = model_wrapper.model.roberta.embeddings(input_ids)
    return input_embedding