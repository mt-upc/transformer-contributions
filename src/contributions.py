import torch
import torch.nn.functional as F
from functools import partial
import collections
import torch.nn as nn

import random
import numpy as np

from pyaml_env import parse_config
config = parse_config("./src/config.yaml")

from captum.attr import IntegratedGradients, Saliency, InputXGradient, InterpretableEmbeddingBase, TokenReferenceBase

device = "cuda" if torch.cuda.is_available() else "cpu"
data_type = torch.float32

def get_module(model, module_name, layer, model_layer_name=None):
    parsed_module_name = module_name.split('.')
    tmp_module = model
    if model_layer_name:
        parsed_layer_name = model_layer_name.split('.')
        # Loop to find layers module
        for sub_module in parsed_layer_name:
            tmp_module = getattr(tmp_module, sub_module)
        # Select specific layer
        tmp_module = tmp_module[layer]
    # Loop over layer module to find module_name
    for sub_module in parsed_module_name:
        tmp_module = getattr(tmp_module, sub_module)
    return tmp_module

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()

        self.model = model

        self.modules_config = config['models'][model.config.model_type]

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

    def get_model_info(self):
        ln1_name = self.modules_config['ln1']
        ln2_name = self.modules_config['ln2']
        values_name = self.modules_config['values']
        model_layer_name = self.modules_config['layer']
        dense_name = self.modules_config['dense']
        pre_layer_norm = self.modules_config['pre_layer_norm']
        lnf_name = self.modules_config['final_layer_norm']


        return {'ln1_name': ln1_name,
                'ln2_name': ln2_name,
                'values_name': values_name,
                'model_layer_name': model_layer_name,
                'dense_name': dense_name,
                'pre_layer_norm': pre_layer_norm,
                'lnf_name': lnf_name}

    def get_modules_model(self, layer):
        model_info_dict = self.get_model_info()  
        model_layer_name = model_info_dict['model_layer_name']

        dense = get_module(self.model, model_info_dict['dense_name'], layer, model_layer_name)
        fc1 = get_module(self.model, model_info_dict['fc1_name'], layer, model_layer_name)
        fc2 = get_module(self.model, model_info_dict['fc2_name'], layer, model_layer_name)
        ln1 = get_module(self.model, model_info_dict['ln1_name'], layer, model_layer_name)
        ln2 = get_module(self.model, model_info_dict['ln2_name'], layer, model_layer_name)

        return {'dense': dense,
                'fc1': fc1,
                'fc2': fc2,
                'ln1': ln1,
                'ln2': ln2}

    @torch.no_grad()
    def get_contributions(self, hidden_states_model, attentions, func_inputs, func_outputs):
        #   hidden_states_model: Representations from previous layer and inputs to self-attention. (batch, seq_length, all_head_size)
        #   attentions: Attention weights calculated in self-attention. (batch, num_heads, seq_length, seq_length)

        ln1_name = self.modules_config['ln1']
        ln2_name = self.modules_config['ln2']
        values_name = self.modules_config['values']
        model_layer_name = self.modules_config['layer']
        dense_name = self.modules_config['dense']
        pre_layer_norm = self.modules_config['pre_layer_norm']
        if pre_layer_norm == 'True':
            pre_layer_norm = True
        elif pre_layer_norm == 'False':
            pre_layer_norm = False

        model_importance_list = []
        transformed_vectors_norm_list = []
        transformed_vectors_list = []
        resultants_list = []

        model_importance_list2 = []
        transformed_vectors_norm_list2 = []
        transformed_vectors_list2 = []
        resultants_list2 = []
        contributions_data = {}

        model_importance_list_l2 = []

        try:
            num_layers = self.model.config.n_layers
        except:
            num_layers = self.model.config.num_hidden_layers

        for layer in range(num_layers):
            hidden_states = hidden_states_model[layer].detach()
            attention_probs = attentions[layer].detach()

            #   value_layer: Value vectors calculated in self-attention. (batch, num_heads, seq_length, head_size)
            #   dense: Dense layer in self-attention. nn.Linear(all_head_size, all_head_size)
            #   LayerNorm: nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            #   pre_ln_states: Vectors just before LayerNorm (batch, seq_length, all_head_size)

            value_layer = self.transpose_for_scores(func_outputs[model_layer_name + '.' + str(layer) + '.' + values_name][0])
            
            dense = get_module(self.model, dense_name, layer, model_layer_name)
            ln1 = get_module(self.model, ln1_name, layer, model_layer_name)
            ln2 = get_module(self.model, ln2_name, layer, model_layer_name)
            pre_ln_states = func_inputs[model_layer_name + '.' + str(layer) + '.' + ln1_name][0][0]
            post_ln_states = func_outputs[model_layer_name + '.' + str(layer) + '.' + ln1_name][0]
            pre_ln2_states = func_inputs[model_layer_name + '.' + str(layer) + '.' + ln2_name][0][0]
            if pre_ln2_states.dim() == 2:
                pre_ln2_states = pre_ln2_states.unsqueeze(0)
            post_LayerNorm_FFN = func_outputs[model_layer_name + '.' + str(layer) + '.' + ln2_name][0]
            if post_LayerNorm_FFN.dim() == 2:
                post_LayerNorm_FFN = post_LayerNorm_FFN.unsqueeze(0)
            
            # VW_O
            dense_bias = dense.bias
            dense = dense.weight.view(self.all_head_size, self.num_attention_heads, self.attention_head_size)
            transformed_layer = torch.einsum('bhsv,dhv->bhsd', value_layer, dense) #(batch, num_heads, seq_length, all_head_size)
            del dense

            # AVW_O
            # (batch, num_heads, seq_length, seq_length, all_head_size)
            weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_probs, transformed_layer)
            del transformed_layer
            
            # Sum each weighted vectors αf(x) over all heads:
            # (batch, seq_length, seq_length, all_head_size)
            summed_weighted_layer = weighted_layer.sum(dim=1) # sum over heads
            del weighted_layer

            # Make residual matrix (batch, seq_length, seq_length, all_head_size)
            hidden_shape = hidden_states.size()
            device = hidden_states.device
            residual = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1]).to(device), hidden_states)

            # AVW_O + residual vectors -> (batch,seq_len,seq_len,embed_dim)
            residual_weighted_layer = summed_weighted_layer + residual
            
            @torch.no_grad()
            def l_transform(x, w_ln):
                '''Computes mean and performs hadamard product with ln weight (w_ln) as a linear transformation.'''
                ln_param_transf = torch.diag(w_ln)
                ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - \
                    1 / w_ln.size(0) * torch.ones_like(ln_param_transf).to(w_ln.device)

                out = torch.einsum(
                    '... e , e f , f g -> ... g',
                    x,
                    ln_mean_transf,
                    ln_param_transf
                )
                return out

            if pre_layer_norm == False:
                # LN 1
                ln1_weight = ln1.weight.data
                ln1_eps = ln1.eps
                ln1_bias = ln1.bias

                # Transformed vectors T_i(x_j)
                transformed_vectors = l_transform(residual_weighted_layer, ln1_weight)
            else:
                transformed_vectors = residual_weighted_layer

            # Output vectors 1 per source token
            attn_output = transformed_vectors.sum(dim=2)

            if pre_layer_norm == False:
                # Lb_O
                dense_bias_term = l_transform(dense_bias, ln1_weight)
                # y_i
                ln_std_coef = 1/(pre_ln_states + ln1_eps).std(-1, unbiased=False).view(1, -1, 1)
                resultant = (attn_output + dense_bias_term)*ln_std_coef + ln1_bias
                transformed_vectors_std = l_transform(residual_weighted_layer, ln1_weight)*ln_std_coef.unsqueeze(-1)
                real_resultant = post_ln_states
            else:
                dense_bias_term = dense_bias
                resultant = attn_output + dense_bias_term
                transformed_vectors_std = transformed_vectors
                real_resultant = pre_ln2_states
                
            # Assert interpretable expression of attention is equal to the output of the attention block
            assert torch.dist(resultant, real_resultant).item() < 1e-3 * real_resultant.numel()
            del real_resultant
            
            transformed_vectors_norm_std = torch.norm(transformed_vectors_std, dim=-1) # (batch, seq_len, seq_len)

            importance_matrix = -F.pairwise_distance(transformed_vectors_std, resultant.unsqueeze(2),p=1)
            importance_matrix_l2 = -F.pairwise_distance(transformed_vectors_std, resultant.unsqueeze(2),p=2)

            
            model_importance_list.append(torch.squeeze(importance_matrix).cpu().detach())
            model_importance_list_l2.append(torch.squeeze(importance_matrix_l2).cpu().detach())

            transformed_vectors_norm_list.append(torch.squeeze(transformed_vectors_norm_std).cpu().detach())
            transformed_vectors_list.append(torch.squeeze(transformed_vectors_std).cpu().detach())
            resultants_list.append(torch.squeeze(resultant).cpu().detach())


            ############################
            # LN 2
            ln2_weight = ln2.weight.data
            ln2_eps = ln2.eps
            ln2_bias = ln2.bias

            ln2_std_coef = 1/(pre_ln2_states + ln2_eps).std(-1, unbiased=False).view(1, -1, 1) # (batch,seq_len,1)
            transformed_vectors_std2 = l_transform(transformed_vectors_std, ln2_weight)*ln2_std_coef.unsqueeze(-1)
            resultant2 = post_LayerNorm_FFN

            transformed_vectors_norm_std2 = torch.norm(transformed_vectors_std2, dim=-1) # (batch, seq_len, seq_len)

            importance_matrix2 = -F.pairwise_distance(transformed_vectors_std2, resultant2.unsqueeze(2),p=1)

            model_importance_list2.append(torch.squeeze(importance_matrix2).cpu().detach())
            transformed_vectors_norm_list2.append(torch.squeeze(transformed_vectors_norm_std2).cpu().detach())
            transformed_vectors_list2.append(torch.squeeze(transformed_vectors_std2).cpu().detach())
            resultants_list2.append(torch.squeeze(resultant2).cpu().detach())

        contributions_model = torch.stack(model_importance_list)
        ###
        contributions_model_l2 = torch.stack(model_importance_list_l2)
        ###
        transformed_vectors_norm_model = torch.stack(transformed_vectors_norm_list)
        transformed_vectors_model = torch.stack(transformed_vectors_list)
        resultants_model = torch.stack(resultants_list)

        contributions_data['contributions'] = contributions_model
        ###
        contributions_data['contributions_l2'] = contributions_model_l2
        ###
        contributions_data['transformed_vectors'] = transformed_vectors_model
        contributions_data['transformed_vectors_norm'] = transformed_vectors_norm_model
        contributions_data['resultants'] = resultants_model

        contributions_model2 = torch.stack(model_importance_list2)
        transformed_vectors_norm_model2 = torch.stack(transformed_vectors_norm_list2)
        transformed_vectors_model2 = torch.stack(transformed_vectors_list2)
        resultants_model2 = torch.stack(resultants_list2)

        contributions_data['contributions2'] = contributions_model2
        contributions_data['transformed_vectors2'] = transformed_vectors_model2
        contributions_data['transformed_vectors_norm2'] = transformed_vectors_norm_model2
        contributions_data['resultants2'] = resultants_model2

        return contributions_data

    def get_prediction(self, input_model):
        with torch.no_grad():
            output = self.model(input_model, output_hidden_states=True, output_attentions=True)
            prediction_scores = output['logits']

            return prediction_scores


    def __call__(self, input_model, contributions=True):
        with torch.no_grad():
            self.handles = {}
            for name, module in self.model.named_modules():
                self.handles[name] = module.register_forward_hook(partial(self.save_activation, name))

            self.func_outputs = collections.defaultdict(list)
            self.func_inputs = collections.defaultdict(list)

            output = self.model(**input_model, output_hidden_states=True, output_attentions=True)
            prediction_scores = output['logits'].detach()
            hidden_states = output['hidden_states']
            attentions = output['attentions']
            
            contributions_data = self.get_contributions(hidden_states, attentions, self.func_inputs, self.func_outputs)

            if contributions:
                contributions_data = self.get_contributions(hidden_states, attentions, self.func_inputs, self.func_outputs)
            else:
                contributions_data = None

            # Clean forward_hooks dictionaries
            self.clean_hooks()
            return prediction_scores, hidden_states, attentions, contributions_data


class ClassificationModelWrapperCaptum(nn.Module):
    def __init__(self, model):
        super(ClassificationModelWrapperCaptum, self).__init__()
        self.model = model
        
    
    def compute_pooled_outputs(self, embedding_output, attention_mask=None, head_mask=None):

        if attention_mask is None:
            attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

        encoder_outputs = self.model(inputs_embeds=embedding_output, attention_mask = attention_mask)
        return encoder_outputs[0]

    def get_prediction(self, input_model):
        output = self.model(input_model, output_hidden_states=True, output_attentions=True)
        logits = output['logits']
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_class_ind = torch.argmax(probs).detach().cpu().item()
        pred = torch.max(probs)#[pred_ind]
        prob_pred_class = probs[0][pred_class_ind].detach().cpu().item()
        
        return pred_class_ind, prob_pred_class

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

    def get_prediction(self, input_model, target_idx):
        output = self.model(input_model, output_hidden_states=True, output_attentions=True)
        logits = output['logits']
        prediction_scores = torch.squeeze(logits)
        prob_pred_class = torch.sigmoid(prediction_scores[target_idx]).detach().cpu().item()
        
        return target_idx, prob_pred_class

    def forward(self, input_embedding):      
        logits = self.compute_pooled_outputs(input_embedding)

        return torch.softmax(logits.squeeze(-1), dim=-1)

def occlusion(model_wrapped, tokenizer, pred_class_index, prob_pred_class, pt_batch):
    # Blank-out method ####
    # Repeating examples and replacing one token at a time with unk
    batch_size = 1
    max_len = pt_batch['input_ids'][0].size(0)

    # Repeat each example 'max_len' times
    x = pt_batch['input_ids'].cpu().detach().numpy()
    extended_x = np.reshape(np.tile(x[:,None,...], (1, max_len, 1)),(-1,x.shape[-1]))

    # Create sequences and mask
    unktoken = tokenizer.mask_token_id
    unks = unktoken * np.eye(max_len)
    unks =  np.tile(unks, (batch_size, 1))

    unk_mask =  (unktoken - unks)/unktoken

    # Replace one token in each repetition with unk
    extended_x = extended_x * unk_mask + unks

    # Get the new output
    extended_logits = model_wrapped.get_prediction(torch.tensor(extended_x, dtype=torch.int64, device=device))
    #prediction_scores = torch.squeeze(extended_logits)
    extended_correct_probs = torch.sigmoid(extended_logits[:,pred_class_index]).squeeze()
    diffs = abs(prob_pred_class - extended_correct_probs)
    return diffs

def interpret_sentence(model_wrapper, tokenizer, input_ids, method, pred_class):

    ig = IntegratedGradients(model_wrapper)
    saliency = Saliency(model_wrapper)
    input_x_gradient = InputXGradient(model_wrapper)

    model_wrapper.eval()
    model_wrapper.zero_grad()
    model_wrapper.to(device)

    #input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)]).to(device)
    input_embedding = get_embedding(model_wrapper,input_ids)
    # Get predicted class (to compute gradients)
    #pred_class, pred_value = model_wrapper.get_prediction(input_embedding)


    if method == 'ig':
        ref_token_id = tokenizer.mask_token_id # A token used for generating token reference
        sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
        cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

        ref_input_ids = [cls_token_id] + [ref_token_id] * (input_embedding.size(1)-2) + [sep_token_id]
        ref_input_ids = torch.tensor([ref_input_ids], device=device)
        ref_input_ids = get_embedding(model_wrapper,ref_input_ids)

        attribution, _delta = ig.attribute(input_embedding, baselines=ref_input_ids, internal_batch_size=10,
                                            target = pred_class, n_steps=100, return_convergence_delta=True)
        attribution_l2 = torch.norm(attribution.squeeze(),dim=-1)
        attribution_abs_mean = torch.mean(torch.abs(attribution.squeeze()), dim=-1)
        attribution_mean = torch.mean(attribution.squeeze(), dim=-1)

        # Dot product
        attribution = torch.squeeze(torch.sum(attribution,dim=-1))
        
        attribution_clip = torch.clip(attribution,min=0)
        attribution_clip = attribution_clip / attribution_clip.sum()[...,None]

        # Absolute value of attributions (Abnar and Zuidema, 2020)
        attribution_abs = torch.abs(attribution)
        attribution_abs = attribution_abs / attribution_abs.sum()[...,None]
        attribution = {'abs': attribution_abs, 'clip': attribution_clip, 'l2': attribution_l2, 'mean': attribution_mean, 'abs_mean': attribution_abs_mean}

    elif method == 'grad':
        attribution = saliency.attribute(input_embedding, target=pred_class, abs=False)
        attribution = torch.norm(attribution.squeeze(),dim=-1)
        attribution = attribution / attribution.sum()[...,None]
        
        
        
    elif method == 'grad_input':
        attribution = input_x_gradient.attribute(input_embedding, target=pred_class)
        attribution_l2 = torch.norm(attribution.squeeze(),dim=-1)
        attribution_abs_mean = torch.mean(torch.abs(attribution.squeeze()), dim=-1)
        attribution_mean = torch.mean(attribution.squeeze(), dim=-1)
        attribution = torch.squeeze(torch.sum(attribution,dim=-1))
        
        attribution_clip = torch.clip(attribution,min=0)
        attribution_clip = attribution_clip / attribution_clip.sum()[...,None]

        # Absolute value of attributions (Abnar and Zuidema, 2020)
        attribution_abs = torch.abs(attribution)
        attribution_abs = attribution_abs / attribution_abs.sum()[...,None]
        attribution = {'abs': attribution_abs, 'clip': attribution_clip, 'l2': attribution_l2, 'mean': attribution_mean, 'abs_mean': attribution_abs_mean}

    return attribution

def get_embedding(model_wrapper, input_ids):
    if model_wrapper.model.config.model_type == 'bert':
        input_embedding = model_wrapper.model.bert.embeddings(input_ids)
    elif model_wrapper.model.config.model_type == 'distilbert':
        input_embedding = model_wrapper.model.distilbert.embeddings(input_ids)
    elif model_wrapper.model.config.model_type == 'roberta':
        input_embedding = model_wrapper.model.roberta.embeddings(input_ids)
    return input_embedding