# Measuring the Mixing of Contextual Information in the Transformer

## Abstract
<em>
The Transformer architecture aggregates input information through the attention mechanism, but there is no clear understanding of how this information is mixed across the entire model. Recent works have demonstrated that attention weights alone are not enough to describe the flow of information. In this paper, we consider the whole attention block --multi-head attention, residual connection and layer normalization-- and define a metric to measure token-to-token interactions within each layer, considering the characteristics of the representation space. Then, we aggregate the layer-wise interpretations to provide input attribution scores for model predictions. Experimentally, we show that our method provides faithful explanations and outperforms similar aggregation methods.
</em>

## Installation
Clone this repostitory to `$CONTRIB_ROOT`:
```bash
!git clone https://github.com/javiferran/transformer_contributions.git ${CONTRIB_ROOT}

pip install -r ${CONTRIB_ROOT}/requirements.txt
```

## Usage with Transformers

In our paper we use BERT, DistilBERT and RoBERTa models from Huggingface's [transformers](https://github.com/huggingface/transformers "Huggingface's transformers github") library, but it can be easily extended to other models.

We compare our method with:
- Attention Rollout ([Abnar and Zuidema., 2020](https://arxiv.org/pdf/2005.00928.pdf))
- Attention Rollout + ([Kobayashi et al., 2021](https://arxiv.org/pdf/2109.07152.pdf))
- Gradient Saliency
- Integrated Gradients
- Gradient x Input

We use [Captum](https://captum.ai/) implementation of gradient-based methods.

### Text Classification
To get correlations between different interpretability methods (Table 2), and special tokens ranks (Figure 6):
```bash
python ${CONTRIB_ROOT}/correlations.py \
  -model bert \         # model: bert/distilbert/roberta
  -dataset sst2 \       # dataset to use: sst2/sva
  -samples 500 \        # number of samples
```
To analyze model predictions with the proposed (and others) intepretability methods in SST2 dataset:
```bash
Text_classification.ipynb
```
### Subject-verb Agreement
To analyze model predictions with the proposed (and others) intepretability methods in Subject-Verb Agreement dataset:

```bash
SVA.ipynb
```