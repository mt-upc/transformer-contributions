# Measuring the Mixing of Contextual Information in the Transformer

### Implementation of the paper [Measuring the Mixing of Contextual Information in the Transformer](https://arxiv.org/pdf/2203.04212.pdf)
## Abstract
<p>
<em>
The Transformer architecture aggregates input information through the self-attention mechanism, but there is no clear understanding of how this information is mixed across the entire model. Additionally, recent works have demonstrated that attention weights alone are not enough to describe the flow of information. In this paper, we consider the whole attention block --multi-head attention, residual connection, and layer normalization-- and define a metric to measure token-to-token interactions within each layer, considering the characteristics of the representation space. Then, we aggregate layer-wise interpretations to provide input attribution scores for model predictions. Experimentally, we show that our method, ALTI (Aggregation of Layer-wise Token-to-token Interactions), provides faithful explanations and outperforms similar aggregation methods.
</em>
</p>

<p align="center"><br>
<img src="./img/layers_relevances_example_bert.png" class="center" title="paper logo" width="800"/>
</p><br>

## Environment Setup

Clone this repostitory:
```bash
!git clone https://github.com/javiferran/transformer_contributions.git
```

Create a conda environment using the `environment.yml` file, and activate it:
```bash
conda env create -f ./environment.yml && \
conda activate alti
```

## Usage with Transformers

In our paper we use BERT, DistilBERT and RoBERTa models from Huggingface's [transformers](https://github.com/huggingface/transformers "Huggingface's transformers github") library, but it can be easily extended to other models.

We compare our method with:
- Attention Rollout ([Abnar and Zuidema., 2020](https://arxiv.org/pdf/2005.00928.pdf))
- Attention Rollout + ([Kobayashi et al., 2021](https://arxiv.org/pdf/2109.07152.pdf))
- Gradient-based methods: Gradient Saliency, Integrated Gradients and Gradient x Input

We use [Captum](https://captum.ai/) implementation of gradient-based methods.

First, get the attributions obtained by each method for the specified model and dataset:
```bash
python ./attributions.py \
  -model bert \         # model: bert/distilbert/roberta
  -dataset sst2 \       # dataset to use: sst2/sva
  -samples 500 \        # number of samples
```
To reproduce Table 2, Figure 6 and 7, run the following command:

```bash
python ./correlations.py \
  -model bert \         # model: bert/distilbert/roberta
  -dataset sst2 \       # dataset to use: sst2/sva
```
### Text Classification
To analyze model predictions with the proposed (and others) intepretability methods in SST2 dataset:
```bash
Text_classification.ipynb
```
### Subject-verb Agreement
To analyze model predictions with the proposed (and others) intepretability methods in Subject-Verb Agreement dataset:
```bash
SVA.ipynb
```
## Citation
If you use ALTI in your work, please consider citing:

```bibtex
@misc{alti,
  title = {Measuring the Mixing of Contextual Information in the Transformer},
  author = {Ferrando, Javier and G??llego, Gerard I. and Costa-juss??, Marta R.},
  doi = {10.48550/ARXIV.2203.04212},
  url = {https://arxiv.org/abs/2203.04212},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

````
