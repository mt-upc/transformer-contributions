# Measuring the Mixing of Contextual Information in the Transformer

## Installation

```bash
!git clone https://github.com/javiferran/transformer_contributions.git

import os
os.chdir(f'./transformer_contributions')

!pip install -r requirements.txt
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
```bash
Text_classification.ipynb
```
### Subject-verb Agreement
```bash
SV.ipynb
```