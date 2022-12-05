import os
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import json
import random
random.seed(10)
from dotenv import load_dotenv
load_dotenv()

from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from pathlib import Path



from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import numpy as np
from datasets import load_metric

def main(args):
    models_dir = Path(os.environ['MODELS_DIR']).as_posix()

    model_path = args.model_path
    #model_name = f'multiberts-seed_{args.bert_seed}'
    model_name = model_path.replace('/','_').split('-')[0]
    
    dataset_name = args.dataset
    #model = BertModel.from_pretrained(f"google/{model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    #tokenizer = BertTokenizer.from_pretrained(f'google/{model_name}')

    metric = load_metric("accuracy")
    metric_name = "accuracy"
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    if dataset_name == 'sst2':
        dataset = load_dataset('glue', dataset_name)
    elif dataset_name == 'yelp':
        dataset = load_dataset("yelp_polarity")

    
    def tokenize_function(examples, text_feature):
        return tokenizer(examples[text_feature], padding="max_length", truncation=True, max_length=256)

    column_name = list(dataset['train'].features.keys())[0]
    tokenized_datasets = dataset.map(tokenize_function, fn_kwargs = {'text_feature' : column_name}, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]#.shuffle(seed=42)#.select(range(1000))
    eval_dataset = tokenized_datasets["test"]#.shuffle(seed=42)#.select(range(1000))

    print('Traning: ', model_name)
    print('Batch size: ', args.bsz)
    print('lr: ', args.lr)

    args = TrainingArguments(
        f"{models_dir}/{model_path}-finetuned-{dataset_name}",
        adafactor=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        label_smoothing_factor=0.0,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        warmup_ratio=0.0,
        warmup_steps=0,
        weight_decay=0.0,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    del model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, 
        help="BERT seed from MultiBerts HF",
    )
    parser.add_argument(
        "--bsz", type=int, default=16,      # 16, 32
        help="batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,   # 5e-5, 3e-5, 2e-5
        help="learning rate",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,    # 2, 3, 4
        help="number of epochs",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="number of workers for data loading",
    )
    parser.add_argument(
         '--dataset', help="sst2/yelp/imdb/sva", choices=['sst2', 'sva', 'imdb', 'yelp'], type=str
         )
    args = parser.parse_args()
    main(args)