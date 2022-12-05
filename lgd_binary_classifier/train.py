#!/usr/bin/env python

import os
import argparse
import pandas as pd
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForMaskedLM

from dataset import LGDBinaryDataset, collate
from metrics import MetricsLogger


def train_step(model, batch, tokenizer, opt, criterion, metrics, device):
    opt.zero_grad()
    batch = {k: v.to(device) for k, v in batch.items()}
    model_output = model(**batch['model_input'])
    mask_pos = (batch['model_input']['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    logits = torch.diagonal(model_output['logits'].index_select(1, mask_pos)).squeeze(0)
    loss = criterion(logits, batch['target'].float())
    loss.backward()
    opt.step()
    metrics.add(logits, batch['target'])


def validate(model, dataloader, tokenizer, metrics, device):
    model.eval()
    with torch.no_grad():
        for val_batch in dataloader:
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            model_output = model(**val_batch['model_input'])
            mask_pos = (val_batch['model_input']['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            logits = torch.diagonal(model_output['logits'].index_select(1, mask_pos)).squeeze(0)
            metrics.add(logits, val_batch['target'])
    model.train()


def print_metrics(metrics, subset):
    m = metrics.get_avg('main')
    print(
        f"SUBSET: {subset}\n"
        f"- Loss:\t\t{round(m['loss'], 2)}\n"
        f"- Accuracy:\t{round(m['accuracy'], 2)}\n"
        f"- F1 Score:\t{round(m['f1'], 2)}\n"
    )


def main(args):
    print(args)

    data_path = Path(args.data_path).resolve()
    assert data_path.exists(), f"{data_path} does not exist"
    assert args.bert_name == data_path.stem.split('_')[-1], \
        "The BERT model is not compatible with the dataset"

    save_dir = Path(args.save_dir).resolve()
    os.makedirs(save_dir.as_posix(), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    df = pd.read_csv(data_path.as_posix(), sep='\t', index_col=[0])

    train_dataset = LGDBinaryDataset(
        df[df.split == 'train'].drop('split', 1).reset_index(drop=True),
        tokenizer,
    )
    val_dataset = LGDBinaryDataset(
        df[df.split == 'val'].drop('split', 1).reset_index(drop=True),
        tokenizer,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.bsz,
        num_workers=args.num_workers,
        collate_fn=partial(collate, tokenizer),
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.bsz,
        num_workers=args.num_workers,
        collate_fn=partial(collate, tokenizer),
        shuffle=True,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForMaskedLM.from_pretrained(args.bert_name)
    if args.bert_name == 'bert-base-uncased':
        model.cls.predictions.decoder = nn.Linear(
            model.cls.predictions.decoder.in_features, 1
        )
    elif args.bert_name == 'distilbert-base-uncased':
        model.vocab_projector = nn.Linear(
            model.vocab_projector.in_features, 1
        )
    elif args.bert_name == 'roberta-base':
        model.lm_head.decoder = nn.Linear(
            model.lm_head.decoder.in_features, 1
        )
    model.to(device)

    criterion = F.binary_cross_entropy_with_logits
    opt = optim.Adam(model.parameters(), lr=args.lr)
    train_metrics = MetricsLogger(
        buffer_names=['main'],
        loss_fn=criterion
    )
    val_metrics = MetricsLogger(
        buffer_names=['main'],
        loss_fn=criterion
    )

    best_loss = float('inf')
    for ep in range(args.epochs):
        for i, batch in enumerate(train_dataloader):
            if ((i + 1) % args.log_interval == 0) or ((i + 1) % args.save_interval == 0):
                validate(model, val_dataloader, tokenizer, val_metrics, device)
                print(f"Epoch {ep+1}\tUpdate {i+1}\n")
                print_metrics(train_metrics, 'train')
                print_metrics(val_metrics, 'val')
                val_loss = val_metrics.get_avg('main')['loss']
                val_metrics.reset('main')
                ckpt_file = save_dir / f"checkpoint_{args.bert_name}_{ep+1}_{i+1}_{round(val_loss, 2)}.pt"
                if val_loss <= best_loss:
                    best_loss = val_loss
                    best_ckpt = ckpt_file.name

                if (i % args.save_interval == 0) or (val_loss <= best_loss):
                    torch.save(model.state_dict(), ckpt_file)
                    print(f"Saved checkpoint to {ckpt_file}\n")

            train_step(model, batch, tokenizer, opt, criterion, train_metrics, device)

        validate(model, val_dataloader, tokenizer, val_metrics, device)
        print(f"Epoch {ep+1}\tUpdate {i+1}\n")
        print_metrics(train_metrics, 'train')
        print_metrics(val_metrics, 'val')
        val_loss = val_metrics.get_avg('main')['loss']
        val_metrics.reset('main')
        ckpt_file = save_dir / f"checkpoint_{args.bert_name}_{ep+1}_{i+1}_{round(val_loss, 2)}.pt"
        if val_loss < best_loss:
            best_loss = val_loss
            best_ckpt = ckpt_file.name
        torch.save(model.state_dict(), ckpt_file)
        print(f"Saved checkpoint to {ckpt_file}\n")

    print(
        f"END OF TRAINING\n"
        f"Best checkpoint: {best_ckpt}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bert-name", type=str, required=True,
        help="name of the BERT model from HF to use",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        default="/home/usuaris/veu/javier.ferrando/emnlp-transformer-contributions/data/lgd_dataset.tsv",
        help="path of the dataframe containing data",
    )
    parser.add_argument(
        "--save-dir", type=str, required=True,
        help="directory to save the checkpoints",
    )
    parser.add_argument(
        "--bsz", type=int, default=32,      # 16, 32
        help="batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,   # 5e-5, 3e-5, 2e-5
        help="learning rate",
    )
    parser.add_argument(
        "--epochs", type=int, default=4,    # 2, 3, 4
        help="number of epochs",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--log-interval", type=int, default=100,
        help="logging interval during training",
    )
    parser.add_argument(
        "--save-interval", type=int, default=200,
        help="checkpoint saving interval",
    )
    args = parser.parse_args()
    main(args)
