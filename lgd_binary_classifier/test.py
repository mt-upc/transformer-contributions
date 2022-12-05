#!/usr/bin/env python

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


def main(args):
    print(args)

    data_path = Path(args.data_path).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()
    bert_name = ckpt_path.stem.split('_')[1]
    assert data_path.exists(), f"{data_path} does not exist"
    assert bert_name == data_path.stem.split('_')[-1], \
        "The BERT model is not compatible with the dataset"

    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    df = pd.read_csv(data_path.as_posix(), sep='\t', index_col=[0])
    test_dataset = LGDBinaryDataset(
        df[df.split == 'test'].drop('split', 1).reset_index(drop=True),
        tokenizer
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.bsz,
        num_workers=args.num_workers,
        collate_fn=partial(collate, tokenizer),
        shuffle=True,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForMaskedLM.from_pretrained(bert_name)
    model.cls.predictions.decoder = nn.Linear(
        model.cls.predictions.transform.dense.out_features, 1
    )
    model.load_state_dict(torch.load(ckpt_path.as_posix()))
    model.to(device)

    criterion = F.binary_cross_entropy_with_logits

    metrics = MetricsLogger(
        buffer_names=['main'],
        loss_fn=criterion
    )

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            model_output = model(**batch['model_input'])
            mask_pos = (batch['model_input']['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            logits = torch.diagonal(model_output['logits'].index_select(1, mask_pos)).squeeze(0)
            metrics.add(logits, batch['target'])


    m_test = metrics.get_avg('main')
    print(
        f"TESTING RESULTS\n"
        f"- Loss:\t\t{round(m_test['loss'], 2)}\n"
        f"- Accuracy:\t{round(m_test['accuracy'], 2)}\n"
        f"- F1 Score:\t{round(m_test['f1'], 2)}\n"
    )
    metrics.reset('main')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="path of the dataframe containing data",
    )
    parser.add_argument(
        "--ckpt-path", type=str, required=True,
        help="path of the model checkpoint",
    )
    parser.add_argument(
        "--bsz", type=int, default=6,      # 16, 32
        help="batch size",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="number of workers for data loading",
    )
    args = parser.parse_args()
    main(args)