#!/usr/bin/env python

from tqdm import tqdm
import argparse
import pandas as pd
from pathlib import Path

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def main(args):
    file_in = Path(args.input).resolve()
    if getattr(args, 'output', None) is None:
        file_out = file_in.parent / f"{file_in.stem}_{args.tokenizer}{file_in.suffix}"
    else:
        file_out = Path(args.output).resolve()
    
    df_in = pd.read_csv(file_in.as_posix(), sep='\t', header=None)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except OSError:
        raise ValueError(f"Tokenizer '{args.tokenizer}' does not exist in HF")
    df_out = pd.DataFrame(columns=['sentence', 'word', 'third'])

    n_isare_sents = 0
    n_unktok_sents = 0
    unk_tokens = []
    for _, row in tqdm(df_in.iterrows(), total=df_in.shape[0]):
        if row[3] in ['is', 'are']:
            n_isare_sents += 1
            continue
        if row[3] not in tokenizer.vocab:
            n_unktok_sents += 1
            unk_tokens.append(row[3])
            continue
        third_singular = True if row[3] in ['is', 'has'] else len(row[3]) > len(row[4])
        df_out.loc[len(df_out.index)] = [row[2], row[3], third_singular]

    if n_isare_sents > 0:
        print(f"{n_isare_sents}/{len(df_in.index)} is/are sentences were skipped")
    if n_unktok_sents > 0:
        print(
            f"{n_unktok_sents}/{len(df_in.index)} sentences were skipped because the masked word was not in the vocabulary:\n" \
            f"\t{', '.join(set(unk_tokens))}"
        )

    idx_train, idx_valtest = train_test_split(
        df_out.index,
        test_size=0.3,
        random_state=48151623,
        shuffle=True,
        stratify=df_out.third
    )
    idx_val, idx_test = train_test_split(
        idx_valtest,
        test_size=0.5,
        random_state=48151623,
        shuffle=True,
        stratify=df_out.loc[idx_valtest, 'third']
    )
    df_out.loc[idx_train, 'split'] = 'train'
    df_out.loc[idx_val, 'split'] = 'val'
    df_out.loc[idx_test, 'split'] = 'test'

    df_out.to_csv(file_out.as_posix(), sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, help="path to the LGD dataset file",
    )
    parser.add_argument(
        "-o", "--output", nargs='?', type=str, help="path to save the binary LGD dataset",
    )
    parser.add_argument(
        "-t", "--tokenizer", type=str, help="HF tokenizer to use",
    )
    args = parser.parse_args()
    main(args)
