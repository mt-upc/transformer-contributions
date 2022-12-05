import torch

class LGDBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        assert df.columns.tolist() == ['sentence', 'word', 'third']
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        item = self.df.loc[i].to_dict()
        item['sentence'] = item['sentence'].replace('***mask***', self.tokenizer.mask_token)
        item['third'] = int(item['third'])
        return item

def collate(tokenizer, items):
    model_input = tokenizer([it['sentence'] for it in items], padding=True, return_tensors='pt')
    return {
        'model_input': model_input,
        'target': torch.tensor([it['third'] for it in items]),
    }
