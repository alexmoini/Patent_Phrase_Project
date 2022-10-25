from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

class PhraseDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.data = pd.read_csv(path, header=0)
        self.tokenizer = tokenizer
    def __getitem__(self, index):
        anchor = self.data.iloc[index]['anchor']
        target = self.data.iloc[index]['target']
        label = self.data.iloc[index]['score']
        label = np.float32(label)
        target_tokens = self.tokenizer(target, padding='max_length', max_length=10,
                                       truncation=True, return_tensors="pt")
        anchor_tokens = self.tokenizer(anchor, padding='max_length', max_length=10,
                                       truncation=True, return_tensors="pt")
        for key in target_tokens.keys():
            target_tokens[key] = target_tokens[key].squeeze(0)
        for key in anchor_tokens.keys():
            anchor_tokens[key] = anchor_tokens[key].squeeze(0)
        
        return anchor_tokens, target_tokens, label
    def __len__(self):
        return len(self.data)