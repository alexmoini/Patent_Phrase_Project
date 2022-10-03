from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
class PhraseDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path, header=0)
    def __getitem__(self, index):
        anchor = self.data.iloc[index]['anchor']
        target = self.data.iloc[index]['target']
        label = self.data.iloc[index]['score']
        label = np.float32(label)
        return anchor, target, label
    def __len__(self):
        return len(self.data)