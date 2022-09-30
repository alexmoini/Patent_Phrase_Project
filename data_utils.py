from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer

class PhraseDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
    def process(self, text):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    def __getitem__(self, index):
        anchor = self.data.iloc[index]['anchor']
        target = self.data.iloc[index]['target']
        label = self.data.iloc[index]['label']
        anchor_input_ids, anchor_attention_masks = self.process(anchor)
        target_input_ids, target_attention_masks = self.process(target)
        return anchor_input_ids, anchor_attention_masks, target_input_ids, target_attention_masks, label
    def __len__(self):
        return len(self.data)

