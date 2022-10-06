import torch
from transformers import AutoModel
import torch.nn.functional as F
import numpy as np

class SimilarityModel(torch.nn.Module):
    def __init__(self, model_name):
        super(SimilarityModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask):
        # encode input sequences
        encoded_anchor = self.encoder(input_ids=anchor_input_ids, attention_mask=anchor_attention_mask)
        encoded_target = self.encoder(input_ids=target_input_ids, attention_mask=target_attention_mask)
        # mean pooling
        anchor_pooled_output = self.mean_pooling(encoded_anchor, anchor_attention_mask)
        target_pooled_output = self.mean_pooling(encoded_target, target_attention_mask)
        # normalize
        anchor_pooled_output = F.normalize(anchor_pooled_output, p=2, dim=1)
        # cosine similarity
        return torch.cosine_similarity(anchor_pooled_output, target_pooled_output, dim=1)
    def predict(self, anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask):
        self.eval()
        with torch.no_grad():
            return self.forward(anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask)

def train(model, tokenizer, train_dataloader, valid_dataloader, device, optimizer, loss_function, epochs):
    train_loss = []
    valid_loss = []
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for anchor_batch, target_batch, label_batch in train_dataloader:

            anchor = tokenizer(list(anchor_batch), padding=True, truncation=True, return_tensors="pt")
            target = tokenizer(list(target_batch), padding=True, truncation=True, return_tensors="pt")
            anchor_input_ids = anchor['input_ids'].to(device)
            anchor_attention_mask = anchor['attention_mask'].to(device)
            target_input_ids = target['input_ids'].to(device)
            target_attention_mask = target['attention_mask'].to(device)
            label = torch.Tensor(label_batch).to(device)
            optimizer.zero_grad()
            output = model(anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask)
            loss = loss_function(output, label)
            train_loss.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Average Train Loss: {np.mean(train_loss)}")
        model.eval()
        with torch.no_grad():
            for anchor_batch, target_batch, label_batch in valid_dataloader:
                anchor = tokenizer(list(anchor_batch), padding=True, truncation=True, return_tensors="pt")
                target = tokenizer(list(target_batch), padding=True, truncation=True, return_tensors="pt")
                anchor_input_ids = anchor['input_ids'].to(device)
                anchor_attention_mask = anchor['attention_mask'].to(device)
                target_input_ids = target['input_ids'].to(device)
                target_attention_mask = target['attention_mask'].to(device)
                label = torch.Tensor(label_batch).to(device)
                output = model(anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask)
                loss = loss_function(output, label)
                valid_loss.append(loss.detach().cpu().numpy())
        print(f'Average Epoch Valid Loss: {np.mean(valid_loss)}')
    return train_loss, valid_loss, model