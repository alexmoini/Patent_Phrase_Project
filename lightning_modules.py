"""
Create a lightning module for training similarity pairs
"""

import pytorch_lightning as pl
import torch

class SimilarityModule(pl.LightningModule):
    def __init__(self, model, hyperparameters):
        super().__init__()
        self.model = model
        self.hyperparameters = hyperparameters
        self.loss = hyperparameters['loss']
        self.optimizer = hyperparameters['optimizer']
    def forward(self, target_input_ids, target_attention_mask, anchor_input_ids, anchor_attention_mask):
        anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)
        target_embeddings = self.model(target_input_ids, target_attention_mask)
        anchor_embeddings = self.__mean_pooling(anchor_embeddings, anchor_attention_mask)
        target_embeddings = self.__mean_pooling(target_embeddings, target_attention_mask)
        logits = torch.cosine_similarity(anchor_embeddings, target_embeddings, dim=1)
        return logits
    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def training_step(self, batch, batch_idx):
        anchor_tokens, target_tokens, labels = batch
        logits = self.forward(target_tokens['input_ids'],
                              target_tokens['attention_mask'],
                              anchor_tokens['input_ids'],
                              anchor_tokens['attention_mask'])
        loss = self.loss(logits, labels)
        self.log('train_loss', loss.item())
        return loss
    def validation_step(self, batch, batch_idx):
        anchor_tokens, target_tokens, labels = batch
        logits = self.forward(target_tokens['input_ids'],
                              target_tokens['attention_mask'],
                              anchor_tokens['input_ids'],
                              anchor_tokens['attention_mask'])
        loss = self.loss(logits, labels)
        self.log('val_loss', loss.item())
        return loss
    def configure_optimizers(self):
        if self.hyperparameters['scheduler'] is not None:
            return [self.optimizer], [self.scheduler]
        return torch.optim.Adam(self.parameters(), self.hyperparameters['lr'])
