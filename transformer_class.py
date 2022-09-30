import torch
from transformers import AutoModel

class SimilarityModel(torch.nn.Module):
    def __init__(self, model_name):
        super(SimilarityModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask):
        _anchor, anchor_pooled_output = self.bert(input_ids=anchor_input_ids, attention_mask=anchor_attention_mask)
        _target, target_pooled_output = self.bert(input_ids=target_input_ids, attention_mask=target_attention_mask)
        return torch.cosine_similarity(anchor_pooled_output, target_pooled_output, dim=1)
    def predict(self, anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask):
        self.eval()
        with torch.no_grad():
            return self.forward(anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask)

def train(model, train_dataloader, valid_dataloader, device, optimizer, loss_function, epochs):
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        model.train()
        for anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask, label in train_dataloader:
            anchor_input_ids = anchor_input_ids.to(device)
            anchor_attention_mask = anchor_attention_mask.to(device)
            target_input_ids = target_input_ids.to(device)
            target_attention_mask = target_attention_mask.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask)
            loss = loss_function(output, label)
            train_loss.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Train Loss: {loss.detach().cpu().numpy()}")
        model.eval()
        with torch.no_grad():
            for anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask, label in valid_dataloader:
                anchor_input_ids = anchor_input_ids.to(device)
                anchor_attention_mask = anchor_attention_mask.to(device)
                target_input_ids = target_input_ids.to(device)
                target_attention_mask = target_attention_mask.to(device)
                label = label.to(device)
                output = model(anchor_input_ids, anchor_attention_mask, target_input_ids, target_attention_mask)
                loss = loss_function(output, label)
                valid_loss.append(loss.detach().cpu().numpy())
                print(f'Valid Loss: {loss.detach().cpu().numpy()}')