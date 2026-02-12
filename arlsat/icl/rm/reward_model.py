"""
This is code for reward model training on AR_LSAT dataset.
A classifier to distinguish if the given response is reasonable or not.

Currently built on top of frozen Qwen3_4b model.
"""


import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import random
import numpy as np
import json
import os
from tqdm.auto import tqdm


random.seed(224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ResponseDataset(Dataset):
    """
    responses: list[str]
    labels:    list[int]  (0 = unreasonable, 1 = reasonable)
    """
    def __init__(self, responses, labels, tokenizer, max_len=512):
        self.responses = responses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        text = self.responses[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",   # we’ll pad in batch; this is simplest
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }
        return item


class RewardModel(nn.Module):
    def __init__(self, model_name: str):
        super(RewardModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Small MLP head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),  # binary logit
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state  # [B, T, H]

        # Masked mean-pooling over sequence length
        mask = attention_mask.unsqueeze(-1).float()      # [B, T, 1]
        summed = (last_hidden * mask).sum(dim=1)         # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)          # [B, 1]
        pooled = summed / denom                          # [B, H]

        logits = self.classifier(pooled).squeeze(-1)     # [B]
        return logits



def train(
    model_name,
    train_responses,
    train_labels,
    val_responses=None,
    val_labels=None,
    epochs=10,
    batch_size=16,
    lr=1e-5,
    max_len=4096,
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RewardModel(model_name).to(DEVICE)

    train_dataset = ResponseDataset(train_responses, train_labels, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_responses is not None and val_labels is not None:
        val_dataset = ResponseDataset(val_responses, val_labels, tokenizer, max_len)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)  # float (0 or 1)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

            print(f'Batch loss: {loss.item():.4f}')

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                total_correct += (preds == labels.long()).sum().item()
                total_examples += input_ids.size(0)

        avg_loss = total_loss / total_examples
        train_acc = total_correct / total_examples

        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, train_acc={train_acc:.4f}")

        # Simple validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_examples = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    labels = batch["label"].to(DEVICE)

                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * input_ids.size(0)

                    preds = (torch.sigmoid(logits) > 0.5).long()
                    val_correct += (preds == labels.long()).sum().item()
                    val_examples += input_ids.size(0)

            print(
                f"  val_loss={val_loss/val_examples:.4f}, "
                f"val_acc={val_correct/val_examples:.4f}"
            )

        if epoch % 5 == 0:
            head_dir = f"checkpoints/reasonable_clf_head_{epoch+1}"
            os.makedirs(head_dir, exist_ok=True)
            torch.save(model.classifier.state_dict(), os.path.join(head_dir, "classifier_head.bin"))
    head_dir = "checkpoints/reasonable_clf_head"
    os.makedirs(head_dir, exist_ok=True)
    torch.save(model.classifier.state_dict(), os.path.join(head_dir, "classifier_head.bin"))

    return model, tokenizer    



def evaluate(
        responses,
        labels,
        batch_size=16,
        max_len=4096,
        ):
    
    print('Evaluating model...')
    base_model_name = "Qwen/Qwen3-4B"
    model = RewardModel(base_model_name)
    clf_state_dict = torch.load("checkpoints/reasonable_clf_head/classifier_head.bin")
    model.classifier.load_state_dict(clf_state_dict)
    model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    test_dataset = ResponseDataset(responses, labels, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    val_loss = 0.0
    val_correct = 0
    val_examples = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            print(f'logits: {logits}\n')
            preds = (torch.sigmoid(logits) > 0.5).long()
            print(f'preds: {preds}\n')
            labels = labels.long()
            print(f'labels: {labels}\n')
            val_correct += (preds == labels.long()).sum().item()
            val_examples += input_ids.size(0)

            print(
                f"test_acc={val_correct/val_examples:.4f}"
            )
        
        print(f"test_acc={val_correct/val_examples:.4f}")


    






def main():
#     with open('data/train/trainset.json', 'r') as f:
#         train_data = json.load(f)
    
#     train_responses = train_data['data']
#     train_labels = train_data['label']

#     train(model_name="Qwen/Qwen3-4B", 
#           train_responses=train_responses, 
#           train_labels=train_labels, 
#           epochs=10, 
#           batch_size=8, 
#           lr=1e-5, 
#           max_len=4096)
    
    with open('data/test/testset_100.json', 'r') as f:
        test_data = json.load(f)
    
    responses = [item['data'] for item in test_data]
    labels = [item['label'] for item in test_data]

    evaluate(
        responses,
        labels,
        batch_size=8,
        max_len=4096,
    )

if __name__ == "__main__":
    main()




