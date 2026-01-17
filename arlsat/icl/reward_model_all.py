
import os
import json
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# 1. Dataset
# --------------------------

class ReasonablenessDataset(Dataset):
    """
    Expects a list of dicts:
        {"prompt": str, "response": str, "label": int (0/1)}
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 2048,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def build_text(self, sequence: str) -> str:
        # TODO: replace this with your real chat template if needed
        # e.g. use system/user/assistant. For now: simple prefix.
        # return f"User: {prompt}\nAssistant: {response}"
        return sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.data[idx]
        text = self.build_text(ex['data'])

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(float(ex["label"]), dtype=torch.float),
        }
        return item


# --------------------------
# 2. Data collator
# --------------------------

@dataclass
class RewardDataCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8  # good for tensor cores

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = torch.stack([f["labels"] for f in features])

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


# --------------------------
# 3. Reward model wrapper
# --------------------------

class QwenRewardModel(nn.Module):
    """
    Wrap Qwen CausalLM + small reward head.
    - Base model is typically frozen.
    - Reward head takes pooled last-layer hidden state.
    """

    def __init__(
        self,
        base_model_name: str,
        freeze_base: bool = True,
        hidden_size: Optional[int] = None,
        two_layer_head: bool = True,
        pooling: str = "last",  # "last" or "mean"
        init_from_config_only: bool = False,
    ):
        super().__init__()

        if init_from_config_only:
            # light init: build model from config, NOT from HF weights
            config = AutoConfig.from_pretrained(base_model_name)
            self.base_model = AutoModelForCausalLM.from_config(config)
        else:
            # heavy init: load actual HF weights (used in training)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
            )
        # self.base_model = AutoModelForCausalLM.from_pretrained(
        #     base_model_name,
        #     torch_dtype=torch.bfloat16,
        # )
        base_dtype = next(self.base_model.parameters()).dtype  # bf16
        print("Base model dtype:", base_dtype)
        self.pooling = pooling

        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False

        if hidden_size is None:
            hidden_size = self.base_model.config.hidden_size
            print("Hidden size:", hidden_size)

        if two_layer_head:
            self.reward_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, dtype=base_dtype),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, dtype=base_dtype),
            )
            # h1 = hidden_size
            # h2 = int(1.5 * hidden_size)  # or int(1.5 * hidden_size)
            # h3 = int(hidden_size // 2)  # you can also shrink: e.g., hidden_size // 2

            # self.reward_head = nn.Sequential(
            #     nn.LayerNorm(hidden_size),

            #     nn.Linear(hidden_size, h1, dtype=base_dtype),
            #     nn.GELU(),
            #     nn.Dropout(0.1),

            #     nn.Linear(h1, h2, dtype=base_dtype),
            #     nn.GELU(),
            #     nn.Dropout(0.1),

            #     nn.Linear(h2, h3, dtype=base_dtype),
            #     nn.GELU(),
            #     nn.Dropout(0.1),

            #     nn.Linear(h3, 1, dtype=base_dtype),
            # )
        else:
            print("Using single-layer reward head.")
            self.reward_head = nn.Sequential(
                nn.LayerNorm(hidden_size, dtype=base_dtype),
                nn.Linear(hidden_size, 1, dtype=base_dtype)
            )
            

        self.loss_fn = nn.BCEWithLogitsLoss()

    def pool_hidden(self, last_hidden: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """
        last_hidden: [B, T, d]
        attention_mask: [B, T]
        Returns pooled: [B, d]
        """
        if self.pooling == "last":
            # Last non-pad token per sequence
            seq_lens = attention_mask.sum(dim=1) - 1  # [B]
            batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            pooled = last_hidden[batch_idx, seq_lens]  # [B, d]
        elif self.pooling == "mean":
            # Mean over all non-pad tokens
            mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)  # [B, d]
            lengths = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            pooled = summed / lengths
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, T, d]

        pooled = self.pool_hidden(last_hidden, attention_mask)  # [B, d]
        logits = self.reward_head(pooled).squeeze(-1)           # [B]

        loss = None
        if labels is not None:
            # labels: [B] with 0/1
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }
    @classmethod
    def from_checkpoint(cls, ckpt_dir: str, base_model_name: str, **kwargs):
        """
        Load reward model from a Trainer checkpoint dir
        without loading HF base weights first.
        """
        # 1) build architecture from config only (random weights)
        model = cls(
            base_model_name=base_model_name,
            init_from_config_only=True,   # <- no HF weight load
            **kwargs,
        )

        # 2) load checkpoint weights (base + head)
        ckpt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)

        return model


# qwen embedding reward model
import torch
import torch.nn as nn

class QwenEmbeddingRewardModel(nn.Module):
    def __init__(self, emb_weight_path: str,
                 pooling: str = "last",
                 two_layer_head: bool = True,
                 head_dtype=torch.bfloat16):
        super().__init__()

        # Load pre-saved embedding
        weight = torch.load(emb_weight_path, map_location="cpu")
        vocab_size, hidden_size = weight.shape

        # Keep embedding in bf16 (or fp16) to save memory
        weight = weight.to(torch.bfloat16)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            _weight=weight,
        )

        # We usually freeze it
        self.embedding.weight.requires_grad_(False)

        self.pooling = pooling
        self.hidden_size = hidden_size

        # Small classifier head; we cast inputs to head_dtype for stability
        if two_layer_head:
            self.reward_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, dtype=head_dtype),
                nn.Tanh(),
                nn.Linear(hidden_size, 256, dtype=head_dtype),
                nn.Tanh(),
                nn.Linear(256, 1, dtype=head_dtype),
            )
        else:
            self.reward_head = nn.Linear(hidden_size, 1, dtype=head_dtype)
        
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _pool(self, token_emb, attention_mask):
        # token_emb: [B, L, H] (float32)
        # attention_mask: [B, L]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(token_emb.dtype)  # [B, L, 1]
            summed = (token_emb * mask).sum(dim=1)                   # [B, H]
            lengths = mask.sum(dim=1).clamp(min=1.0)                 # [B, 1]
            return summed / lengths
        elif self.pooling == "last":
            # last non-pad token
            idx = attention_mask.sum(dim=1) - 1                      # [B]
            idx = idx.clamp(min=0)
            return token_emb[torch.arange(token_emb.size(0)), idx]   # [B, H]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, input_ids, attention_mask, labels):
        # input_ids: [B, L], attention_mask: [B, L]
        # Embedding in bf16, no grad to save memory/compute
        with torch.no_grad():
            token_emb = self.embedding(input_ids)  # [B, L, H], bf16

        # Cast to float32 for head computations
        token_emb = token_emb.float()

        seq_emb = self._pool(token_emb, attention_mask)  # [B, H]

        # head is the only trainable part
        logits = self.reward_head(seq_emb).squeeze(-1)          # [B]

        loss = None
        if labels is not None:
            # labels: [B] with 0/1
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }


# reward model with backbone of qwen3
    
class Qwen3backbone_RewardModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Small MLP head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),  # binary logit
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels: Optional[torch.Tensor] = None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]

        # Masked mean-pooling over sequence length
        mask = attention_mask.unsqueeze(-1).float()      # [B, T, 1]
        summed = (last_hidden * mask).sum(dim=1)         # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)          # [B, 1]
        pooled = summed / denom                          # [B, H]

        logits = self.reward_head(pooled).squeeze(-1)     # [B]

        loss = None
        if labels is not None:
            # labels: [B] with 0/1
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }

# --------------------------
# 4. Metrics
# --------------------------

def compute_metrics(eval_pred):
    import numpy as np

    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int32)
    labels = labels.astype(np.int32)

    acc = (preds == labels).mean()
    return {"accuracy": acc}



def load_data(args):
    with open(args.trainset_path, "r") as f:
        train_data = json.load(f)

    with open(args.testset_path, "r") as f:
        test_data = json.load(f)
    
    print(f'Number of training samples: {len(train_data)}')
    print(f'Number of testing samples: {len(test_data)}')
    return train_data, test_data

def load_data_response(args):
    with open("data/train/trainset_res_50.json", "r") as f:
        train_data = json.load(f)

    with open("data/test/testset_res_100.json", "r") as f:
        test_data = json.load(f)
    
    print(f'Number of training samples: {len(train_data)}')
    print(f'Number of testing samples: {len(test_data)}')
    return train_data, test_data



def get_hidden_states(args):
    train_data, _ = load_data()
    base_model_name = "Qwen/Qwen3-4B"  # or your qwen3_4b path

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()


    # ------------------------------------------------
    # 5. Build dataloader (NO shuffle, so order = dataset index)
    # ------------------------------------------------
    dataset = ReasonablenessDataset(train_data, tokenizer, max_length=args.max_length)
    datacollator = RewardDataCollator(tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=datacollator,
    )

    print(len(loader), "batches in total.")

    # get hidden states of all, last token, mean
    all_embs = []
    last_embs = []
    mean_embs = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f'Processing batch {i+1}\n')
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            last_hidden = outputs.hidden_states[-1]  # [B, T, d]

            # all embs
            pooled = last_hidden
            # all_embs.append(pooled.cpu().float())

            # Last non-pad token per sequence
            seq_lens = attention_mask.sum(dim=1) - 1  # [B]
            bsz = input_ids.size(0)
            idxs = torch.arange(bsz, device=device)
            pooled = last_hidden[idxs, seq_lens]  # [B, d]
            last_embs.append(pooled.cpu().float())
            print(f'Last token embeddings: {pooled}')
            print(f'Last token embeddings shape: {pooled.shape}')
            
            # mean embs
            pooled = last_hidden.mean(dim=1)
            mean_embs.append(pooled.cpu().float())  
                

    # [N, d]
    # all_embs = torch.cat(all_embs, dim=0)
    last_embs = torch.cat(last_embs, dim=0)
    mean_embs = torch.cat(mean_embs, dim=0)
    print("Saved embeddings shape:", last_embs.shape)

    # torch.save(all_embs, "qwen3_4b/train_last_hidden_all.pt")
    torch.save(mean_embs, "data/qwen3_4b/train_last_hidden_mean.pt")
    torch.save(last_embs, "data/qwen3_4b/train_last_hidden_last.pt")
    print("train_last_hidden.pt written")






# --------------------------
# 5. Main
# --------------------------

def train(args):
    base_model_name = "Qwen/Qwen3-4B"  # or your qwen3_4b path
    base_model_name = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"
    output_dir = args.output_dir

    # Dummy example: replace with your actual list of dicts
    # Each example: {"prompt": ..., "response": ..., "label": 0/1}
    train_data = [
        {"prompt": "problem 1", "response": "CoT + answer 1", "label": 1},
        {"prompt": "problem 2", "response": "CoT + answer 2", "label": 0},
        # ...
    ]
    val_data = [
        {"prompt": "val problem 1", "response": "val CoT + answer 1", "label": 1},
        # ...
    ]

    train_data, val_data = load_data(args)
    print(f'Loading trainset from {args.trainset_path}\ntestset from {args.testset_path}')

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Make sure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ReasonablenessDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = ReasonablenessDataset(val_data, tokenizer, max_length=args.max_length)

    data_collator = RewardDataCollator(tokenizer=tokenizer)

    if args.model == 'hidden':
        model = QwenRewardModel(
            base_model_name=base_model_name,
            freeze_base=True,           # only train head
            pooling=args.hidden_pooling,             # or "mean"
            two_layer_head=False,
        )
    elif args.model == 'embed':
        embed_path = 'qwen3_4b_embed_weight.pt'
        model = QwenEmbeddingRewardModel(
            emb_weight_path=embed_path, 
            pooling="mean",
            two_layer_head=True,
            )
    elif args.model == 'backbone':
        model = Qwen3backbone_RewardModel(
            model_name=base_model_name,
        )
    
    print(f'Model initialized: {args.model}')
    print(f'base model: {base_model_name}')



    # Let Trainer move model to device(s)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        logging_steps=1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        # save_total_limit=2,
        load_best_model_at_end=True,
        save_safetensors=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,
        bf16=True,                  # if you have A100 / similar
        gradient_accumulation_steps=2,
        report_to=['wandb'],               # or ["wandb"]
        ddp_find_unused_parameters=False,  # safer for custom models
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save full model (base + head)
    trainer.save_model(output_dir)

    # Also save only the classifier head
    head_path = os.path.join(output_dir, "reward_head.pt")
    torch.save(model.reward_head.state_dict(), head_path)
    print(f"Saved reward head to {head_path}")



def evaluate(model):
    """
    Reward model takes prompt and response as input.
    Or it tasks pooled hidden states / embeddings as input.
    """
    base_model_name = "Qwen/Qwen3-4B"
    reward_model = model(
            base_model_name=base_model_name,
            freeze_base=True,           # only train head
            pooling="last",             # or "mean"
            two_layer_head=True,
        )
    reward_state_dict = torch.load("checkpoints/reward_model_hidden/reward_head.pt")
    reward_model.reward_head.load_state_dict(reward_state_dict)
    reward_model.eval()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="model name or path")
    parser.add_argument("--model", type=str, default="hidden", choices=["hidden", "embed", "backbone"], help="type of reward model")
    
    parser.add_argument("--trainset_path", type=str, default="data/train/rm_trainset.json", help="path to training dataset")
    parser.add_argument("--testset_path", type=str, default="data/test/rm_testset.json", help="path to testing dataset")

    parser.add_argument("--hidden_pooling", type=str, default="last", choices=["last", "mean"], help="pooling method for hidden state reward model")
    parser.add_argument("--data_type", type=str, default="full", help="type of data to use")
    
    parser.add_argument("--output_dir", type=str, default="checkpoints/reward_model_hidden_state", help="output directory")
    parser.add_argument("--logging_dir", type=str, default="logs/rm_hidden_embed", help="logging directory")

    parser.add_argument("--max_length", type=int, default=4096, help="max token length")
    parser.add_argument("--train_batch_size", type=int, default=8, help="train batch size")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="per device train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="per device eval batch size")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")

    # parser.add_argument("--project_name", type=str, default="reward_model", help="wandb project name")
    # parser.add_argument("--experiment_name", type=str, default="reward_model_hidden_state", help="wandb experiment name")

    args = parser.parse_args()



    train(args)
    # get_hidden_states(args)
