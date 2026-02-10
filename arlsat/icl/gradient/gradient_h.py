"""
The script is to get gradients with respect to hidden states.

First, get the most critical layer of hidden states related to semantic meaning
- get the gap of cos similarity of each layer to next
--> the most effective layers are [35, 22, 23, 21, 24] for tokenwise cos similarity
--> pooling cos similarity:
reasonable set: [35, 22, 23, 21, 24]
unreasonable set: [35, 4, 22, 24, 5] 

Second, extract gradients with respect to hidden states to the specific layer
- Lora or not
- Last layer at first

Dataset: problem based true/false response sets 

"""


import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

from icl.gradient.gradient import (
    QADatasetTorch, load_data, load_model_and_tokenizer, iter_lora_trainable_params,
    flatten_lora_grads, make_dense_pi
)

random.seed(224)
SEED=224



MODEL_PATH = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"   # <- change to your checkpoint
DATASET_NAME = "path/to/your/dataset"     # HF repo or local dir
TEXT_COLUMN = "data"                    # change if your column name is different
OUT_COLUMN = 'output'
INPUT_COLUMN = 'input'
BATCH_SIZE = 4
MAX_LENGTH = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _get_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Include embedding output as layer 0 .
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    # tuple length = n_layers + 1 (embeddings + each transformer block)
    return out.hidden_states


def _make_token_selector(
    attention_mask: torch.Tensor,
    response_mask: Optional[torch.Tensor] = None,
    ignore_prompt: bool = False,
) -> torch.Tensor:
    
    sel = attention_mask.bool()
    if ignore_prompt:
        if response_mask is None:
            raise ValueError("ignore_prompt=True requires response_mask (B, T) to be provided.")
        sel = sel & response_mask.bool()
    return sel


def _tokenwise_cosine_mean(
    H_prev: torch.Tensor,  # (B, T, d)
    H_cur: torch.Tensor,   # (B, T, d)
    token_sel: torch.Tensor,  # (B, T) bool
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes per-token cosine similarity between adjacent layers, then averages over selected tokens.
    """
    # Normalize along hidden dim
    a = H_prev / (H_prev.norm(dim=-1, keepdim=True) + eps)
    b = H_cur  / (H_cur.norm(dim=-1, keepdim=True) + eps)

    # Cosine per token: (B, T)
    cos_tok = (a * b).sum(dim=-1)

    # Mask + average per sample
    mask = token_sel.float()
    denom = mask.sum(dim=1).clamp_min(1.0)  # avoid div0
    return (cos_tok * mask).sum(dim=1) / denom


def _pooled_cosine(
    H_prev: torch.Tensor,  # (B, T, d)
    H_cur: torch.Tensor,   # (B, T, d)
    token_sel: torch.Tensor,  # (B, T) bool
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mean-pools token vectors per layer (over selected tokens), then cosine.
    """
    mask = token_sel.float().unsqueeze(-1)  # (B, T, 1)
    denom = mask.sum(dim=1).clamp_min(1.0)  # (B, 1)

    h_prev = (H_prev * mask).sum(dim=1) / denom  # (B, d)
    h_cur  = (H_cur  * mask).sum(dim=1) / denom  # (B, d)

    h_prev = h_prev / (h_prev.norm(dim=-1, keepdim=True) + eps)
    h_cur  = h_cur  / (h_cur.norm(dim=-1, keepdim=True) + eps)

    return (h_prev * h_cur).sum(dim=-1)  # (B,)


def find_phase_transition_layer(
    model,
    dataloader,
    device: Optional[torch.device] = None,
    ignore_prompt: bool = False,
    method: str = "tokenwise",  # "tokenwise" or "pooled"
    fallback_to_last_if_tie: bool = True,
) -> Dict[str, object]:
    """
    Computes adjacent-layer cosine similarity curve S_l over D_probe and selects l* = argmin S_l.

    Expected batch keys:
      - input_ids: (B, T)
      - attention_mask: (B, T)

    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    sum_sims = None
    count = 0

    for batch in tqdm(dataloader, desc="Batch: "):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            attention_mask = torch.ones_like(input_ids, device=device)

        response_mask = batch.get("response_mask", None)
        if response_mask is not None:
            response_mask = response_mask.to(device)

        hidden_states = _get_hidden_states(model, input_ids, attention_mask)  # tuple (L+1) of (B,T,d)
        L = len(hidden_states)  # includes embeddings at index 0

        token_sel = _make_token_selector(attention_mask, response_mask, ignore_prompt=ignore_prompt)  # (B,T)

        # Initialize accumulator on first batch
        if sum_sims is None:
            sum_sims = torch.zeros(L - 1, device=device, dtype=torch.float64)

        # For each adjacent pair, compute (B,) and add mean over batch
        for l in range(1, L):
            H_prev = hidden_states[l - 1]
            H_cur = hidden_states[l]
            
            # different cos calculation
            if method == "tokenwise":
                sims_b = _tokenwise_cosine_mean(H_prev, H_cur, token_sel)  # (B,)
            elif method == "pooled":
                sims_b = _pooled_cosine(H_prev, H_cur, token_sel)          # (B,)
            else:
                raise ValueError(f"Unknown method={method}. Use 'tokenwise' or 'pooled'.")

            sum_sims[l - 1] += sims_b.double().sum()
        count += input_ids.size(0)

    # if sum_sims is None or count == 0:
    #     raise RuntimeError("Empty dataloader or no samples processed.")

    # Dataset-level mean similarity 
    curve = (sum_sims / float(count)).cpu().tolist()  

    # rank the curve and return the rank 
    idx = np.argsort(curve)                 
    rank = np.empty_like(idx)
    rank[idx] = np.arange(len(curve))

    # Choose l* = argmin 

    min_val = min(curve)
    mins = [k for k, v in enumerate(curve) if abs(v - min_val) < 1e-12]

    if len(mins) == 1:
        k_star = mins[0]
    else:
        # choose last layer pair if requested
        k_star = (len(curve) - 1) if fallback_to_last_if_tie else mins[0]

    l_star = k_star + 1  # convert curve index to hidden_states layer index

    return {
        "curve": curve,
        "rank": rank.tolist(),
        "l_star": l_star,
        "num_layers_with_embeddings": len(curve) + 1,
        "method": method,
        "ignore_prompt": ignore_prompt,
    }



def get_gradients_over_dataset(model, tokenizer, dataset, LORA=True, LAST=False, 
                               layers=[35, 22, 23, 21, 24], HIDDEN=False, Pi=None):

    dataloader = make_dataloader(dataset, tokenizer)

    model.train()  # enable grads; dropout is fine or switch to eval() if you want deterministic

    # device = next(model.parameters()).device
    device = 'cpu'

    if LORA: 
        print("Using LAST layer of LoRA for gradient extraction.")

        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = any(f"layers.{s}." in n for s in layers)

        trainable = iter_lora_trainable_params(model)
        total_dim = sum(p.numel() for _, p in trainable)
        print(f"Trainable params (LoRA): {total_dim:,} in {len(trainable)} tensors")
        d = 1024  # projected dimension
        D = total_dim

        if Pi is not None:
            d_pi, D_pi = Pi.shape
            print(f'Pi shape: {Pi.shape}, lora grad shape: {d}, {D}')
            assert D_pi == D, f"Pi expects D={D_pi}, but current LoRA grad dim is D={D}"
        else:
            Pi = make_dense_pi(d, D, device=device, dtype=torch.float16, seed=SEED)  # [d, D]
            # torch.save(Pi, f'data_h/pi_matrix_hid.pt')
    
    #----------------------------------------------- projection slice matrix ------------------------------------

    sketches = []
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Batches"):
        # if get gradient wrt to hidden states
        if HIDDEN:
            for n, p in model.named_parameters():
                p.requires_grad = True
            # Get hidden-state gradients as fixed vectors
            vecs, loss = grad_hidden_to_fixed_vector(
                model,
                batch,
                layers=layers,
                use_response_only=True,
                normalize=None,
            )
            # Concatenate selected layers' vectors
            g = torch.cat([vecs[l] for l in layers], dim=-1)  # [B, D]
            g = g[0]

            sketches.append(g.detach().to('cpu'))

        # normal gradient wrt parameters
        else: 
            device = next(model.parameters()).device
            print(f'model device: {device}')
            batch = {k: v.to(device) for k, v in batch.items()} 

            # print(hasattr(model, "hf_device_map"))
            model.zero_grad(set_to_none=True)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss  # scalar
            loss.backward()

            g = flatten_lora_grads(trainable)        # [D] fp16

            # if LAST:
            #     y = g.to('cpu', torch.float16)
            #     print(f'gradient shape: {y.shape}')
            # else:
            g = g.to(Pi.device, torch.float16)
            y = (Pi @ g).detach().to("cpu")          # [d]
            # print(f'gradient shape: {g.shape}, projected shape: {y.shape}')
            
            sketches.append(y)
            
            print(f"[batch {batch_idx}] loss={loss.item():.4f}")

    sketches = torch.stack(sketches, dim=0)  # [N, d]
    return sketches, Pi


import torch
from typing import Dict, List, Tuple, Optional

def masked_mean_pool_token_vectors(X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    X:    (B, T, d)
    mask: (B, T) bool or 0/1
    returns: (B, d)
    """
    mask = mask.to(dtype=X.dtype, device=X.device).unsqueeze(-1)  # (B,T,1)
    denom = mask.sum(dim=1).clamp_min(1.0)                        # (B,1)
    return (X * mask).sum(dim=1) / denom                          # (B,d)

def grad_hidden_to_fixed_vector(
    model,
    batch: Dict[str, torch.Tensor],
    layers: List[int],                 # hidden_states indices (HF: 0=emb, 1..)
    use_response_only: bool = True,
    normalize: Optional[str] = None,   # None | "l2" | "sign"
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Returns:
      vecs: {layer_idx: (B,d)}  fixed d-dim vectors per sample from dLoss/dH^(layer)
      loss: scalar loss tensor (detached)

    Requires batch contains:
      input_ids, attention_mask, labels
      plus response_mask if use_response_only=True
    """
    model.train()
    model.zero_grad(set_to_none=True)

    print(any(p.requires_grad for p in model.parameters()))


    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = torch.ones_like(input_ids, device=device)

    labels = batch["labels"].to(device)

    # Mask: response-only tokens (recommended)
    if use_response_only:
        if "response_mask" not in batch:
            raise ValueError("use_response_only=True requires batch['response_mask'] (B,T).")
        token_sel = batch["response_mask"].to(device).bool() & attention_mask.bool()
    else:
        token_sel = attention_mask.bool()

    # Forward with hidden states
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states  # tuple of (B,T,d)
    for l in layers[1:]:
        hidden_states[l].retain_grad()
    

    loss = outputs.loss


    vecs: Dict[int, torch.Tensor] = {}
    for l in layers:
        # g = hidden_states[l].grad  # (B,T,d) = dLoss/dH^(l)
        g = torch.autograd.grad(
            loss, hidden_states[l],
            retain_graph=True,
            allow_unused=False,   # set True to debug; if True and g is None => disconnected graph
        )[0]

        # optional normalization at token-level before pooling
        if normalize == "l2":
            g = g / (g.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        elif normalize == "sign":
            g = g.sign()

        v = masked_mean_pool_token_vectors(g, token_sel)  # (B,d)
        vecs[l] = v.detach()

    return vecs, loss.detach()



def make_dataloader(ds, tokenizer):
    dataset = QADatasetTorch(ds)

    def collate_fn(batch):
        input_text  = [ex[INPUT_COLUMN] for ex in batch]
        output_text = [ex[OUT_COLUMN] for ex in batch]

        sep = "\n"
        text = [i + sep + o for i, o in zip(input_text, output_text)]

        # encode full text
        enc = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )

        # encode prompt text to get prompt lens
        prompt_enc = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )

        prompt_lens = prompt_enc.attention_mask.sum(dim=1)  # (B,)

        labels = enc.input_ids.clone()
        B, T = labels.shape

        # response_mask: 1 for response tokens, 0 for prompt
        response_mask = torch.zeros((B, T), dtype=torch.long)

        # masking prompt, only response for gradient
        for i, Lp in enumerate(prompt_lens.tolist()):
            # mask prompt labels
            labels[i, :Lp] = -100
            # mark response tokens (only where not padding)
            response_mask[i, Lp:] = 1

        # mask padding from both
        labels[enc.attention_mask == 0] = -100
        response_mask[enc.attention_mask == 0] = 0

        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": labels,                 # gradient only on response tokens
            "response_mask": response_mask,   
            "prompt_lens": prompt_lens,       
        }

    return DataLoader(dataset, 
                      batch_size=1, 
                      shuffle=False, 
                      collate_fn=collate_fn
                      )



# ---------------------------
# Example usage
# ---------------------------
"""
# Suppose your D_probe loader yields dicts like:
# {
#   "input_ids": (B,T),
#   "attention_mask": (B,T),
#   "response_mask": (B,T)  # optional, required if ignore_prompt=True
# }
"""

def layer_selection(ds, model_name, K=5, Pi=None):
    """
    k=5: number of layers (use 6 in actual way but embedding layer is removed)
    """

    # refile='data_resp/true_resps.json'
    # unfile='data_resp/false_resps.json'

    # reasonable, unreasonable = load_data(refile=refile, unfile=unfile)

    model, tokenizer = load_model_and_tokenizer(model_name, LORA=True)

    probe_loader = make_dataloader(ds, tokenizer)
    result = find_phase_transition_layer(
        model=model,
        dataloader=probe_loader,
        ignore_prompt=True,      # use response_mask to ignore prompt tokens
        method="tokenwise",      # or "pooled"
    )

    rank = result['rank']
    print("Selected l* =", result["l_star"])
    print("Curve length =", len(result["curve"]))
    print(f"Rank of layers: {rank}")
    # print("First 10 curve values:", result["curve"])
    layers = [rank.index(i) for i in range(K+1)]
    layers = [l for l in layers if l != 0]
    print(f'selected layers: {layers}')

    return layers

    # probe_loader = make_dataloader(unreasonable, tokenizer)
    # result = find_phase_transition_layer(
    #     model=model,
    #     dataloader=probe_loader,
    #     ignore_prompt=True,      # use response_mask to ignore prompt tokens
    #     method="pooled",      # or "pooled"
    # )
    # rank = result['rank']

    # print("Selected l* =", result["l_star"])
    # print("Curve length =", len(result["curve"]))
    # print(f"Rank of layers: {rank}")

    # layers = [rank.index(i) for i in range(6)]
    # print(layers)


def main():
    refile='data_resp/true_resps.json'
    unfile='data_resp/false_resps.json'

    reasonable, unreasonable = load_data(refile=refile, unfile=unfile)
    input()

    # Pi = torch.load(f'data/pi_matrix.pt')
    Pi = None
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, LORA=True)

    split = 'true'
    stats, _ = get_gradients_over_dataset(model, tokenizer, reasonable, LORA=False, HIDDEN=True, Pi=Pi)
    torch.save({"sketches": stats}, f'data_h/{split}_gradient_wrt_hid')

    split = 'false'
    stats, _ = get_gradients_over_dataset(model, tokenizer, unreasonable, LORA=False, HIDDEN=True, Pi=Pi)
    torch.save({"sketches": stats}, f'data_h/{split}_gradient_wrt_hid')

if __name__ == "__main__":
    main()