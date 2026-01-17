
import torch
from transformers import AutoModelForCausalLM

base_model_name = "Qwen/Qwen3-4B"  # or your local path

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

emb_weight = model.get_input_embeddings().weight.detach().cpu()
print("Embedding weight:", emb_weight.shape, emb_weight.dtype)  # [vocab, hidden]

torch.save(emb_weight, "qwen3_4b_embed_weight.pt")

del model
torch.cuda.empty_cache()
