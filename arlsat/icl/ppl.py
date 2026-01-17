"""
This is code for evaluating perplexity on AR_LSAT task.

Definition:
positive trajectories (T+): pos icl prompt(from gpt5-mini or Qwen3) 
negative trajectories (T-): neg icl prompt(from Qwen3)

Target: perplexity of (Y,Z|X, T+)  perplexity of (Y,Z|X, T-) and for ppl(Y, Z|X, T) or ppl(Y, Z|X)
To check ppl(T) is close to ppl(T+) or ppl(T-)

Firstly, check ppl(T+) and ppl(T-) over small set of X, see the difference.

Model: Qwen3_4b base model / sft model / ICL model
Dataset: AR_LSAT subset of training set.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
import math
import random
import json
import numpy as np
import time

from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

import argparse

from icl_prompt_template import ICL_POSOITIVE_PROMPT, ICL_NEG_PROMPT, ICL_POS_Q3_PROMPT, INSTRUCTION_PROMPT

from utils_.data_process import *


random.seed(224)

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_ds():
    """
    train_ds: dict of multiple prompts list
    """
    # load subset of training set here: 100 samples
    data_source = "tasksource/lsat-ar"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, revision="refs/convert/parquet")

    # idx = np.load("arlsat_train_random_idx.npy")
    
    train_dataset = dataset["train"]
    # selected_idx = [i for i in range(len(train_dataset)) if i not in idx]

    # train_dataset = train_dataset.select(selected_idx)
    test_dataset = dataset["test"]

    train_ds = {'pos_prompt': [], 'neg_prompt': [], 'pos_q3_prompt': [], 'instruction_prompt': []}

    for example in train_dataset:
        context = example.pop("context")
        query = example.pop("question")
        answer = example.pop("label")

        options = example.pop("answers")
        opt = ""
        for i, option in enumerate(options):
            opt = opt + str(i) + ": " + option + "\n"


        neg_prompt = "system:\n" + ICL_NEG_PROMPT['system'] + "\n" + "user:" + ICL_NEG_PROMPT['user'].format(context=context, options=opt, question=query)
        pos_prompt = "system:\n" + ICL_POSOITIVE_PROMPT['system'] + "\n" + "user:" + ICL_POSOITIVE_PROMPT['user'].format(context=context, options=opt, question=query)
        q3_prompt = "system:\n" + ICL_POS_Q3_PROMPT['system'] + "\n" + "user:" + ICL_POS_Q3_PROMPT['user'].format(context=context, options=opt, question=query)
        raw_prompt = "system:\n" + INSTRUCTION_PROMPT['system'] + "\n" + "user:" + INSTRUCTION_PROMPT['user'].format(context=context, options=opt, question=query)

        train_ds['pos_prompt'].append(pos_prompt)
        train_ds['neg_prompt'].append(neg_prompt)
        train_ds['pos_q3_prompt'].append(q3_prompt)
        train_ds['instruction_prompt'].append(raw_prompt)
    
    with open('data/qwen3_4b/train_ds_all_prompts.json', 'w') as f:
        json.dump(train_ds, f, indent=4)

    return train_ds, test_dataset


class Perplexity():
    def __init__(self, train_ds=None, test_ds=None):
        self.train_ds = train_ds
        self.test_ds = test_ds
    
    def load_model(self, model_name, sft=False):
        if not sft:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                              torch_dtype=torch.bfloat16,
                                                              device_map=None,
                                                              attn_implementation="eager").eval()

            self.model.to('cpu')
            # self.model.to("cuda")
        else:
            sft_dir = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"
            self.tokenizer = AutoTokenizer.from_pretrained(sft_dir)
            self.model = AutoModelForCausalLM.from_pretrained(sft_dir,
                                                              torch_dtype=torch.bfloat16,
                                                              device_map="auto",
                                                              attn_implementation="eager").eval()
            self.model.to('cpu')
            # self.model.to("cuda")

    def load_model_gpu(self, model_name, sft=False):
        if not sft:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                              torch_dtype=torch.bfloat16,
                                                              device_map=None,
                                                              ).eval()

            # self.model.to('cpu')
            self.model.to("cuda")
        else:
            sft_dir = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"
            self.tokenizer = AutoTokenizer.from_pretrained(sft_dir)
            self.model = AutoModelForCausalLM.from_pretrained(sft_dir,
                                                              torch_dtype=torch.bfloat16,
                                                              device_map=None,
                                                              ).eval()
            # self.model.to('cpu')
            self.model.to("cuda")


    def inference(self, args, num_samples, ds_save_dir, num_inf=32):
        llm = LLM(
        model=args.model_name, 
        tensor_parallel_size=4,
        )
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            n=num_inf,                   # one sample per prompt
            stop=None              # add stop tokens if you have a special format
        )

        print('Loading model done...')

        def chunk_list(lst, chunk_size: int):
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]
        
        inference_ds = {}
        ds = {k: v[:num_samples] for k, v in self.train_ds.items()}
        # model generation over batch here:
        for idx, (k, v) in enumerate(tqdm(ds.items(), desc=f'Generating responses...')):
            inference_ds[k] = []
            for i, batched_ds in enumerate(list(chunk_list(v, args.inference_batch_size))):
                outputs = llm.generate(batched_ds, sampling_params=sampling_params)
                for ds_, out in zip(batched_ds, outputs):
                    generated_text = [cand.text for cand in out.outputs]
                    ds_dict = {'prompt': ds_, 'gen': generated_text}
                    inference_ds[k].append(ds_dict)
                print(f'Completed {i}th batch for prompt type: {k} ...')

                if i % 5 == 0:
                    print(f'Saving inference ds file')
                    with open(ds_save_dir, 'w') as f:
                        json.dump(inference_ds, f, indent=4)
        
        with open(ds_save_dir, 'w') as f:
            json.dump(inference_ds, f, indent=4)
        return inference_ds


    def inference_raw(self, model_name, num_samples=50):
        llm = LLM(
        model=model_name, 
        tensor_parallel_size=2,
        )
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            n=1,                   # one sample per prompt
            stop=None              # add stop tokens if you have a special format
        )

        print('Loading model done...')

        def chunk_list(lst, chunk_size: int):
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]
        

        inference_ds = {}

        ds = {'instruction_prompt': self.train_ds['instruction_prompt'][:num_samples]}
        # model generation over batch here:
        for k, v in ds.items():
            inference_ds[k] = []
            print(f'Generating responses for prompt type: {k} ...')
            for batched_ds in tqdm(list(chunk_list(v, args.inference_batch_size))):
                outputs = llm.generate(batched_ds, sampling_params=sampling_params)
                for out in outputs:
                    generated_text = out.outputs[0].text
                    inference_ds[k].append(generated_text)
        
        with open('data/qwen3_4b/raw_inference.json', 'w') as f:
            json.dump(inference_ds, f, indent=4)

        return inference_ds


    def gpt_eval(self, prompt_ds, responses_ds):
        with open('data/qwen3_4b/icl_inference.json', 'r') as f:
            ds = json.load(f)
        
        for prompt, responses in zip(prompt_ds, ds):
            gpt_eval = eval_thinking_steps(prompt, responses['pos_prompt'])
        
            


    def eval_cos_representation(self, train_ds, raw_ds, inference_ds):
        """
        trian_ds: list of dict with 4 prompts
        raw_ds: list of raw prompts responses
        inference_dataset: list of dict of 3 icl prompts responses
        """
        repre_list = []
        cos_rep_list = []
        with torch.no_grad():
            # train_ds: 
            for x, y_raw, y_inf in tqdm(zip(train_ds, raw_ds, inference_ds)):
                repre = {}
                # getting raw ds re[presentation
                # print(x.keys())
                x_raw = x['instruction_prompt']

                context_str = self._build_context(x_raw)    # includes system + T_block + X
                target_str  = self._build_context(y_raw)         # Z+Y
                full_str = context_str + target_str
                # print(full_str)       
                tokens = self.tokenizer(full_str, return_tensors="pt")
                input_ids = tokens["input_ids"].to("cuda")
                outputs = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = outputs.hidden_states[-1]
                x_rep_last_token = last_hidden[:, -1, :].squeeze(0)  # [hidden_size]
                
                repre["raw"] = x_rep_last_token.cpu().tolist()

                # getting inference ds representation
                in_rep_dict = {}
                print(y_inf.keys())

                for k, v in y_inf.items():
                    x_ = x[k]

                    context_str = self._build_context(x_)
                    target_str  = self._build_context(v) 
                    full_str = context_str + target_str
                    # print(full_str)       
                    tokens = self.tokenizer(full_str, return_tensors="pt")
                    input_ids = tokens["input_ids"].to("cuda")
                    outputs = self.model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    last_hidden = outputs.hidden_states[-1]
                    rep_last_token = last_hidden[:, -1, :].squeeze(0)  # [hidden_size]
                    
                    in_rep_dict[k] = rep_last_token.cpu().tolist()
                
                repre['inf'] = in_rep_dict

                repre_list.append(repre)
                cos_dict = {}
                for k, v in in_rep_dict.items():
                    # print(repre["raw"])
                    cos_sim = self.cosine_similarity(repre["raw"], v)
                    cos_dict[k] = cos_sim
                
                cos_rep_list.append(cos_dict)

            with open('data/qwen3_4b/last_token_repre.json', 'w') as f:
                json.dump(repre_list, f, indent=4)
            
            with open('data/qwen3_4b/last_token_repre_cos.json', 'w') as f:
                json.dump(cos_rep_list, f, indent=4)
            return cos_rep_list, repre_list


    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


    def eval_delta_logprob(self, dataset):
        """
        dataset: list of [x_t, z_y], e.g. [prompt with system template, response]
        We compute:
        logp_full  = log p(answer | context + thinking)
        logp_no_th = log p(answer | context only)
        delta      = logp_full - logp_no_th
        """
        delta_list = []
        logp_full_list = []
        logp_no_thinking_list = []

        print(f'Evaluating delta logprob over {len(dataset)} samples...\n')

        device = "cuda"

        with torch.no_grad():
            for (x_t, z_y) in tqdm(dataset, desc="Evaluating delta logp"):
                # Build strings
                context_str = self._build_context(x_t)    # includes system + T_block + X
                target_str  = self._build_context(z_y)    # Z+Y

                # Split target_str into thinking / answer
                thinking_str, answer_str = self.get_segment_indices_under_sequence(target_str)
                # full context (prompt + thinking)
                ctx_full_str = context_str + thinking_str

                # ----------------------- FULL: context + thinking + answer -----------------------
                full_str = context_str + thinking_str + answer_str

                tokens = self.tokenizer(full_str, return_tensors="pt")
                input_ids = tokens["input_ids"].to(device)        # [1, T]
                # (we ignore attention_mask here; full contiguous sequence)

                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits                           # [1, T, V]

                # Teacher forcing alignment
                shift_logits = logits[:, :-1, :]                  # [1, T-1, V]
                shift_labels = input_ids[:, 1:]                   # [1, T-1]

                log_probs = torch.log_softmax(shift_logits, dim=-1)  # [1, T-1, V]
                token_log_probs = log_probs.gather(
                    dim=-1,
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)                                     # [1, T-1]

                # Token lengths (without special tokens) to locate answer segment
                answer_ids = self.tokenizer(
                    answer_str, return_tensors="pt", add_special_tokens=False
                )["input_ids"]    # [1, ans_len]
                ctx_full_ids = self.tokenizer(
                    ctx_full_str, return_tensors="pt", add_special_tokens=False
                )["input_ids"]    # [1, ctx_len_full]

                ctx_len = ctx_full_ids.shape[1]   # number of context+thinking tokens
                ans_len = answer_ids.shape[1]     # number of answer tokens

                # Indexing: first answer token is token index ctx_len in the *no-specials* sequence.
                # In teacher-forcing log_probs, position t corresponds to predicting token_ids[t+1].
                # So we want log probs at positions t = ctx_len-1 ... ctx_len+ans_len-2
                start = max(ctx_len - 1, 0)
                end = ctx_len + ans_len - 1       # exclusive in slicing

                answer_token_log_probs = token_log_probs[:, start:end]    # [1, ans_len]
                logp_full = answer_token_log_probs.sum().item()

                # ------------------- NO THINKING: context only + answer -------------------
                no_thinking_full_str = context_str + answer_str

                input_ids2 = self.tokenizer(
                    no_thinking_full_str, return_tensors="pt"
                )["input_ids"].to(device)         # [1, T2]

                outputs2 = self.model(input_ids=input_ids2)
                logits2 = outputs2.logits         # [1, T2, V]

                shift_logits2 = logits2[:, :-1, :]
                shift_labels2 = input_ids2[:, 1:]

                log_probs2 = torch.log_softmax(shift_logits2, dim=-1)  # [1, T2-1, V]
                token_log_probs2 = log_probs2.gather(
                    dim=-1,
                    index=shift_labels2.unsqueeze(-1)
                ).squeeze(-1)                                      # [1, T2-1]

                # Context length for "context only" case (no thinking)
                ctx_ids = self.tokenizer(
                    context_str, return_tensors="pt", add_special_tokens=False
                )["input_ids"]    # [1, ctx_len_ctx_only]

                ctx_len2 = ctx_ids.shape[1]
                ans_len2 = ans_len             # same answer string

                start2 = max(ctx_len2 - 1, 0)
                end2 = ctx_len2 + ans_len2 - 1

                answer_token_log_probs2 = token_log_probs2[:, start2:end2]  # [1, ans_len2]
                logp_no_thinking = answer_token_log_probs2.sum().item()

                # ----------------------------- delta -----------------------------
                delta = logp_full - logp_no_thinking

                print(f'logp_full: {logp_full:.4f}, '
                    f'logp_no_thinking: {logp_no_thinking:.4f}, '
                    f'delta: {delta:.4f}')

                delta_list.append(delta)
                logp_full_list.append(logp_full)
                logp_no_thinking_list.append(logp_no_thinking)

        print(f'Length of delta list: {len(delta_list)}')
        return delta_list, logp_full_list, logp_no_thinking_list


    def eval_ppl(self, dataset):  # T_block is either T_pos_text or T_neg_text
        """
        dataset: list of [x_t, z_y], eg [prompt with system template, response]
        """
        total_logprob = 0.0
        logprob_list = []
        total_tokens = 0

        print(f'Evaluating perplexity over {len(dataset)} samples...\n')

        with torch.no_grad():
            # [x-t, z-y] in dataset: [prompt with system template, response]
            for (x_t, z_y) in tqdm(dataset, desc="Evaluating PPL"):
                context_str = self._build_context(x_t)    # includes system + T_block + X
                target_str  = self._build_context(z_y)         # Z+Y
                
                full_str = context_str + target_str
                # print(full_str)
                
                tokens = self.tokenizer(full_str, return_tensors="pt")
                input_ids = tokens["input_ids"].to("cuda")
                
                # Run model in teacher-forcing mode
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits  # [1, seq_len, vocab]
                
                # We want log p of each target token given previous
                # Shift logits by one to align with next token
                shift_logits = logits[:, :-1, :]            # predict token t+1
                shift_labels = input_ids[:, 1:]             # ground truth token t+1

                # Find answer token positions

                # Mask out thinking tokens (mask out all target str except for answer part)
                # Mask out context positions (only target tokens count)
                ctx_len = self.tokenizer(context_str, return_tensors="pt")["input_ids"].shape[1]
                # full_length = shift_labels.shape[1]
                # positions >= ctx_len-1 are target tokens (because of the shift)
                mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                mask[:, ctx_len-1:] = True

                log_probs = torch.log_softmax(shift_logits, dim=-1)
                # gather log probs of correct tokens
                token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # only keep target positions
                # log prob for one sample
                token_log_probs = token_log_probs[mask]
                
                total_logprob += token_log_probs.sum().item()
                total_tokens  += mask.sum().item()

                # save nll per sample
                logprob_list.append(-token_log_probs.sum().item() / mask.sum().item())

        # save total nll for the dataset
        nll = - total_logprob / (total_tokens + 1e-8)
        # save and compute ppl for the dataset
        ppl = math.exp(nll)

        print(f'Perplexity: {ppl}, NLL: {nll}, Total tokens: {total_tokens}')
        print(f'NLL per sample: {logprob_list}')
        return ppl, nll, logprob_list
    

    def eval_reverse_ppl(self, dataset):  # T_block is either T_pos_text or T_neg_text
        """
        reverse ppl: p(X|Z, Y)
        dataset: list of [x_t, z_y], eg [prompt with system template, response]
        """
        total_logprob = 0.0
        logprob_list = []
        total_tokens = 0

        print(f'Evaluating perplexity over {len(dataset)} samples...\n')

        with torch.no_grad():
            # [x-t, z-y] in dataset: [prompt with system template, response]
            for (x_t, z_y) in tqdm(dataset, desc="Evaluating reverse PPL"):
                context_str = self._build_context(z_y)   # includes system + T_block + X
                target_str  = self._build_context(x_t)          # Z+Y
                
                full_str = context_str + target_str
                # print(full_str)
                
                tokens = self.tokenizer(full_str, return_tensors="pt")
                input_ids = tokens["input_ids"].to("cuda")
                
                # Run model in teacher-forcing mode
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits  # [1, seq_len, vocab]
                
                # We want log p of each target token given previous
                # Shift logits by one to align with next token
                shift_logits = logits[:, :-1, :]            # predict token t+1
                shift_labels = input_ids[:, 1:]             # ground truth token t+1

                # Find answer token positions

                # Mask out thinking tokens (mask out all target str except for answer part)
                # Mask out context positions (only target tokens count)
                ctx_len = self.tokenizer(context_str, return_tensors="pt")["input_ids"].shape[1]
                # full_length = shift_labels.shape[1]
                # positions >= ctx_len-1 are target tokens (because of the shift)
                mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                mask[:, ctx_len-1:] = True

                log_probs = torch.log_softmax(shift_logits, dim=-1)
                # gather log probs of correct tokens
                token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # only keep target positions
                # log prob for one sample
                token_log_probs = token_log_probs[mask]
                
                total_logprob += token_log_probs.sum().item()
                total_tokens  += mask.sum().item()

                # save nll per sample
                logprob_list.append(-token_log_probs.sum().item() / mask.sum().item())

        # save total nll for the dataset
        nll = - total_logprob / (total_tokens + 1e-8)
        # save and compute ppl for the dataset
        ppl = math.exp(nll)

        print(f'Perplexity: {ppl}, NLL: {nll}, Total tokens: {total_tokens}')
        print(f'NLL per sample: {logprob_list}')
        return ppl, nll, logprob_list
    
    def _build_context(self, text):
        return text.strip()


    def contribution_score(self, dataset, layers=None):
        """
        dataset: list of [x_t, z_y]
        contribution score: attention based score
        Ratio of thinking tokens attention score to answer tokens over all tokens
        Eliminating the influence of different lengths of target tokens
        score = (attention score to answer tokens) * (distance to answer token)
        """

        device='cpu'

        contrib_list = []
        for (x_t, z_y) in tqdm(dataset, desc="Evaluating contribution score"):
            s_t = time.time()
            context_str = self._build_context(x_t)    # includes system + T_block + X
            target_str  = self._build_context(z_y)         # Z+Y
            
            full_str = context_str + target_str
            # compute attention of full context + target
            tokens = self.tokenizer(full_str, return_tensors="pt")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            T = input_ids.size(1)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=False,  
            )
            attentions = outputs.attentions  # tuple of (L,)[B, H, T, T]
            L = len(attentions)
            B, H = attentions[0].shape[:2]
            assert B == 1, "This function assumes batch size 1."

            # Pick which layers to use
            if layers is None:
                layer_indices = list(range(L))
            elif layers == 1/3:
                layer_indices = list(range(2*L//3, L))

            # Stack: [L, B, H, T, T]
            attn_stack = torch.stack(
                [attentions[l] for l in layer_indices],
                dim=0
            )  # [L_used, B, H, T, T]
            L_used = attn_stack.size(0)

            # get context length to get prompt idx: 0, ctx_len-1
            ctx_len = self.tokenizer(context_str, return_tensors="pt")["input_ids"].shape[1]
            resp_len = T - ctx_len
            prompt_idx = [0, ctx_len-1]

            # get segment of thinking and answer tokens idx
            think_idx, answer_idx = self.get_segment_indices_with_tags(target_str, self.tokenizer)
            think_idx = [idx+ctx_len for idx in think_idx]  # shift by ctx_len
            answer_idx = answer_idx + ctx_len

            # print(f'context len: {ctx_len}, response len: {resp_len}')
            print(f'answer idx: {answer_idx}')
            print(f'Total tokens: {T}')

            # getting attention to answer tokens: A
            answer_positions = torch.arange(
                answer_idx, T
            )  # [A]
            if answer_positions.numel() == 0:
                # fallback: last token is the "answer"
                answer_positions = torch.tensor([T - 1], device=device)

            # Extract attention: [L_used, H, A, T]
            # (batch idx = 0)
            attn_to_all = attn_stack[:, 0, :, answer_positions, :]  # [L_used, H, A, T]

            # Average over layers, heads, and answer positions -> [T]
            A = attn_to_all.mean(dim=(0, 1, 2))  # [T]

            # We only care about contributions from *previous* tokens:
            # zero out attention to tokens at or after first answer-body token.
            A = A.clone()
            A[answer_idx:] = 0.0

            # Prompt segment:
            p_start, p_end = prompt_idx
            p_start = max(0, p_start)
            p_end = min(T - 1, p_end)
            if p_start <= p_end:
                prompt_mask = torch.zeros(T, dtype=torch.bool, device=device)
                prompt_mask[p_start:p_end + 1] = True
                A_prompt = A[prompt_mask].sum().item()
            else:
                A_prompt = 0.0

            # Think segment:
            if think_idx is not None:
                t_start, t_end = think_idx
                t_start = max(0, t_start)
                t_end = min(T - 1, t_end)
                if t_start <= t_end:
                    think_mask = torch.zeros(T, dtype=torch.bool, device=device)
                    think_mask[t_start:t_end + 1] = True
                    A_think = A[think_mask].sum().item()
                else:
                    A_think = 0.0
            else:
                A_think = 0.0
            
            # Local segement:
            local_mask = torch.zeros(T, dtype=torch.bool, device=device)
            local_mask[-10:] = True
            A_local = A[local_mask].sum().item()
            
            # calculating attention ratio
            eps = 1e-8
            denom1 = A_think + A_prompt + eps
            denom2 = A_think + A_prompt + A_local + eps
            R1 = A_think / denom1
            R2 = A_think / denom2

            attention_dict = {'A_think': A_think, 
                              'A_prompt': A_prompt, 
                              'A_local': A_local, 
                            #   'A_all': A,
                              'R1': R1, 'R2': R2,
                              'resp_len': resp_len
                              }
            e_t = time.time()
            print(f'Time for one sample contribution score: {e_t - s_t} seconds.')
        contrib_list.append(attention_dict)

        return contrib_list
        
    def get_segment_indices_under_sequence(self, text):
        """
        Get thinking tokens and answers tokens start and end idx
        thinking tokens start after prompt, end before <answer>pattern</answer>
        answer tokens idx is derived by extracting <answer>pattern</answer>

        Return: 
        thinking_range: (start_idx, end_idx)
        answer_start_idx: int
        """

        answer_pattern = r"<answer>(.*?)</answer>"
        matches = list(re.finditer(answer_pattern, text, re.DOTALL))

        if not matches:
            thinking_range = (0, -1)
            answer_range = -1
            return None

        last_match = matches[-1]


        # Character span of the FULL pattern: "<answer>...</answer>"
        full_start, full_end = last_match.span(0)

        # Character span of JUST the inner group "(.*?)"
        inner_start, inner_end = last_match.span(1)
        # print(full_start, full_end)
        thinking = text[0: full_start]
        answer = text[full_start: full_end]
        print(answer)

        return thinking, answer
    
    def get_segment_indices_with_tags(self, text, tokenizer):
        """
        Get thinking tokens and answers tokens start and end idx
        thinking tokens start after prompt, end before <answer>pattern</answer>
        answer tokens idx is derived by extracting <answer>pattern</answer>

        Return: 
        thinking_range: (start_idx, end_idx)
        answer_start_idx: int
        """

        answer_pattern = r"<answer>(.*?)</answer>"
        matches = list(re.finditer(answer_pattern, text, re.DOTALL))

        if not matches:
            thinking_range = (0, -1)
            answer_range = -1
            return thinking_range, answer_range

        last_match = matches[-1]


        # Character span of the FULL pattern: "<answer>...</answer>"
        full_start, full_end = last_match.span(0)

        # Character span of JUST the inner group "(.*?)"
        inner_start, inner_end = last_match.span(1)

        thinking = [0, full_start]
        answer = inner_start


        enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,   # important: offsets align directly with `text`
        return_tensors="pt",
        )

        # print(f'resp tokens: {enc["input_ids"].shape[1]}' )

        offset_mapping = enc["offset_mapping"][0].tolist()  # List[(char_start, char_end)]
        T = len(offset_mapping)

        # 1) Find thinking token range: tokens fully before the <answer>... pattern
        thinking_start_tok = 0
        thinking_end_tok = -1
        for i, (c_start, c_end) in enumerate(offset_mapping):
            # token ends before the full <answer> tag starts
            if c_end <= full_start:
                thinking_end_tok = i
            else:
                break

        if thinking_end_tok < thinking_start_tok:
            # no thinking tokens
            thinking_token_range = (0, -1)
        else:
            thinking_token_range = (thinking_start_tok, thinking_end_tok)

        # 2) Find first token belonging to the answer *content*
        answer_token_start = None
        for i, (c_start, c_end) in enumerate(offset_mapping):
            # token overlaps the inner answer content region
            if not (c_end <= inner_start or c_start >= inner_end):
                answer_token_start = i
                break

        if answer_token_start is None:
            # fallback: last token
            answer_token_start = T - 1

        return thinking_token_range, answer_token_start




        


def main():
    train_ds, test_ds = load_ds()
    print('Loading data done.')
    ppl_evaluator = Perplexity(train_ds, test_ds)
    ppl_evaluator.load_model(args.model_name)
    inference_ds = ppl_evaluator.inference(args.model_name, num_samples=200)

    with open('data/qwen3_4b/icl_inference.json', 'r') as f:
        inference_ds = json.load(f)
    ppl_results = {}
    for k in inference_ds.keys():
        print(f'Inference dataset key: {k}, number of samples: {len(inference_ds[k])}')
        ppl_ds = [[tr, inf] for tr, inf in zip(train_ds[k], inference_ds[k])]
        ppl, nll, logprob_list = ppl_evaluator.eval_ppl(ppl_ds)
        ppl_results[k] = {'ppl': ppl, 'nll': nll, 'logprob_list': logprob_list}
    
    with open('data/qwen3_4b/icl_ppl_results.json', 'w') as f:
        json.dump(ppl_results, f, indent=4)




def raw_inference():
    train_ds, test_ds = load_ds()
    print('Loading data done.')
    ppl_evaluator = Perplexity(train_ds, test_ds)
    ppl_evaluator.load_model(args.model_name)
    # inference_ds = ppl_evaluator.inference_raw(args.model_name, num_samples=200)

    with open('data/qwen3_4b/raw_inference.json', 'r') as f:
        inference_ds = json.load(f)

    ppl_results = {}
    for k in inference_ds.keys():
        print(f'Inference dataset key: {k}, number of samples: {len(inference_ds[k])}')
        ppl_ds = [[tr, inf] for tr, inf in zip(train_ds[k], inference_ds[k])]
        ppl, nll, logprob_list = ppl_evaluator.eval_ppl(ppl_ds)
        ppl_results[k] = {'ppl': ppl, 'nll': nll, 'logprob_list': logprob_list}
    
    with open('data/qwen3_4b/raw_ppl_results.json', 'w') as f:
        json.dump(ppl_results, f, indent=4)
    

def eval_cos():
    train_ds, test_ds = load_ds()
    print('Loading data done.')
    ppl_evaluator = Perplexity(train_ds, test_ds)
    ppl_evaluator.load_model(args.model_name)

    with open('data/qwen3_4b/raw_inference.json', 'r') as f:
        raw_ds = json.load(f)
    
    with open('data/qwen3_4b/icl_inference.json', 'r') as f:
        inference_ds = json.load(f)
    
    print(inference_ds.keys())
    
    
    train_ds = [{k: train_ds[k][i] for k in train_ds.keys()} for i in range(50)]
    # print(train_ds[0])
    inference_ds = [{k: inference_ds[k][i]} for k in inference_ds.keys() for i in range(50)]

    raw_ds = raw_ds['instruction_prompt']
    
    ppl_evaluator.eval_cos_representation(train_ds, raw_ds, inference_ds)



def inference_32(args):
    train_ds, test_ds = load_ds()
    print('Loading data done.')
    lens = len(train_ds['pos_prompt'])
    print(f'Total number of training samples: {lens}')
    inf_idx = random.sample(range(lens), 200)
    np.save('data/qwen3_4b/inference/inf_idx_200.npy', np.array(inf_idx))

    train_ds = {k: [v[i] for i in inf_idx] for k, v in train_ds.items()}
    print(f'Selected {len(train_ds["pos_prompt"])} samples for inference.')
    with open('data/qwen3_4b/train_ds_all_prompts_200.json', 'w') as f:
        json.dump(train_ds, f, indent=4)
    
    # input()

    ppl_evaluator = Perplexity(train_ds, test_ds)
    print(args.sft)
    input()
    if not args.sft:
        ppl_evaluator.load_model(args.model_name)
        ds_save_dir = 'data/qwen3_4b/inference/base_icl_inference32_200.json'
        inference_ds = ppl_evaluator.inference(args, args.model_name, num_samples=200, ds_save_dir=ds_save_dir, num_inf=32)

    else:
        sft_save_dir = 'data/qwen3_4b/inference/sft_icl_inference32_200.json'
        ppl_evaluator.load_model(args.model_name, sft=True)
        inference_ds = ppl_evaluator.inference(args, args.model_name, num_samples=200, ds_save_dir=sft_save_dir, num_inf=32)



def gpt_eval():
    with open('data/qwen3_4b/train_ds_all_prompts_200.json', 'r') as f:
        train_ds = json.load(f)
    with open('data/qwen3_4b/inference/sft_icl_inference32_200.json', 'r') as f:
        ds = json.load(f)
    
    inf_idx = np.load('data/qwen3_4b/inference/inf_idx_200.npy')
    original_prompts = [train_ds['instruction_prompt'][i] for i in inf_idx]
    
    gpt_eval = {}
    for k, inf_list in ds.items():
        gpt_eval[k] = []
        for origin_p, responses in zip(original_prompts, inf_list):
            gpt_eval_dict = {'eval_num': [], 'eval_res': []}
            prompt = origin_p
            for response in responses['gen']:
                eval_res, eval_num = eval_thinking_steps(prompt, response)
                gpt_eval_dict['eval_res'].append(eval_res)
                gpt_eval_dict['eval_num'].append(eval_num)
                print(f'The evaluation result is: {eval_num}')
            
            gpt_eval[k].append(gpt_eval_dict)
            print('Finished one sample evaluation...')
    
    with open('data/qwen3_4b/gpt_eval_results.json', 'w') as f:
        json.dump(gpt_eval, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="model name or path")
    parser.add_argument("--max_tokens", type=int, default=4096, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p sampling")
    parser.add_argument("--inference_batch_size", type=int, default=16, help="batch size for inference")
    
    parser.add_argument("--sft", action='store_true', help="whether to use sft model")

    args = parser.parse_args()

    # raw_inference()
    # eval_cos()
    
    gpt_eval()
    # inference_32(args)

    

    







