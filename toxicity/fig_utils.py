"""
Utility functions for figures.
"""

import torch
import torch.nn.functional as F
import einops
from transformer_lens import (
    HookedTransformer,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, BitsAndBytesConfig


def convert(orig_state_dict, cfg):
    state_dict = {}

    state_dict["embed.W_E"] = orig_state_dict["transformer.wte.weight"]
    state_dict["pos_embed.W_pos"] = orig_state_dict["transformer.wpe.weight"]

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = orig_state_dict[
            f"transformer.h.{l}.ln_1.weight"
        ]
        state_dict[f"blocks.{l}.ln1.b"] = orig_state_dict[
            f"transformer.h.{l}.ln_1.bias"
        ]

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = orig_state_dict[f"transformer.h.{l}.attn.c_attn.weight"]
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        qkv_bias = orig_state_dict[f"transformer.h.{l}.attn.c_attn.bias"]
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = orig_state_dict[f"transformer.h.{l}.attn.c_proj.weight"]
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = orig_state_dict[
            f"transformer.h.{l}.attn.c_proj.bias"
        ]

        state_dict[f"blocks.{l}.ln2.w"] = orig_state_dict[
            f"transformer.h.{l}.ln_2.weight"
        ]
        state_dict[f"blocks.{l}.ln2.b"] = orig_state_dict[
            f"transformer.h.{l}.ln_2.bias"
        ]

        W_in = orig_state_dict[f"transformer.h.{l}.mlp.c_fc.weight"]
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = orig_state_dict[
            f"transformer.h.{l}.mlp.c_fc.bias"
        ]

        W_out = orig_state_dict[f"transformer.h.{l}.mlp.c_proj.weight"]
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = orig_state_dict[
            f"transformer.h.{l}.mlp.c_proj.bias"
        ]
    state_dict["unembed.W_U"] = orig_state_dict["lm_head.weight"].T

    state_dict["ln_final.w"] = orig_state_dict["transformer.ln_f.weight"]
    state_dict["ln_final.b"] = orig_state_dict["transformer.ln_f.bias"]
    return state_dict


def load_hooked(model_name, weights_path):
    _model = HookedTransformer.from_pretrained(model_name)
    cfg = _model.cfg

    _weights = torch.load(weights_path, map_location=torch.device("cuda"))[
        "state"
    ]
    weights = convert(_weights, cfg)
    model = HookedTransformer(cfg)
    model.load_and_process_state_dict(weights)
    model.tokenizer.padding_side = "left"
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    return model



def load_model(config):
    """
    Load model, tokenizer and distribute across multiple GPUs if available.
    """
    assert "model_or_path" in config
    assert "tokenizer" in config

    tokenizer_name = config["tokenizer"]
    model_name = config["model_or_path"]
    state_dict_path = config.get("state_dict_path")
    state_dict = None

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)["state"]

    # Load model config
    # model_config = AutoConfig.from_pretrained(model_name)

    # Apply 8-bit quantization only for LLaMA 3 models
    if "llama-3" in model_name.lower():
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, state_dict=state_dict
        )
    elif "gemma" in model_name.lower():
        # Load model with correct attention implementation
        model = AutoModelForCausalLM.from_pretrained(
            model_name, state_dict=state_dict, 
            attn_implementation="eager").to(config["device"])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, state_dict=state_dict
            ).to(config["device"])

    # Distribute model across multiple GPUs if available 
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # Load tokenizer
    if tokenizer_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    return model, tokenizer




# def load_model(model_name, weights_path=None, device="cuda"):
#     """
#     Loads the DPO-ed version of Hugging Face transformer model for causal LM.
#     """
#     # Load model
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
#     # Move to device
#     model.to(device)

#     # If custom weights are provided, load them
#     if weights_path:
#         state_dict = torch.load(weights_path, map_location=device)
#         model.load_state_dict(state_dict, strict=False)  # strict=False allows missing keys

#     return model



def get_svd(_model, toxic_vector, num_mlp_vecs):
    scores = []
    for layer in range(_model.cfg.n_layers):
        mlp_outs = _model.blocks[layer].mlp.W_out
        cos_sims = F.cosine_similarity(
            mlp_outs, toxic_vector.unsqueeze(0), dim=1
        )
        _topk = cos_sims.topk(k=300)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    top_vecs = [
        _model.blocks[x[2]].mlp.W_out[x[1]]
        for x in sorted_scores[:num_mlp_vecs]
    ]
    top_vecs = [x / x.norm() for x in top_vecs]
    _top_vecs = torch.stack(top_vecs)

    svd = torch.linalg.svd(_top_vecs.transpose(0, 1))
    return svd, sorted_scores


def get_negative_svd(_model, toxic_vector, num_mlp_vecs):
    scores = []
    for layer in range(_model.cfg.n_layers):
        mlp_outs = _model.blocks[layer].mlp.W_out
        cos_sims = F.cosine_similarity(
            mlp_outs, toxic_vector.unsqueeze(0), dim=1
        ) # the anti-toxic vector
        _topk = cos_sims.topk(k=300)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=False) # reverse the ordering
    top_vecs = [
        _model.blocks[x[2]].mlp.W_out[x[1]]
        for x in sorted_scores[:num_mlp_vecs]
    ]
    top_vecs = [x / x.norm() for x in top_vecs]
    _top_vecs = torch.stack(top_vecs)

    svd = torch.linalg.svd(_top_vecs.transpose(0, 1))
    return svd, sorted_scores