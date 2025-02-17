import os
import sys

os.chdir('/data/kebl6672/dpo-toxic-general/toxicity')
sys.path.append('/data/kebl6672/dpo-toxic-general/toxicity')

import json
import torch
import torch.nn.functional as F
# from transformer_lens import (
#     HookedTransformer,
# )
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
# from fig_utils import load_hooked
from fig_utils import load_model

device = torch.device("cuda") 
ROOT_DIR = '/data/kebl6672/dpo-toxic-general/checkpoints'

model_name = "meta-llama/Llama-3.1-8B" # "gpt2-medium" # "meta-llama/Llama-3.1-8B" # "google/gemma-2-2b", # "gpt2-medium", # "mistralai/Mistral-7B-v0.1",
dpo_model_name = "llama3_dpo_final.pt" # "gpt2_dpo.pt" # "llama3_dpo_2.pt"
probe_name = "llama3_probe.pt" # "gpt2_probe.pt" # "llama3_probe.pt"
model_short_name = "llama3" #"gpt2" # "llama3"

# Load the tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the pre-trained model
model = load_model(model_name)
model.to(device)  

# Load the DPO-ed model
dpo_model = load_model(model_name, os.path.join(ROOT_DIR, dpo_model_name))
dpo_model.to(device)

# Load the toxic probe vector
toxic_vector = torch.load(os.path.join(ROOT_DIR, probe_name)).to(device)  


# load evaluation data
DATA_DIR = '/data/kebl6672/dpo-toxic-neuron/data/intervene_data'

with open(
    os.path.join(DATA_DIR, "challenge_prompts.jsonl"), "r"
) as file_p:
    data = file_p.readlines()

prompts = [json.loads(x.strip())["prompt"] for x in data]

# Tokenizing the prompts correctly
tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)






# Compute the neuron toxicity projection 
def compute_neuron_toxic_projection(model, tokenized_prompts, toxic_vector, batch_size=2):
    """
    Computes neuron toxicity projections by extracting activations after non-linearity
    using hooks on up_proj.
    """
    model_neuron_projections = defaultdict(list)
    sample_size = tokenized_prompts.size(0)

    print("Computing MLP neuron projections...")

    device = next(model.parameters()).device  # Get model's device

    # Storage for activations extracted from inputs to down_proj
    neuron_acts_storage = {}

    # Hook function to capture activations after non-linearity
    def hook_fn(module, input, output, layer_idx):
        neuron_acts_storage[layer_idx] = input[0]  # Store activations (B, T, d_mlp)
        print(f"Layer {layer_idx} activation shape (B, T, d_mlp): {neuron_acts_storage[layer_idx].shape}") 

    # Register hooks for all layers
    hooks = []
    for layer_idx in range(len(model.model.layers)):
        hook = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(
            lambda module, input, output, l=layer_idx: hook_fn(module, input, output, l)
        )
        hooks.append(hook)

    for idx in tqdm(range(0, sample_size, batch_size)):
        batch = tokenized_prompts[idx: idx + batch_size, :].to(device)

        for timestep in range(20):  # Generate 20 tokens
            with torch.inference_mode():
                outputs = model(batch, output_hidden_states=True, return_dict=True)
                cache = outputs.hidden_states  # Extract the hidden states

            # Ensure cache tensors are moved to the correct device
            cache = [v.to(device).detach().clone() for v in cache]  # Convert to list if needed

            print(f"Number of hidden state layers in cache: {len(cache)}")
            for i, layer_cache in enumerate(cache):
                print(f"Layer {i} hidden state shape: {layer_cache.shape}")  # (B, T, d)

            sampled = model.lm_head(cache[-1]).argmax(-1).detach().to(device)[:, -1] # Generate the next token (B, 1)

            for layer_idx in range(len(model.model.layers)):
                if layer_idx in neuron_acts_storage:
                    neuron_acts = neuron_acts_storage[layer_idx]  # Get stored activations (B, T, d_mlp)
                    print(f"Neuron activations shape (B, T, d_mlp) at layer {layer_idx}: {neuron_acts.shape}")
                else:
                    continue

                value_vectors = model.model.layers[layer_idx].mlp.down_proj.weight.to(device).T # (d_mlp, d)
                print(f"Down_proj weight shape (d_mlp, d) at layer {layer_idx}: {value_vectors.shape}")


                # Compute neuron outputs
                neuron_outputs = torch.einsum('btd,dm->btdm', neuron_acts.clone(), value_vectors.clone())  # (B, T, d_mlp, d) 
                print(f"Neuron outputs shape (B, T, d_mlp, d) at layer {layer_idx}: {neuron_outputs.shape}")

                toxic_vector = toxic_vector.to(device, dtype=torch.float16).squeeze(0) # (d, )
                print(f"Toxic vector shape (d, ): {toxic_vector.shape}")
      
                neuron_projections = torch.matmul(neuron_outputs, toxic_vector.to(device)) / torch.norm(toxic_vector.to(device)) # (B, T, d_mlp)
                print(f"Neuron projections shape (B, T, d_mlp) at layer {layer_idx}: {neuron_projections.shape}")

                for neuron_idx in range(neuron_projections.size(2)):
                    print(f"Neuron {neuron_idx} projections shape (B, T) at layer {layer_idx}: {neuron_projections[:, :, neuron_idx].shape}")
                    model_neuron_projections[(layer_idx, neuron_idx)].extend(neuron_projections[:, :,neuron_idx].tolist()) # flatted (B, T) to (B*T) in list

            batch = torch.concat([batch, sampled.unsqueeze(-1)], dim=-1) # Add the generated token, (B, T+1)

    # Remove all hooks
    for hook in hooks:
        hook.remove()

    # Compute final average neuron projections over B, T, and the 20 generated tokens
    avg_neuron_projections = {
        (layer_idx, neuron_idx): np.mean(projections)
        for (layer_idx, neuron_idx), projections in model_neuron_projections.items()
    }

    return avg_neuron_projections





# # Compute the neuron toxicity projection 
# def compute_neuron_toxic_projection_test(model, tokenized_prompts, toxic_vector, batch_size=64):
#     """
#     Computes neuron toxicity projections by extracting activations after non-linearity
#     using hooks on up_proj.
#     """
#     model_neuron_projections = defaultdict(list)
#     sample_size = tokenized_prompts.size(0)

#     print("Computing MLP neuron projections...")

#     device = next(model.parameters()).device  # Get model's device

#     # Storage for activations extracted from up_proj
#     neuron_acts_storage = {}

#     # Hook function to capture activations after non-linearity
#     def hook_fn(module, input, output, layer_idx):
#         neuron_acts_storage[layer_idx] = input[0]  # Store activations
#         print(f"Layer {layer_idx} activation shape: {neuron_acts_storage[layer_idx].shape}")

#     # Register hooks for all layers
#     hooks = []
#     for layer_idx in range(len(model.model.layers)):
#         hook = model.model.layers[layer_idx].mlp.up_proj.register_forward_hook(
#             lambda module, input, output, l=layer_idx: hook_fn(module, input, output, l)
#         )
#         hooks.append(hook)

#     for idx in tqdm(range(0, sample_size, batch_size)):
#         batch = tokenized_prompts[idx: idx + batch_size, :].to(device)

#         for timestep in range(20):  # Generate 20 tokens
#             with torch.inference_mode():
#                 _, cache = model(batch, output_hidden_states=True, return_dict=True)

#             # Ensure cache tensors are moved to the correct device
#             cache = {k: v.to(device).detach().clone() for k, v in cache.items()}

#             sampled = model.lm_head(cache.hidden_states[-1]).argmax(-1).detach().to(device)[:, -1]

#             for layer_idx in range(len(model.model.layers)):
#                 if layer_idx in neuron_acts_storage:
#                     neuron_acts = neuron_acts_storage[layer_idx]  # Get stored activations
#                 else:
#                     continue

#                 value_vectors = model.model.layers[layer_idx].mlp.down_proj.to(device)

#                 # Compute neuron outputs
#                 neuron_outputs = neuron_acts.unsqueeze(-1) * value_vectors
#                 neuron_projections = torch.matmul(neuron_outputs, toxic_vector.to(device)) / torch.norm(toxic_vector.to(device))

#                 for neuron_idx in range(neuron_projections.size(1)):
#                     model_neuron_projections[(layer_idx, neuron_idx)].extend(neuron_projections[:, neuron_idx].tolist())

#             batch = torch.concat([batch, sampled.unsqueeze(-1)], dim=-1)

#     # Remove all hooks
#     for hook in hooks:
#         hook.remove()

#     # Compute final average neuron projections
#     avg_neuron_projections = {
#         (layer_idx, neuron_idx): np.mean(projections)
#         for (layer_idx, neuron_idx), projections in model_neuron_projections.items()
#     }

#     return avg_neuron_projections





# Save results to csv file
def save_neuron_projections_to_csv(avg_neuron_projections, model_name):
    # Convert the dictionary to a list of tuples (layer_idx, neuron_idx, projection_value)
    data = [
        {"layer_idx": layer_idx, "neuron_idx": neuron_idx, "projection_value": projection}
        for (layer_idx, neuron_idx), projection in avg_neuron_projections.items()
    ]

    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    filename = f"{model_name}_neuron_projections.csv"
    df.to_csv(filename, index=False)
    print(f"Neuron projections saved to {filename}")





# Compute cossims of all neurons - very similar before and after DPO
def compute_all_neuron_cossims(model, toxic_vector, model_name):
    model_neuron_cossims = []

    # Iterate over all layers and all neurons in each layer
    for layer_idx in range(len(model.blocks)):
        # Get the weight matrix W_out for the current layer's MLP
        W_out = model.blocks[layer_idx].mlp.W_out  # [d_mlp, d_model]

        for neuron_idx in range(W_out.shape[0]):
            # Get the value vector for the specified neuron
            value_vector = W_out[neuron_idx]  # [d_model]

            # Compute the cosine similarity between the value vector and the toxic vector
            cossim = F.cosine_similarity(value_vector.unsqueeze(0), toxic_vector.unsqueeze(0), dim=1).item()

            # Store the layer index, neuron index, and computed cosine similarity
            model_neuron_cossims.append({
                "layer_idx": layer_idx,
                "neuron_idx": neuron_idx,
                "cosine_similarity": cossim
            })
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(model_neuron_cossims)

    # Generate the CSV filename using the model name
    csv_filename = f"{model_name}_neuron_cossims.csv"

    # Save the DataFrame to a CSV file with the generated filename
    df.to_csv(csv_filename, index=False)
    print(f"Cosine similarities saved to {csv_filename}")

    return df





def main():
    """Main execution pipeline."""
    # Compute and save neuron projections for the base model
    print("Processing pre-trained model...")
    avg_neuron_projections = compute_neuron_toxic_projection(model, tokenized_prompts, toxic_vector)
    save_neuron_projections_to_csv(avg_neuron_projections, model_short_name)
    # compute_all_neuron_cossims(model, toxic_vector, model_short_name)
    # Compute and save neuron projections for the DPO-trained model
    # print("Processing DPO model...")
    # avg_neuron_projections_dpo = compute_neuron_toxic_projection(dpo_model, tokenized_prompts, toxic_vector)
    # save_neuron_projections_to_csv(avg_neuron_projections_dpo, model_short_name + "_dpo")
    # compute_all_neuron_cossims(dpo_model, toxic_vector, model_short_name + "_dpo")


if __name__ == "__main__":
    main()