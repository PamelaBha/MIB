import os
import sys

os.chdir('/data/kebl6672/dpo-toxic-general/toxicity')
sys.path.append('/data/kebl6672/dpo-toxic-general/toxicity')

import json
import torch
import torch.nn.functional as F
from transformer_lens import (
    HookedTransformer,
)

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
from fig_utils import load_hooked
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer

device = torch.device("cuda") 
ROOT_DIR = '/data/kebl6672/dpo-toxic-general/checkpoints'

model_name = "meta-llama/Llama-2-7b-hf" # "meta-llama/Llama-3.1-8B" # "google/gemma-2-2b", # "gpt2-medium", # "mistralai/Mistral-7B-v0.1",
dpo_model_name = "llama_dpo.pt"
probe_name = "llama_probe.pt"
model_short_name = "llama2"

# Load the pre-trained model
# For GPT2-medium
# model = HookedTransformer.from_pretrained(model_name, hf_model=True)
# model.tokenizer.padding_side = "left"
# model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
# model.to(device)  

# For llama-2-7B
hf_model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
model = HookedTransformer.from_pretrained(model_name, hf_model=hf_model, device="cuda" if torch.cuda.is_available() else "cpu", fold_ln=False, center_writing_weights=False, center_unembed=False)
model.to(device)  

# Set tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model.set_tokenizer(tokenizer)


# Load the DPO-ed model
# dpo_model = load_hooked(model_name, os.path.join(ROOT_DIR, dpo_model_name))
# dpo_model.to(device)

# Load the toxic probe vector
toxic_vector = torch.load(os.path.join(ROOT_DIR, probe_name)).to(device)  


# Ensure toxic_vector has correct dimensions
if toxic_vector.dim() == 1:
    toxic_vector = toxic_vector.unsqueeze(1)  # Convert (d_model,) to (d_model, 1)
elif toxic_vector.shape[0] == 1:
    toxic_vector = toxic_vector.T  # Transpose (1, d_model) to (d_model, 1)


# load evaluation data
DATA_DIR = '/data/kebl6672/dpo-toxic-general/data/intervene_data'

with open(
    os.path.join(DATA_DIR, "challenge_prompts.jsonl"), "r"
) as file_p:
    data = file_p.readlines()

prompts = [json.loads(x.strip())["prompt"] for x in data]

tokenized_prompts = model.to_tokens(prompts, prepend_bos=True).cuda()     




# Compute the neuron toxicity projection 
def compute_neuron_toxic_projection(model, tokenized_prompts, toxic_vector, batch_size=16):
    # Initialize dictionaries to store projections and activations for all layers
    model_neuron_projections = defaultdict(list)

    sample_size = tokenized_prompts.size(0)

    print("Computing MLP neuron projections...")
    
    device = next(model.parameters()).device  # Get the model's device

    for idx in tqdm(range(0, sample_size, batch_size)):
        batch = tokenized_prompts[idx : idx + batch_size, :].to(device)

        for timestep in range(20):  # generate 20 tokens
            with torch.inference_mode():
                _, cache = model.run_with_cache(batch)

            # Ensure cache tensors are moved to the correct device
            cache = {k: v.to(device).detach().clone() for k, v in cache.items()}

            sampled = model.unembed(cache["ln_final.hook_normalized"]).argmax(-1).detach().to(device)[:, -1]

            for layer_idx in range(len(model.blocks)):
                neuron_acts = cache[f"blocks.{layer_idx}.mlp.hook_post"][:, -1, :].to(device)
                value_vectors = model.blocks[layer_idx].mlp.W_out.to(device)

                neuron_outputs = neuron_acts.unsqueeze(-1) * value_vectors
                neuron_projections = torch.matmul(neuron_outputs, toxic_vector.to(device)) / torch.norm(toxic_vector.to(device))

                for neuron_idx in range(neuron_projections.size(1)):
                    model_neuron_projections[(layer_idx, neuron_idx)].extend(neuron_projections[:, neuron_idx].tolist())

            batch = torch.concat([batch, sampled.unsqueeze(-1)], dim=-1)

    # Compute final average neuron projections and average activations across all batches and tokens 
    avg_neuron_projections = {
        (layer_idx, neuron_idx): np.mean(projections)
        for (layer_idx, neuron_idx), projections in model_neuron_projections.items()
    }

    return avg_neuron_projections





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
            cossim = F.cosine_similarity(value_vector.unsqueeze(0), toxic_vector.T, dim=1).item()

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
    compute_all_neuron_cossims(model, toxic_vector, model_short_name)
    # Compute and save neuron projections for the DPO-trained model
    # print("Processing DPO model...")
    # avg_neuron_projections_dpo = compute_neuron_toxic_projection(dpo_model, tokenized_prompts, toxic_vector)
    # save_neuron_projections_to_csv(avg_neuron_projections_dpo, model_short_name + "_dpo")
    # compute_all_neuron_cossims(dpo_model, toxic_vector, model_short_name + "_dpo")


if __name__ == "__main__":
    main()