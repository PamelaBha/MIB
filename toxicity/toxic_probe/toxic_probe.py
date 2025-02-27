import os
import torch
import numpy as np
import pandas as pd
import datasets
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change as needed
MODEL_NAME = "google/gemma-2-2b" # google/gemma-7b # "google/gemma-2-2b" #"meta-llama/Llama-3.1-8B" # "mistralai/Mistral-7B-v0.1" # "google/gemma-2-2b","gpt2-medium", "meta-llama/Llama-3.1-8B"
PROBE_NAME = "gemma_2_2b_probe.pt" # "gemma_probe.pt" # "llama3_probe.pt" # "mistral_probe.pt" 
BATCH_SIZE = 64 # Control memory usage

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Load model with float32 precision
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)

# Set model to evaluation mode
model.eval()

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Ensure tokenizer has a padding token
# if tokenizer.pad_token is None:
#     # if tokenizer.eos_token:
#     tokenizer.pad_token = tokenizer.eos_token
    # else:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to account for new token

# Load Jigsaw Toxicity dataset
dataset = load_dataset("jigsaw_toxicity_pred", data_dir="/data/kebl6672/dpo-toxic-general/data/jigsaw-toxic-comment-classification-challenge")

# Tokenize the dataset before splitting
def tokenize_batch(batch):
    return tokenizer(batch["comment_text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize_batch, batched=True)  # Ensure all data has 'input_ids' and 'attention_mask'

# Convert dataset to Pandas DataFrame
train_df = pd.DataFrame(dataset["train"])

# Split dataset (90:10 for training and validation)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Reset indices to avoid misalignment issues
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Extract text and labels
train_texts, train_labels = train_df["comment_text"].tolist(), train_df["toxic"].tolist()
val_texts, val_labels = val_df["comment_text"].tolist(), val_df["toxic"].tolist()


def extract_features(texts):
    """ Extracts residual stream features in batches. """
    all_features = []
    
    for i, text in enumerate(texts):
        inputs = tokenizer([text], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}  # Move batch to GPU

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # Check NaNs in Layer 1
            nan_count = torch.isnan(outputs.hidden_states[1]).sum().item()
            if nan_count > 0:
                print(f"WARNING: NaNs detected in Layer 1 for input {i}")
                print("Problematic text:", text)
                print("Tokenized input IDs:", inputs['input_ids'])
                print("Tokenized attention mask:", inputs['attention_mask'])

            last_hidden_states = outputs.hidden_states[-1].to(torch.float32)
            avg_hidden_state = last_hidden_states.mean(dim=1).cpu().numpy()

        all_features.append(avg_hidden_state)

        del inputs, outputs, last_hidden_states  # Free memory
        torch.cuda.empty_cache()  # Clear CUDA cache
    
    return np.vstack(all_features)


# def extract_features(texts):
#     """ Extracts residual stream features in batches. """
#     all_features = []
    
#     for i in range(0, len(texts), BATCH_SIZE):
#         batch_texts = texts[i : i + BATCH_SIZE]
#         inputs = tokenizer(batch_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
#         inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}  # Move batch to GPU

#         # with torch.no_grad():
#         #     outputs = model(**inputs, output_hidden_states=True)
#             # last_hidden_states = outputs.hidden_states[-1]  # Last layer activations
#             # # avg_hidden_state = last_hidden_states.mean(dim=1).cpu().numpy()  # Mean across all timesteps
            
#             # # Debug: Check for NaN values in hidden states
#             # nan_count = torch.isnan(last_hidden_states).sum().item()
#             # if nan_count > 0:
#             #     print(f"WARNING: Found {nan_count} NaNs in hidden states!")
            
#             last_hidden_states = outputs.hidden_states[-1]  # Last layer activations
#             # Convert to float32 explicitly to avoid BF16/FP16 issues
#             last_hidden_states = last_hidden_states.to(torch.float32)

#             # Compute mean across all timesteps
#             avg_hidden_state = last_hidden_states.mean(dim=1).cpu().numpy()

#         all_features.append(avg_hidden_state)
#         del inputs, outputs, last_hidden_states  # Free memory
#         torch.cuda.empty_cache()  # Clear CUDA cache
    
#     return np.vstack(all_features)

# Extract features for training and validation
print("Extracting train features...")
train_features = extract_features(train_texts)
print("Extracting validation features...")
val_features = extract_features(val_texts)

# Debug: Check for NaN values in extracted features
if np.isnan(train_features).sum() > 0:
    print("WARNING: NaN values detected in train features!")

if np.isnan(val_features).sum() > 0:
    print("WARNING: NaN values detected in validation features!")

# # Handle NaN values: Replace NaNs with 0
# train_features = np.nan_to_num(train_features)
# val_features = np.nan_to_num(val_features)

# # Ensure train_labels matches train_features
# if len(train_features) != len(train_labels):
#     print(f"WARNING: Mismatch in train set! Features: {len(train_features)}, Labels: {len(train_labels)}")
#     min_len = min(len(train_features), len(train_labels))
#     train_features = train_features[:min_len]
#     train_labels = train_labels[:min_len]

# # Ensure val_labels matches val_features
# if len(val_features) != len(val_labels):
#     print(f"WARNING: Mismatch in validation set! Features: {len(val_features)}, Labels: {len(val_labels)}")
#     min_len = min(len(val_features), len(val_labels))
#     val_features = val_features[:min_len]
#     val_labels = val_labels[:min_len]


# Convert labels to NumPy array
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

# Train a linear probe (Logistic Regression)
clf = LogisticRegression(max_iter=500)
clf.fit(train_features, train_labels)

# Save the learned probe vector (weights of logistic regression)
probe_vector = torch.tensor(clf.coef_, dtype=torch.float32)  # Shape: (1, hidden_dim)
torch.save(probe_vector, PROBE_NAME)
print("Toxicity probe vector saved.")

# Evaluate the probe model
val_preds = clf.predict(val_features)
accuracy = accuracy_score(val_labels, val_preds)
print(f"Validation Accuracy: {accuracy:.4f}")