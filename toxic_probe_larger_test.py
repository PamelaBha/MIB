import os
import torch
import numpy as np
import datasets
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose model
MODEL_NAME = "meta-llama/Llama-3.1-8B"  

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()  

# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to account for new token

# Load Jigsaw Toxicity dataset (train + test set) using `load_dataset`
dataset = load_dataset("jigsaw_toxicity_pred", data_dir="/data/kebl6672/dpo-toxic-general/data/jigsaw-toxic-comment-classification-challenge")

# Check available splits
print("Dataset splits available:", dataset.keys())

# Tokenize function
def tokenize_batch(batch):
    return tokenizer(batch["comment_text"], truncation=True, padding="max_length", max_length=128)

# Apply tokenization to train and test datasets
train_dataset = dataset["train"].map(tokenize_batch, batched=True)
test_dataset = dataset["test"].map(tokenize_batch, batched=True)

# Convert to DataFrame
train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_dataset)

# Filter out -1 labels from test set (since -1 indicates unlabeled samples)
test_df = test_df[test_df["toxic"] != -1]

# Reset indices to avoid misalignment issues
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Check dataset sizes
print(f"Train dataset size: {len(train_df)}")
print(f"Test dataset size: {len(test_df)}")

# Extract text and labels
train_texts, train_labels = train_df["comment_text"].tolist(), train_df["toxic"].tolist()
test_texts, test_labels = test_df["comment_text"].tolist(), test_df["toxic"].tolist()

# Process data in small batches to avoid memory issues
BATCH_SIZE = 64  # Adjust as needed

def extract_features(texts):
    """ Extracts residual stream features in batches. """
    all_features = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}  

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]  # Last layer activations
            avg_hidden_state = last_hidden_states.mean(dim=1).cpu().numpy()  # Mean across all timesteps

        all_features.append(avg_hidden_state)
        del inputs, outputs, last_hidden_states  # Free memory
        torch.cuda.empty_cache()  # Clear CUDA cache
    
    return np.vstack(all_features)

# Extract features for training and testing
print("Extracting train features...")
train_features = extract_features(train_texts)
print("Extracting test features...")
test_features = extract_features(test_texts)

# Convert labels to NumPy array
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Train a linear probe (Logistic Regression)
clf = LogisticRegression(max_iter=500)
clf.fit(train_features, train_labels)

# Save the learned probe vector (weights of logistic regression)
probe_vector = torch.tensor(clf.coef_, dtype=torch.float32)  # Shape: (1, hidden_dim)
torch.save(probe_vector, "llama_probe.pt")
print("Toxicity probe vector saved as 'llama_probe.pt'.")

# Evaluate the probe model
test_preds = clf.predict(test_features)
accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {accuracy:.4f}")
