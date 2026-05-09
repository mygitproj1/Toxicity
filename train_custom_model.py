import os
os.environ["MLFLOW_TRACKING_URI"] = "file:///C:/mlruns"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import mlflow
import os


# Import the architecture we built in the previous step
from custom_transformer import CustomToxicityTransformer 

print("--- Hardware Verification ---")
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Device: {str(device).upper()}")
print("-" * 27)

print("1. Loading data and tokenizing...")
df = pd.read_csv("data/train.csv").sample(10000, random_state=42)
toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

X = df['comment_text'].fillna("").astype(str).tolist()
y = df[toxicity_labels].astype(float).values

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenized_X = tokenizer(X, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

print("2. Building PyTorch DataLoaders...")
dataset = TensorDataset(tokenized_X['input_ids'], torch.tensor(y, dtype=torch.float32))

# Split 80/20 for Train/Validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Batch size of 16 (Lower this to 8 if your GPU throws an OutOfMemory error)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("3. Initializing custom transformer model...")
model = CustomToxicityTransformer(
    vocab_size=tokenizer.vocab_size, 
    embed_dim=256, 
    num_heads=8, 
    hidden_dim=512, 
    num_layers=3, 
    num_classes=6
).to(device)

criterion = nn.BCEWithLogitsLoss() # Multi-label classification loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

## Removed ipex.optimize for cross-platform compatibility

EPOCHS = 3
print(f"\nStarting {EPOCHS} epochs of training...")

os.environ["MLFLOW_EXPERIMENT_NAME"] = "Custom_Transformer_Architecture"

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("optimizer", "AdamW")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (input_ids, labels) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(input_ids)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"End of epoch {epoch+1} | Average training loss: {avg_train_loss:.4f}")
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

print("\nTraining complete. Saving model weights...")
torch.save(model.state_dict(), "custom_transformer_weights.pt")