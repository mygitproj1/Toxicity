import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

from custom_transformer import CustomToxicityTransformer

print("--- Hardware Verification ---")
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Evaluation executing on: {str(device).upper()}")
print("-" * 27)

# 1. Load the Exact Same Data and Split
print("Loading Validation Data...")
df = pd.read_csv("data/train.csv").sample(10000, random_state=42)
toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

X = df['comment_text'].fillna("").astype(str).tolist()
y = df[toxicity_labels].astype(float).values

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenized_X = tokenizer(X, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Recreate the exact 80/20 split using the same random seed logic
dataset = TensorDataset(tokenized_X['input_ids'], torch.tensor(y, dtype=torch.float32))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 2. Initialize the Blank Architecture
print("Initializing Custom Transformer Architecture...")
model = CustomToxicityTransformer(
    vocab_size=tokenizer.vocab_size, embed_dim=256, num_heads=8, hidden_dim=512, num_layers=3, num_classes=6
)

# 3. Load the Trained Weights
print("Loading trained weights into the architecture...")
model.load_state_dict(torch.load("custom_transformer_weights.pt", weights_only=True))
model = model.to(device)

# 4. Set to Evaluation Mode (CRITICAL)
model.eval()

# 5. The Evaluation Loop
all_predictions = []
all_labels = []

print("Running Inference on unseen validation data...")
# Disable gradient calculations to save memory and compute
with torch.no_grad():
    for input_ids, labels in val_loader:
        input_ids = input_ids.to(device)
        
        # Forward Pass
        logits = model(input_ids)
        
        # Apply Sigmoid to get probabilities between 0 and 1
        probs = torch.sigmoid(logits)
        
        # Convert probabilities to 1 or 0 using a 0.5 threshold
        preds = (probs > 0.5).int()
        
        # Move back to CPU for Scikit-Learn evaluation
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# 6. Generate Metrics
macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

print("\n=== Custom Transformer Evaluation Results ===")
print(f"Macro F1-Score: {macro_f1:.4f}\n")
print(classification_report(all_labels, all_predictions, target_names=toxicity_labels, zero_division=0))

# --- CONFUSION MATRIX CODE ---
print("\nGenerating Multi-Label Confusion Matrices...")

# 1. Calculate the matrices
mcm = multilabel_confusion_matrix(all_labels, all_predictions)

# 2. Set up the visual plot (2 rows, 3 columns to fit all 6 categories)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Custom Transformer: Toxicity Confusion Matrices', fontsize=16)

# 3. Loop through each of the 6 categories and plot them
for i, (ax, matrix, label) in enumerate(zip(axes.flatten(), mcm, toxicity_labels)):
    # Create a heatmap for each 2x2 matrix
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    ax.set_title(label.capitalize().replace('_', ' '))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

# 4. Show the plot
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

print("Pipeline Complete!")