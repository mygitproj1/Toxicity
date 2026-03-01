import pandas as pd
import numpy as np
import torch
import intel_extension_for_pytorch as ipex  # The magic key to unlock your Arc GPU!
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report

print("--- Hardware Verification ---")
device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"Deep Learning will execute on: {device.upper()}")
print("-" * 27)

print("\n1. Loading the local train.csv dataset...")
df = pd.read_csv("data/train.csv").sample(10000, random_state=42)

toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("2. Splitting data and preparing for Hugging Face...")
X = df['comment_text'].fillna("").astype(str)
y = df[toxicity_labels].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert Pandas Dataframes to Hugging Face Datasets
train_dataset = Dataset.from_dict({"text": X_train.values, "labels": y_train.values})
test_dataset = Dataset.from_dict({"text": X_test.values, "labels": y_test.values})

print("3. Tokenizing text...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

print("4. Loading the pre-trained RoBERTa Model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", 
    problem_type="multi_label_classification", 
    num_labels=len(toxicity_labels)
)

print("5. Defining evaluation metrics...")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits)) 
    predictions = (probs > 0.5).astype(int)
    
    print("\n--- RoBERTa Classification Report ---")
    print(classification_report(labels, predictions, target_names=toxicity_labels, zero_division=0))
    return {}

print("6. Setting up the Training Engine...")
training_args = TrainingArguments(
    output_dir="./roberta_checkpoints",
    eval_strategy="epoch",       
    learning_rate=2e-5,          
    per_device_train_batch_size=16, # If your GPU runs out of VRAM, lower this to 8
    num_train_epochs=3,          
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

print("7. Fine-tuning RoBERTa on Intel Arc GPU...")
trainer.train()

print("8. Saving the final deep learning model...")
trainer.save_model("./roberta_toxicity_final")
tokenizer.save_pretrained("./roberta_toxicity_final")
print("Pipeline Complete! Model saved for inference.")
