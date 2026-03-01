import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report

print("1. Loading the local train.csv dataset...")
df = pd.read_csv("data/train.csv").sample(10000, random_state=42)

toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("2. Splitting data and preparing for Hugging Face...")
# Convert labels to floats (Strict requirement by PyTorch for multilabel math)
X = df['comment_text'].fillna("").astype(str)
y = df[toxicity_labels].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert standard Pandas dataframes directly into Hugging Face Datasets
train_dataset = Dataset.from_dict({"text": X_train.values, "labels": y_train.values})
test_dataset = Dataset.from_dict({"text": X_test.values, "labels": y_test.values})

print("3. Tokenizing text...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    # RoBERTa requires every sentence to be padded/truncated to a uniform length
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

print("4. Loading the pre-trained RoBERTa Model...")
# Setting problem_type forces the model to use the correct Loss Function (BCEWithLogitsLoss)
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", 
    problem_type="multi_label_classification", 
    num_labels=len(toxicity_labels)
)

print("5. Defining evaluation metrics...")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply sigmoid to convert raw model outputs (logits) into percentages, then threshold at 50%
    probs = 1 / (1 + np.exp(-logits)) 
    predictions = (probs > 0.5).astype(int)
    
    print("\n--- RoBERTa Classification Report ---")
    print(classification_report(labels, predictions, target_names=toxicity_labels, zero_division=0))
    return {}

print("6. Setting up the Training Engine...")
training_args = TrainingArguments(
    output_dir="./roberta_checkpoints",
    eval_strategy="epoch",       # Run our metrics function at the end of each epoch
    learning_rate=2e-5,          # Standard learning rate for fine-tuning
    per_device_train_batch_size=16,
    num_train_epochs=3,          # 3 passes over the data is the sweet spot
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

print("7. Fine-tuning RoBERTa (This requires heavy computation!)...")
trainer.train()

print("8. Saving the final deep learning model...")
trainer.save_model("./roberta_toxicity_final")
tokenizer.save_pretrained("./roberta_toxicity_final")
print("Pipeline Complete!")