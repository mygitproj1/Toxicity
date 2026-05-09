import torch
from transformers import AutoTokenizer
from custom_transformer import CustomToxicityTransformer

print("--- Hardware Verification ---")
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Device: {str(device).upper()}")
print("-" * 27)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = CustomToxicityTransformer(
    vocab_size=tokenizer.vocab_size, embed_dim=256, num_heads=8, hidden_dim=512, num_layers=3, num_classes=6
)

# Load trained weights
print("Loading model weights...")
model.load_state_dict(torch.load("custom_transformer_weights.pt", weights_only=True))
model = model.to(device)

# Set model to evaluation mode
model.eval()

toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


print("="*50)
print("Toxicity Inference Sandbox")
print("Type 'quit' or 'exit' to stop.")
print("="*50 + "\n")

while True:
    user_text = input("Enter a chat message to evaluate (type 'quit' or 'exit' to stop): ")
    if user_text.lower() in ['quit', 'exit']:
        print("Shutting down inference engine...")
        break

    # Tokenize input and move to device
    inputs = tokenizer(user_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)

    # Run model inference
    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Show results
    threshold = 0.5
    if (probabilities > threshold).any():
        print("\n--- Toxicity Report ---")
        for label, prob in zip(toxicity_labels, probabilities):
            print(f"{label.capitalize().replace('_', ' ')}: {prob * 100:.2f}%")
        print("-" * 23 + "\n")
    else:
        print("Non-toxic.\n")