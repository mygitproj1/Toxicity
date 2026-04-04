import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoTokenizer
from custom_transformer import CustomToxicityTransformer

print("--- Hardware Verification ---")
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Inference Engine executing on: {str(device).upper()}")
print("-" * 27)

# 1. Initialize the Tokenizer and Blank Architecture
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = CustomToxicityTransformer(
    vocab_size=tokenizer.vocab_size, embed_dim=256, num_heads=8, hidden_dim=512, num_layers=3, num_classes=6
)

# 2. Load Your Custom Trained Weights
print("Loading 'brain' into the architecture...")
model.load_state_dict(torch.load("custom_transformer_weights.pt", weights_only=True))
model = model.to(device)

# 3. Lock the Model (CRITICAL)
model.eval()

toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("\n" + "="*50)
print("🛡️ ToxicShield AI: Live Inference Sandbox")
print("Type 'quit' or 'exit' to stop.")
print("="*50 + "\n")

# 4. The Interactive Loop
while True:
    user_text = input("Enter a chat message to evaluate: ")
    
    if user_text.lower() in ['quit', 'exit']:
        print("Shutting down inference engine...")
        break

    # Step A: Tokenize the user's text into exact tensor shapes
    inputs = tokenizer(user_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)

    # Step B: The Forward Pass (No Backpropagation!)
    with torch.no_grad():
        logits = model(input_ids)
        
        # Step C: Mathematical translation from raw numbers to probabilities
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Step D: Print the final enterprise report
    print("\n--- Toxicity Report ---")
    for label, prob in zip(toxicity_labels, probabilities):
        print(f"{label.capitalize().replace('_', ' ')}: {prob * 100:.2f}%")
    print("-" * 23 + "\n")