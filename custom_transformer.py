import torch
import torch.nn as nn
import math

print("--- Hardware Verification ---")
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Custom Architecture will execute on: {str(device).upper()}")
print("-" * 27)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of [max_len, d_model] to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the embedded tokens
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CustomToxicityTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_seq_length=128):
        super(CustomToxicityTransformer, self).__init__()
        
        # 1. Token Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_length)
        
        # 3. The Transformer Encoder Blocks
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=0.1, 
            batch_first=True # Keeps our data shape as [batch_size, sequence_length, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, 
                                                         enable_nested_tensor=False  # <--- THIS DISABLES THE BUGGY FAST-PATH
                                                         )
        
        # 4. Classification Head (Maps to our 6 toxicity labels)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        
        # Create a mask so the model ignores empty <PAD> tokens
        padding_mask = (x == 0)
        
        # Pass through embedding and positional encoding
        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoder(embedded)
        
        # Pass through the Transformer
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # Extract the representation of the very first token (often called the [CLS] token) 
        # to represent the entire sentence's meaning
        sentence_representation = transformer_out[:, 0, :]
        
        # Pass through final linear layers to get our 6 toxicity scores
        logits = self.classifier(sentence_representation)
        return logits

# --- Let's test if the architecture compiles! ---
if __name__ == "__main__":
    # Hyperparameters for a "Mini" Transformer (much smaller than RoBERTa)
    VOCAB_SIZE = 30522 # Standard BERT vocabulary size
    EMBED_DIM = 256
    NUM_HEADS = 8
    HIDDEN_DIM = 512
    NUM_LAYERS = 3
    NUM_CLASSES = 6
    
    # Instantiate the model and move it to the Intel Arc GPU
    model = CustomToxicityTransformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS, 
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS, 
        num_classes=NUM_CLASSES
    ).to(device)
    
    # Create some fake dummy data (e.g., a batch of 4 sentences, each 128 words long)
    dummy_input = torch.randint(0, VOCAB_SIZE, (4, 128)).to(device)
    
    print("Feeding dummy data into the custom Transformer...")
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape} -> [Batch Size, Number of Labels]")
    print("Architecture compiled successfully!")