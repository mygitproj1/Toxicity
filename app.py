import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Configuration
st.set_page_config(page_title="ToxicShield AI (RoBERTa)", page_icon="🛡️", layout="centered")
st.title("🛡️ ToxicShield: Deep Learning Moderation")
st.markdown("Powered by RoBERTa. Enter a message below to evaluate it against the toxicity taxonomy.")

# 2. Load Deep Learning Model (Cached for speed)
@st.cache_resource
def load_model():
    model_path = "./roberta_toxicity_final"
    
    # Load tokenizer and model from your local folder
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Automatically detect Intel Arc GPU (XPU) for fast real-time inference
    device = "cpu"
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            device = "xpu"
            model = model.to(device)
    except ImportError:
        pass # Fallback to CPU if Intel extension isn't found
        
    return tokenizer, model, device

try:
    tokenizer, model, device = load_model()
except Exception as e:
    st.error("⚠️ Model not found! Please wait for the PyTorch training script to finish generating the 'roberta_toxicity_final' folder.")
    st.stop()

toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 3. The User Interface
user_input = st.text_area("Live Sandbox:", placeholder="Type a simulated chat message here...")

if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Step A: Tokenize the input text
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128, padding=True)
        
        # Move inputs to the same hardware (XPU or CPU) as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Step B: Run the prediction
        with torch.no_grad(): # Disable gradient calculation for faster inference
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Step C: Convert raw logits to percentages using the Sigmoid function
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        st.markdown("### RoBERTa Toxicity Analysis")
        
        # Display results with visual progress bars
        for label, prob in zip(toxicity_labels, probs):
            display_label = label.replace('_', ' ').title()
            percentage = int(prob * 100)
            
            st.write(f"**{display_label}:** {percentage}%")
            
            if percentage > 50:
                st.progress(float(prob), text="🚨 Flagged")
            elif percentage > 20:
                st.progress(float(prob), text="⚠️ Warning")
            else:
                st.progress(float(prob), text="✅ Safe")