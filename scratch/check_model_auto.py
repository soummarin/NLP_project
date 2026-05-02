from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_REPO = "soumia11111/summarizer"

try:
    print(f"Attempting to load tokenizer and model from {MODEL_REPO} using Auto classes...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO)
    print("✅ Model and tokenizer loaded successfully!")
    
    # Try a simple summarization
    text = "The quick brown fox jumps over the lazy dog. This is a test sentence for summarization."
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Summary test: {summary}")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
