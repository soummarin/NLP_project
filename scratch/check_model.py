from transformers import BartTokenizer, BartForConditionalGeneration
import torch

MODEL_REPO = "soumia11111/summarizer"

try:
    print(f"Attempting to load tokenizer and model from {MODEL_REPO}...")
    tokenizer = BartTokenizer.from_pretrained(MODEL_REPO)
    model = BartForConditionalGeneration.from_pretrained(MODEL_REPO)
    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
