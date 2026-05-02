from huggingface_hub import list_repo_files

MODEL_REPO = "soumia11111/summarizer"

try:
    files = list_repo_files(MODEL_REPO)
    print(f"Files in {MODEL_REPO}:")
    for f in files:
        print(f" - {f}")
except Exception as e:
    print(f"❌ Failed to list files: {e}")
