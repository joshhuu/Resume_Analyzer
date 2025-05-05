from transformers import pipeline

# Hugging Face login handled via CLI or env token
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device="cpu")
response = pipe("Tell me what you do.", max_new_tokens=100, do_sample=True)
print(response)
