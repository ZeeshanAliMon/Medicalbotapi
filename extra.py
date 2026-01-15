import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()
client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

def get_embedding(text: str):
    embedding = client.feature_extraction(
        text,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding
print(get_embedding("hello world"))