import requests
from openai import OpenAI
import ollama

# ===================
# 1. getting embeddings using ollama and OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',  # Ollama's endpoint
    api_key='ollama'  # Required but unused by Ollama
)
def get_ollama_embedding_openai(text):
    response = client.embeddings.create(
        input=text,
        model="nomic-embed-text:latest"  # Use any embedding model you have installed
    )
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding

# ===================
# 2. getting embeddings directly from Ollama

def get_ollama_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text:latest",
        input=text
    )
    embedding = response["embeddings"][0]  # Extract the embedding vector
    return embedding

# Batch processing (more efficient)
def get_ollama_embeddings_batch(texts):
    response = ollama.embed(
        model="nomic-embed-text:latest",
        input=texts  # Pass list of texts
    )
    embeddings = response["embeddings"]
    print(f"==== Generated {len(embeddings)} embeddings... ====")
    return embeddings

# ===================
# 3. Using REST APIs

def get_ollama_embedding_api(text):
    response = requests.post(
        'http://localhost:11434/api/embed',
        json={
            'model': 'nomic-embed-text:latest',
            'input': text
        }
    )
    embedding = response.json()['embeddings'][0]
    print("==== Generating embeddings... ====")
    return embedding