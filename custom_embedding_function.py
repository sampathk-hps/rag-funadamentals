import os
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

# 1. to get OpenAI embedding function
# openai_key = os.getenv("OPENAI_API_KEY")
# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#     api_key=openai_key, model_name="text-embedding-3-small"
# )

# ===================
# 2. Using sentence-transformers models
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # model_name="BAAI/bge-base-en-v1.5"  # Better performance
)

# ===================
# 3. Custom embedding function
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # Convert numpy arrays to Python lists (required by ChromaDB)
        return self.model.encode(input).tolist()

# Usage
# custom_ef = LocalEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")

# ===================
# 4. Hugging Face embedding function
# Hugging Face models via transformers
hf_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="your_hf_token",  # Optional for public models
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===================
# 5. Ollama embedding function
# Requires Ollama to be running locally
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text:latest",
)
