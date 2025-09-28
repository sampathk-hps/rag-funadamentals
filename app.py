import os
import chromadb
import ollama

from custom_embedding_function import ollama_ef
from embeddings import get_ollama_embedding

# get the embedding function
custom_ef = ollama_ef

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="news_article_storage")
collection_name = "news_article_collection"
chroma_collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=custom_ef
)

# Function to load documents from a directory
def load_documents_from_directory(directory_path_name):
    print("==== Loading documents from directory ====")
    documents_arr = []
    for filename in os.listdir(directory_path_name):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path_name, filename), "r", encoding="utf-8"
            ) as file:
                documents_arr.append({"id": filename, "text": file.read()})
    return documents_arr

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks_arr = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks_arr.append(text[start:end])
        start = end - chunk_overlap
    return chunks_arr

# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print("Loaded {} documents".format(len(documents)))

# Split documents into chunks
chunked_documents = []
print("==== Splitting docs into chunks ====")
for doc in documents:
    chunks = split_text(doc["text"])
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

print("Chunk size: {}".format(len(chunked_documents)))

# Generate embeddings for the document chunks
print("==== Generating embeddings... ====")
for doc in chunked_documents:
    doc["embedding"] = get_ollama_embedding(doc["text"])

# Upsert documents with embeddings into Chroma
print("==== Inserting chunks into db ====")
for doc in chunked_documents:
    chroma_collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )

# Function to query documents
def query_documents(question, n_results=2):
    """
    Perform semantic search on the ChromaDB collection and
    return the top N relevant document texts.
    """
    results = chroma_collection.query(query_texts=question, n_results=n_results)
    # Flatten and return the list of document texts
    relevant_chunks = sum(results["documents"], [])
    print("==== Returning relevant chunks ====")
    return relevant_chunks

# def query_documents(question, n_results=2):
#     # query_embedding = get_openai_embedding(question)
#     results = chroma_collection.query(query_texts=question, n_results=n_results)
#
#     # Extract the relevant chunks
#     relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
#     print("==== Returning relevant chunks ====")
#     return relevant_chunks
#     # for idx, document in enumerate(results["documents"][0]):
#     #     doc_id = results["ids"][0][idx]
#     #     distance = results["distances"][0][idx]
#     #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")

# Generate response using LLM
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = ollama.chat(
        model="qwen2.5-coder:7b",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.message.content
    return answer

# Example query and response generation
qn = "tell me about AI replacing TV writers strike."
# qn = "tell me about databricks"
# qn = "who is Justin Pihlaja?"
# qn = "tell me more about review of antitrust regulator"
relevant_chunks_retrieved = query_documents(qn)
answer_received = generate_response(qn, relevant_chunks_retrieved)

print(answer_received)