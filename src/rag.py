from pathlib import Path
import requests
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB_DIR = BASE_DIR / "vector_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

COLLECTION_NAME = "data_mining_unit1"

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# -----------------------------
# Load models
# -----------------------------

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading reranker...")
reranker = CrossEncoder(RERANKER_MODEL)


# -----------------------------
# Connect to ChromaDB
# -----------------------------

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Loaded {collection.count()} vectors.")


# -----------------------------
# Retrieve + rerank
# -----------------------------

def retrieve_and_rerank(query, initial_k=10, final_k=5):

    # Initial vector retrieval
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Create query-document pairs
    pairs = [
        [query, document]
        for document in documents
    ]

    # Reranker scores
    scores = reranker.predict(pairs)

    # Sort by reranker score
    ranked_results = sorted(
        zip(scores, documents, metadatas),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked_results[:final_k]


# -----------------------------
# Generate answer
# -----------------------------

def generate_answer(query, ranked_results):

    context_parts = []

    for i, (score, document, metadata) in enumerate(
        ranked_results,
        start=1
    ):

        context_parts.append(
            f"""
[Source {i}]
File: {metadata['source']}
Slide: {metadata['slide']}

Content:
{document}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a Data Mining Unit 1 assistant.

Answer the user's question ONLY using the provided context.

Do not use outside knowledge.
Do not invent information.

If the answer is not available in the context, say:

"I could not find this information in the provided Unit 1 materials."

Use the terminology and information from the provided material.

At the end of your answer, provide the relevant sources:

Sources:
- [filename, Slide X]
- [filename, Slide Y]

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
    )

    response.raise_for_status()

    return response.json()["response"]


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    query = input("\nAsk your question: ")

    print("\nRetrieving and reranking relevant information...")

    ranked_results = retrieve_and_rerank(
        query,
        initial_k=10,
        final_k=5
    )

    print("Generating answer using Llama 3.2...\n")

    answer = generate_answer(
        query,
        ranked_results
    )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)