from pathlib import Path
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

print(f"Loaded collection with {collection.count()} vectors.")


# -----------------------------
# Search + rerank
# -----------------------------

def search(query, initial_k=10, final_k=5):

    # Step 1: Vector retrieval
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Step 2: Create query-document pairs
    pairs = [
        [query, document]
        for document in documents
    ]

    # Step 3: Cross-encoder scoring
    scores = reranker.predict(pairs)

    # Step 4: Sort by reranker score
    ranked_results = sorted(
        zip(scores, documents, metadatas),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked_results[:final_k]


# -----------------------------
# Test retrieval
# -----------------------------

if __name__ == "__main__":

    query = "What are the steps involved in the KDD process?"

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = search(
        query,
        initial_k=10,
        final_k=5
    )

    for rank, (score, document, metadata) in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 70)
        print(f"RESULT {rank}")
        print(f"Reranker Score : {score:.4f}")
        print(f"Source         : {metadata['source']}")
        print(f"Slide          : {metadata['slide']}")

        print("\nTEXT:")
        print(document)