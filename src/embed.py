from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
import chromadb


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"

VECTOR_DB_DIR = BASE_DIR / "vector_db"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTION_NAME = "data_mining_unit1"


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

def load_chunks():

    chunks = []

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            chunks.append(
                json.loads(line)
            )

    return chunks


# --------------------------------------------------
# CREATE VECTOR DATABASE
# --------------------------------------------------

def create_vector_database():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    print("\nLoading chunks...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    # --------------------------------------------------
    # CHROMA CLIENT
    # --------------------------------------------------

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    # Delete existing collection if it exists
    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print("Existing collection deleted.")

    except Exception:

        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        chunk["metadata"]["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    # --------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------

    print("\nCreating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    print("Embeddings created.")

    # --------------------------------------------------
    # STORE IN CHROMA
    # --------------------------------------------------

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("\nVector database created successfully.")

    print(f"Collection: {COLLECTION_NAME}")

    print(
        f"Number of vectors: "
        f"{collection.count()}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    create_vector_database()