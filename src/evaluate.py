from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

COLLECTION_NAME = "data_mining_unit1"

INITIAL_K = 10
FINAL_K = 5


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading reranker...")
reranker = CrossEncoder(RERANKER_MODEL)


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Loaded {collection.count()} vectors.\n")


# ============================================================
# EVALUATION DATASET
# ============================================================

evaluation_data = [

    {
        "question": "What is data mining?",
        "relevant_slides": {
            ("1.1 DM-intro.pptx", 7),
            ("1.1 DM-intro.pptx", 8)
        }
    },

    {
        "question": "Why is data mining needed?",
        "relevant_slides": {
            ("1.1 DM-intro.pptx", 2)
        }
    },

    {
        "question": "What are the steps involved in the KDD process?",
        "relevant_slides": {
            ("1.1 DM-intro.pptx", 15),
            ("1.1 DM-intro.pptx", 16),
            ("1.1 DM-intro.pptx", 17),
            ("1.1 DM-intro.pptx", 25)
        }
    },

    {
        "question": "What is the role of data cleaning in KDD?",
        "relevant_slides": {
            ("1.1 DM-intro.pptx", 15)
        }
    },

    {
        "question": "What are the major applications of data mining?",
        "relevant_slides": {
            ("1.1 DM-intro.pptx", 11)
        }
    },

    {
        "question": "What is classification in data mining?",
        "relevant_slides": {
            ("1.3 DM-functionalities22.pptx", 12)
        }
    },

    {
        "question": "What is clustering in data mining?",
        "relevant_slides": {
            ("1.3 DM-functionalities22.pptx", 17)
        }
    },

    {
        "question": "What are some biological applications of data mining?",
        "relevant_slides": {
            ("1.1 DM-intro.pptx", 22),
            ("1.1 DM-intro.pptx", 23)
        }
    },

    {
        "question": "What is the relationship between KDD and data mining?",
        "relevant_slides": {
            ("1.1 DM-intro.pptx", 8),
            ("1.1 DM-intro.pptx", 17),
            ("1.1 DM-intro.pptx", 25)
        }
    },

    {
        "question": "How is fraud detection an application of data mining?",
        "relevant_slides": {
            ("1.1 DM-intro.pptx", 11)
        }
    }
]


# ============================================================
# EVALUATION
# ============================================================

recall_scores = []
reciprocal_ranks = []


for number, item in enumerate(evaluation_data, start=1):

    question = item["question"]
    relevant_slides = item["relevant_slides"]

    # --------------------------------------------------------
    # Initial vector retrieval
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=INITIAL_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    pairs = [
        [question, document]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(scores, documents, metadatas),
        key=lambda x: x[0],
        reverse=True
    )

    top_results = ranked_results[:FINAL_K]

    # --------------------------------------------------------
    # Find relevant results
    # --------------------------------------------------------

    retrieved_relevant = []

    first_relevant_rank = None

    for rank, (score, document, metadata) in enumerate(
        top_results,
        start=1
    ):

        slide_key = (
            metadata["source"],
            metadata["slide"]
        )

        if slide_key in relevant_slides:

            retrieved_relevant.append(slide_key)

            if first_relevant_rank is None:
                first_relevant_rank = rank

    # --------------------------------------------------------
    # Recall@5
    # --------------------------------------------------------

    recall_at_5 = (
        len(set(retrieved_relevant))
        / len(relevant_slides)
    )

    # --------------------------------------------------------
    # Reciprocal Rank
    # --------------------------------------------------------

    if first_relevant_rank is not None:
        reciprocal_rank = 1 / first_relevant_rank
    else:
        reciprocal_rank = 0

    recall_scores.append(recall_at_5)
    reciprocal_ranks.append(reciprocal_rank)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("=" * 80)
    print(f"QUESTION {number}")
    print(question)
    print("=" * 80)

    print(
        f"Recall@5: {recall_at_5:.2f}"
    )

    if first_relevant_rank:
        print(
            f"First relevant result: Rank {first_relevant_rank}"
        )
    else:
        print(
            "First relevant result: Not found"
        )

    print("\nRetrieved:")

    for rank, (score, document, metadata) in enumerate(
        top_results,
        start=1
    ):

        marker = ""

        slide_key = (
            metadata["source"],
            metadata["slide"]
        )

        if slide_key in relevant_slides:
            marker = "  <-- RELEVANT"

        print(
            f"{rank}. "
            f"{metadata['source']} | "
            f"Slide {metadata['slide']} | "
            f"Score: {score:.4f}"
            f"{marker}"
        )

    print()


# ============================================================
# FINAL METRICS
# ============================================================

mean_recall = sum(recall_scores) / len(recall_scores)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)


print("\n" + "=" * 80)
print("FINAL RETRIEVAL EVALUATION")
print("=" * 80)

print(
    f"Number of questions : {len(evaluation_data)}"
)

print(
    f"Mean Recall@5      : {mean_recall:.3f}"
)

print(
    f"MRR                 : {mrr:.3f}"
)

print("=" * 80)