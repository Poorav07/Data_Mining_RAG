from pathlib import Path
import requests
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

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


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

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Loaded {collection.count()} vectors.\n")


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

evaluation_data = [

    {
        "question": "What is data mining?",
        "expected": [
            "extracting knowledge from data",
            "patterns",
            "large amounts of data"
        ]
    },

    {
        "question": "What are the steps involved in the KDD process?",
        "expected": [
            "Data Cleaning",
            "Data Integration",
            "Data Selection",
            "Data Transformation",
            "Data Mining",
            "Pattern Evaluation",
            "Knowledge Presentation"
        ]
    },

    {
        "question": "What are the major applications of data mining?",
        "expected": [
            "Market Analysis",
            "Risk Management",
            "Fraud Detection"
        ]
    },

    {
        "question": "What is classification in data mining?",
        "expected": [
            "classification",
            "known class labels",
            "training set"
        ]
    },

    {
        "question": "What is clustering in data mining?",
        "expected": [
            "clustering",
            "groups",
            "similarities"
        ]
    },

    {
        "question": "What are some biological applications of data mining?",
        "expected": [
            "sequence analysis",
            "genome annotation",
            "gene",
            "protein"
        ]
    },

    {
        "question": "What is the relationship between KDD and data mining?",
        "expected": [
            "KDD",
            "data mining",
            "knowledge discovery"
        ]
    },

    {
        "question": "What is data cleaning in the KDD process?",
        "expected": [
            "remove noise",
            "inconsistent data"
        ]
    },

    {
        "question": "What is pattern evaluation in KDD?",
        "expected": [
            "interesting patterns",
            "knowledge",
            "interestingness"
        ]
    },

    {
        "question": "What is knowledge presentation in KDD?",
        "expected": [
            "visualization",
            "knowledge representation",
            "user"
        ]
    }
]


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_and_rerank(query):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    pairs = [
        [query, document]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(scores, documents, metadatas),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked_results[:5]


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, ranked_results):

    context = ""

    for i, (score, document, metadata) in enumerate(
        ranked_results,
        start=1
    ):

        context += f"""
[Source {i}]
File: {metadata['source']}
Slide: {metadata['slide']}

{document}

"""


    prompt = f"""
You are a Data Mining Unit 1 assistant.

Answer the question ONLY using the supplied context.

Do not use outside knowledge.
Do not invent information.

Question:
{question}

Context:
{context}

Answer:
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


# ============================================================
# EVALUATION
# ============================================================

scores = []

for number, item in enumerate(
    evaluation_data,
    start=1
):

    question = item["question"]
    expected = item["expected"]

    print("=" * 80)
    print(f"QUESTION {number}")
    print(question)
    print("=" * 80)

    ranked_results = retrieve_and_rerank(
        question
    )

    answer = generate_answer(
        question,
        ranked_results
    )

    answer_lower = answer.lower()

    matched = []

    for keyword in expected:

        if keyword.lower() in answer_lower:
            matched.append(keyword)

    score = len(matched) / len(expected)

    scores.append(score)

    print("\nANSWER:")
    print(answer)

    print("\nExpected concepts found:")

    for keyword in expected:

        if keyword in matched:
            print(f"✓ {keyword}")
        else:
            print(f"✗ {keyword}")

    print(
        f"\nConcept Coverage: {score:.2f}"
    )

    print()


# ============================================================
# FINAL RESULT
# ============================================================

average_score = sum(scores) / len(scores)

print("=" * 80)
print("FINAL ANSWER EVALUATION")
print("=" * 80)

print(
    f"Questions evaluated : {len(evaluation_data)}"
)

print(
    f"Average Concept Coverage : {average_score:.3f}"
)

print("=" * 80)