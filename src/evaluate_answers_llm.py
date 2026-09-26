from pathlib import Path
import requests
import json
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

questions = [
    "What is data mining?",
    "What are the steps involved in the KDD process?",
    "What are the major applications of data mining?",
    "What is classification in data mining?",
    "What is clustering in data mining?",
    "What are some biological applications of data mining?",
    "What is the relationship between KDD and data mining?",
    "What is data cleaning in the KDD process?",
    "What is pattern evaluation in KDD?",
    "What is knowledge presentation in KDD?"
]


# ============================================================
# RETRIEVE + RERANK
# ============================================================

def retrieve_and_rerank(question):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

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

Answer the question ONLY using the context below.

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

    return response.json()["response"], context


# ============================================================
# LLM JUDGE
# ============================================================

def evaluate_answer(question, context, answer):

    judge_prompt = f"""
You are evaluating a RAG system for a Data Mining Unit 1
educational assistant.

Evaluate the answer ONLY against the provided context.

Do not use outside knowledge.

Question:
{question}

Context:
{context}

Generated Answer:
{answer}

Evaluate three things:

1. Correctness:
Is the answer factually supported by the context?

2. Completeness:
Does the answer cover the important information needed to answer
the question based on the context?

3. Faithfulness:
Does the answer avoid introducing information that is not supported
by the context?

Give each score from 1 to 5.

5 = excellent
4 = good
3 = acceptable
2 = weak
1 = poor

Return ONLY valid JSON in this exact format:

{{
    "correctness": 1,
    "completeness": 1,
    "faithfulness": 1,
    "reason": "short explanation"
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": judge_prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        }
    )

    response.raise_for_status()

    raw_response = response.json()["response"].strip()

    # Remove markdown code fences if the model adds them
    raw_response = raw_response.replace(
        "```json", ""
    ).replace(
        "```", ""
    ).strip()

    try:
        evaluation = json.loads(raw_response)

    except json.JSONDecodeError:

        print("\nWARNING: Could not parse judge response:")
        print(raw_response)

        evaluation = {
            "correctness": 0,
            "completeness": 0,
            "faithfulness": 0,
            "reason": "Judge output could not be parsed."
        }

    return evaluation


# ============================================================
# RUN EVALUATION
# ============================================================

results = []


for number, question in enumerate(
    questions,
    start=1
):

    print("=" * 80)
    print(f"QUESTION {number}")
    print(question)
    print("=" * 80)

    ranked_results = retrieve_and_rerank(
        question
    )

    answer, context = generate_answer(
        question,
        ranked_results
    )

    print("\nGENERATED ANSWER:")
    print(answer)

    evaluation = evaluate_answer(
        question,
        context,
        answer
    )

    print("\nLLM JUDGE:")

    print(
        f"Correctness  : "
        f"{evaluation['correctness']}/5"
    )

    print(
        f"Completeness : "
        f"{evaluation['completeness']}/5"
    )

    print(
        f"Faithfulness : "
        f"{evaluation['faithfulness']}/5"
    )

    print(
        f"Reason       : "
        f"{evaluation['reason']}"
    )

    results.append(evaluation)

    print()


# ============================================================
# FINAL METRICS
# ============================================================

if results:

    avg_correctness = sum(
        r["correctness"]
        for r in results
    ) / len(results)

    avg_completeness = sum(
        r["completeness"]
        for r in results
    ) / len(results)

    avg_faithfulness = sum(
        r["faithfulness"]
        for r in results
    ) / len(results)


    print("=" * 80)
    print("FINAL LLM-BASED ANSWER EVALUATION")
    print("=" * 80)

    print(
        f"Questions evaluated : {len(results)}"
    )

    print(
        f"Average Correctness  : "
        f"{avg_correctness:.2f}/5"
    )

    print(
        f"Average Completeness : "
        f"{avg_completeness:.2f}/5"
    )

    print(
        f"Average Faithfulness : "
        f"{avg_faithfulness:.2f}/5"
    )

    print("=" * 80)