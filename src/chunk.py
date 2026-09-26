import json
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "documents.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"


# --------------------------------------------------
# CHUNK SETTINGS
# --------------------------------------------------

MAX_CHARS = 1500


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

def load_documents():

    documents = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            documents.append(
                json.loads(line)
            )

    return documents


# --------------------------------------------------
# CHUNK ONE DOCUMENT
# --------------------------------------------------

def chunk_document(document):

    text = document["text"]
    metadata = document["metadata"]

    # --------------------------------------------------
    # CASE 1: Slide fits into one chunk
    # --------------------------------------------------

    if len(text) <= MAX_CHARS:

        chunk_metadata = metadata.copy()

        chunk_metadata["chunk_id"] = (
            f"{metadata['source']}"
            f"_slide_{metadata['slide']}"
            f"_chunk_1"
        )

        return [
            {
                "text": text,
                "metadata": chunk_metadata
            }
        ]

    # --------------------------------------------------
    # CASE 2: Slide needs multiple chunks
    # --------------------------------------------------

    chunks = []
    current_chunk = []
    current_length = 0

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        line_length = len(line)

        if (
            current_length + line_length > MAX_CHARS
            and current_chunk
        ):

            chunks.append("\n".join(current_chunk))

            current_chunk = []
            current_length = 0

        current_chunk.append(line)
        current_length += line_length

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    # --------------------------------------------------
    # Add metadata to every chunk
    # --------------------------------------------------

    final_chunks = []

    for index, chunk in enumerate(chunks, start=1):

        chunk_metadata = metadata.copy()

        chunk_metadata["chunk_id"] = (
            f"{metadata['source']}"
            f"_slide_{metadata['slide']}"
            f"_chunk_{index}"
        )

        final_chunks.append(
            {
                "text": chunk,
                "metadata": chunk_metadata
            }
        )

    return final_chunks


# --------------------------------------------------
# CREATE ALL CHUNKS
# --------------------------------------------------

def create_chunks():

    documents = load_documents()

    all_chunks = []

    for document in documents:

        chunks = chunk_document(document)

        all_chunks.extend(chunks)

    return all_chunks


# --------------------------------------------------
# SAVE CHUNKS
# --------------------------------------------------

def save_chunks(chunks):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"Saved {len(chunks)} chunks to:"
    )

    print(OUTPUT_FILE)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    chunks = create_chunks()

    print(
        f"Created {len(chunks)} chunks "
        f"from the extracted documents."
    )

    save_chunks(chunks)