import json
from pathlib import Path


file_path = Path("data/processed/documents.jsonl")

with open(file_path, "r", encoding="utf-8") as f:
    documents = [json.loads(line) for line in f]


print(f"Total documents: {len(documents)}")


for i, document in enumerate(documents[:10], start=1):

    metadata = document["metadata"]

    print("\n" + "=" * 60)
    print(f"Document {i}")
    print(f"Source : {metadata['source']}")
    print(f"Slide  : {metadata['slide']}")
    print("-" * 60)
    print(document["text"])