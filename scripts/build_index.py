import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.document_loader import load_all_documents
from src.text_chunker import chunk_documents
from src.vector_store import HybridVectorStore
from config import VECTOR_STORE_PATH, METADATA_PATH


def main():
    print("=== Starting Vector Store Index Build ===")
    start_time = time.time()

    # 1. Load documents
    docs = load_all_documents()
    print(f"Loaded {len(docs)} documents")

    # 2. Chunk documents
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # 3. Build hybrid store
    store = HybridVectorStore()
    store.build(chunks)

    # 4. Confirm files exist
    if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(METADATA_PATH):
        print(f"Verification: Index files confirmed at '{VECTOR_STORE_PATH}' and '{METADATA_PATH}'")
    else:
        print("Error: Index files were not found on disk!")

    # 5. Test with 3 sample queries
    test_queries = [
        "What fertilizer is recommended for wheat crop?",
        "How can I apply for PM-KISAN scheme?",
        "How to control sucking pests and aphids in cotton?"
    ]

    print("\n=== Testing Search on 3 Queries ===")
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = store.search(q, k=2)
        for i, res in enumerate(results, 1):
            print(f"  Result {i} [Score: {res['score']:.5f}] (Source: {res['source']}, ID: {res['chunk_id']}):")
            snippet = res['text'].replace('\n', ' ')[:120]
            print(f"    \"{snippet}...\"")

    elapsed = time.time() - start_time
    print(f"\nIndex build and verification completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
