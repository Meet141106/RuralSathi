import os
import sys
import numpy as np
from sentence_transformers import CrossEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K
from src.vector_store import HybridVectorStore
from src.document_loader import load_all_documents
from src.text_chunker import chunk_documents


class Retriever:
    def __init__(self):
        self.store = HybridVectorStore()
        if not self.store.load():
            print("Vector store not found. Building index from documents...")
            docs = load_all_documents()
            chunks = chunk_documents(docs)
            self.store.build(chunks)
        self.reranker = None

    def _load_reranker(self):
        if self.reranker is None:
            # Use CPU on macOS to avoid Apple Metal MPS command buffer concurrency issues in Streamlit
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    def retrieve(self, query: str, top_k: int = RERANK_TOP_K) -> list:
        """Retrieve candidates using hybrid search and rerank using CrossEncoder."""
        # Step 1: Get top 8 candidates from HybridVectorStore
        candidates = self.store.search(query, k=8)
        if not candidates:
            return []

        # Step 2: Rerank using CrossEncoder
        self._load_reranker()
        pairs = [(query, chunk["text"]) for chunk in candidates]
        scores = self.reranker.predict(pairs)

        # Step 3: Add reranker_score to each candidate
        for i, score in enumerate(scores):
            candidates[i]["reranker_score"] = float(score)

        # Sort by reranker_score descending
        ranked_results = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)
        return ranked_results[:top_k]

    def is_relevant(self, top_result: dict, threshold: float = 0.1) -> bool:
        """Decide whether to answer or abstain based on reranker score."""
        if not top_result or "reranker_score" not in top_result:
            return False
        return top_result["reranker_score"] > threshold
