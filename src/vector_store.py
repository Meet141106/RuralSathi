import os
import sys
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTOR_STORE_PATH, METADATA_PATH, TOP_K
from src.embeddings import encode_texts, encode_single


class HybridVectorStore:
    def __init__(self):
        self.index = None          # FAISS IndexFlatIP
        self.metadata = []         # list of chunk dicts
        self.bm25 = None           # BM25Okapi instance
        self.tokenized_corpus = [] # for BM25

    def build(self, chunks: list):
        """Build FAISS dense index and BM25 sparse index from chunk documents."""
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        self.metadata = chunks
        texts = [c["text"] for c in chunks]

        # 1. FAISS dense index
        vectors = encode_texts(texts)
        # Normalize vectors for cosine similarity via inner product
        faiss.normalize_L2(vectors)
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

        # 2. BM25 sparse index
        self.tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 3. Save to disk
        self.save()
        print(f"Index built: {len(chunks)} vectors, FAISS + BM25 ready")

    def save(self):
        """Save FAISS index and metadata (including BM25) to disk."""
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

        # Write FAISS index
        faiss.write_index(self.index, VECTOR_STORE_PATH)

        # Write metadata and BM25 data
        payload = {
            "metadata": self.metadata,
            "bm25": self.bm25,
            "tokenized_corpus": self.tokenized_corpus
        }
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(payload, f)

    def load(self) -> bool:
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists(METADATA_PATH):
            return False

        try:
            self.index = faiss.read_index(VECTOR_STORE_PATH)
            with open(METADATA_PATH, "rb") as f:
                payload = pickle.load(f)
            self.metadata = payload["metadata"]
            self.bm25 = payload["bm25"]
            self.tokenized_corpus = payload.get("tokenized_corpus", [])
            print(f"Index loaded: {len(self.metadata)} documents")
            return True
        except Exception as e:
            print(f"Failed to load vector store: {e}")
            return False

    def search(self, query: str, k: int = TOP_K) -> list:
        """Dense + Sparse hybrid search with Reciprocal Rank Fusion (RRF)."""
        if self.index is None or self.bm25 is None or not self.metadata:
            raise ValueError("Index not loaded or empty. Call load() or build() first.")

        fetch_k = min(k * 2, len(self.metadata))

        # 1. Dense Search (FAISS)
        query_vec = encode_single(query).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        _, dense_indices = self.index.search(query_vec, fetch_k)
        dense_rankings = dense_indices[0].tolist()

        # 2. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Top fetch_k indices sorted by score descending
        sparse_rankings = np.argsort(bm25_scores)[::-1][:fetch_k].tolist()

        # 3. Reciprocal Rank Fusion (RRF)
        # score = sum(1 / (rank + 60))
        rrf_scores = {}

        for rank, doc_idx in enumerate(dense_rankings):
            if doc_idx < 0 or doc_idx >= len(self.metadata):
                continue
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rank + 60))

        for rank, doc_idx in enumerate(sparse_rankings):
            if doc_idx < 0 or doc_idx >= len(self.metadata):
                continue
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rank + 60))

        # Sort doc indices by aggregated RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:k]

        results = []
        for doc_idx, score in sorted_docs:
            chunk = self.metadata[doc_idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk.get("chunk_id", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {})
            })

        return results
