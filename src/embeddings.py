import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL

# Singleton model pattern
_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of texts into a float32 numpy array of shape (N, 384)."""
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_single(text: str) -> np.ndarray:
    """Encode a single text string into a float32 numpy array of shape (384,)."""
    model = get_embedding_model()
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    return np.asarray(embedding, dtype=np.float32)
