from sentence_transformers import SentenceTransformer
import numpy as np
from src.utils.caching import get_cached_vector, store_vector

_model = None


def get_model():
    """
    Loads the embedding model once, then reuses it (faster).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Embeds a single text string and caches the vector.
    """
    # 1️⃣ Try cache first
    cached = get_cached_vector(text)
    if cached:
        return np.array(cached)

    # 2️⃣ Compute new embedding
    model = get_model()
    vec = model.encode([text], convert_to_numpy=True)[0]

    # 3️⃣ Store in cache for next time
    store_vector(text, vec)

    return vec


def embed_batch(texts):
    """
    Embeds a list of texts at once (faster for FAISS building).
    No caching is applied here because FAISS is usually batch-based.
    """
    model = get_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
