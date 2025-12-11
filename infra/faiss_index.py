import faiss
import numpy as np
from src.utils.embeddings import embed_batch

def build_faiss_index(context_texts):
    """
    context_texts: list of strings
    returns: (index, vectors)
    """
    vectors = embed_batch(context_texts)
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index, vectors


def search_index(index, query_vector, top_k=3):
    distances, indices = index.search(np.array([query_vector]), top_k)
    return distances, indices
