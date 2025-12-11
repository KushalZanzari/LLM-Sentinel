from src.utils.embeddings import embed_text, cosine_similarity
from infra.faiss_index import build_faiss_index, search_index


def relevance_score(user_question: str, answer: str) -> float:
    """
    Measures semantic similarity between the user question and assistant answer.
    """
    q_vec = embed_text(user_question)
    a_vec = embed_text(answer)
    return cosine_similarity(q_vec, a_vec)


def completeness_check(answer: str, contexts: list) -> float:
    """
    Uses FAISS to quickly find the most relevant context chunk.
    Converts FAISS L2 distance into a similarity-like score.
    """
    # Prepare list of context strings
    ctx_texts = [c.text for c in contexts]

    # 1️⃣ Build FAISS index (fast, uses embed_batch internally)
    index, vectors = build_faiss_index(ctx_texts)

    # 2️⃣ Embed the answer
    ans_vec = embed_text(answer)

    # 3️⃣ Find top-1 closest context chunk
    distances, indices = search_index(index, ans_vec, top_k=1)

    # FAISS returns L2 distance:
    # distance = 0 → perfect match
    # distance = large → unrelated
    d = distances[0][0]

    # Convert L2 distance to a similarity-like score (bounded 0..1)
    sim_score = 1 / (1 + d)

    return float(sim_score)
