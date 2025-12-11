import re
from typing import List, Dict
from src.utils.embeddings import embed_text, cosine_similarity


def split_into_claims(text: str) -> List[str]:
    """
    Very basic sentence splitter.
    Each sentence = one potential claim.
    """
    # Split on . ! ? but keep text clean
    parts = re.split(r'[.!?]+', text)
    claims = [p.strip() for p in parts if p.strip()]
    return claims


def score_claim_against_context(claim: str, contexts: List) -> float:
    """
    Computes maximum similarity between a claim and all context chunks.
    """
    claim_vec = embed_text(claim)
    sims = []

    for ctx in contexts:
        ctx_vec = embed_text(ctx.text)
        sims.append(cosine_similarity(claim_vec, ctx_vec))

    return max(sims) if sims else 0.0


def factuality_report(answer: str, contexts: List) -> Dict:
    """
    Returns:
    {
        "claims": [...],
        "claim_scores": [...],
        "avg_score": float,
        "hallucinated_claims": [...]
    }
    """

    claims = split_into_claims(answer)
    scores = []

    for c in claims:
        s = score_claim_against_context(c, contexts)
        scores.append(s)

    # Determine hallucinations (threshold = 0.55 but adjustable later)
    hallucinated = [claims[i] for i, s in enumerate(scores) if s < 0.55]

    return {
        "claims": claims,
        "claim_scores": scores,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "hallucinated_claims": hallucinated,
    }
