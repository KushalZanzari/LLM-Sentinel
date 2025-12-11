from src.evaluators.relevance import relevance_score

def test_relevance_basic():
    s = relevance_score("What is AI?", "AI means artificial intelligence.")
    assert s > 0.3
