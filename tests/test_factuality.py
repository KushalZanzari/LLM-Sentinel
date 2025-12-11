from src.evaluators.factuality import factuality_report
from src.utils.parsers import parse_context

def test_factuality():
    ctx = parse_context("data/samples/sample_context_vectors-01.json")
    report = factuality_report("AI means artificial intelligence.", ctx.contexts)
    assert report["avg_score"] > 0.3
