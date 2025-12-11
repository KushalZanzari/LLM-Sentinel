from src.evaluators.reporter import make_verdict

def test_verdict_pass():
    assert make_verdict(0.9, 0.8, 0.9) == "PASS"

def test_verdict_warn_due_to_relevance():
    assert make_verdict(0.5, 0.8, 0.8) == "WARN"

def test_verdict_fail_due_to_factuality():
    assert make_verdict(0.9, 0.9, 0.2) == "FAIL"
