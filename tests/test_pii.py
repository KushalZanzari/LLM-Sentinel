from src.utils.pii import detect_pii, redact_pii

def test_detect_email():
    res = detect_pii('Contact: alice@example.com')
    assert 'alice@example.com' in res['emails']

def test_redact_phone():
    s = 'Call at +1 555 222 3333'
    out = redact_pii(s)
    assert '[REDACTED_PHONE]' in out
