import re

# Basic PII patterns
EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_PATTERN = r"\+?\d[\d\s\-]{7,}\d"
NAME_PATTERN  = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b"
ID_PATTERN    = r"\b\d{6,}\b"


def detect_pii(text: str):
    """
    Detect possible PII: emails, phone numbers, IDs, names.
    Returns a dictionary.
    """
    return {
        "emails": re.findall(EMAIL_PATTERN, text),
        "phones": re.findall(PHONE_PATTERN, text),
        "ids": re.findall(ID_PATTERN, text),
        "names": re.findall(NAME_PATTERN, text)
    }


def redact_pii(text: str):
    """
    Replace PII with placeholder tokens.
    """
    text = re.sub(EMAIL_PATTERN, "[REDACTED_EMAIL]", text)
    text = re.sub(PHONE_PATTERN, "[REDACTED_PHONE]", text)
    text = re.sub(ID_PATTERN, "[REDACTED_ID]", text)
    text = re.sub(NAME_PATTERN, "[REDACTED_NAME]", text)
    return text
