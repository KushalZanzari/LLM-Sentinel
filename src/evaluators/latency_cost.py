import tiktoken
from typing import Dict
from datetime import datetime


# ------------------------------
# Token counting
# ------------------------------

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Uses free tiktoken library to approximate token count.
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def estimate_cost(tokens: int, price_per_1k_tokens: float) -> float:
    """
    Simple cost estimation.
    Example: price_per_1k_tokens = 0.002 → $0.002 per 1000 tokens.
    """
    return (tokens / 1000) * price_per_1k_tokens


# ------------------------------
# Latency calculation
# ------------------------------

def calculate_latency(start_ts: str, end_ts: str) -> float:
    """
    Timestamps must be ISO strings like '2024-01-01T12:00:05'.
    Returns latency in seconds.
    """
    t1 = datetime.fromisoformat(start_ts)
    t2 = datetime.fromisoformat(end_ts)
    delta = (t2 - t1).total_seconds()
    return float(delta)
