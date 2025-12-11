import json
import yaml
from src.utils.parsers import parse_chat, parse_context
from src.evaluators.relevance import relevance_score, completeness_check
from src.evaluators.factuality import factuality_report
from src.evaluators.latency_cost import count_tokens, estimate_cost, calculate_latency
from src.evaluators.reporter import build_report
from src.utils.pii import detect_pii, redact_pii



def evaluate(chat_path: str, ctx_path: str):
    """
    Full evaluation pipeline producing a canonical evaluation report.
    """

    # Load chat + context files
    chat = parse_chat(chat_path)
    ctx = parse_context(ctx_path)

    
    # Extract raw messages
    user_msg_raw = chat.messages[0].content
    assistant_msg_raw = chat.messages[1].content

    # Detect and redact PII
    user_pii = detect_pii(user_msg_raw)
    assistant_pii = detect_pii(assistant_msg_raw)

    user_msg = redact_pii(user_msg_raw)
    assistant_msg = redact_pii(assistant_msg_raw)


    # 1️⃣ Relevance
    rel = relevance_score(user_msg, assistant_msg)

    # 2️⃣ Completeness
    comp = completeness_check(assistant_msg, ctx.contexts)

    # 3️⃣ Factuality
    fact = factuality_report(assistant_msg, ctx.contexts)

    # 4️⃣ Token usage
    user_tokens = count_tokens(user_msg)
    assistant_tokens = count_tokens(assistant_msg)
    total_tokens = user_tokens + assistant_tokens

    with open("configs/thresholds.yaml", "r") as f:
        config = yaml.safe_load(f)

    cost_est = estimate_cost(total_tokens, config["price_per_1k_tokens"])

    token_usage = {
        "user_tokens": user_tokens,
        "assistant_tokens": assistant_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost_est
    }

    # 5️⃣ Latency (placeholder timestamps)
    latency = calculate_latency("2024-01-01T12:00:00", "2024-01-01T12:00:05")

    # 6️⃣ Build complete report with verdict system
    final_report = build_report(
    rel,
    comp,
    fact,
    latency,
    token_usage
)

    final_report["pii_detected"] = {
    "user": user_pii,
    "assistant": assistant_pii
}
    return final_report


# -------------------------------
# CLI ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Evaluation CLI")
    parser.add_argument("--chat", required=True, help="Path to chat JSON")
    parser.add_argument("--ctx", required=True, help="Path to context JSON")

    args = parser.parse_args()

    result = evaluate(args.chat, args.ctx)
    print(json.dumps(result, indent=2))
