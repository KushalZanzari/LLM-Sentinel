import json
import csv
from pathlib import Path
from src.main import evaluate

BATCH_FOLDER = Path("data/samples")
OUTPUT_FILE = Path("data/batch_results.csv")

def find_pairs():
    """
    Finds chat + context file pairs by matching index numbers.
    Example:
    sample-chat-conversation-01.json with sample_context_vectors-01.json
    """
    chat_files = sorted(BATCH_FOLDER.glob("sample-chat-conversation-*.json"))
    ctx_files  = sorted(BATCH_FOLDER.glob("sample_context_vectors-*.json"))

    pairs = []
    for c in chat_files:
        idx = c.stem.split("-")[-1]
        ctx = BATCH_FOLDER / f"sample_context_vectors-{idx}.json"
        if ctx.exists():
            pairs.append((c, ctx))
    return pairs


def run_batch():
    pairs = find_pairs()
    print(f"Found {len(pairs)} pairs.")

    rows = []

    for chat, ctx in pairs:
        print(f"Evaluating {chat.name} ...")

        report = evaluate(str(chat), str(ctx))

        rows.append({
            "chat_file": chat.name,
            "context_file": ctx.name,
            "relevance": report["scores"]["relevance"],
            "completeness": report["scores"]["completeness"],
            "factuality": report["scores"]["factuality"]["avg_score"],
            "verdict": report["verdict"],
            "latency": report["latency_seconds"],
            "total_tokens": report["token_usage"]["total_tokens"]
        })

    # Save CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Batch results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_batch()
