import json
import hashlib
from pathlib import Path


CACHE_PATH = Path("data/cache")
CACHE_PATH.mkdir(exist_ok=True)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_cached_vector(text: str):
    key = _hash_text(text)
    file = CACHE_PATH / f"{key}.json"

    if file.exists():
        with open(file, "r") as f:
            return json.load(f)["vector"]

    return None


def store_vector(text: str, vector):
    key = _hash_text(text)
    file = CACHE_PATH / f"{key}.json"

    with open(file, "w") as f:
        json.dump({"vector": vector.tolist()}, f)
