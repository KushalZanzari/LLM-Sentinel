import json
from pathlib import Path
from pydantic import BaseModel, ValidationError, Field
from typing import List


# -----------------------------
# Canonical internal schemas
# -----------------------------

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ChatDocument(BaseModel):
    messages: List[Message]

class ContextItem(BaseModel):
    id: str
    text: str

class ContextDocument(BaseModel):
    contexts: List[ContextItem]


# -----------------------------
# Parsing utilities
# -----------------------------

def load_json(path: str):
    """Safe JSON loader."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_chat(path: str) -> ChatDocument:
    """Parses and normalizes chat conversation JSON."""
    raw = load_json(path)

    if "messages" not in raw:
        raise ValueError("Chat JSON must contain a 'messages' field.")

    try:
        return ChatDocument(**raw)
    except ValidationError as e:
        raise ValueError(f"Invalid chat schema: {e}")


def parse_context(path: str) -> ContextDocument:
    """Parses and normalizes context vector JSON."""
    raw = load_json(path)

    if "contexts" not in raw:
        raise ValueError("Context JSON must contain a 'contexts' field.")

    try:
        return ContextDocument(**raw)
    except ValidationError as e:
        raise ValueError(f"Invalid context schema: {e}")


# -----------------------------
# Example "one call" helper
# -----------------------------

def load_all(chat_path: str, context_path: str):
    """Load both chat and context using canonical schema."""
    chat = parse_chat(chat_path)
    ctx = parse_context(context_path)
    return chat, ctx
