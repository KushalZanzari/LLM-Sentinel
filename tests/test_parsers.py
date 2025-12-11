from src.utils.parsers import parse_chat, parse_context

def test_parse_chat():
    chat = parse_chat("data/samples/sample-chat-conversation-01.json")
    assert len(chat.messages) >= 1

def test_parse_context():
    ctx = parse_context("data/samples/sample_context_vectors-01.json")
    assert len(ctx.contexts) >= 1
