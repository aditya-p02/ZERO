from unittest.mock import MagicMock

import agents.router as router
from agents.router import _groq_classify, _is_followup, _keyword_match, classify


def test_fresh_question_is_not_followup():
    assert not _is_followup("what is python", has_prior_conversation=False)


def test_short_followup_with_prior_context_is_general():
    assert classify("what about that", has_prior_conversation=True) == "general"


def test_screen_keywords_route_to_screen():
    assert _keyword_match("what's on my screen") == "screen"
    assert _keyword_match("read this error") == "screen"


def test_code_keywords_route_to_code():
    assert _keyword_match("write python code to sort a list") == "code"
    assert _keyword_match("run this code") == "code"


def test_automation_keywords_route_to_automation():
    assert _keyword_match("open notepad") == "automation"
    assert _keyword_match("press enter") == "automation"


def test_research_keywords_route_to_research():
    assert _keyword_match("what is the latest news on AI") == "research"


def test_memory_keywords_route_to_memory():
    assert _keyword_match("remember this") == "memory"
    assert _keyword_match("what do you know about me") == "memory"


def test_system_keywords_route_to_system():
    assert _keyword_match("how's the system doing") == "system"
    assert _keyword_match("battery status") == "system"


def test_groq_fallback_uses_injected_client(monkeypatch):
    """
    Phase 2 proof: before the shared-client refactor, swapping in a fake
    Groq client meant intercepting `Groq(api_key=os.getenv(...))` at
    import time in whichever of the 7 files you were testing — order-
    dependent and fragile once a module was already imported elsewhere
    in the test session. Now there's exactly one place the client comes
    from, so injecting a fake is a single, ordinary attribute patch on
    this module's own reference — no import-time interception needed.
    """
    fake_response = MagicMock()
    fake_response.choices[0].message.content = "research"

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    monkeypatch.setattr(router, "groq_client", fake_client)

    result = _groq_classify("some ambiguous message with no keyword hits")

    assert result == "research"
    fake_client.chat.completions.create.assert_called_once()