from agents.router import _is_followup, _keyword_match, classify


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
