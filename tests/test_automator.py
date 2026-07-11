import types

import core.automator as automator
from agents.automation import _normalize_key


def test_resolve_exact_app_alias():
    assert automator._resolve_app_command("google chrome") == "chrome"
    assert automator._resolve_app_command("vs code") == "code"


def test_resolve_fuzzy_app_alias():
    assert automator._resolve_app_command("notepadd") == "notepad"


def test_unknown_app_falls_back_to_lower_text():
    assert automator._resolve_app_command("Some Unknown App") == "some unknown app"


def test_key_normalization():
    assert _normalize_key("Escape") == "esc"
    assert _normalize_key("page down") == "pagedown"


def test_find_window_requires_reasonable_match(monkeypatch):
    windows = [types.SimpleNamespace(title="Completely Different Window")]
    monkeypatch.setattr(automator.gw, "getAllWindows", lambda: windows)
    assert automator._find_window("notepad") is None
