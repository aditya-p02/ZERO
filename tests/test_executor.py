import pytest

from core.executor import _check_code_safety, _run_code


def test_safe_code_runs():
    result = _run_code("print(2 + 3)")
    assert result["success"] is True
    assert result["output"].strip() == "5"


@pytest.mark.parametrize("code", ["import os", "import sys", "import subprocess"])
def test_dangerous_imports_are_blocked(code):
    safe, reason = _check_code_safety(code)
    assert safe is False
    assert "blocked" in reason.lower()


def test_open_builtin_is_unavailable():
    result = _run_code("open('x.txt', 'w')")
    assert result["success"] is False
    assert "name 'open' is not defined" in result["error"]


def test_syntax_error_returns_clean_result():
    result = _run_code("if True print('bad')")
    assert result["success"] is False
    assert "Syntax error" in result["error"]


def test_timeout_returns_timeout_result():
    result = _run_code("while True:\n    pass", timeout=1)
    assert result["timed_out"] is True
    assert result["success"] is False
