from agents.developer import _extract_code_block, _strip_input_calls


def test_extract_python_code_block():
    text = "Here:\n```python\nprint('hi')\n```"
    assert _extract_code_block(text) == "print('hi')"


def test_extract_generic_code_block():
    text = "```\nprint('hi')\n```"
    assert _extract_code_block(text) == "print('hi')"


def test_strip_input_calls_keeps_chained_expression_valid():
    code = "values = list(input('nums: ').split())\nprint(values)"
    stripped = _strip_input_calls(code)
    compile(stripped, "<test>", "exec")
    assert "input" not in stripped


def test_strip_int_input_replaces_with_number():
    stripped = _strip_input_calls("n = int(input('n: '))\nprint(n)")
    assert "n = 0" in stripped
