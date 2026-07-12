# agents/developer.py
# ZERO's code agent — write, explain, debug, and execute code

import asyncio
import re

from core.clients import groq_client
from core.config import settings
from core.executor import execute
from core.logger import log

USER_NAME = settings.user_name

# ── Session state ──────────────────────────────────────────────────────────────
_last_written_code: str | None = None   # raw code from last write
_last_written_full: str | None = None   # full response (code + explanation)
_last_request:      str | None = None   # the user message that produced the code
# ─────────────────────────────────────────────────────────────────────────────

# Phrases that clearly mean "run what you just wrote"
_RUN_TRIGGERS = [
    "run this", "run it", "run the code", "run that",
    "execute this", "execute it", "execute the code", "execute that",
    "can you run", "go ahead and run", "now run", "test it", "test this",
    "try it", "try this", "let's run", "let's execute",
]

# Phrases that are obviously write requests — skip the classify API call
_WRITE_TRIGGERS = [
    "write", "create", "make", "build", "implement", "code", "script",
    "function", "class", "program", "algorithm", "generate",
    "debug", "fix", "explain", "refactor", "optimise", "optimize",
    "show me how", "how do i", "how to",
]

CODE_WRITER_PROMPT = f"""
You are ZERO — {USER_NAME}'s personal AI and coding assistant.

When asked to write code:
- Write clean, working code. No placeholders, no "TODO" comments.
- Use Python unless another language is explicitly requested.
- Always wrap executable code in a ```python block so it can be run.
- NEVER use input() — hardcode example values instead (e.g. n = 20)
  with a comment like # change this
- NEVER use open(), file I/O, subprocess, os.system, os, sys, socket, or requests
- NEVER use infinite loops without a clear exit condition
- For demos, always use hardcoded test values so the code runs and produces output immediately.
- Add brief inline comments only where genuinely needed.
- After the code, explain what it does in 1-2 sentences max.
- If asked to debug, identify the exact issue and provide the fixed version.

When asked to explain code:
- Be direct. Explain what it does, not just what each line says.
- Point out anything clever or potentially problematic.
"""

CODE_DECISION_PROMPT = """
You are deciding whether the user wants code WRITTEN or EXECUTED.

Reply with exactly one word — "execute" or "write".

- "execute" → user wants to RUN code and see the OUTPUT (compute, calculate, find)
- "write"   → user wants code written, explained, or debugged

Examples:
"write me a function to check primes" → write
"what does this code output" → execute
"calculate the factorial of 10" → execute
"can you make a sorting algorithm" → write
"debug this code" → write
"compute fibonacci up to 100" → execute
"what is 2 to the power of 32" → execute

Request: {message}
"""


# ── Intent detection ───────────────────────────────────────────────────────────

def _is_run_request(message: str) -> bool:
    lower = message.lower()
    return any(trigger in lower for trigger in _RUN_TRIGGERS)


def _is_obvious_write(message: str) -> bool:
    """Fast-path: skip the classify API call for obvious write requests."""
    lower = message.lower()
    return any(trigger in lower for trigger in _WRITE_TRIGGERS)


def _decide_action(user_message: str) -> str:
    """Groq fallback only for genuinely ambiguous cases."""
    try:
        completion = groq_client.chat.completions.create(
            model=settings.groq_code_model,
            messages=[
                {"role": "user", "content": CODE_DECISION_PROMPT.format(message=user_message)},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        result = completion.choices[0].message.content.strip().lower()
        return "execute" if "execute" in result else "write"
    except Exception:
        log.error("Groq code-action classifier failed", exc_info=True)
        return "write"


# ── Code stripping ─────────────────────────────────────────────────────────────

def _strip_input_calls(code: str) -> str:
    """
    Replace input() calls with safe defaults so the sandbox never hangs.

    The old regex + manual paren-counter was blind to string contents and
    corrupted chained calls like list(input("...").split()), producing
    unmatched parens and immediate SyntaxErrors in the executor.

    This version uses Python's own AST — structurally correct, string-safe.
    Falls back to returning code unchanged on SyntaxError so the executor
    surfaces the real error rather than a mangled one.
    """
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code

    class _InputReplacer(_ast.NodeTransformer):
        def visit_Call(self, node):
            func = node.func

            if (
                isinstance(func, _ast.Name)
                and func.id in ("int", "float", "str")
                and len(node.args) == 1
                and isinstance(node.args[0], _ast.Call)
                and isinstance(node.args[0].func, _ast.Name)
                and node.args[0].func.id == "input"
            ):
                replacements = {"int": 0, "float": 0.0, "str": ""}
                return _ast.Constant(value=replacements[func.id])

            self.generic_visit(node)

            if isinstance(func, _ast.Name) and func.id == "input":
                return _ast.Constant(value="")

            return node

    new_tree = _InputReplacer().visit(tree)
    _ast.fix_missing_locations(new_tree)
    return _ast.unparse(new_tree)
def _write_code(user_message: str, prior_code: str | None = None) -> str:
    """
    Ask Groq to write or explain the code.
    Injects prior code as context for follow-up requests ("make it recursive").
    """
    messages = [{"role": "system", "content": CODE_WRITER_PROMPT}]

    if prior_code:
        messages.append({
            "role": "assistant",
            "content": f"Here's the code I wrote previously:\n```python\n{prior_code}\n```"
        })

    messages.append({"role": "user", "content": user_message})

    try:
        completion = groq_client.chat.completions.create(
            model=settings.groq_code_model,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        log.error("Groq code generation failed", exc_info=True)
        return f"Code generation failed: {str(e)}"


def _extract_code_block(text: str) -> str | None:
    """Pull the Python code out of a markdown code block."""
    match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ── Result formatting ──────────────────────────────────────────────────────────

def _format_execution_result(result: dict, voice_mode: bool = False) -> str:
    """Format executor output. voice_mode=True returns clean spoken text, no markdown."""
    if result["timed_out"]:
        return "The code hit the time limit — might be an infinite loop or heavy computation."

    # Surface stderr warnings even on success
    stderr_note = ""
    if result.get("stderr", "").strip():
        if not voice_mode:
            stderr_note = f"\nWarnings:\n```\n{result['stderr'].strip()}\n```"

    if not result["success"]:
        error = result["error"].strip()
        # Trim long tracebacks to last 8 lines
        lines = error.split("\n")
        if len(lines) > 8:
            error = "\n".join(lines[-8:])
        if voice_mode:
            # Extract just the last error line for speaking
            last_line = [line for line in lines if line.strip()][-1] if lines else error
            return f"The code hit an error: {last_line}"
        return f"Code ran but hit an error:\n```\n{error}\n```{stderr_note}"

    output = result["output"].strip()
    if output:
        if voice_mode:
            # Trim output for speaking — first 200 chars
            spoken = output[:200]
            if len(output) > 200:
                spoken += "... and more"
            return f"Done. The output was: {spoken}"
        return f"Done. Output:\n```\n{output}\n```{stderr_note}"
    else:
        return "Code ran successfully — no output produced."


# ── Main entry point ───────────────────────────────────────────────────────────

async def develop(user_message: str, voice_mode: bool = False) -> str:
    """
    Main entry point for the developer agent.

    Flow:
      1. "Run this" → execute last stored code, return result
      2. Obvious write trigger → skip classify API call, go straight to write
      3. Ambiguous → Groq classifies write vs execute
      4. Write code (with prior code as context for follow-ups)
      5. Store code + request in session state
      6. Return code only (write) or code + output (execute)
    """
    global _last_written_code, _last_written_full, _last_request

    loop = asyncio.get_running_loop()

    # ── Step 1: "Run this" ────────────────────────────────────────────────────
    if _is_run_request(user_message):
        if _last_written_code is None:
            return (
                "I don't have any code from this session to run. "
                "Ask me to write something first."
            )

        safe_code = _strip_input_calls(_last_written_code)
        print("[ZERO] Executing stored code...")
        result = await execute(safe_code)
        return _format_execution_result(result, voice_mode=voice_mode)

    # ── Step 2 & 3: Decide write vs execute ───────────────────────────────────
    if _is_obvious_write(user_message):
        action = "write"
    else:
        action = await loop.run_in_executor(None, _decide_action, user_message)

    # ── Step 4: Write the code (inject prior context for follow-ups) ──────────
    # Detect follow-up: short message + we have prior code
    is_followup = (
        _last_written_code is not None
        and len(user_message.split()) < 12
        and not _is_obvious_write(user_message)
    )
    prior = _last_written_code if is_followup else None
    written = await loop.run_in_executor(None, _write_code, user_message, prior)

    # ── Step 5: Store in session state ────────────────────────────────────────
    code = _extract_code_block(written)
    if code:
        _last_written_code = code
        _last_written_full = written
        _last_request = user_message

    # ── Step 6: Return ────────────────────────────────────────────────────────
    if action == "write" or not code:
        return written

    # Execute path
    safe_code = _strip_input_calls(code)
    print("[ZERO] Executing code...")
    result = await execute(safe_code)
    execution_summary = _format_execution_result(result, voice_mode=voice_mode)

    return f"{written}\n\n{execution_summary}"