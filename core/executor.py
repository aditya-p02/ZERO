# core/executor.py
# ZERO's sandboxed code executor
# Runs Python code safely with timeout, captured output, no system damage

import ast
import asyncio
import io
import os
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout

# signal is Unix-only
if os.name != 'nt':
    import signal

import builtins

# ── Safe stdlib modules allowed inside exec ────────────────────────────────────
# These are explicitly imported into the exec namespace so 'import math' works.
_SAFE_MODULES = {
    "math", "random", "datetime", "json", "re", "itertools",
    "functools", "collections", "string", "decimal", "fractions",
    "statistics", "time", "calendar", "hashlib", "base64",
    "textwrap", "pprint", "copy",
}

# ── Blocked at AST level ───────────────────────────────────────────────────────
_BLOCKED_IMPORTS = {
    "subprocess", "shutil", "ctypes", "socket", "requests", "httpx",
    "urllib", "multiprocessing", "importlib", "pickle", "marshal",
    "pty", "nt", "winreg", "msvcrt", "resource", "signal",
    "os", "sys",   # os and sys blocked — too many escape hatches
}

_BLOCKED_BUILTINS = {
    "__import__", "eval", "exec", "compile",
    "open", "input", "breakpoint",
    "__loader__", "__spec__", "__builtins__",
}

# Build safe builtins dict
_SAFE_BUILTINS: dict = {}
for _name in dir(builtins):
    if _name not in _BLOCKED_BUILTINS:
        _SAFE_BUILTINS[_name] = getattr(builtins, _name)

# Inject a controlled __import__ that only allows safe modules
def _safe_import(name, *args, **kwargs):
    base = name.split(".")[0]
    if base not in _SAFE_MODULES:
        raise ImportError(
            f"Import of '{name}' is blocked for security. "
            f"Allowed modules: {', '.join(sorted(_SAFE_MODULES))}"
        )
    return __import__(name, *args, **kwargs)

_SAFE_BUILTINS["__import__"] = _safe_import


# ── AST-based safety check ─────────────────────────────────────────────────────

def _check_code_safety(code: str) -> tuple[bool, str]:
    """
    Parse the code as an AST and check for dangerous constructs.
    Much more reliable than string matching — won't trip on comments or strings.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error in code: {e}"

    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in _BLOCKED_IMPORTS:
                        return False, f"Import of '{alias.name}' is blocked."
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                if module in _BLOCKED_IMPORTS:
                    return False, f"Import from '{node.module}' is blocked."

        # Block attribute access on os/sys even if somehow available
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in ("os", "sys", "subprocess"):
                    return False, f"Access to '{node.value.id}.{node.attr}' is blocked."

        # Block __dunder__ attribute fishing (class hierarchy escape)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                dangerous = {"__class__", "__bases__", "__subclasses__",
                             "__globals__", "__builtins__", "__import__",
                             "__code__", "__func__", "__self__"}
                if node.attr in dangerous:
                    return False, f"Access to '{node.attr}' is blocked."

    return True, ""


# ── Code runner ────────────────────────────────────────────────────────────────

def _build_exec_namespace() -> dict:
    """Fresh execution namespace with safe builtins and pre-imported safe modules."""
    ns = {"__builtins__": _SAFE_BUILTINS}
    # Pre-import safe stdlib so 'import math' inside exec just works
    for mod_name in _SAFE_MODULES:
        try:
            ns[mod_name] = __import__(mod_name)
        except ImportError:
            pass
    return ns


def _run_code_unix(code: str, timeout: int, result: dict,
                   stdout_cap: io.StringIO, stderr_cap: io.StringIO):
    """Unix path — SIGALRM gives a hard kill."""
    def _handler(signum, frame):
        raise TimeoutError("Execution timed out.")

    try:
        with redirect_stdout(stdout_cap), redirect_stderr(stderr_cap):
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(timeout)
            exec(code, _build_exec_namespace())
            signal.alarm(0)
        result["success"] = True
        result["output"] = stdout_cap.getvalue()
        result["stderr"] = stderr_cap.getvalue()
    except TimeoutError:
        result["timed_out"] = True
        result["error"] = "Code took too long and was stopped."
    except Exception:
        result["error"] = traceback.format_exc()
        result["stderr"] = stderr_cap.getvalue()
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass


def _run_code_windows(code: str, timeout: int, result: dict,
                      stdout_cap: io.StringIO, stderr_cap: io.StringIO):
    """
    Windows path — run in daemon thread with join timeout.
    Thread can't be killed but is daemon so it dies with the process.
    stdout/stderr captured in thread-local StringIO to avoid race conditions.
    """
    local_stdout = io.StringIO()
    local_stderr = io.StringIO()
    exec_done = threading.Event()
    exec_result = {"error": None}

    def _target():
        try:
            with redirect_stdout(local_stdout), redirect_stderr(local_stderr):
                exec(code, _build_exec_namespace())
        except Exception:
            exec_result["error"] = traceback.format_exc()
        finally:
            exec_done.set()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    finished = exec_done.wait(timeout=timeout)

    if not finished:
        result["timed_out"] = True
        result["error"] = "Code took too long and was stopped."
        return

    result["stderr"] = local_stderr.getvalue()
    if exec_result["error"]:
        result["error"] = exec_result["error"]
    else:
        result["success"] = True
        result["output"] = local_stdout.getvalue()


def _run_code(code: str, timeout: int = 10) -> dict:
    result = {
        "success": False,
        "output": "",
        "stderr": "",
        "error": "",
        "timed_out": False,
    }

    is_safe, reason = _check_code_safety(code)
    if not is_safe:
        result["error"] = f"Execution blocked: {reason}"
        return result

    stdout_cap = io.StringIO()
    stderr_cap = io.StringIO()

    if os.name == "nt":
        _run_code_windows(code, timeout, result, stdout_cap, stderr_cap)
    else:
        _run_code_unix(code, timeout, result, stdout_cap, stderr_cap)

    return result


async def execute(code: str, timeout: int = 10) -> dict:
    """Async wrapper — runs in thread pool so it doesn't block the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_code, code, timeout)