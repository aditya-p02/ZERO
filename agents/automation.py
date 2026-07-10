# agents/automation.py
# ZERO's automation agent — desktop control via LLM-based command interpretation
# Groq interprets messy/natural phrasing into structured actions, instead of
# requiring exact keyword matches.

import os
import re
import json
import asyncio
from groq import Groq
from dotenv import load_dotenv
from core.logger import log
from core.automator import (
    open_app_async, close_app_async, focus_app_async,
    minimize_app_async, maximize_app_async,
    type_text_async, press_key_async, hotkey_async,
    click_text_async, scroll_async, list_open_windows,
)

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_KEY_ALIASES = {
    "enter": "enter", "return": "enter",
    "escape": "esc", "esc": "esc",
    "backspace": "backspace", "delete": "delete",
    "tab": "tab", "space": "space", "spacebar": "space",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "page up": "pageup", "page down": "pagedown",
    "home": "home", "end": "end",
}

_HOTKEY_MAP = {
    "copy": ("ctrl", "c"),
    "paste": ("ctrl", "v"),
    "cut": ("ctrl", "x"),
    "undo": ("ctrl", "z"),
    "redo": ("ctrl", "y"),
    "save": ("ctrl", "s"),
    "select_all": ("ctrl", "a"),
    "switch_window": ("alt", "tab"),
    "new_tab": ("ctrl", "t"),
    "close_tab": ("ctrl", "w"),
    "refresh": ("ctrl", "r"),
    "find": ("ctrl", "f"),
}

# ── The interpreter prompt — this is the actual "understanding" layer ─────────
_COMMAND_INTERPRETER_PROMPT = """
You are a command interpreter for ZERO, a desktop automation assistant.
The user's speech was transcribed and may contain errors, filler words, or unusual phrasing.
Your job: figure out what desktop action they actually want, even if the wording is messy.

Available actions:
- open_app: target = app name (e.g. "chrome", "notepad", "spotify")
- close_app: target = app name
- focus_app: target = app name (switch to / bring up an already-open app)
- minimize_app: target = app name
- maximize_app: target = app name
- type_text: target = the exact text to type
- press_key: target = key name (enter, escape, tab, backspace, etc)
- hotkey: target = one of [copy, paste, cut, undo, redo, save, select_all, switch_window, new_tab, close_tab, refresh, find]
- click_text: target = the visible text/label to click on screen
- scroll_up: target = null
- scroll_down: target = null
- list_windows: target = null
- unclear: target = null (use this ONLY if you genuinely cannot determine any reasonable action)

Rules:
- Be generous in interpretation. Transcription errors are common — "one notepad" almost certainly means "open notepad". "Crome browser" means "chrome". Use context and common sense.
- If the user names an app with a typo or partial word, still extract it as the target — don't reject it.
- Only use "unclear" if the message has nothing to do with desktop control at all.
- Respond ONLY with JSON, no explanation, no markdown.

Format: {"action": "action_name", "target": "extracted target or null"}

Examples:
"one notepad" → {"action": "open_app", "target": "notepad"}
"can you pull up chrome for me" → {"action": "open_app", "target": "chrome"}
"close this spotify thing" → {"action": "close_app", "target": "spotify"}
"type hello how are you" → {"action": "type_text", "target": "hello how are you"}
"hit enter" → {"action": "press_key", "target": "enter"}
"copy that" → {"action": "hotkey", "target": "copy"}
"scroll down a bit" → {"action": "scroll_down", "target": null}
"what's open right now" → {"action": "list_windows", "target": null}
"click on the submit button" → {"action": "click_text", "target": "submit button"}
"""


def _interpret_command(user_message: str) -> dict:
    """Send the raw message to Groq, get back a structured action."""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _COMMAND_INTERPRETER_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        raw = completion.choices[0].message.content.strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)
        if "action" not in parsed:
            return {"action": "unclear", "target": None}
        return parsed
    except Exception:
        log.error("Automation command interpretation failed", exc_info=True)
        return {"action": "unclear", "target": None}


def _normalize_key(key: str) -> str:
    key_lower = key.lower().strip()
    return _KEY_ALIASES.get(key_lower, key_lower)


# ── Main dispatch ────────────────────────────────────────────────────────────────

async def automate(user_message: str) -> str:
    """
    Main entry point. Groq interprets the command, then we execute the
    matching automator.py function.
    """
    loop = asyncio.get_running_loop()
    parsed = await loop.run_in_executor(None, _interpret_command, user_message)

    action = parsed.get("action", "unclear")
    target = parsed.get("target")

    if action == "open_app" and target:
        result = await open_app_async(target)
        return result["message"]

    if action == "close_app" and target:
        result = await close_app_async(target)
        return result["message"]

    if action == "focus_app" and target:
        result = await focus_app_async(target)
        return result["message"]

    if action == "minimize_app" and target:
        result = await minimize_app_async(target)
        return result["message"]

    if action == "maximize_app" and target:
        result = await maximize_app_async(target)
        return result["message"]

    if action == "type_text" and target:
        result = await type_text_async(target)
        return result["message"]

    if action == "press_key" and target:
        key = _normalize_key(target)
        result = await press_key_async(key)
        return result["message"]

    if action == "hotkey" and target:
        combo = _HOTKEY_MAP.get(target.lower())
        if combo:
            result = await hotkey_async(*combo)
            return result["message"]

    if action == "click_text" and target:
        result = await click_text_async(target)
        return result["message"]

    if action == "scroll_up":
        result = await scroll_async(300)
        return result["message"]

    if action == "scroll_down":
        result = await scroll_async(-300)
        return result["message"]

    if action == "list_windows":
        windows = list_open_windows()
        if not windows:
            return "Nothing's open right now."
        return "Currently open: " + ", ".join(windows[:8])

    return "Not sure what you want me to do on the desktop — try rephrasing."