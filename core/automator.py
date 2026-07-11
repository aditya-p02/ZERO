# core/automator.py
# ZERO's hands — low-level desktop input controller
# App launching, keyboard input, mouse control, OCR-based click targeting
#
# Design principles:
#   - App open/close/focus/type/shortcuts are INSTANT — no vision involved
#   - Clicking by text only triggers OCR when explicitly needed, runs in thread pool
#   - Everything async-wrapped so it never blocks ZERO's main loop

import asyncio
import os
import shutil
import subprocess
import time
from difflib import SequenceMatcher

import pyautogui
import pygetwindow as gw

# Keep PyAutoGUI's corner-abort enabled. Automation controls the real machine.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


# ── Common app name → executable/command mapping ──────────────────────────────
# Extend this as you find apps that don't resolve via plain name
_APP_ALIASES = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "notepad": "notepad",
    "calculator": "calc",
    "spotify": "spotify",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "explorer": "explorer",
    "file explorer": "explorer",
    "cmd": "cmd",
    "command prompt": "cmd",
    "powershell": "powershell",
    "settings": "ms-settings:",
    "task manager": "taskmgr",
    "paint": "mspaint",
    "word": "winword",
    "excel": "excel",
}

# Minimum similarity to accept a fuzzy alias match instead of using the raw
# (possibly mis-transcribed) text as a literal command.
_FUZZY_ALIAS_THRESHOLD = 0.72


# ── App control ──────────────────────────────────────────────────────────────

def _resolve_app_command(app_name: str) -> str:
    """
    Map a spoken app name to its launch command.
    Tries exact alias match first, then fuzzy match against known aliases
    (catches mis-transcriptions like "notepower" -> notepad), and only
    falls back to using the raw text as a literal command if nothing's close.
    """
    lower = app_name.lower().strip()

    if lower in _APP_ALIASES:
        return _APP_ALIASES[lower]

    best_key, best_score = None, 0.0
    for key in _APP_ALIASES:
        score = SequenceMatcher(None, lower, key).ratio()
        if score > best_score:
            best_score = score
            best_key = key

    if best_key and best_score >= _FUZZY_ALIAS_THRESHOLD:
        return _APP_ALIASES[best_key]

    return lower


def _command_exists(command: str) -> bool:
    """
    Check whether a command actually resolves to something runnable
    before we attempt to launch it and claim success.
    """
    if command in _APP_ALIASES.values():
        return True
    return shutil.which(command) is not None


def open_app(app_name: str) -> dict:
    """
    Launch an application by name.
    Returns {"success": bool, "message": str}

    Validates the command BEFORE claiming success — subprocess.Popen with
    shell=True does not raise just because the inner command fails, so we
    can't trust "no exception" as proof anything actually opened.
    """
    command = _resolve_app_command(app_name)

    if command.startswith("ms-settings:"):
        try:
            os.startfile(command)
            time.sleep(0.8)
            return {"success": True, "message": f"Opened {app_name}."}
        except Exception as e:
            return {"success": False, "message": f"Couldn't open {app_name}: {e}"}

    if not _command_exists(command):
        return {
            "success": False,
            "message": f"I don't recognize '{app_name}' as an app — didn't open anything.",
        }

    try:
        proc = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        time.sleep(0.6)  # give it a moment — real apps keep running, bad commands exit fast

        if proc.poll() is not None and proc.returncode != 0:
            return {
                "success": False,
                "message": f"Couldn't open {app_name} — '{command}' isn't a recognized command.",
            }

        time.sleep(0.2)
        return {"success": True, "message": f"Opened {app_name}."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't open {app_name}: {e}"}


def close_app(app_name: str) -> dict:
    """
    Close a window matching the app name (closest title match).
    Returns {"success": bool, "message": str}
    """
    window = _find_window(app_name)
    if window is None:
        return {"success": False, "message": f"Couldn't find a window matching '{app_name}'."}
    try:
        window.close()
        return {"success": True, "message": f"Closed {app_name}."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't close {app_name}: {e}"}


def _find_window(app_name: str):
    """
    Find the window whose title best matches app_name.
    Uses fuzzy matching since window titles rarely match the app name exactly
    (e.g. "Notepad" app might show as "Untitled - Notepad").
    """
    all_windows = [w for w in gw.getAllWindows() if w.title.strip()]
    if not all_windows:
        return None

    lower_target = app_name.lower()
    best_window = None
    best_score = 0.0

    for w in all_windows:
        title_lower = w.title.lower()
        if lower_target in title_lower:
            return w  # direct substring match — good enough, return immediately
        score = SequenceMatcher(None, lower_target, title_lower).ratio()
        if score > best_score:
            best_score = score
            best_window = w

    # Only accept fuzzy match if reasonably confident
    return best_window if best_score > 0.4 else None


def focus_app(app_name: str) -> dict:
    """Bring a window to the foreground."""
    window = _find_window(app_name)
    if window is None:
        return {"success": False, "message": f"Couldn't find a window matching '{app_name}'."}
    try:
        if window.isMinimized:
            window.restore()
        window.activate()
        return {"success": True, "message": f"Switched to {app_name}."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't focus {app_name}: {e}"}


def minimize_app(app_name: str) -> dict:
    window = _find_window(app_name)
    if window is None:
        return {"success": False, "message": f"Couldn't find a window matching '{app_name}'."}
    try:
        window.minimize()
        return {"success": True, "message": f"Minimized {app_name}."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't minimize {app_name}: {e}"}


def maximize_app(app_name: str) -> dict:
    window = _find_window(app_name)
    if window is None:
        return {"success": False, "message": f"Couldn't find a window matching '{app_name}'."}
    try:
        window.maximize()
        return {"success": True, "message": f"Maximized {app_name}."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't maximize {app_name}: {e}"}


def list_open_windows() -> list:
    """Return titles of all currently open windows with visible titles."""
    return [w.title for w in gw.getAllWindows() if w.title.strip()]


# ── Keyboard input ─────────────────────────────────────────────────────────────

def type_text(text: str, interval: float = 0.02) -> dict:
    """Type text into whatever currently has focus."""
    try:
        pyautogui.write(text, interval=interval)
        return {"success": True, "message": "Typed."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't type: {e}"}


def press_key(key: str) -> dict:
    """Press a single key — 'enter', 'esc', 'tab', 'backspace', etc."""
    try:
        pyautogui.press(key)
        return {"success": True, "message": f"Pressed {key}."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't press {key}: {e}"}


def hotkey(*keys: str) -> dict:
    """Press a key combination — hotkey('ctrl', 'c') for copy, etc."""
    try:
        pyautogui.hotkey(*keys)
        return {"success": True, "message": f"Pressed {'+'.join(keys)}."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't press hotkey: {e}"}


# ── Mouse control ────────────────────────────────────────────────────────────

def click_at(x: int, y: int, button: str = "left") -> dict:
    """Click at exact screen coordinates."""
    try:
        pyautogui.click(x=x, y=y, button=button)
        return {"success": True, "message": f"Clicked at ({x}, {y})."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't click: {e}"}


def double_click_at(x: int, y: int) -> dict:
    try:
        pyautogui.doubleClick(x=x, y=y)
        return {"success": True, "message": f"Double-clicked at ({x}, {y})."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't double-click: {e}"}


def move_to(x: int, y: int, duration: float = 0.2) -> dict:
    try:
        pyautogui.moveTo(x, y, duration=duration)
        return {"success": True, "message": f"Moved to ({x}, {y})."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't move mouse: {e}"}


def scroll(amount: int) -> dict:
    """Positive = scroll up, negative = scroll down."""
    try:
        pyautogui.scroll(amount)
        return {"success": True, "message": "Scrolled."}
    except Exception as e:
        return {"success": False, "message": f"Couldn't scroll: {e}"}


# ── OCR-based click targeting ────────────────────────────────────────────────
# Finds clickable text on screen using existing screen.py OCR.
# Only called when explicit text-based clicking is requested — never runs
# automatically, so it never causes the "always loading" problem.

def _find_text_on_screen(target_text: str) -> dict:
    """
    Capture screen, OCR it, find approximate location of target_text.
    Tesseract gives word-level bounding boxes via image_to_data.
    Returns {"found": bool, "x": int, "y": int, "confidence": float}
    """
    try:
        import pytesseract
        from PIL import Image

        from core.screen import _setup_tesseract, _take_screenshot

        _setup_tesseract()
        path = _take_screenshot()
        img = Image.open(path)

        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        target_lower = target_text.lower().strip()
        best_match = None
        best_score = 0.0

        n = len(data["text"])
        for i in range(n):
            word = data["text"][i].strip()
            if not word:
                continue
            word_lower = word.lower()

            # Direct substring match — strong signal
            if target_lower in word_lower or word_lower in target_lower:
                score = 1.0
            else:
                score = SequenceMatcher(None, target_lower, word_lower).ratio()

            if score > best_score:
                best_score = score
                cx = data["left"][i] + data["width"][i] // 2
                cy = data["top"][i] + data["height"][i] // 2
                best_match = (cx, cy)

        if best_match and best_score > 0.6:
            return {"found": True, "x": best_match[0], "y": best_match[1], "confidence": best_score}
        return {"found": False, "x": 0, "y": 0, "confidence": best_score}

    except Exception as e:
        return {"found": False, "x": 0, "y": 0, "confidence": 0.0, "error": str(e)}


def click_text(target_text: str) -> dict:
    """
    Find text on screen via OCR and click it.
    Synchronous — wrap with click_text_async for non-blocking use.
    """
    result = _find_text_on_screen(target_text)
    if not result["found"]:
        return {"success": False, "message": f"Couldn't find '{target_text}' on screen."}

    click_result = click_at(result["x"], result["y"])
    if click_result["success"]:
        return {"success": True, "message": f"Clicked on '{target_text}'."}
    return click_result


# ── Async wrappers — keep everything off the event loop ────────────────────────

async def open_app_async(app_name: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, open_app, app_name)


async def close_app_async(app_name: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, close_app, app_name)


async def focus_app_async(app_name: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, focus_app, app_name)


async def minimize_app_async(app_name: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, minimize_app, app_name)


async def maximize_app_async(app_name: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, maximize_app, app_name)


async def type_text_async(text: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, type_text, text)


async def press_key_async(key: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, press_key, key)


async def hotkey_async(*keys: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, hotkey, *keys)


async def click_at_async(x: int, y: int) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, click_at, x, y)


async def click_text_async(target_text: str) -> dict:
    """OCR-based click — the only operation with real latency (screenshot + OCR)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, click_text, target_text)


async def scroll_async(amount: int) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, scroll, amount)
