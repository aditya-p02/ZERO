# core/platform_tools.py
# Cross-platform helpers for optional OS/desktop integrations.

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Sequence

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"


class OptionalDependencyError(RuntimeError):
    """Raised when an optional platform dependency is unavailable."""


class UnsupportedPlatformError(RuntimeError):
    """Raised when a feature is not implemented for the current OS."""


def platform_name() -> str:
    return platform.system() or "Unknown"


def require_module(module_name: str, install_hint: str | None = None):
    """Import an optional dependency only when the feature actually needs it."""
    try:
        return __import__(module_name)
    except ImportError as exc:
        hint = f" Install it with: {install_hint}." if install_hint else ""
        raise OptionalDependencyError(
            f"Missing optional dependency '{module_name}'.{hint}"
        ) from exc


def command_exists(command: str) -> bool:
    if IS_WINDOWS and command.lower().startswith("ms-settings:"):
        return True
    return shutil.which(command) is not None


def run_detached(command: Sequence[str]) -> subprocess.Popen:
    """Start a GUI/CLI program without invoking a shell."""
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.PIPE,
        "shell": False,
    }
    if IS_WINDOWS:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(list(command), **kwargs)


def open_uri(uri: str) -> None:
    if IS_WINDOWS:
        os.startfile(uri)  # type: ignore[attr-defined]
        return
    if IS_LINUX:
        if not command_exists("xdg-open"):
            raise OptionalDependencyError(
                "Missing 'xdg-open', needed to open desktop URIs on Linux."
            )
        run_detached(["xdg-open", uri])
        return
    if IS_MACOS:
        run_detached(["open", uri])
        return
    raise UnsupportedPlatformError(f"Opening URIs is not supported on {platform_name()}.")


def linux_desktop_session_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def get_linux_window_titles() -> list[str]:
    """Best-effort Linux window title listing without importing pygetwindow."""
    if command_exists("wmctrl"):
        proc = subprocess.run(["wmctrl", "-l"], text=True, capture_output=True, check=False)
        if proc.returncode == 0:
            titles = []
            for line in proc.stdout.splitlines():
                parts = line.split(None, 3)
                if len(parts) == 4 and parts[3].strip():
                    titles.append(parts[3].strip())
            return titles

    if command_exists("xdotool"):
        proc = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", "."],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            titles = []
            for window_id in proc.stdout.splitlines():
                title_proc = subprocess.run(
                    ["xdotool", "getwindowname", window_id],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                title = title_proc.stdout.strip()
                if title:
                    titles.append(title)
            return titles

    raise UnsupportedPlatformError(
        "Linux window listing needs either 'wmctrl' or 'xdotool' installed."
    )


def focus_linux_window(title: str) -> None:
    if command_exists("wmctrl"):
        proc = subprocess.run(["wmctrl", "-a", title], text=True, capture_output=True, check=False)
        if proc.returncode == 0:
            return
        raise UnsupportedPlatformError(proc.stderr.strip() or f"Couldn't focus window '{title}'.")

    if command_exists("xdotool"):
        proc = subprocess.run(
            ["xdotool", "search", "--name", title, "windowactivate"],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return
        raise UnsupportedPlatformError(proc.stderr.strip() or f"Couldn't focus window '{title}'.")

    raise UnsupportedPlatformError("Linux window focusing needs either 'wmctrl' or 'xdotool'.")


def close_linux_window(title: str) -> None:
    if command_exists("wmctrl"):
        proc = subprocess.run(["wmctrl", "-c", title], text=True, capture_output=True, check=False)
        if proc.returncode == 0:
            return
        raise UnsupportedPlatformError(proc.stderr.strip() or f"Couldn't close window '{title}'.")

    if command_exists("xdotool"):
        proc = subprocess.run(
            ["xdotool", "search", "--name", title, "windowclose"],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return
        raise UnsupportedPlatformError(proc.stderr.strip() or f"Couldn't close window '{title}'.")

    raise UnsupportedPlatformError("Linux window closing needs either 'wmctrl' or 'xdotool'.")
