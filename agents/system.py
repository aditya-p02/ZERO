# agents/system.py
# ZERO's system control agent
# Handles: volume, brightness, battery/CPU/RAM status, sleep
# Deliberately excludes: shutdown, restart, anything touching personal files/data

import re
import asyncio
import psutil

# ── Volume control (Windows — pycaw) ───────────────────────────────────────────

def _get_volume_interface():
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def _set_volume(level: int) -> str:
    """level: 0-100"""
    level = max(0, min(100, level))
    try:
        volume = _get_volume_interface()
        volume.SetMasterVolumeLevelScalar(level / 100, None)
        return f"Volume set to {level}%."
    except Exception as e:
        return f"Couldn't change volume: {e}"


def _get_volume() -> str:
    try:
        volume = _get_volume_interface()
        current = round(volume.GetMasterVolumeLevelScalar() * 100)
        return f"Volume is at {current}%."
    except Exception as e:
        return f"Couldn't read volume: {e}"


def _mute(state: bool) -> str:
    try:
        volume = _get_volume_interface()
        volume.SetMute(1 if state else 0, None)
        return "Muted." if state else "Unmuted."
    except Exception as e:
        return f"Couldn't change mute state: {e}"


def _adjust_volume(delta: int) -> str:
    """delta: positive or negative, e.g. +10 or -10"""
    try:
        volume = _get_volume_interface()
        current = volume.GetMasterVolumeLevelScalar() * 100
        new_level = max(0, min(100, current + delta))
        volume.SetMasterVolumeLevelScalar(new_level / 100, None)
        return f"Volume {'up' if delta > 0 else 'down'} to {round(new_level)}%."
    except Exception as e:
        return f"Couldn't adjust volume: {e}"


# ── Brightness control ──────────────────────────────────────────────────────────

def _set_brightness(level: int) -> str:
    """level: 0-100"""
    import screen_brightness_control as sbc
    level = max(0, min(100, level))
    try:
        sbc.set_brightness(level)
        return f"Brightness set to {level}%."
    except Exception as e:
        return f"Couldn't change brightness: {e}"


def _get_brightness() -> str:
    import screen_brightness_control as sbc
    try:
        current = sbc.get_brightness()[0]
        return f"Brightness is at {current}%."
    except Exception as e:
        return f"Couldn't read brightness: {e}"


def _adjust_brightness(delta: int) -> str:
    import screen_brightness_control as sbc
    try:
        current = sbc.get_brightness()[0]
        new_level = max(0, min(100, current + delta))
        sbc.set_brightness(new_level)
        return f"Brightness {'up' if delta > 0 else 'down'} to {new_level}%."
    except Exception as e:
        return f"Couldn't adjust brightness: {e}"


# ── System status ────────────────────────────────────────────────────────────────

def _get_battery() -> str:
    battery = psutil.sensors_battery()
    if battery is None:
        return "No battery detected — probably a desktop."
    plugged = "charging" if battery.power_plugged else "on battery"
    return f"Battery is at {round(battery.percent)}%, {plugged}."


def _get_cpu() -> str:
    usage = psutil.cpu_percent(interval=0.5)
    return f"CPU usage is at {usage}%."


def _get_ram() -> str:
    mem = psutil.virtual_memory()
    used_gb = round(mem.used / (1024 ** 3), 1)
    total_gb = round(mem.total / (1024 ** 3), 1)
    return f"RAM usage is {mem.percent}% — {used_gb}GB of {total_gb}GB."


def _get_disk() -> str:
    disk = psutil.disk_usage('/')
    used_gb = round(disk.used / (1024 ** 3), 1)
    total_gb = round(disk.total / (1024 ** 3), 1)
    return f"Disk is {disk.percent}% full — {used_gb}GB of {total_gb}GB used."


def _get_full_status() -> str:
    parts = [_get_battery(), _get_cpu(), _get_ram(), _get_disk()]
    return " ".join(parts)


# ── Sleep ──────────────────────────────────────────────────────────────────────

def _sleep_system() -> str:
    try:
        import ctypes
        ctypes.windll.powrprof.SetSuspendState(0, 1, 0)
        return "Going to sleep now."
    except Exception as e:
        return f"Couldn't put the system to sleep: {e}"


# ── Intent parsing ───────────────────────────────────────────────────────────────

def _extract_number(text: str) -> int | None:
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


def _handle(user_message: str) -> str:
    lower = user_message.lower()

    # ── Volume ──────────────────────────────────────────────────────────────
    if "volume" in lower or "mute" in lower:
        if "unmute" in lower:
            return _mute(False)
        if "mute" in lower:
            return _mute(True)
        if "up" in lower or "increase" in lower or "raise" in lower:
            num = _extract_number(lower)
            return _adjust_volume(num if num else 10)
        if "down" in lower or "decrease" in lower or "lower" in lower:
            num = _extract_number(lower)
            return _adjust_volume(-(num if num else 10))
        if "set" in lower or "%" in lower:
            num = _extract_number(lower)
            if num is not None:
                return _set_volume(num)
        if "what" in lower or "current" in lower or "check" in lower:
            return _get_volume()
        return _get_volume()

    # ── Brightness ──────────────────────────────────────────────────────────
    if "brightness" in lower:
        if "up" in lower or "increase" in lower or "raise" in lower or "brighter" in lower:
            num = _extract_number(lower)
            return _adjust_brightness(num if num else 10)
        if "down" in lower or "decrease" in lower or "lower" in lower or "dimmer" in lower or "dim" in lower:
            num = _extract_number(lower)
            return _adjust_brightness(-(num if num else 10))
        if "set" in lower or "%" in lower:
            num = _extract_number(lower)
            if num is not None:
                return _set_brightness(num)
        if "what" in lower or "current" in lower or "check" in lower:
            return _get_brightness()
        return _get_brightness()

    # ── Sleep ──────────────────────────────────────────────────────────────
    if "sleep" in lower:
        return _sleep_system()

    # ── Status checks ─────────────────────────────────────────────────────
    if "battery" in lower:
        return _get_battery()
    if "cpu" in lower:
        return _get_cpu()
    if "ram" in lower or "memory usage" in lower:
        return _get_ram()
    if "disk" in lower or "storage" in lower:
        return _get_disk()
    if "status" in lower or "stats" in lower or "system" in lower:
        return _get_full_status()

    return "Not sure what system action you want — try volume, brightness, battery, CPU, RAM, disk, or sleep."


async def handle_system(user_message: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _handle, user_message)