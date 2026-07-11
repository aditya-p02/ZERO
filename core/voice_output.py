# core/voice_output.py
# ZERO's voice — ElevenLabs primary, edge-tts fallback (auto, no credits needed)

import asyncio
import os
import re
import tempfile
import time

from dotenv import load_dotenv

from core.config import settings

load_dotenv()

ELEVENLABS_API_KEY = settings.elevenlabs_api_key
VOICE_ID = "iP95p4xoKVk53GoZ742B"  # Josh — deep, calm, works on free tier

EDGE_FALLBACK_VOICE = "en-US-GuyNeural"

_FAILURE_THRESHOLD = 3

_el_client = None
_el_failed = False
_el_fail_count = 0


def _get_el_client():
    global _el_client
    if _el_client is None and ELEVENLABS_API_KEY:
        from elevenlabs.client import ElevenLabs
        _el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return _el_client


def _is_quota_error(e: Exception) -> bool:
    s = str(e)
    return (
        "quota_exceeded" in s
        or "paid_plan_required" in s
        or "status_code: 402" in s
        or ("status_code: 401" in s and "quota" in s.lower())
    )


def _clean_for_speech(text: str) -> str:
    """Remove things that sound weird when spoken."""
    # Strip asterisk actions like *laughs*, *chuckles*, *sighs*
    text = re.sub(r'\*[^*]+\*', '', text)
    # Strip markdown bold/italic
    text = re.sub(r'\*+', '', text)
    # Strip backticks and code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)
    # Strip URLs
    text = re.sub(r'https?://\S+', '', text)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _speak_elevenlabs(text: str) -> bool:
    global _el_failed, _el_fail_count

    client = _get_el_client()
    if client is None:
        return False

    try:
        from elevenlabs import VoiceSettings
        audio = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.55,
                similarity_boost=0.85,
                style=0.2,
                use_speaker_boost=True
            )
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            for chunk in audio:
                tmp.write(chunk)
            tmp_path = tmp.name

        _play(tmp_path)
        _el_fail_count = 0
        return True

    except Exception as e:
        if _is_quota_error(e):
            print("[ZERO] ElevenLabs quota exhausted — switching to edge-tts fallback.")
            _el_failed = True
        else:
            _el_fail_count += 1
            print(f"[ZERO] ElevenLabs error ({_el_fail_count}/{_FAILURE_THRESHOLD}): {e}")
            if _el_fail_count >= _FAILURE_THRESHOLD:
                print("[ZERO] Too many ElevenLabs failures — switching to edge-tts fallback.")
                _el_failed = True
        return False


def _speak_edge_tts(text: str):
    try:
        import edge_tts

        async def _generate():
            communicate = edge_tts.Communicate(text, EDGE_FALLBACK_VOICE)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            await communicate.save(tmp_path)
            return tmp_path

        loop = asyncio.new_event_loop()
        try:
            tmp_path = loop.run_until_complete(_generate())
        finally:
            loop.close()

        _play(tmp_path)

    except ImportError:
        print("[ZERO] edge-tts not installed. Run: pip install edge-tts")
        print(f"[ZERO would say]: {text}")
    except Exception as e:
        print(f"[ZERO] edge-tts error: {e}")
        print(f"[ZERO would say]: {text}")


def speak(text: str):
    global _el_failed

    # Clean text before speaking
    text = _clean_for_speech(text)

    if not text:
        return

    if not _el_failed:
        success = _speak_elevenlabs(text)
        if success:
            return

    _speak_edge_tts(text)


async def speak_async(text: str):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, speak, text)


def _play(path: str):
    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.quit()
    try:
        os.unlink(path)
    except Exception:
        pass
