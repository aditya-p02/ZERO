# core/speaker_verify.py
# ZERO's voice identity system
#
# Enrollment  — first run, records Aditya's voice, saves voiceprint to disk
# Verification — every listen cycle, checks if speaker is Aditya
# Modes        — OWNER (only Aditya) / GUEST (anyone)
# Switching    — guest mode only activates if Aditya's voice authorizes it

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

# Suppress SpeechBrain's broken k2_fsa lazy-import warning — it's an optional
# Kaldi integration that isn't installed, and it pollutes the import system
# in a way that crashes unrelated packages (including Resemblyzer) if not muted.
warnings.filterwarnings("ignore", message=".*speechbrain.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Module.*deprecated.*", category=UserWarning)

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000
ENROLLMENT_SECS   = 30        # how long to record during enrollment
VERIFY_THRESHOLD  = 0.68      # similarity score to accept as Aditya (0.0–1.0)
VOICEPRINT_PATH   = Path(os.getenv("VOICEPRINT_PATH", "data/voiceprint.npy"))

# ZERO's rejection lines — rotated so it doesn't repeat
_REJECTION_LINES = [
    "That's not Aditya's voice. Don't play dumb with me.",
    "Nice try. You're not him.",
    "I know what Aditya sounds like. That isn't it.",
    "Not authorized. I don't take orders from just anyone.",
    "Wrong voice. I'm not switching for you.",
    "You're not Aditya. I wasn't built yesterday.",
    "That voice doesn't match. Try again with the right person.",
    "I recognize Aditya's voice. That's not it. Move on.",
]
_rejection_index = 0

# ── Eager encoder load ─────────────────────────────────────────────────────────
# Load Resemblyzer immediately at import time — before SpeechBrain (tone
# detection) ever runs and leaves broken lazy-import stubs in sys.modules.
# Those stubs (k2_fsa, huggingface.wordemb, nlp, etc.) regenerate on every
# SpeechBrain call and crash _embed() if Resemblyzer hasn't already claimed
# its namespace. Loading eagerly here wins the race permanently.
print("[ZERO] Loading voice encoder...")
from resemblyzer import VoiceEncoder  # noqa: E402
from resemblyzer import preprocess_wav as _preprocess_wav  # noqa: E402

_encoder = VoiceEncoder(device="cpu")
print("[ZERO] Voice encoder ready.")


def _get_encoder():
    return _encoder


# ── Voiceprint storage ─────────────────────────────────────────────────────────

def voiceprint_exists() -> bool:
    return VOICEPRINT_PATH.exists()


def _save_voiceprint(embedding: np.ndarray):
    VOICEPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(VOICEPRINT_PATH), embedding)
    print(f"[ZERO] Voiceprint saved → {VOICEPRINT_PATH}")


def _load_voiceprint() -> np.ndarray:
    return np.load(str(VOICEPRINT_PATH))


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _record(seconds: float) -> np.ndarray:
    """Record audio for a fixed duration. Blocking."""
    n = int(SAMPLE_RATE * seconds)
    audio = sd.rec(n, samplerate=SAMPLE_RATE, channels=1,
                   dtype="float32", blocking=True)
    return audio.flatten()


def _to_wav(audio: np.ndarray) -> str:
    """Save numpy audio to temp WAV, return path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        return f.name


def _embed(audio: np.ndarray) -> np.ndarray:
    """
    Get a 256-dim speaker embedding from audio.
    Resemblyzer expects float32 audio at 16kHz.
    """
    encoder = _get_encoder()
    wav_path = _to_wav(audio)
    try:
        wav = _preprocess_wav(wav_path)
        return encoder.embed_utterance(wav)
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings. Returns 0.0–1.0."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── Enrollment ─────────────────────────────────────────────────────────────────

def enroll() -> bool:
    """
    Record Aditya's voice and save his voiceprint.
    Called once on first run. Returns True on success.
    """
    print("\n[ZERO] ── Voice Enrollment ──────────────────────────────")
    print("[ZERO] I need to learn your voice. This happens once.")
    print(f"[ZERO] Speak naturally for {ENROLLMENT_SECS} seconds.")
    print("[ZERO] Talk about anything — introduce yourself, describe your day.")
    print("[ZERO] Starting in 3 seconds...\n")

    import time
    time.sleep(3)

    print("[ZERO] Recording... speak now.")
    audio = _record(ENROLLMENT_SECS)
    print("[ZERO] Recording done. Processing voiceprint...")

    try:
        embedding = _embed(audio)
        _save_voiceprint(embedding)
        print("[ZERO] Enrollment complete. I'll recognize your voice from now on.")
        return True
    except Exception as e:
        print(f"[ZERO] Enrollment failed: {e}")
        return False


async def enroll_async() -> bool:
    """Non-blocking enrollment — runs in thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, enroll)


# ── Verification ───────────────────────────────────────────────────────────────

def verify(audio: np.ndarray) -> tuple[bool, float]:
    """
    Check if the audio matches Aditya's voiceprint.
    Returns (is_aditya: bool, confidence: float).
    Audio too short or silent → (False, 0.0).
    """
    if not voiceprint_exists():
        # No voiceprint — fail open so ZERO still works before enrollment
        return True, 1.0

    # Skip silent audio
    if float(np.abs(audio).mean()) < 0.005:
        return False, 0.0

    # Too short to embed reliably (< 1 second)
    if len(audio) < SAMPLE_RATE:
        return False, 0.0

    try:
        stored = _load_voiceprint()
        live   = _embed(audio)
        score  = _similarity(live, stored)
        return score >= VERIFY_THRESHOLD, score
    except Exception as e:
        # Encoder is loaded at startup — an error here is a real failure.
        # Fail closed to keep owner mode secure.
        print(f"[ZERO] Verify error: {e} — rejecting for safety.")
        return False, 0.0


async def verify_async(audio: np.ndarray) -> tuple[bool, float]:
    """Non-blocking verification — runs in thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, verify, audio)


# ── Rejection lines ────────────────────────────────────────────────────────────

def get_rejection_line() -> str:
    """Returns the next rejection line, rotating through the list."""
    global _rejection_index
    line = _REJECTION_LINES[_rejection_index % len(_REJECTION_LINES)]
    _rejection_index += 1
    return line
