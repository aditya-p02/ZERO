# core/voice_input.py
# ZERO's ears — Continuous listening + transcription
#
# Pipeline:
#   1. Wait for speech (silence-gated, no wake word needed)
#   2. Transcribe via Groq Whisper (cloud) or local Whisper (private mode)
#
# No speaker verification. No guest/owner mode. No noise reduction. Anyone can talk to ZERO.

import os
import tempfile

import numpy as np

from core.clients import groq_client
from core.config import settings

# Audio settings
SAMPLE_RATE     = 16000
VAD_SPEECH_PROB = 0.5   # Silero VAD speech-probability threshold.
                         # Replaces the old fixed amplitude SILENCE_THRESH=0.004,
                         # which discarded quiet/far speech regardless of content.

# Silero VAD (onnx) hard-requires EXACTLY 512 samples per call at 16kHz
# (256 at 8kHz). It's not a suggestion — the model raises ValueError on any
# other size. So the mic block size has to be this, not an arbitrary "nice"
# duration like 100ms. See silero_vad.utils_vad.OnnxWrapper.__call__.
VAD_FRAME_SAMPLES = 512
_BLOCK_DUR        = VAD_FRAME_SAMPLES / SAMPLE_RATE   # ~0.032s per block

SILENCE_SECS       = 1.5   # silence duration that ends recording
MAX_CMD_SECS       = 30    # hard cap on recording length
SPEECH_START_SECS  = 0.4   # sustained speech needed before recording starts
MIN_SPEECH_SECS    = 1.0   # min total speech time in a recording to bother transcribing

# Lazy-loaded local Whisper model
_local_model = None

# Lazy-loaded Silero VAD model
_vad_model = None


def _get_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Voice input needs the optional 'sounddevice' package. "
            "Install project dependencies for your OS before using voice mode."
        ) from exc
    return sd


def _get_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Voice transcription needs the optional 'soundfile' package. "
            "Install project dependencies for your OS before using voice mode."
        ) from exc
    return sf


def _get_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Silero VAD needs the optional 'torch' package. "
            "Install project dependencies for your OS before using voice mode."
        ) from exc
    return torch


# ── Local model loader ─────────────────────────────────────────────────────────

def _get_local_model():
    global _local_model
    if _local_model is None:
        print("[ZERO] Loading local Whisper model... (one-time)")
        from faster_whisper import WhisperModel
        _local_model = WhisperModel("medium.en", device="cpu", compute_type="int8")
        print("[ZERO] Local model ready.")
    return _local_model


# ── VAD model loader ────────────────────────────────────────────────────────────

def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        print("[ZERO] Loading Silero VAD... (one-time)")
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad(onnx=True)  # onnx — lighter than full torch inference
        print("[ZERO] VAD ready.")
    return _vad_model


def _is_speech(block: np.ndarray) -> bool:
    """
    Classify one audio block as speech using Silero VAD — real speech-pattern
    recognition, not raw loudness. This is what makes quiet or far-from-mic
    speech reliable; a fixed amplitude threshold cannot distinguish "quiet
    speech" from "silence" the way a trained VAD model can.

    block MUST be exactly VAD_FRAME_SAMPLES (512) samples at SAMPLE_RATE
    (16000) — Silero enforces this internally and raises ValueError otherwise.
    """
    model = _get_vad_model()
    torch = _get_torch()
    with torch.no_grad():
        prob = model(torch.from_numpy(block), SAMPLE_RATE).item()
    return prob > VAD_SPEECH_PROB


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _record_until_silence() -> tuple[np.ndarray, bool]:
    """
    Wait for speech to start, then record until silence.
    Returns (audio, speech_detected).

    Phase A — wait: discard blocks until speech onset confirmed for
              SPEECH_START_SECS of consecutive speech.
    Phase B — record: capture until SILENCE_SECS of silence or MAX_CMD_SECS.

    speech_detected=False means it was a cough/thump — skip transcription.

    Speech/silence classification uses Silero VAD (_is_speech), not raw
    amplitude — this is what makes quiet or far-from-mic speech reliable.

    Block size is fixed at VAD_FRAME_SAMPLES because Silero VAD only accepts
    that exact frame size per call — everything else here (onset/silence/max
    thresholds) is defined in seconds and converted to block counts so the
    actual timing behavior doesn't depend on that implementation detail.
    """
    block_size          = VAD_FRAME_SAMPLES
    max_blocks          = int(MAX_CMD_SECS / _BLOCK_DUR)
    silence_blocks      = max(1, round(SILENCE_SECS / _BLOCK_DUR))
    speech_start_blocks = max(1, round(SPEECH_START_SECS / _BLOCK_DUR))
    min_speech_blocks   = max(1, round(MIN_SPEECH_SECS / _BLOCK_DUR))

    # Fresh session: clear Silero's recurrent state (_state/_context) so
    # audio from a previous listen() call can't bleed into this one's
    # speech/silence decisions.
    _get_vad_model().reset_states()

    sd = _get_sounddevice()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=block_size) as stream:

        # ── Phase A: wait for speech onset ────────────────────────────────
        onset_count = 0
        while True:
            block, _ = stream.read(block_size)
            block = block.flatten()
            if _is_speech(block):
                onset_count += 1
                if onset_count >= speech_start_blocks:
                    recorded = [block]
                    break
            else:
                onset_count = 0

        # ── Phase B: record until silence ─────────────────────────────────
        silent_count  = 0
        speech_blocks = speech_start_blocks

        for _ in range(max_blocks):
            block, _ = stream.read(block_size)
            block = block.flatten()
            recorded.append(block)
            if _is_speech(block):
                speech_blocks += 1
                silent_count = 0
            else:
                silent_count += 1
                if silent_count >= silence_blocks:
                    break

    audio = np.concatenate(recorded)
    speech_detected = speech_blocks >= min_speech_blocks
    return audio, speech_detected


def _audio_to_wav(audio: np.ndarray) -> str:
    """Write numpy audio to a temp WAV file, return the path."""
    sf = _get_soundfile()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        return tmp.name


# ── Transcription ──────────────────────────────────────────────────────────────

def _transcribe_groq(audio: np.ndarray) -> str:
    """Transcribe via Groq Whisper API — fast, accurate, cloud."""
    wav_path = _audio_to_wav(audio)
    try:
        with open(wav_path, "rb") as f:
            result = groq_client.audio.transcriptions.create(
                model=settings.groq_transcribe_model,
                file=f,
                language="en"
            )
        return result.text.strip()
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


def _transcribe_local(audio: np.ndarray) -> str:
    """Transcribe locally via faster-whisper — private mode, nothing leaves machine."""
    wav_path = _audio_to_wav(audio)
    try:
        model = _get_local_model()
        segments, _ = model.transcribe(
            wav_path,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return " ".join(seg.text for seg in segments).strip()
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


# ── Main listen function ───────────────────────────────────────────────────────

def listen(private: bool = False) -> str:
    """
    Continuous listening pipeline — no wake word, no speaker verification,
    no noise reduction.

    Waits for speech → records until silence → transcribes.

    private=True  → uses local Whisper, nothing leaves the machine
    private=False → uses Groq Whisper API

    Returns transcribed text, or "" if nothing was captured.
    """
    print("[ZERO] Listening...")

    audio, speech_detected = _record_until_silence()

    if not speech_detected:
        return ""

    if private:
        text = _transcribe_local(audio)
    else:
        text = _transcribe_groq(audio)

    return text.strip() if text else ""