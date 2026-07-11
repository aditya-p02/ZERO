# ZERO

ZERO is a local personal AI assistant prototype with text chat, voice chat, memory, desktop automation, screen OCR, system controls, research, coding help, and a browser HUD.

This repo is currently focused on making the existing capabilities reliable. It does not add new features beyond the current build.

## Current Capabilities

- Text and voice interaction from `main.py`
- Cloud chat through Groq
- Private chat through local Ollama
- PostgreSQL memory with pgvector image
- Intent routing across current agents
- Web research through DDGS plus Groq synthesis
- Code writing and limited local code execution
- Desktop automation through PyAutoGUI
- System status and basic volume/brightness control
- Screen capture plus Tesseract OCR
- WebSocket/browser HUD
- ElevenLabs voice output with Edge TTS fallback

## Setup

1. Create and activate a Python 3.11+ virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
python -m pip install -r requirements.txt
```

3. Create your local environment file.

```powershell
Copy-Item .env.example .env
```

4. Fill in `.env` with your keys and local paths.

5. Start PostgreSQL memory.

```powershell
docker compose up -d
```

6. Run ZERO.

```powershell
python main.py
```

## Required Services

- Groq API key is required for cloud chat, routing fallback, research synthesis, memory extraction, and cloud transcription.
- Docker is required for the included PostgreSQL memory database.
- Ollama is required only for private mode.
- Tesseract OCR is required only for screen reading.
- ElevenLabs is optional because `edge-tts` is used as fallback.

## Windows OCR Setup

Install Tesseract from the UB Mannheim Windows builds, then set this in `.env` if it is not already on PATH:

```env
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Verification

Run syntax and unit checks:

```powershell
python -m compileall .
python -m pytest
```

Optional lint check:

```powershell
python -m ruff check .
```

## Notes

- The code executor is a limited local safety layer, not a true security sandbox.
- Desktop automation controls the real machine. Keep PyAutoGUI failsafe enabled.
- `.env`, logs, caches, and virtual environments should stay uncommitted.
