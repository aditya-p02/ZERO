# core/screen.py
# ZERO's eyes — screenshot capture + OCR
#
# Two modes:
#   capture()     → take a screenshot, run OCR, return structured result
#   describe()    → capture + ask Groq to describe/analyse what's on screen
#
# Dependencies:
#   pip install mss pillow pytesseract
#   Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki  (Windows)
#   After install, set TESSERACT_PATH in .env if not on system PATH
#   e.g. TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

import os
import asyncio
import tempfile
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Optional: override tesseract path via .env
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")

# Screenshot save dir — kept for session, cleaned on next capture
_SCREENSHOT_DIR = os.path.join(tempfile.gettempdir(), "zero_screenshots")
os.makedirs(_SCREENSHOT_DIR, exist_ok=True)

# Last screenshot path — agents can reference it
last_screenshot_path: str | None = None


def _setup_tesseract():
    """Point pytesseract at the right binary if TESSERACT_PATH is set."""
    if TESSERACT_PATH:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def _take_screenshot(monitor_index: int = 1) -> str:
    """
    Capture the screen using mss.
    monitor_index: 1 = primary monitor, 0 = all monitors combined
    Returns the path to the saved PNG.
    """
    import mss
    import mss.tools

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_SCREENSHOT_DIR, f"screen_{timestamp}.png")

    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        img = sct.grab(monitor)
        mss.tools.to_png(img.rgb, img.size, output=path)

    return path


def _ocr_image(image_path: str) -> str:
    """
    Run Tesseract OCR on the screenshot.
    Returns extracted text — may be noisy for complex UIs.
    """
    try:
        import pytesseract
        from PIL import Image

        _setup_tesseract()

        img = Image.open(image_path)

        # Upscale slightly for better OCR on small text
        w, h = img.size
        if w < 1920:
            scale = 1920 / w
            img = img.resize((int(w * scale), int(h * scale)))

        text = pytesseract.image_to_string(img, lang="eng")
        # Clean up: collapse excessive blank lines
        lines = [l.rstrip() for l in text.splitlines()]
        cleaned = "\n".join(l for l in lines if l.strip())
        return cleaned

    except ImportError:
        return "[OCR unavailable — install pytesseract and Pillow]"
    except Exception as e:
        return f"[OCR error: {e}]"


def _encode_image_base64(image_path: str) -> str:
    """Encode screenshot as base64 for sending to vision APIs."""
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def capture(monitor_index: int = 1) -> dict:
    """
    Main capture function — synchronous.
    Takes a screenshot, runs OCR, returns structured result.

    Returns:
        {
            "success": bool,
            "path": str,           # path to PNG file
            "ocr_text": str,       # raw OCR output
            "word_count": int,
            "error": str           # only on failure
        }
    """
    global last_screenshot_path

    try:
        path = _take_screenshot(monitor_index)
        last_screenshot_path = path
        ocr_text = _ocr_image(path)
        word_count = len(ocr_text.split())

        return {
            "success": True,
            "path": path,
            "ocr_text": ocr_text,
            "word_count": word_count,
            "error": "",
        }

    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        return {
            "success": False,
            "path": "",
            "ocr_text": "",
            "word_count": 0,
            "error": f"Missing dependency: {missing}. Run: pip install mss pillow pytesseract",
        }
    except Exception as e:
        return {
            "success": False,
            "path": "",
            "ocr_text": "",
            "word_count": 0,
            "error": str(e),
        }


async def capture_async(monitor_index: int = 1) -> dict:
    """Async wrapper for capture() — doesn't block the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, capture, monitor_index)


def get_last_screenshot() -> str | None:
    """Return path to the most recent screenshot, or None."""
    return last_screenshot_path


# Public alias so agents can import ocr_image without touching private internals.
# agents/screen.py uses this for passive screen context injection.
ocr_image = _ocr_image