# agents/screen.py
# ZERO's screen vision agent
# Handles: "what's on my screen", "read this", "what does this error say", "describe my screen"

import asyncio

from core.clients import groq_client
from core.config import settings
from core.screen import capture_async

USER_NAME = settings.user_name

SCREEN_ANALYST_PROMPT = f"""
You are ZERO — {USER_NAME}'s personal AI with the ability to see his screen.
You have just been given the text extracted from a screenshot via OCR.

Rules:
- Describe or analyse what's on screen based ONLY on the OCR text provided.
- Be direct and useful. Don't pad the answer.
- If it's an error message — identify it, explain it, suggest a fix.
- If it's code — read it and explain what it does or what's wrong.
- If it's a webpage or document — summarize the key content.
- If the OCR text is empty or garbled — say so honestly.
- Keep it conversational — {USER_NAME} is listening, not reading.
"""


def _analyse(user_message: str, ocr_text: str) -> str:
    """Send OCR text + user question to Groq for analysis."""
    try:
        screen_context = f"=== SCREEN CONTENT (via OCR) ===\n{ocr_text}\n=== END SCREEN CONTENT ==="

        completion = groq_client.chat.completions.create(
            model=settings.groq_screen_model,
            messages=[
                {"role": "system", "content": SCREEN_ANALYST_PROMPT},
                {"role": "user", "content": f"{screen_context}\n\nUser question: {user_message}"},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Screen captured but analysis failed: {e}"


async def handle_screen(user_message: str) -> str:
    """
    Main entry point for screen agent.
    1. Capture screenshot
    2. Run OCR
    3. Analyse with Groq based on user's question
    """
    loop = asyncio.get_running_loop()

    print("[ZERO] Capturing screen...")
    result = await capture_async()

    if not result["success"]:
        return f"Couldn't capture the screen. {result['error']}"

    ocr_text = result["ocr_text"]
    word_count = result["word_count"]

    print(f"[ZERO] Screen captured — {word_count} words read via OCR.")

    if word_count == 0:
        return (
            "I took the screenshot but couldn't read any text from it. "
            "The screen might be showing mostly images or graphics with no readable text."
        )

    # Analyse in thread so we don't block
    response = await loop.run_in_executor(None, _analyse, user_message, ocr_text)
    return response


def get_screen_context() -> str | None:
    """
    Called by brain.py to inject current screen content into ZERO's context.
    Returns a short summary string, or None if no screenshot taken yet.
    Used passively — ZERO is aware of the screen without being explicitly asked.
    """
    from core.screen import last_screenshot_path, ocr_image
    if last_screenshot_path is None:
        return None
    try:
        text = ocr_image(last_screenshot_path)
        if not text.strip():
            return None
        # Truncate to avoid bloating context — first 800 chars is enough for awareness
        truncated = text[:800]
        return f"[Screen context — what's currently visible]\n{truncated}"
    except Exception:
        return None