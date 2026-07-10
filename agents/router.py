# agents/router.py
# ZERO's intent classifier — Hybrid mode
# Keywords first (instant), Groq fallback (when ambiguous)

import os
import re
from groq import Groq
from dotenv import load_dotenv
from core.logger import log

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INTENTS = [
    "general",
    "research",
    "code",
    "automation",
    "memory",
    "system",
    "screen",
]

# Keyword scoring weights — longer/more specific phrases score higher
# so "write a script" beats "script" alone in close calls.
# Generic single words removed where they caused ties without helping
# disambiguation (e.g. "app", "type", "start", "find", "close" all appear
# naturally in sentences meant for a different intent).
_KEYWORD_MAP = {
    "research": [
        "search for", "look up", "google", "browse", "research",
        "what is the latest", "news about", "who is", "when did",
        "latest news", "recent news", "web search", "internet",
        "current news", "find out", "look into",
    ],
    "code": [
        "write code", "write a script", "write a function", "write a program",
        "create a function", "build a function", "implement",
        "code", "script", "algorithm", "debug", "fix this code",
        "fix the bug", "error in", "run this", "run the code",
        "execute", "python", "javascript", "html", "css", "sql",
        "refactor", "optimise", "optimize",
    ],
    "automation": [
        "open app", "open the app", "close app", "close the",
        "click on", "launch", "minimize", "maximize",
        "move the mouse", "press", "hotkey", "key combination",
        "desktop", "switch window", "switch to", "application",
        "automate", "control the", "type this", "type out",
    ],
    "memory": [
        "remember this", "don't forget", "add to memory", "save this",
        "what do you know about me", "what have i told you",
        "forget this", "update my", "my facts", "recall",
        "store this", "keep this in mind",
    ],
    "system": [
        "volume", "brightness", "shutdown", "restart", "sleep mode",
        "wifi", "bluetooth", "battery", "disk space", "cpu", "ram",
        "memory usage", "turn off", "turn on", "mute", "unmute",
        "night mode", "dark mode", "how's the system", "how is the system",
        "system doing", "system status", "system stats", "check the system",
    ],
    "screen": [
        "what's on my screen", "what is on my screen", "look at my screen",
        "read my screen", "read the screen", "what does this say",
        "what does this error say", "what error is this", "see my screen",
        "describe my screen", "describe what's on screen", "what am i looking at",
        "analyse my screen", "analyze my screen", "screenshot",
        "what's on screen", "read this error", "what's on the screen",
    ],
}

# Tie-breaking priority — when two intents score equally, the one appearing
# earlier in this list wins. More specific/actionable intents beat broader ones.
_INTENT_PRIORITY = ["screen", "memory", "system", "code", "automation", "research", "general"]

# Genuine continuation phrases — feel wrong as opening lines with no prior context.
# Removed: "what is/was", "who/where/when/how/why was", "what day/year/time/were"
# — all valid fresh questions that were bypassing keyword matching entirely.
_FOLLOWUP_PHRASES = [
    "and what", "and when", "and who", "and where", "and how", "and why",
    "what about", "how about", "which one", "which day", "which year",
    "tell me more", "more about", "so what", "then what", "but what",
    "i mean", "i'm asking", "asking about",
]

_CLASSIFIER_PROMPT = f"""
You are an intent classifier for an AI assistant called ZERO.
Classify the user's message into exactly one of these intents:

- general    → conversation, opinions, explanations, advice, casual talk, follow-up questions about something already discussed
- research   → needs live web search or current information not yet in conversation
- code       → writing, debugging, running, or explaining code
- automation → controlling the desktop, opening apps, clicking things
- memory     → saving or recalling personal facts about the user (ONLY when user explicitly says "remember this" or "what do you know about me")
- system     → OS-level controls like volume, brightness, shutdown, AND system status checks (battery, CPU, RAM, disk, "how's the system doing")
- screen     → look at the screen, read what's on screen, describe screen, read an error

IMPORTANT:
- If the message is a short follow-up to something already discussed (like "what day was it?" after talking about India's independence), classify as "general" — ZERO already has the context.
- Only use "memory" if the user explicitly asks to save or recall personal facts.
- When in doubt between research and general, prefer "general" for follow-ups.

Reply with ONLY the intent word. Nothing else. No punctuation. No explanation.
"""


def _is_followup(text: str, has_prior_conversation: bool = False) -> bool:
    """
    True only when: prior conversation exists AND message is short AND
    starts with a continuation phrase. Mid-sentence matching removed —
    it was firing on fresh questions like 'what is the weather today'.
    """
    if not has_prior_conversation:
        return False
    lower = text.lower().strip()
    if len(lower.split()) > 10:
        return False
    return any(lower.startswith(phrase) for phrase in _FOLLOWUP_PHRASES)


def _keyword_match(text: str) -> str | None:
    """
    Score each intent by keyword hits, then return the winner.

    Old behaviour: return None on any tie, silently falling through to a
    Groq API call for something the keyword list almost certainly handled.

    New behaviour: on a tie, pick the highest-priority intent from
    _INTENT_PRIORITY (more specific beats more general). Only falls to
    Groq when score is zero — i.e. genuinely no keywords matched at all.
    """
    lower = text.lower()
    scores = {intent: 0 for intent in INTENTS}

    for intent, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if kw in lower:
                # Longer keywords are more specific — give them more weight
                scores[intent] += len(kw.split())

    best_score = max(scores.values())

    if best_score == 0:
        return None  # nothing matched — let Groq handle it

    top_intents = [i for i, s in scores.items() if s == best_score]

    if len(top_intents) == 1:
        return top_intents[0]

    # Tie — pick highest priority intent instead of falling to Groq
    for intent in _INTENT_PRIORITY:
        if intent in top_intents:
            return intent

    return top_intents[0]


def _groq_classify(text: str) -> str:
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _CLASSIFIER_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        result = completion.choices[0].message.content.strip().lower()
        return result if result in INTENTS else "general"
    except Exception:
        log.error("Groq classify failed", exc_info=True)
        return "general"


def classify(user_message: str, has_prior_conversation: bool = False) -> str:
    clean = re.sub(r'\[tone:.*?\]', '', user_message).strip()

    if _is_followup(clean, has_prior_conversation=has_prior_conversation):
        return "general"

    # Keyword match
    intent = _keyword_match(clean)
    if intent:
        return intent

    # Groq fallback
    return _groq_classify(clean)