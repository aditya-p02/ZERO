# agents/research.py
# ZERO's research agent — DDGS search + Groq synthesis

import asyncio
import os
import re

from ddgs import DDGS
from dotenv import load_dotenv

from core.clients import groq_client
from core.config import settings
from core.logger import log
from core.memory import get_recent_conversation

load_dotenv()

USER_NAME = os.getenv("USER_NAME", "Aditya")

RESEARCH_PROMPT = f"""
You are ZERO — {USER_NAME}'s personal AI.
You have just searched the web and retrieved real results below.

Rules:
- Answer using ONLY the search results provided. Do not add anything from memory or training.
- Be direct and conversational. No bullet points unless it genuinely helps.
- If the results don't contain the answer, say so honestly — don't guess.
- Cite nothing formally — just speak naturally like a knowledgeable friend.
- Keep it concise unless the question needs depth.
- ALWAYS consider the conversation history for context. If the user's question is vague or refers to something mentioned before, use that context to understand what they mean.
"""

QUERY_EXTRACTOR_PROMPT = """
Extract a clean, short web search query from the user's message.
Consider that this may be a follow-up question — if it refers to something vague like "what year" or "who was he", use the conversation context to make the query specific.
Return ONLY the search query — no explanation, no punctuation at the end, no extra words.

Examples:
"tell me the current price of Bitcoin" → "Bitcoin price today"
"well can you tell me who won IPL 2026" → "IPL 2026 winner"
"what is the latest news on AI" → "latest AI news 2026"
"search for JEE 2026 exam date" → "JEE 2026 exam date"
"who is the prime minister of India" → "current Prime Minister of India"
"what year did that happen" → use prior context to make specific query
"""


def _extract_query(user_message: str, history: list = None) -> str:
    clean = re.sub(r'\[tone:.*?\]', '', user_message).strip()
    
    messages = [{"role": "system", "content": QUERY_EXTRACTOR_PROMPT}]
    
    if history:
        messages.extend(history[-4:])
    
    messages.append({"role": "user", "content": clean})
    
    try:
        completion = groq_client.chat.completions.create(
            model=settings.groq_research_model,
            messages=messages,
            temperature=0.0,
            max_tokens=20,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        log.error("Query extraction failed — using raw message as query", exc_info=True)
        return clean


def _search(query: str, max_results: int = 5) -> list:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception:
        log.error("DDGS search failed", exc_info=True)
        return []


def _format_results(results: list) -> str:
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip()
        body  = r.get("body", "").strip()
        url   = r.get("href", "").strip()
        lines.append(f"[{i}] {title}\n{body}\nSource: {url}")

    return "\n\n".join(lines)


def _synthesize(query: str, results_text: str, history: list = None) -> str:
    messages = [{"role": "system", "content": RESEARCH_PROMPT}]
    
    if history:
        messages.extend(history[-6:])
    
    messages.append({
        "role": "user",
        "content": f"Question: {query}\n\nSearch Results:\n{results_text}"
    })
    
    try:
        completion = groq_client.chat.completions.create(
            model=settings.groq_research_model,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        log.error("Groq synthesis failed", exc_info=True)
        return f"Search worked but synthesis failed: {str(e)}"


async def research(user_message: str) -> str:
    loop = asyncio.get_running_loop()

    history = await get_recent_conversation(limit=6)

    query = await loop.run_in_executor(None, _extract_query, user_message, history)
    print(f"[ZERO] Searching → {query}")

    results = await loop.run_in_executor(None, _search, query)

    if not results:
        from core.brain import think
        return await think(user_message)

    results_text = _format_results(results)
    response = await loop.run_in_executor(None, _synthesize, query, results_text, history)
    return response