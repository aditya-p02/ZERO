# core/memory.py
# ZERO's memory system — PostgreSQL + pgvector

import os
import json
import asyncio
from datetime import datetime
from typing import Optional
import asyncpg
from dotenv import load_dotenv
from core.logger import log

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5433)),
    "database": os.getenv("POSTGRES_DB", "zero_memory"),
    "user": os.getenv("POSTGRES_USER", "zero"),
    "password": os.getenv("POSTGRES_PASSWORD", "zero_secure_pass"),
}

_pool = None

# How many times to retry pool creation before giving up (e.g. Docker
# is still starting up when ZERO launches)
_POOL_CONNECT_RETRIES = 3
_POOL_RETRY_DELAY_SECS = 2.0


async def get_pool():
    """
    Returns the shared asyncpg connection pool, creating it if needed.

    Two failure modes handled here that the original code silently ignored:

    1. Pool is None but Postgres is unreachable (Docker not running, crash) —
       retried up to _POOL_CONNECT_RETRIES times with a short delay, then
       raises with an actionable error message.

    2. Pool exists but connections are dead (Postgres restarted while ZERO
       was running) — callers catch asyncpg errors and call reset_pool(),
       which sets _pool back to None so the next get_pool() call rebuilds it.
    """
    global _pool
    if _pool is not None:
        return _pool

    last_error = None
    for attempt in range(1, _POOL_CONNECT_RETRIES + 1):
        try:
            _pool = await asyncpg.create_pool(
                **DB_CONFIG,
                min_size=1,
                max_size=5,
                command_timeout=10,       # fail fast if Postgres is unresponsive
            )
            log.info("PostgreSQL pool connected (attempt %d)", attempt)
            return _pool
        except Exception as e:
            last_error = e
            log.warning(
                "PostgreSQL connection attempt %d/%d failed — retrying in %.0fs",
                attempt, _POOL_CONNECT_RETRIES, _POOL_RETRY_DELAY_SECS
            )
            if attempt < _POOL_CONNECT_RETRIES:
                await asyncio.sleep(_POOL_RETRY_DELAY_SECS)

    log.error(
        "Failed to connect to PostgreSQL at %s:%s after %d attempts. "
        "Is Docker running? Try: docker compose up -d",
        DB_CONFIG["host"], DB_CONFIG["port"], _POOL_CONNECT_RETRIES,
        exc_info=True
    )
    raise last_error


async def reset_pool():
    """
    Discard the current pool so the next get_pool() call rebuilds it.
    Called by memory functions when they catch a connection-level error,
    which means Postgres restarted while ZERO was running.
    """
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:
            pass  # pool is already broken — closing best-effort only
        _pool = None
    log.warning("Connection pool reset — will reconnect on next memory call")


async def init_memory():
    """Create all tables ZERO needs. Run once on startup."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Conversation history
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Long-term facts ZERO remembers about Aditya
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id SERIAL PRIMARY KEY,
                category TEXT NOT NULL,
                fact TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Tasks and goals
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                priority TEXT DEFAULT 'normal',
                notes TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # ZERO's general knowledge store (vector searchable later)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id SERIAL PRIMARY KEY,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """)

    print("[ZERO] Memory initialized.")


async def save_message(role: str, content: str):
    """Save a conversation turn."""
    for attempt in range(2):
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO conversations (role, content) VALUES ($1, $2)",
                    role, content
                )
            return
        except (asyncpg.PostgresConnectionError, OSError) as e:
            if attempt == 0:
                log.warning("save_message: connection lost, resetting pool and retrying")
                await reset_pool()
            else:
                log.error("save_message: failed after pool reset", exc_info=True)
                raise


async def get_recent_conversation(limit: int = 20) -> list:
    """Get the last N messages for context injection."""
    for attempt in range(2):
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT role, content FROM conversations ORDER BY timestamp DESC LIMIT $1",
                    limit
                )
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        except (asyncpg.PostgresConnectionError, OSError) as e:
            if attempt == 0:
                log.warning("get_recent_conversation: connection lost, resetting pool and retrying")
                await reset_pool()
            else:
                log.error("get_recent_conversation: failed after pool reset", exc_info=True)
                return []  # degrade gracefully — ZERO runs without history rather than crashing


async def save_fact(category: str, fact: str):
    """Save something ZERO learns about Aditya."""
    for attempt in range(2):
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO user_facts (category, fact) VALUES ($1, $2)",
                    category, fact
                )
            return
        except (asyncpg.PostgresConnectionError, OSError) as e:
            if attempt == 0:
                log.warning("save_fact: connection lost, resetting pool and retrying")
                await reset_pool()
            else:
                log.error("save_fact: failed after pool reset", exc_info=True)
                raise


async def get_all_facts() -> list:
    """Pull all known facts about Aditya for context."""
    for attempt in range(2):
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT category, fact FROM user_facts ORDER BY timestamp DESC"
                )
            return [{"category": r["category"], "fact": r["fact"]} for r in rows]
        except (asyncpg.PostgresConnectionError, OSError) as e:
            if attempt == 0:
                log.warning("get_all_facts: connection lost, resetting pool and retrying")
                await reset_pool()
            else:
                log.error("get_all_facts: failed after pool reset", exc_info=True)
                return []  # degrade gracefully — ZERO runs without facts rather than crashing


async def save_task(title: str, priority: str = "normal", notes: str = ""):
    """Log a task or goal."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO tasks (title, priority, notes) VALUES ($1, $2, $3)",
            title, priority, notes
        )


async def get_tasks(status: str = "pending") -> list:
    """Get tasks by status."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, priority, notes FROM tasks WHERE status = $1 ORDER BY created_at DESC",
            status
        )
    return [dict(r) for r in rows]


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None