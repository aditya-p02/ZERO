# core/logger.py
# ZERO's centralised logger — import this everywhere instead of bare print()
#
# Usage:
#   from core.logger import log
#   log.info("Searching → {query}")
#   log.warning("Groq returned unexpected format")
#   log.error("Groq classify failed", exc_info=True)   # <-- includes full traceback
#
# Set LOG_LEVEL in .env to control verbosity:
#   LOG_LEVEL=DEBUG    → everything (noisy, for active debugging)
#   LOG_LEVEL=INFO     → normal operation (default)
#   LOG_LEVEL=WARNING  → only problems
#   LOG_LEVEL=ERROR    → only failures
#
# Log output goes to:
#   - Console (always)
#   - zero.log in the project root (always, survives restarts, great for post-mortem)

import logging
import os
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

load_dotenv()

_LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_STR, logging.INFO)

_LOG_FORMAT = "[ZERO] %(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def _build_logger(name: str = "zero") -> logging.Logger:
    logger = logging.getLogger(name)

    # Don't add handlers repeatedly if already configured (e.g. on re-import)
    if logger.handlers:
        return logger

    logger.setLevel(_LOG_LEVEL)

    # ── Console handler ─────────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(_LOG_LEVEL)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console)

    # ── File handler — rotates at 2MB, keeps 3 backups ─────────────────────
    # zero.log sits at project root. If the directory isn't writable, we
    # skip the file handler silently rather than crashing startup.
    try:
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "zero.log"
        )
        file_handler = RotatingFileHandler(
            log_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(_LOG_LEVEL)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)
    except Exception:
        pass  # log file unavailable — console is enough

    return logger


# The single shared instance every module imports
log = _build_logger("zero")