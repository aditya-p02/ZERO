# core/clients.py
# Shared client instances for external services.
#
# Previously, 7 different modules each ran `Groq(api_key=os.getenv(...))` at
# import time, constructing 7 separate client objects. Building it once here
# instead means: one client to configure, and tests can inject a fake client
# instead of monkeypatching an import-time global in every module that used
# to build its own (see Phase 6).

from groq import Groq

from core.config import settings

groq_client = Groq(api_key=settings.groq_api_key)