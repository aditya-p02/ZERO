# main.py
# ZERO — Entry point

import asyncio
import re
import webbrowser

from dotenv import load_dotenv

from agents.router import classify
from core.brain import extract_and_save_facts, think
from core.config import settings
from core.memory import get_all_facts, init_memory, save_fact, save_message
from core.voice_input import listen
from core.voice_output import speak_async
from hud.server import broadcast as _hud_broadcast
from hud.server import start_server as _hud_start_server

load_dotenv()

USER_NAME = settings.user_name

BANNER = """
╔══════════════════════════════════════════╗
║                                          ║
║         Z  E  R  O                       ║
║                                          ║
║     Your AI. Your brother. Always.       ║
║                                          ║
╚══════════════════════════════════════════╝
"""

EXIT_PHRASES = ["exit", "quit", "bye", "shutdown", "zero shutdown", "goodbye zero"]

def _is_exit(clean: str) -> bool:
    """
    Check if the cleaned input is an exit command.

    Both modes previously used different matching logic:
    - Text mode: exact match (clean in EXIT_PHRASES) — correct but too strict
    - Voice mode: substring match (phrase in clean) — caused "how do I exit vim"
      and "can you shutdown a process" to instantly kill ZERO

    This uses whole-word regex matching so "exit" only triggers when it
    appears as a standalone word, not inside "explain", "exit codes", etc.
    Multi-word phrases like "zero shutdown" are matched as exact full strings.
    Single words like "bye", "quit" require word boundaries so they don't
    fire inside longer words or sentences like "what does bye mean in TCP".
    """
    import re as _re
    for phrase in EXIT_PHRASES:
        if " " in phrase:
            # Multi-word: must match the whole cleaned string exactly
            if clean == phrase:
                return True
        else:
            # Single word: whole-word boundary match — won't fire mid-sentence
            if _re.search(rf'\b{_re.escape(phrase)}\b', clean) and len(clean.split()) <= 3:
                return True
    return False


async def hud_update(update: dict):
    try:
        await _hud_broadcast(update)
    except Exception:
        pass


async def start_hud():
    try:
        asyncio.create_task(_hud_start_server())
        await asyncio.sleep(2.0)
        webbrowser.get().open("http://localhost:8766/index.html", new=0, autoraise=True)
        print("[ZERO] HUD launched in browser.")
    except Exception as e:
        print(f"[ZERO] HUD unavailable: {e}")


async def handle_intent(intent: str, user_input: str, private: bool) -> str:
    """
    Dispatch to the right agent based on classified intent.
    Agents not yet built fall back to brain.py gracefully.

    Every path through this function ends with the exchange saved to
    conversation history — this is the ONLY place that saves now. Agents
    themselves (research, developer, automation, system, screen) don't
    save anything internally, so without this wrapper, anything routed
    to those agents would be invisible to future get_recent_conversation()
    calls — which is exactly what was happening before this fix.
    """
    response = await _dispatch_intent(intent, user_input, private)

    await save_message("user", user_input)
    await save_message("assistant", response)

    return response


async def _dispatch_intent(intent: str, user_input: str, private: bool) -> str:
    """Actual routing logic — pulled out of handle_intent so saving stays
    in exactly one place above, no matter how many branches get added here."""
    if intent == "research":
        try:
            from agents.research import research
            return await research(user_input)
        except ImportError:
            pass

    elif intent == "code":
        try:
            from agents.developer import develop
            return await develop(user_input)
        except ImportError:
            pass

    elif intent == "automation":
        try:
            from agents.automation import automate
            return await automate(user_input)
        except ImportError:
            pass

    elif intent == "system":
        try:
            from agents.system import handle_system
            return await handle_system(user_input)
        except ImportError:
            pass

    elif intent == "screen":
        try:
            from agents.screen import handle_screen
            return await handle_screen(user_input)
        except ImportError:
            pass

    elif intent == "memory":
        pass

    return await think(user_input, private=private)


async def main():
    print(BANNER)

    print("Mode:")
    print("  [1] Text  — type to ZERO")
    print("  [2] Voice — speak to ZERO")
    print()

    mode = input("Choose (1 or 2): ").strip()
    voice_mode = mode == "2"

    print()
    print("[ZERO] Initializing memory...")
    await init_memory()

    await start_hud()

    if voice_mode:
        print("[ZERO] Connecting to Groq voice API...")

    greeting = f"Hey {USER_NAME}, what's up?"
    print(f"\nZERO: {greeting}\n")

    await hud_update({"status": "speaking", "zero_response": greeting, "new_zero_msg": greeting})

    if voice_mode:
        await speak_async(greeting)
        await hud_update({"status": "online"})

    private = False
    _turn_count = 0  # tracks turns so the router knows if prior conversation exists

    while True:
        try:
            if voice_mode:
                await hud_update({"status": "listening", "user_input": ""})
                user_input = listen(private=private)

                if not user_input:
                    continue

                clean = re.sub(r'[^\w\s]', '', user_input.lower().strip())

                if clean == "go private":
                    private = True
                    ack = "Private mode on. Staying local."
                    print(f"\n[ZERO] 🔒 {ack}")
                    await hud_update({"status": "speaking", "private": True})
                    await speak_async(ack)
                    await hud_update({"status": "online", "private": True})
                    continue

                if clean in ["go cloud", "cloud mode", "go public", "exit private"]:
                    private = False
                    ack = "Back on cloud. Groq is live."
                    print(f"\n[ZERO] ☁️ {ack}")
                    await hud_update({"status": "speaking", "private": False})
                    await speak_async(ack)
                    await hud_update({"status": "online", "private": False})
                    continue

                if _is_exit(clean):
                    farewell = "Later. I'll be here when you need me."
                    print(f"\nZERO: {farewell}")
                    await hud_update({"status": "speaking", "zero_response": farewell})
                    await speak_async(farewell)
                    break

                print(f"{'🔒 ' if private else ''}{USER_NAME}: {user_input}")
                await hud_update({
                    "status": "thinking",
                    "user_input": user_input,
                    "new_user_msg": user_input,
                })

            else:
                user_input = input(f"{USER_NAME}: ").strip()
                if not user_input:
                    continue

                clean = re.sub(r'[^\w\s]', '', user_input.lower().strip())

                if user_input.lower() == "/private":
                    private = True
                    print("[ZERO] 🔒 Private mode on — staying local.\n")
                    await hud_update({"private": True})
                    continue

                if user_input.lower() == "/cloud":
                    private = False
                    print("[ZERO] ☁️ Back on cloud — Groq is live.\n")
                    await hud_update({"private": False})
                    continue

                if _is_exit(clean):
                    farewell = "Later. I'll be here when you need me."
                    print(f"\nZERO: {farewell}")
                    break

                await hud_update({
                    "status": "thinking",
                    "user_input": user_input,
                    "new_user_msg": user_input,
                })

            # Manual remember command
            if user_input.lower().startswith("/remember "):
                parts = user_input[10:].strip().split(":", 1)
                if len(parts) == 2:
                    category, fact = parts[0].strip(), parts[1].strip()
                    await save_fact(category, fact)
                    msg = f"Got it. Filed under '{category}'."
                    print(f"\nZERO: {msg}\n")
                    facts = await get_all_facts()
                    await hud_update({"facts": facts})
                    if voice_mode:
                        await hud_update({"status": "speaking"})
                        await speak_async(msg)
                        await hud_update({"status": "online"})
                else:
                    print("[ZERO] Format: /remember category: fact\n")
                continue

            # Classify intent — tell the router whether prior conversation
            # exists so _is_followup() doesn't fire on a fresh session
            intent = classify(user_input, has_prior_conversation=_turn_count > 0)
            print(f"[ZERO] Intent → {intent}")

            # Think / dispatch
            print("\nZERO: ", end="", flush=True)
            await hud_update({"status": "thinking"})
            response = await handle_intent(intent, user_input, private)
            _turn_count += 1
            print(response)
            print()

            # Auto-save any facts from this exchange
            await extract_and_save_facts(user_input, response)

            facts = await get_all_facts()
            await hud_update({
                "status": "speaking",
                "zero_response": response,
                "new_zero_msg": response,
                "facts": facts,
            })

            if voice_mode:
                await speak_async(response)

            await hud_update({"status": "listening" if voice_mode else "online"})

        except KeyboardInterrupt:
            farewell = "Caught that. Later, Aditya."
            print(f"\n\nZERO: {farewell}")
            if voice_mode:
                await speak_async(farewell)
            break


if __name__ == "__main__":
    asyncio.run(main())
