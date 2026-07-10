# hud/server.py
# ZERO HUD — HTTP + WebSocket server

import asyncio
import json
import websockets
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import os

# ── Shared state ───────────────────────────────────────────────────────────────
_clients: set = set()
_counter: int = 0

state = {
    "status": "online",
    "mode": "cloud",
    "private": False,
    "user_input": "",
    "zero_response": "",
    "facts": [],
    "uptime_start": datetime.now().isoformat(),
    "messages": [],
}
# ──────────────────────────────────────────────────────────────────────────────


def _next_id() -> int:
    global _counter
    _counter += 1
    return _counter


async def _send_all():
    dead = set()
    msg = json.dumps(state)
    for ws in _clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


async def broadcast(update: dict):
    if "new_user_msg" in update:
        state["messages"].append({"id": _next_id(), "role": "user", "text": update.pop("new_user_msg")})
    if "new_zero_msg" in update:
        state["messages"].append({"id": _next_id(), "role": "zero", "text": update.pop("new_zero_msg")})
    state.update(update)
    await _send_all()


async def _ws_handler(websocket):
    _clients.add(websocket)
    try:
        await websocket.send(json.dumps(state))
        async for _ in websocket:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _clients.discard(websocket)


def _http_thread():
    hud_dir = os.path.dirname(os.path.abspath(__file__))

    class Silent(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=hud_dir, **kw)
        def log_message(self, *a):
            pass

    HTTPServer(("localhost", 8766), Silent).serve_forever()


async def start_server():
    threading.Thread(target=_http_thread, daemon=True).start()

    # Use serve_forever() instead of asyncio.Future() — works on Windows Proactor
    async with websockets.serve(_ws_handler, "localhost", 8765) as server:
        print("[HUD] WebSocket server ready on ws://localhost:8765")
        await server.serve_forever()