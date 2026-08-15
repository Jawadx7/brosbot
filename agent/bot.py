# -*- coding: utf-8 -*-

"""
agent/bot.py  --  Step 10
Main bot process. Connects to the chat server (app/server) as a
participant named BOT_NAME, listens to every message, decides whether
to respond via trigger.py, builds context via prompt.py, generates a
reply via inference.py, and POSTs the reply back to the server's
HTTP endpoint.

Run (with the chat server, and Ollama, already running):
    python agent/bot.py

Requires:
    pip install python-socketio requests

Env vars:
    SERVER_URL                 default http://localhost:3000
    BOT_NAME                   default "BrosBot"
    INACTIVITY_CHECK_SECONDS   how often to poll check_inactivity, default 30
"""

import os
import time
import logging
import threading

import requests
import socketio

import trigger
import prompt
import inference


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bot")


SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:3000")
BOT_NAME = os.environ.get("BOT_NAME", "BrosBot")
INACTIVITY_CHECK_SECONDS = int(os.environ.get("INACTIVITY_CHECK_SECONDS", 30))

# Keep local history from growing unbounded during a long-running session.
# Only the last N messages ever matter (prompt.py only looks at the last
# 14 anyway), so this is just a memory-hygiene cap, not a functional limit.
MAX_LOCAL_HISTORY = 200


class BrosBotAgent:
    """
    Wraps a Socket.io client connection plus the decision/generation
    pipeline. One instance = one running bot participant.
    """

    def __init__(self):
        self.sio = socketio.Client(reconnection=True, reconnection_attempts=0)
        self.history = []
        self.history_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._register_handlers()

    # -----------------------------------------------------------------
    # Socket.io event handlers
    # -----------------------------------------------------------------

    def _register_handlers(self):

        @self.sio.event
        def connect():
            logger.info("Connected to server, joining as '%s'", BOT_NAME)
            self.sio.emit("user:join", {"name": BOT_NAME})

        @self.sio.event
        def disconnect():
            logger.warning("Disconnected from server.")

        @self.sio.on("history")
        def on_history(history):
            with self.history_lock:
                self.history = list(history)
            logger.info("Synced history on join: %d messages", len(history))

        @self.sio.on("user:message")
        def on_user_message(message):
            self._handle_incoming(message)

        @self.sio.on("bot:message")
        def on_bot_message(message):
            # This fires for the bot's OWN messages too (server broadcasts
            # every bot reply back to all connected clients, including us).
            # We only use it to keep local history in sync for accurate
            # cooldown tracking -- we never trigger a response from here,
            # which is what prevents the bot from ever replying to itself.
            self._append_history(message)

    # -----------------------------------------------------------------
    # History helpers
    # -----------------------------------------------------------------

    def _append_history(self, message):
        with self.history_lock:
            self.history.append(message)
            if len(self.history) > MAX_LOCAL_HISTORY:
                self.history = self.history[-MAX_LOCAL_HISTORY:]

    def _history_snapshot(self):
        with self.history_lock:
            return list(self.history)

    # -----------------------------------------------------------------
    # Core decision + reply flow
    # -----------------------------------------------------------------

    def _handle_incoming(self, message):
        # Evaluate the trigger against history BEFORE this message is
        # appended -- matches how trigger.py's should_respond() expects
        # to be called (message + prior context), and keeps cooldown
        # counting consistent with the module's own tests.
        history_before = self._history_snapshot()

        fire, mode = trigger.should_respond(message, history_before)

        self._append_history(message)

        if not fire:
            return

        logger.info("Triggering response (mode=%s) to: %r", mode, message.get("text"))
        self._generate_and_send()

    def _generate_and_send(self):
        history_snapshot = self._history_snapshot()
        context = prompt.build_context(history_snapshot)

        if not context:
            logger.warning("Empty context built, skipping generation.")
            return

        reply = inference.generate(context)

        if not reply:
            logger.info("Model produced no usable reply this time, staying silent.")
            return

        self._post_reply(reply)

    def _post_reply(self, text):
        try:
            resp = requests.post(
                "{}/api/bot-message".format(SERVER_URL),
                json={"text": text},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("Posted reply: %r", text)
        except requests.exceptions.RequestException as e:
            logger.error("Failed to post reply to server: %s", e)

    # -----------------------------------------------------------------
    # Inactivity polling loop -- runs on its own background thread.
    # Kept separate from should_respond's per-message path, same as
    # trigger.py's check_inactivity() was designed to be called
    # independently.
    # -----------------------------------------------------------------

    def _inactivity_loop(self):
        while not self._stop_event.wait(INACTIVITY_CHECK_SECONDS):
            history_snapshot = self._history_snapshot()

            fire, mode = trigger.check_inactivity(history_snapshot)
            if fire:
                logger.info("Triggering spontaneous message (silence timeout).")
                self._generate_and_send()

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self):
        if not inference.is_ollama_running():
            logger.error("Ollama is not ready. Fix that before starting the bot.")
            return

        inactivity_thread = threading.Thread(target=self._inactivity_loop, daemon=True)
        inactivity_thread.start()

        logger.info("Connecting to %s ...", SERVER_URL)
        self.sio.connect(SERVER_URL)

        try:
            self.sio.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        finally:
            self._stop_event.set()
            if self.sio.connected:
                self.sio.disconnect()


def main():
    agent = BrosBotAgent()
    agent.start()


if __name__ == "__main__":
    main()
