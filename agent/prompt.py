# -*- coding: utf-8 -*-

"""
Builds the context string sent to the model.

Takes the last N messages from conversation history and joins them
into plain newline-separated text -- the exact same shape as a
training window (see data/format.py: format_window()). No speaker
names, no timestamps, no system prompt, no instructions. Just raw
conversation flow, because that is the only format the model has
ever seen. Anything more "helpful" (labels, instructions, a persona
prompt) would push the input outside the training distribution and
degrade output quality.

Usage (from bot.py):
    from agent.prompt import build_context
    context = build_context(history)
    reply = generate(context)   # agent/inference.py
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Matches the training window size in data/format.py (window=14).
# Keeping inference context the same length as training windows keeps
# the model's input distribution as close to training as possible.
DEFAULT_CONTEXT_MESSAGES = 14


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def build_context(
    history      : List[Dict],
    window_size  : int = DEFAULT_CONTEXT_MESSAGES,
) -> str:
    """
    Build the raw text context to feed the model, from the last
    `window_size` messages in history.

    Args:
        history:     full conversation history, oldest first. Each item
                     needs a "text" field. Both human AND bot messages
                     are included and treated identically -- the model
                     was trained on undifferentiated group flow, and a
                     live conversation the bot is part of is exactly
                     that: the bot's own prior replies are just more
                     messages in the same collective voice.
        window_size: how many trailing messages to include. Defaults to
                     14 to match the training window size in format.py.

    Returns:
        A single string: messages joined by "\\n", always ending in
        exactly one trailing "\\n" so the model reads the context as
        "complete, generate the next message now" (same fix applied in
        inference.py -- kept here too as a second, independent layer so
        the empty-output bug can't reappear if this function is ever
        called from somewhere that skips inference.py's normalization).

        Returns an empty string if history is empty.
    """
    if not history:
        logger.warning("build_context called with empty history.")
        return ""

    window = history[-window_size:] if window_size > 0 else history

    lines = []
    for msg in window:
        text = msg.get("text", "").strip()
        if text:
            lines.append(text)

    if not lines:
        logger.warning("build_context: no usable text found in history window.")
        return ""

    context = "\n".join(lines)

    # Defensive trailing-newline enforcement (see docstring). Idempotent --
    # safe to call even if the caller (inference.py) also normalizes this.
    context = context.rstrip("\n") + "\n"

    return context


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------

def _test():
    logging.basicConfig(level=logging.INFO)

    history = [
        {"sender": "Kofi",    "text": "guys assignment due tomorrow", "is_bot": False},
        {"sender": "Ama",     "text": "yeah lecturer just posted it",  "is_bot": False},
        {"sender": "BrosBot", "text": "rip nobody's ready for that",   "is_bot": True},
        {"sender": "Kofi",    "text": "fr fr",                          "is_bot": False},
        {"sender": "",        "text": "   ",                            "is_bot": False},  # blank, should be skipped
    ]

    context = build_context(history)
    print("--- Context (window=14, only 4 usable messages exist) ---")
    print(repr(context))
    print()
    print(context, end="")
    print("--- end ---")

    print("\n--- Context with window_size=2 (last 2 only) ---")
    context_small = build_context(history, window_size=2)
    print(repr(context_small))

    print("\n--- Empty history ---")
    print(repr(build_context([])))


if __name__ == "__main__":
    _test()
