# -*- coding: utf-8 -*-

"""
agent/trigger.py  --  Step 6
Decides WHEN the bot should speak.

Exposes:
    should_respond(message, history) -> (bool, mode)
    check_inactivity(history)        -> (bool, mode)

should_respond() is called by bot.py every time a new human message
arrives. check_inactivity() is called separately, on a timer loop
inside bot.py, to catch silence-based spontaneous triggers -- it is
NOT called from inside should_respond(), so it can be tested and
tuned independently of the per-message path.

Expected message / history shape (matches the Socket.io event
payloads bot.py will receive from server.ts in Step 8):

    {
        "sender":    "Kofi",          # str, human sender name or "BrosBot"
        "text":      "bro did you see that",
        "is_bot":    False,           # True only for BrosBot's own messages
        "timestamp": 1737000000.0,    # unix seconds
    }

history is a list of these dicts, oldest first, e.g. the last 100
messages server.ts keeps in memory.
"""

import os
import re
import random
import logging
import time
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config -- env vars match .env.example from scaffold.py
# ---------------------------------------------------------------------------

BOT_NAME = os.environ.get("BOT_NAME", "BrosBot")

TRIGGER_SPONTANEOUS_PROB    = float(os.environ.get("TRIGGER_SPONTANEOUS_PROB", 0.20))
TRIGGER_KEYWORD_PROB        = float(os.environ.get("TRIGGER_KEYWORD_PROB", 0.60))
TRIGGER_INACTIVITY_SECONDS  = int(os.environ.get("TRIGGER_INACTIVITY_SECONDS", 180))
BOT_COOLDOWN_MESSAGES       = int(os.environ.get("BOT_COOLDOWN_MESSAGES", 2))

# Tag detection -- matches "@BrosBot", "@brosbot", "@bros bot", case-insensitive
TAG_PATTERN = re.compile(
    r"@\s*" + re.escape(BOT_NAME).replace(r"\ ", r"\s*"),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Keyword groups
# Sample set for an anime + "boys boys" friend group that is also a
# university class group chat. Tune freely -- these are starting points,
# not exhaustive. Keep everything lowercase; matching is case-insensitive.
# ---------------------------------------------------------------------------

KEYWORDS_ANIME = {
    "anime", "manga", "weeb", "waifu", "husbando", "isekai", "arc",
    "filler", "sub", "dub", "op", "ed", "senpai", "sensei", "shonen",
    "shounen", "seinen", "naruto", "one piece", "luffy", "sasuke",
    "goku", "dragon ball", "jujutsu kaisen", "jjk", "gojo", "demon slayer",
    "attack on titan", "aot", "eren", "chainsaw man", "spy x family",
    "solo leveling", "my hero academia", "mha", "bleach", "hunter x hunter",
    "hxh", "tokyo revengers", "crunchyroll", "episode", "season",
    "power scaling", "power scale", "mc", "protagonist", "villain arc",
}

KEYWORDS_BOYS_CHAT = {
    "gist", "yawa", "shatap", "shaddup", "charley", "chale", "guy",
    "guys", "bros", "bro", "cuh", "walahi", "trust me bro", "banter",
    "roast", "opp", "opps", "beef", "smoke", "cap", "no cap", "based",
    "mid", "l take", "w take", "ratio", "girlfriend", "girl", "shoot your shot",
    "situationship", "body count", "hookup", "party", "hangout", "spot",
    "place", "crib", "hostel", "room", "vibe", "vibes", "grind",
}

KEYWORDS_UNIVERSITY = {
    "exam", "exams", "quiz", "midsem", "endsem", "assignment", "assign",
    "deadline", "lecturer", "lecture", "class", "classes", "lab",
    "practical", "project", "group work", "presentation", "slides",
    "handout", "sandwich", "resit", "gpa", "cgpa", "semester", "trimester",
    "registration", "fees", "hall", "hostel", "campus", "uenr", "csi",
    "invigilator", "tut", "tutorial", "syllabus", "coursework", "fyp",
    "supervisor", "defense", "viva",
}

ALL_KEYWORDS = KEYWORDS_ANIME | KEYWORDS_BOYS_CHAT | KEYWORDS_UNIVERSITY

# Sort longest-first so multi-word phrases (e.g. "no cap") are checked
# before their substrings could cause false negatives in the regex build.
_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ALL_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Cooldown helper
# ---------------------------------------------------------------------------

def _messages_since_last_bot_reply(history: List[Dict]) -> int:
    """
    Count how many human messages have arrived since BrosBot last spoke.
    Returns len(history) if the bot has never spoken (no cooldown active).
    """
    count = 0
    for msg in reversed(history):
        if msg.get("is_bot"):
            return count
        count += 1
    return count


def _in_cooldown(history: List[Dict]) -> bool:
    """True if the bot spoke too recently to fire again (tag always overrides this)."""
    return _messages_since_last_bot_reply(history) < BOT_COOLDOWN_MESSAGES


# ---------------------------------------------------------------------------
# Tag / keyword detection
# ---------------------------------------------------------------------------

def _is_tagged(text: str) -> bool:
    return bool(TAG_PATTERN.search(text))


def _matches_keyword(text: str) -> bool:
    return bool(_KEYWORD_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Main entry point -- called per incoming human message
# ---------------------------------------------------------------------------

def should_respond(message: Dict, history: List[Dict]) -> Tuple[bool, Optional[str]]:
    """
    Decide whether BrosBot should respond to this message.

    Args:
        message: the incoming message dict (see module docstring for shape)
        history: prior messages, oldest first, used for cooldown tracking

    Returns:
        (should_respond: bool, mode: str or None)
        mode is one of "tagged", "keyword", "spontaneous", or None.
    """
    text = message.get("text", "")

    if message.get("is_bot"):
        # Never trigger off the bot's own messages
        return False, None

    # -- Tag always wins, and bypasses cooldown entirely.
    # If someone directly calls the bot, it should answer regardless of
    # how recently it last spoke.
    if _is_tagged(text):
        return True, "tagged"

    # -- Everything else respects cooldown
    if _in_cooldown(history):
        return False, None

    # -- Keyword mode: topic words -> probabilistic response
    if _matches_keyword(text):
        if random.random() < TRIGGER_KEYWORD_PROB:
            return True, "keyword"
        return False, None

    # -- Spontaneous mode: small chance to chime in on any message,
    # even with no keyword match, so the bot doesn't feel purely reactive.
    if random.random() < TRIGGER_SPONTANEOUS_PROB:
        return True, "spontaneous"

    return False, None


# ---------------------------------------------------------------------------
# Inactivity-based trigger -- called on a timer loop by bot.py,
# NOT from should_respond(). Lets bot.py poll this on whatever cadence
# makes sense (e.g. every 30s) without needing a thread inside this module.
# ---------------------------------------------------------------------------

def check_inactivity(history: List[Dict], now: Optional[float] = None) -> Tuple[bool, Optional[str]]:
    """
    Check whether the chat has gone quiet long enough for the bot to
    initiate a new message on its own.

    Args:
        history: prior messages, oldest first. Needs "timestamp" on each.
        now: unix timestamp to compare against (defaults to time.time()).
             Pass this explicitly in tests for determinism.

    Returns:
        (should_respond: bool, mode: str or None)
        mode is "spontaneous" if triggered, else None.
    """
    if not history:
        return False, None

    if _in_cooldown(history):
        return False, None

    now = now if now is not None else time.time()
    last_timestamp = history[-1].get("timestamp")

    if last_timestamp is None:
        logger.warning("check_inactivity: last message has no timestamp, skipping.")
        return False, None

    silence_seconds = now - last_timestamp

    if silence_seconds >= TRIGGER_INACTIVITY_SECONDS:
        return True, "spontaneous"

    return False, None


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------

def _test():
    logging.basicConfig(level=logging.INFO)

    history = [
        {"sender": "Kofi", "text": "guys assignment due tomorrow", "is_bot": False, "timestamp": time.time() - 300},
        {"sender": "Ama",  "text": "yeah lecturer just posted it", "is_bot": False, "timestamp": time.time() - 250},
        {"sender": BOT_NAME, "text": "yoo that's rough", "is_bot": True, "timestamp": time.time() - 200},
    ]

    test_messages = [
        "@BrosBot what do you think",
        "bro this jjk episode was insane",
        "did you finish the fyp draft",
        "random unrelated text with no signal",
    ]

    print("Cooldown active (bot spoke {} messages ago): {}".format(
        _messages_since_last_bot_reply(history), _in_cooldown(history)))
    print()

    for text in test_messages:
        msg = {"sender": "Kofi", "text": text, "is_bot": False, "timestamp": time.time()}
        fire, mode = should_respond(msg, history)
        print("MSG : {!r}".format(text))
        print("  -> respond={}, mode={}\n".format(fire, mode))

    print("Inactivity check (silent for 5 min, cooldown clear):")
    old_history = history[:-1]  # drop the recent bot message so cooldown is clear
    old_history[-1]["timestamp"] = time.time() - 400
    fire, mode = check_inactivity(old_history)
    print("  -> respond={}, mode={}".format(fire, mode))


if __name__ == "__main__":
    _test()
