"""
agent/inference.py  --  Step 5
Exposes a single generate(context) -> str function that the bot agent calls.

Calls the locally running Ollama server (localhost:11434).
Ollama must be running before the bot agent starts:
    ollama serve

The model must be registered with Ollama:
    ollama create brosbot -f Modelfile

Usage (from bot agent):
    from agent.inference import generate
    reply = generate("bro did you see that\nlmaooo no way")
    # -> "what happened 😭"
"""

import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "brosbot"
TIMEOUT_SECS = 30

# Generation defaults -- can be overridden per call
DEFAULT_PARAMS = {
    "temperature"    : 0.85,
    "top_p"          : 0.92,
    "repeat_penalty" : 1.1,
    "num_predict"    : 80,     # max tokens to generate
    "stop"           : ["\n"], # stop at first newline -> one message only
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_ollama(payload: dict) -> str:
    """
    POST to Ollama and return the raw response text (unstripped of lines,
    just whitespace-stripped). Raises requests exceptions on failure --
    caller is responsible for catching them.
    """
    response = requests.post(
        OLLAMA_URL,
        json    = payload,
        timeout = TIMEOUT_SECS,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _first_line(result: str) -> Optional[str]:
    """
    Take only the first non-empty line from a raw Ollama response.
    (Safety net in case the stop token didn't fire and multiple lines
    came back.) Returns None if there's nothing usable.
    """
    lines = [line.strip() for line in result.split("\n") if line.strip()]
    return lines[0] if lines else None


# ---------------------------------------------------------------------------
# Core generate function
# ---------------------------------------------------------------------------

def generate(
    context      : str,
    temperature  : Optional[float] = None,
    max_tokens   : Optional[int]   = None,
) -> Optional[str]:
    """
    Generate a single message continuation given conversation context.

    Args:
        context:     The last N messages from the chat, joined by newlines.
                     The model continues from here and stops at the next newline.
        temperature: Override the default sampling temperature (0.0 - 2.0).
                     Higher = more random. Default: 0.85
        max_tokens:  Override max tokens to generate. Default: 80

    Returns:
        A single message string (one line), or None if generation failed.

    Example:
        context = "whose place are we going to\nbro I can't host tonight"
        reply   = generate(context)
        # -> "same here tbh 💀"
    """
    params = {**DEFAULT_PARAMS}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["num_predict"] = max_tokens

    # Critical: training windows are messages joined by "\n". A prompt that
    # does NOT end in "\n" reads to the model as an unfinished line, not a
    # cue to start a new message -- which is why short prompts like
    # "BRO I SWEAR" were frequently returning empty output (the model's
    # top choice was often just "\n", i.e. "this message is done").
    # Always end the context with a single newline so the model reads it
    # as "context complete, generate the next message now."
    normalized_context = context.rstrip("\n") + "\n"

    payload = {
        "model"   : MODEL_NAME,
        "prompt"  : normalized_context,
        "stream"  : False,
        "options" : params,
    }

    try:
        result = _call_ollama(payload)
        reply  = _first_line(result)

        if reply:
            return reply

        # Bounded retry: sampling is stochastic, so an empty first draw
        # doesn't mean the model has nothing to say. One retry with a
        # slightly higher temperature gives it another shot before we
        # give up and return None.
        logger.info("Empty generation on first attempt, retrying once.")
        retry_params = {**params, "temperature": min(params["temperature"] + 0.15, 1.2)}
        retry_payload = {**payload, "options": retry_params}

        result = _call_ollama(retry_payload)
        reply  = _first_line(result)

        return reply  # None if still empty -- caller treats as "bot chose not to reply"

    except requests.exceptions.ConnectionError:
        logger.error(
            "Cannot reach Ollama at %s -- is 'ollama serve' running?", OLLAMA_URL
        )
        return None

    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out after %ds", TIMEOUT_SECS)
        return None

    except requests.exceptions.RequestException as e:
        logger.error("Ollama request failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def is_ollama_running() -> bool:
    """
    Check if the Ollama server is reachable and the model is loaded.
    Call this at bot agent startup to fail fast with a clear error.
    """
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout = 5,
        )
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]

        if not any(MODEL_NAME in m for m in models):
            logger.error(
                "Ollama is running but model '%s' is not found.\n"
                "Register it with: ollama create brosbot -f Modelfile",
                MODEL_NAME,
            )
            return False

        return True

    except requests.exceptions.ConnectionError:
        logger.error(
            "Ollama is not running.\n"
            "Start it with: ollama serve"
        )
        return False

    except Exception as e:
        logger.error("Ollama health check failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def _test():
    print("Checking Ollama...")

    if not is_ollama_running():
        print("\nFix the issue above then re-run.")
        return

    print("Ollama is running. Model '{}' found.\n".format(MODEL_NAME))

    test_contexts = [
        "bro did you see that",
        "lmaooo no way",
        "whose place are we doing it at",
        "BRO I SWEAR",
        "guys",
    ]

    for ctx in test_contexts:
        result = generate(ctx)
        print("PROMPT : {}".format(ctx))
        print("REPLY  : {}\n".format(result if result else "[no output]"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test()
