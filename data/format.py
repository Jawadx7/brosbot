"""
Convert cleaned messages into sliding-window training sequences.

Reads:   data/processed/messages.json
Writes:  data/processed/train.txt

Each training example is a plain-text block of consecutive messages
joined by newlines. No speaker names -- collective group voice.
Examples are separated by <|endoftext|> so the model knows where
one conversation window ends and the next begins.

Usage:
    python data/format.py
    python data/format.py --window 14 --step 7 --session-gap 8
    python data/format.py --preview 3
"""

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Timestamp parsing
# Handles the varied formats parse.py produces:
#   "12/05/2025 14:32"
#   "12/05/2025 14:32:45"
#   "5/12/25 2:32 PM"
#   "12.05.2025 14:32"
# ---------------------------------------------------------------------------

TIMESTAMP_PATTERNS = [
    "%m/%d/%y %I:%M %p",  # 1/12/26 6:03 AM
    "%m/%d/%y %I:%M:%S %p",  # 1/12/26 6:03:45 AM
    "%d/%m/%y %I:%M %p",  # 12/1/26 6:03 AM
    "%d/%m/%Y %I:%M %p",  # 12/01/2026 6:03 AM
    "%m/%d/%Y %I:%M %p",  # 01/12/2026 6:03 AM
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d/%m/%y %H:%M",
    "%m/%d/%y %H:%M",
    "%d.%m.%Y %H:%M",
    "%d-%m-%Y %H:%M",
]


def parse_timestamp(ts_str):
    # Collapse all whitespace first (prevents double-space with AM/PM)
    ts_str = " ".join(ts_str.strip().split())

    # Add space before AM/PM only if one isn't already there
    ts_str = re.sub(r"(?<!\s)([AaPp][Mm])$", r" \1", ts_str)

    for fmt in TIMESTAMP_PATTERNS:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


# Session splitting
# A "session" is a run of messages with no gap larger than session_gap_hours.
# Never create training windows that cross a session boundary.


def split_into_sessions(messages, session_gap_hours=8):
    """
    Split the message list into sessions based on time gaps.
    Returns a list of lists (each inner list is one session).
    Falls back to a single session if timestamps can't be parsed.
    """
    if not messages:
        return []

    sessions = []
    current_session = [messages[0]]
    gap = timedelta(hours=session_gap_hours)
    parse_failures = 0

    for i in range(1, len(messages)):
        prev_dt = parse_timestamp(messages[i - 1]["timestamp"])
        curr_dt = parse_timestamp(messages[i]["timestamp"])

        if prev_dt is None or curr_dt is None:
            parse_failures += 1
            # Can't compute gap -- keep messages together
            current_session.append(messages[i])
            continue

        if curr_dt - prev_dt > gap:
            if len(current_session) > 0:
                sessions.append(current_session)
            current_session = [messages[i]]
        else:
            current_session.append(messages[i])

    if current_session:
        sessions.append(current_session)

    if parse_failures > 0:
        print(
            "  Note: {} timestamps could not be parsed for gap detection.".format(
                parse_failures
            )
        )

    return sessions


# Sliding windows


def sliding_windows(session, window_size, step):
    """
    Generate overlapping windows of `window_size` messages,
    advancing by `step` messages each time.

    session:     list of message dicts
    window_size: number of messages per window
    step:        how many messages to advance per window (step < window_size = overlap)
    """
    windows = []
    i = 0
    while i + window_size <= len(session):
        windows.append(session[i : i + window_size])
        i += step

    # Capture a final window from the tail if there are leftover messages
    # (only if the tail is at least half a window -- avoids tiny fragments)
    tail_start = i
    tail = session[tail_start:]
    if len(tail) >= window_size // 2:
        windows.append(tail)

    return windows


# Format a window into a training text block


def format_window(window):
    """
    Join the messages in a window as plain newline-separated text.
    No speaker names. No timestamps. Just the raw text -- collective voice.
    """
    return "\n".join(m["text"] for m in window)


# Main pipeline


def build_training_data(
    messages: list, window_size: int = 14, step: int = 7, session_gap_hours: int = 8
):
    """
    Full pipeline: split sessions -> slide windows -> format text blocks.
    Returns a list of training text strings.
    """
    sessions = split_into_sessions(messages, session_gap_hours)

    examples = []
    session_stats = []

    for session in sessions:
        if len(session) < 3:
            # Too short to be meaningful
            continue

        windows = sliding_windows(session, window_size, step)
        session_stats.append(
            {
                "messages": len(session),
                "windows": len(windows),
            }
        )

        for w in windows:
            text = format_window(w)
            if text.strip():
                examples.append(text)

    return examples, sessions, session_stats


# Stats


def print_stats(messages, examples, sessions, session_stats, window_size, step):
    total_windows = sum(s["windows"] for s in session_stats)
    total_chars = sum(len(e) for e in examples)
    avg_tokens = int(total_chars / 4)  # rough approximation: 1 token ~ 4 chars

    print("\n" + "-" * 50)
    print("  Input messages      :  {:>6,}".format(len(messages)))
    print(
        "  Sessions found      :  {:>6,}  (gaps > {}h split sessions)".format(
            len(sessions), 8
        )
    )
    print(
        "  Training examples   :  {:>6,}  (windows size={}, step={})".format(
            total_windows, window_size, step
        )
    )
    print("  Approx token count  :  {:>6,}  (rough estimate)".format(avg_tokens))
    print("  Avg tokens/example  :  {:>6,}".format(avg_tokens // max(total_windows, 1)))

    if session_stats:
        largest = max(session_stats, key=lambda s: s["messages"])
        print(
            "\n  Largest session     :  {:>6,} messages  ->  {} windows".format(
                largest["messages"], largest["windows"]
            )
        )

    print("-" * 50 + "\n")


def preview_examples(examples, n=3):
    """Print the first N examples so you can visually verify the format."""
    print(
        "\n--- Preview ({} of {} examples) ---\n".format(
            min(n, len(examples)), len(examples)
        )
    )
    for i, ex in enumerate(examples[:n]):
        lines = ex.split("\n")
        print("Example {}  ({} messages):".format(i + 1, len(lines)))
        print("  " + "\n  ".join(lines))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Format cleaned messages into sliding-window training sequences."
    )
    parser.add_argument(
        "--input",
        default="data/processed/messages.json",
        help="Path to cleaned messages JSON (default: data/processed/messages.json)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/train.txt",
        help="Output path for training text (default: data/processed/train.txt)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=14,
        help="Number of messages per training window (default: 14)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=7,
        help="Sliding step between windows -- lower = more overlap (default: 7)",
    )
    parser.add_argument(
        "--session-gap",
        type=int,
        default=8,
        help="Hours of silence that starts a new session (default: 8)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="Print N example windows after formatting (default: 0)",
    )
    args = parser.parse_args()

    # -- Load messages
    input_path = Path(args.input)
    if not input_path.exists():
        print("ERROR: {} not found. Run parse.py first.".format(args.input))
        return

    messages = json.loads(input_path.read_text(encoding="utf-8"))
    print("Loaded {:,} messages from {}".format(len(messages), args.input))

    # -- Validate step vs window
    if args.step >= args.window:
        print(
            "WARNING: step ({}) >= window ({}). No overlap between windows.".format(
                args.step, args.window
            )
        )
        print("         Consider a step of about half the window size for overlap.")

    # Build training data
    examples, sessions, session_stats = build_training_data(
        messages,
        window_size=args.window,
        step=args.step,
        session_gap_hours=args.session_gap,
    )

    if not examples:
        print("No training examples generated. Check your messages.json.")
        return

    # Print stats
    print_stats(messages, examples, sessions, session_stats, args.window, args.step)

    # Preview
    if args.preview > 0:
        preview_examples(examples, n=args.preview)

    # Write train.txt
    # Each example is separated by <|endoftext|> on its own line.
    # This tells the model that windows are independent documents
    # it should not learn to connect the end of one window to the start of the next.
    separator = "\n<|endoftext|>\n"
    output_text = separator.join(examples) + "\n"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output_text, encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(
        "Done. Saved {:,} training examples  ->  {}  ({:.1f} KB)".format(
            len(examples), args.output, size_kb
        )
    )
    print("     Next: upload train.txt to Google Colab for training\n")


if __name__ == "__main__":
    main()
