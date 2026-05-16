import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Regex
# Matches the opening line of any WhatsApp message across regional formats:
#
#   "12/05/2025, 14:32 - Name: text"          <- most international (Ghana)
#   "[12/05/2025, 14:32:45] Name: text"       <- bracket variant with seconds
#   "5/12/25, 2:32 PM - Name: text"           <- US AM/PM format
#   "12.05.2025, 14:32 - Name: text"          <- dot-separated dates

MSG_PATTERN = re.compile(
    r"^\[?"  # optional [
    r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})"  # date
    r",?\s+"  # comma + space
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?)"  # time (HH:MM or HH:MM:SS, optional AM/PM)
    r"\]?\s*[-\u2013]\s*"  # ] or - separator
    r"([^:]+):\s"  # sender
    r"(.+)$"  # message text
)

# Content to drop
DROP_IF_CONTAINS = [
    "<media omitted>",
    "<Media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "contact card omitted",
    "<View once voice message omitted>",
    "this message was deleted",
    "This message was deleted",
    "<This message was edited>",
    "you deleted this message",
    "missed voice call",
    "missed video call",
    "missed group video call",
    "messages and calls are end-to-end encrypted",
    "Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them. *Learn more*",
    "security code changed",
    "changed the group",
    "changed the subject",
    "changed the group description",
    "changed the group name",
    "created group",
    "added you",
    "you were added",
    "You were added",
    "joined using this group",
    "joined using a group link",
    "pinned a message",
]

MIN_WORDS = 2


# Core parsing


def parse_export(filepath: str) -> list:
    """
    Read the raw WhatsApp .txt export and return a list of raw message dicts.
    Multi-line messages are merged into a single text field.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"\n  File not found: {filepath}"
            f"\n  Drop your WhatsApp export into data/raw/ and name it chat.txt"
            f"\n  (In WhatsApp: open the group -> menu -> More -> Export chat -> Without media)"
        )

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.splitlines()

    messages = []
    current = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = MSG_PATTERN.match(line)

        if match:
            if current is not None:
                messages.append(current)

            date_str, time_str, sender, text = match.groups()
            current = {
                "sender": sender.strip(),
                "timestamp": f"{date_str} {time_str.strip()}",
                "text": text.strip(),
            }
        else:
            # Continuation line - append to current message
            if current is not None:
                current["text"] += f" {line}"

    if current is not None:
        messages.append(current)

    return messages


def clean_messages(messages: list, min_words: int = MIN_WORDS) -> list:
    """
    Remove noise: system events, media placeholders, very short texts.
    Also normalises internal whitespace.
    """
    clean = []

    for msg in messages:
        text_lower = msg["text"].lower()

        if any(pattern in text_lower for pattern in DROP_IF_CONTAINS):
            continue

        words = msg["text"].split()
        if len(words) < min_words:
            continue

        clean.append({**msg, "text": " ".join(words)})

    return clean


# Stats


def print_stats(raw: list, clean: list) -> None:
    dropped = len(raw) - len(clean)
    print(f"\n{'#' * 46}")
    print(f"  Raw messages parsed :  {len(raw):>6,}")
    print(f"  After cleaning      :  {len(clean):>6,}")
    print(
        f"  Dropped (noise)     :  {dropped:>6,}  ({dropped / max(len(raw), 1) * 100:.1f}%)"
    )

    if not clean:
        print("\n  WARNING: No messages survived cleaning.")
        print("  Check that your export matches the expected format:")
        print('  "12/05/2025, 14:32 - Name: message text"')
        print(f"{'#' * 46}\n")
        return

    counts = Counter(m["sender"] for m in clean)
    # max_count = max(counts.values())
    # bar_scale = max(1, max_count // 25)

    print("\n  Messages per member:")
    for sender, count in counts.most_common():
        # bar = "&#9608;" * (count // bar_scale)
        # print(f"    {sender:<22} {count:>5,}  {bar}")
        print(f"    {sender:<22} {count:>5,}")

    print(f"\n  Date range: {clean[0]['timestamp']}  ->  {clean[-1]['timestamp']}")
    print(f"{'#' * 46}\n")


# Entry point


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a WhatsApp group chat export into structured JSON."
    )
    parser.add_argument(
        "--input", default="data/raw/chat.txt", help="Path to the WhatsApp .txt export"
    )
    parser.add_argument(
        "--output",
        default="data/processed/messages.json",
        help="Output path for the JSON file",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=2,
        help="Drop messages shorter than N words (default: 2)",
    )
    parser.add_argument(
        "--show-senders",
        action="store_true",
        help="Print the full list of unique senders found",
    )
    args = parser.parse_args()

    print(f"Parsing:  {args.input}")

    raw = parse_export(args.input)
    clean = clean_messages(raw, min_words=args.min_words)

    print_stats(raw, clean)

    if args.show_senders:
        senders = sorted({m["sender"] for m in clean})
        print("  Unique senders found:")
        for s in senders:
            print(f"    . {s}")
        print()

    if not clean:
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"OK  Saved {len(clean):,} messages -> {args.output}")
    print("    Next: python data/format.py\n")


if __name__ == "__main__":
    main()
