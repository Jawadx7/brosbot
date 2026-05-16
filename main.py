import argparse
import json
import re
from pathlib import Path

MATCH_PATTERN = re.compile(
    r"([^:]+):\s"
    r"(.+)$"
)

DROP_IF_CONTAINS = ["shit"]


def parse_text(filepath: str) -> list:
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found at {path}")

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.splitlines()

    messages = []
    current = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = MATCH_PATTERN.match(line)

        if match:
            if current is not None:
                messages.append(current)

            sender, text = match.groups()

            current = {"sender": sender.strip(), "text": text.strip()}
        else:
            if current is not None:
                current["text"] += f" {line}"

    if current is not None:
        messages.append(current)

    return messages


def clean_raw(messages: list, min_words: int = 2) -> list:
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


def print_stats(raw: list, clean: list) -> None:
    dropped = len(raw) - len(clean)
    print(f"Raw messages parsed: {len(raw)}")
    print(f"After cleaning: {len(clean)}")
    print(f"Dropped messages: {dropped}   ---   {dropped / len(raw) * 100}%")


def main():
    parser = argparse.ArgumentParser(description="parse a text file to a dics format")

    parser.add_argument("--input", default="i-test.txt")
    parser.add_argument("--output", default="o-test.txt")

    args = parser.parse_args()

    raw = parse_text(args.input)

    clean = clean_raw(raw, 2)

    print_stats(raw, clean)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))

    print(f"Done: Wrote json dump to {out_path}")


if __name__ == "__main__":
    main()
