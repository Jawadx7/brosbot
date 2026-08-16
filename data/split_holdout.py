# -*- coding: utf-8 -*-

"""
data/split_holdout.py  --  Step 13a (evaluation prep)
Splits messages.json into a 90% training set and a held-out 10% test set,
BEFORE any windowing happens -- so the held-out set is genuinely unseen.

Why this matters:
  Your existing adapter (model/adapter) was trained on 100% of the data.
  There is no "untouched" subset left to test it on. This script creates
  a SEPARATE split so you can train a second, evaluation-only adapter
  and get a legitimate perplexity comparison. Your original adapter
  (used for the live demo) is never touched by this process.

Split method:
  Splits at the SESSION level, not the message or window level.
  Because windows never cross session boundaries (see format.py), a
  session-level split guarantees no training window and no test window
  share any overlapping messages. The most recent ~10% of sessions
  (chronologically) are held out.

Usage:
    python data/split_holdout.py
    python data/split_holdout.py --holdout-fraction 0.1

Output:
    data/processed/train_holdout.txt   <- 90% of data, use this to train
                                           the evaluation-only adapter
    data/processed/test_holdout.txt    <- 10% of data, NEVER used in
                                           training, used only to compute
                                           perplexity
"""

import json
import argparse
from pathlib import Path

# Reuse the exact same session-splitting / windowing logic as format.py
# so the held-out windows are constructed identically to training windows.
from format import (
    split_into_sessions,
    sliding_windows,
    format_window,
)


def build_examples_from_sessions(sessions, window_size, step):
    examples = []
    for session in sessions:
        if len(session) < 3:
            continue
        for w in sliding_windows(session, window_size, step):
            text = format_window(w)
            if text.strip():
                examples.append(text)
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Split messages.json into train/holdout sets at the session level."
    )
    parser.add_argument("--input", default="data/processed/messages.json")
    parser.add_argument("--train-output", default="data/processed/train_holdout.txt")
    parser.add_argument("--test-output", default="data/processed/test_holdout.txt")
    parser.add_argument("--window", type=int, default=14)
    parser.add_argument("--step", type=int, default=7)
    parser.add_argument("--session-gap", type=int, default=8)
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.10,
        help="Fraction of total messages (chronologically, from the end) to hold out (default: 0.10)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print("ERROR: {} not found.".format(args.input))
        return

    messages = json.loads(input_path.read_text(encoding="utf-8"))
    print("Loaded {:,} messages.".format(len(messages)))

    sessions = split_into_sessions(messages, args.session_gap)
    print("Found {:,} sessions.".format(len(sessions)))

    # Walk sessions in chronological order, accumulating message counts,
    # to find the session index where the last `holdout_fraction` of
    # messages begins.
    total_messages = sum(len(s) for s in sessions)
    holdout_target = total_messages * args.holdout_fraction

    running_total = 0
    split_index = len(sessions)
    for i in range(len(sessions) - 1, -1, -1):
        running_total += len(sessions[i])
        if running_total >= holdout_target:
            split_index = i
            break

    train_sessions = sessions[:split_index]
    holdout_sessions = sessions[split_index:]

    train_msg_count = sum(len(s) for s in train_sessions)
    holdout_msg_count = sum(len(s) for s in holdout_sessions)

    print("\nSplit:")
    print("  Train sessions   : {:,}  ({:,} messages, {:.1f}%)".format(
        len(train_sessions), train_msg_count,
        100 * train_msg_count / total_messages))
    print("  Holdout sessions : {:,}  ({:,} messages, {:.1f}%)".format(
        len(holdout_sessions), holdout_msg_count,
        100 * holdout_msg_count / total_messages))

    train_examples = build_examples_from_sessions(train_sessions, args.window, args.step)
    holdout_examples = build_examples_from_sessions(holdout_sessions, args.window, args.step)

    print("\nTraining windows  : {:,}".format(len(train_examples)))
    print("Holdout windows   : {:,}".format(len(holdout_examples)))

    if len(holdout_examples) < 20:
        print("\nWARNING: fewer than 20 holdout windows. Perplexity estimate")
        print("         will be noisy. Consider a larger --holdout-fraction.")

    separator = "\n<|endoftext|>\n"

    train_out = Path(args.train_output)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.write_text(separator.join(train_examples) + "\n", encoding="utf-8")

    test_out = Path(args.test_output)
    test_out.write_text(separator.join(holdout_examples) + "\n", encoding="utf-8")

    print("\nSaved:")
    print("  {}  ({:,} windows)".format(args.train_output, len(train_examples)))
    print("  {}  ({:,} windows)".format(args.test_output, len(holdout_examples)))
    print("\nNext: train an EVALUATION-ONLY adapter on train_holdout.txt")
    print("      (do not overwrite your production model/adapter/ folder --")
    print("       point output_dir at model/adapter_eval instead)")


if __name__ == "__main__":
    main()
