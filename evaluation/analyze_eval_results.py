# -*- coding: utf-8 -*-

"""
Parses a Google Forms CSV export where the form was built as ONE ROW
PER RATER, with all N items repeated as column blocks across that row
(rather than one row per item). This matches how Google Forms exports
data when a form has repeated question groups.

Expected column layout per row:
    [Timestamp, item1_tone_a, item1_coherence_a, item1_fluency_a,
     item1_tone_b, item1_coherence_b, item1_fluency_b, item1_preferred,
     item2_tone_a, ... (repeats for each item) ...]

IMPORTANT: this reads columns BY POSITION, not by header name. Google
Forms exports repeated question blocks with IDENTICAL header text for
every item, so reading by header name silently collapses all items
down to just the last one -- this is why the original DictReader-based
script produced wrong/missing data. Position-based reading avoids that.

Assumes the form's item blocks appear in the SAME ORDER as the prompts
in generate_eval_samples.py's DEFAULT_PROMPTS list (i.e. block 1 in the
form = item_id 1 in answer_key.csv, block 2 = item_id 2, etc.). If you
built the form in a different order, adjust ITEM_ORDER below.

Usage:
    python evaluation/analyze_eval_results.py --responses evaluation/responses.csv
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict

COLUMNS_PER_ITEM = 7  # tone_a, coherence_a, fluency_a, tone_b, coherence_b, fluency_b, preferred


def load_answer_key(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["item_id"]: row for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True,
                         help="CSV export from your Google Form (wide format, one row per rater)")
    parser.add_argument("--answer-key", default="evaluation/answer_key.csv")
    parser.add_argument("--num-items", type=int, default=15)
    args = parser.parse_args()

    answer_key = load_answer_key(args.answer_key)

    with open(args.responses, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    header, data_rows = rows[0], rows[1:]

    expected_cols = 1 + args.num_items * COLUMNS_PER_ITEM
    if len(header) != expected_cols:
        print("WARNING: expected {} columns ({} items x {} + timestamp), found {}.".format(
            expected_cols, args.num_items, COLUMNS_PER_ITEM, len(header)))
        print("         Check --num-items matches your actual form, or the form")
        print("         layout doesn't match the expected block structure.\n")

    print("Loaded {} rater response row(s), each covering up to {} items.\n".format(
        len(data_rows), args.num_items))

    if len(data_rows) < 5:
        print("NOTE: fewer than 5 raters so far -- treat these numbers as")
        print("      preliminary until more responses come in.\n")

    scores = defaultdict(lambda: defaultdict(list))  # scores[model][criterion] = [values]
    preference_counts = defaultdict(int)
    skipped = 0

    for row in data_rows:
        for item_num in range(1, args.num_items + 1):
            item_id = str(item_num)
            key = answer_key.get(item_id)
            if key is None:
                skipped += 1
                continue

            block_start = 1 + (item_num - 1) * COLUMNS_PER_ITEM
            try:
                tone_a, coherence_a, fluency_a, tone_b, coherence_b, fluency_b, preferred = \
                    row[block_start:block_start + COLUMNS_PER_ITEM]
            except ValueError:
                skipped += 1
                continue

            a_is = key["a_is"]
            b_is = key["b_is"]

            try:
                scores[a_is]["tone"].append(float(tone_a))
                scores[b_is]["tone"].append(float(tone_b))
                scores[a_is]["coherence"].append(float(coherence_a))
                scores[b_is]["coherence"].append(float(coherence_b))
                scores[a_is]["fluency"].append(float(fluency_a))
                scores[b_is]["fluency"].append(float(fluency_b))
            except ValueError:
                skipped += 1
                continue

            preferred_clean = preferred.strip().upper()
            if preferred_clean == "A":
                preference_counts[a_is] += 1
            elif preferred_clean == "B":
                preference_counts[b_is] += 1

    if skipped:
        print("(skipped {} item entries due to missing/malformed data)\n".format(skipped))

    print("=" * 55)
    print("AVERAGE SCORES BY MODEL  (1-5 scale)")
    print("=" * 55)
    for model in ["base", "finetuned"]:
        print("\n{}:".format(model.upper()))
        for criterion in ["tone", "coherence", "fluency"]:
            vals = scores[model][criterion]
            if vals:
                avg = sum(vals) / len(vals)
                print("  {:<12}: {:.2f}  (n={})".format(criterion, avg, len(vals)))
            else:
                print("  {:<12}: no data".format(criterion))

    total_pref = sum(preference_counts.values())
    print("\n" + "=" * 55)
    print("FORCED-CHOICE PREFERENCE  ('sounds more like the group')")
    print("=" * 55)
    for model in ["base", "finetuned"]:
        count = preference_counts[model]
        pct = 100 * count / total_pref if total_pref else 0
        print("  {:<12}: {} votes  ({:.1f}%)".format(model, count, pct))


if __name__ == "__main__":
    main()
