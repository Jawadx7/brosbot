# -*- coding: utf-8 -*-

"""
evaluation/generate_eval_samples.py  --  Step 14a
Generates a blind, randomized set of paired outputs (base model vs.
fine-tuned model) for human evaluators to score, WITHOUT revealing
which response came from which model.

Run this on Colab (needs both the base model and your PRODUCTION
adapter -- model/adapter, the one trained on 100% of the data, since
this is testing final output quality, not doing a leakage-sensitive
comparison like the perplexity script).

Usage:
    !python evaluation/generate_eval_samples.py

Outputs two files:
    evaluation/rater_form_data.csv   <- give this to raters (no model labels)
    evaluation/answer_key.csv        <- keep this PRIVATE, used for scoring later
"""

import csv
import random
import argparse
from pathlib import Path

# Prompts to test. Mix of prompts already used in sample.py verification
# (Step 4) plus a few pulled from the held-out test set for variety.
# Edit this list -- more prompts = more statistical power, but longer
# for raters to complete. 15-20 is a reasonable range for an FYP.
DEFAULT_PROMPTS = [
    "bro did you see that",
    "lmaooo no way",
    "whose place are we doing it at",
    "BRO I SWEAR",
    "guys",
    "wait what happened",
    "nah that's crazy",
    "who's coming today",
    "I can't believe this",
    "someone explain",
    "we going out today",
    "did anyone else see this",
    "this is wild",
    "what time we meeting",
    "no cap",
]


def generate_response(model, tokenizer, prompt, max_new_tokens=60, temperature=0.85):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.92,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Take first line only -- matches the bot's actual inference behaviour
    first_line = text.split("\n")[0].strip()
    return first_line if first_line else "[no output]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--adapter", default="model/adapter",
                         help="Your PRODUCTION adapter (trained on 100% of data)")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    from unsloth import FastLanguageModel

    print("Loading base model...")
    base_model, base_tok = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(base_model)

    print("Generating base model responses...")
    base_responses = {}
    for p in DEFAULT_PROMPTS:
        base_responses[p] = generate_response(base_model, base_tok, p)
        print("  [base] {} -> {}".format(p, base_responses[p]))

    del base_model, base_tok
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    print("\nLoading fine-tuned model...")
    ft_model, ft_tok = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(ft_model)

    print("Generating fine-tuned model responses...")
    ft_responses = {}
    for p in DEFAULT_PROMPTS:
        ft_responses[p] = generate_response(ft_model, ft_tok, p)
        print("  [fine-tuned] {} -> {}".format(p, ft_responses[p]))

    # -- Build blind, randomized pairs --
    rater_rows = []
    answer_key_rows = []

    for i, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        base_text = base_responses[prompt]
        ft_text = ft_responses[prompt]

        # Randomize which side (A/B) each model lands on, per prompt,
        # so raters can't learn a pattern across the form.
        if random.random() < 0.5:
            resp_a, resp_b = base_text, ft_text
            a_model, b_model = "base", "finetuned"
        else:
            resp_a, resp_b = ft_text, base_text
            a_model, b_model = "finetuned", "base"

        rater_rows.append({
            "item_id": i,
            "prompt": prompt,
            "response_a": resp_a,
            "response_b": resp_b,
        })
        answer_key_rows.append({
            "item_id": i,
            "prompt": prompt,
            "response_a": resp_a,
            "response_b": resp_b,
            "a_is": a_model,
            "b_is": b_model,
        })

    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)

    with open(eval_dir / "rater_form_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "prompt", "response_a", "response_b"])
        writer.writeheader()
        writer.writerows(rater_rows)

    with open(eval_dir / "answer_key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "prompt", "response_a", "response_b", "a_is", "b_is"])
        writer.writeheader()
        writer.writerows(answer_key_rows)

    print("\nDone.")
    print("  evaluation/rater_form_data.csv  -- give this to raters (blind)")
    print("  evaluation/answer_key.csv       -- KEEP PRIVATE, do not share")
    print("\nNext: build the Google Form from rater_form_data.csv (see instructions).")


if __name__ == "__main__":
    main()
