# -*- coding: utf-8 -*-
#
"""
training/evaluate_perplexity.py  --  Step 13b (evaluation)
Computes perplexity for BOTH the base model and the fine-tuned (LoRA)
model on the same held-out test set, and reports the comparison.

Perplexity = exp(average negative log-likelihood per token).
Lower perplexity = the model assigns higher probability to the actual
held-out text = better fit to that text's distribution.

IMPORTANT: run this against the EVALUATION-ONLY adapter (trained on
train_holdout.txt), not your production adapter (trained on 100% of
the data) -- otherwise the held-out set isn't actually held out.

Usage (on Colab, after training the eval adapter):
    !python training/evaluate_perplexity.py \\
        --base-model Qwen/Qwen2.5-1.5B \\
        --adapter model/adapter_eval \\
        --test-file data/processed/test_holdout.txt
"""

import argparse
import json
from pathlib import Path


def load_test_windows(path, separator="<|endoftext|>"):
    text = Path(path).read_text(encoding="utf-8")
    windows = [w.strip() for w in text.split(separator) if w.strip()]
    return windows


def compute_perplexity(model, tokenizer, windows, max_seq_length=1024, device="cuda"):
    """
    Computes corpus-level perplexity: total negative log-likelihood
    summed across ALL tokens in ALL windows, divided by total token
    count, then exponentiated. This is the statistically correct way
    to average perplexity across variable-length examples -- averaging
    per-window perplexities directly would over-weight short windows.
    """
    import torch

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, window in enumerate(windows):
            inputs = tokenizer(
                window,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            ).to(device)

            input_ids = inputs["input_ids"]
            if input_ids.shape[1] < 2:
                continue  # skip degenerate windows too short to score

            outputs = model(**inputs, labels=input_ids)

            # outputs.loss is the MEAN cross-entropy loss per token for
            # this window. Multiply back up by token count to get total
            # NLL for correct corpus-level averaging.
            num_scored_tokens = input_ids.shape[1] - 1  # first token has no prediction target
            total_nll += outputs.loss.item() * num_scored_tokens
            total_tokens += num_scored_tokens

            if (i + 1) % 20 == 0:
                print("  scored {}/{} windows...".format(i + 1, len(windows)))

    import math
    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity, total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs. fine-tuned model perplexity on a held-out test set."
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--adapter", default="model/adapter_eval",
                         help="Path to the EVALUATION-ONLY adapter (trained on train_holdout.txt)")
    parser.add_argument("--test-file", default="data/processed/test_holdout.txt")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--output", default="evaluation/perplexity_results.json")
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    windows = load_test_windows(args.test_file)
    print("Loaded {} held-out test windows from {}\n".format(len(windows), args.test_file))

    # -- Base model perplexity --
    print("=" * 50)
    print("Evaluating BASE model: {}".format(args.base_model))
    print("=" * 50)
    base_model, base_tok = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(base_model)
    base_ppl, base_tokens = compute_perplexity(
        base_model, base_tok, windows, args.max_seq_length
    )
    print("\nBase model perplexity: {:.3f}  (scored {:,} tokens)\n".format(base_ppl, base_tokens))

    # Free memory before loading the second model
    del base_model, base_tok
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    # -- Fine-tuned model perplexity --
    print("=" * 50)
    print("Evaluating FINE-TUNED (eval) model: {}".format(args.adapter))
    print("=" * 50)
    ft_model, ft_tok = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(ft_model)
    ft_ppl, ft_tokens = compute_perplexity(
        ft_model, ft_tok, windows, args.max_seq_length
    )
    print("\nFine-tuned model perplexity: {:.3f}  (scored {:,} tokens)\n".format(ft_ppl, ft_tokens))

    # -- Summary --
    improvement = 100 * (base_ppl - ft_ppl) / base_ppl
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print("  Base model perplexity       : {:.3f}".format(base_ppl))
    print("  Fine-tuned model perplexity : {:.3f}".format(ft_ppl))
    print("  Relative improvement        : {:.1f}%".format(improvement))
    if ft_ppl < base_ppl:
        print("  -> Fine-tuned model fits the held-out group-chat text better.")
    else:
        print("  -> WARNING: fine-tuned model did NOT improve on base. Check")
        print("     training config / holdout split before reporting this.")
    print("=" * 50)

    results = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "test_file": args.test_file,
        "num_test_windows": len(windows),
        "base_perplexity": base_ppl,
        "finetuned_perplexity": ft_ppl,
        "relative_improvement_percent": improvement,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("\nSaved results to {}".format(args.output))


if __name__ == "__main__":
    main()
