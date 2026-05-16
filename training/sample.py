"""
Load the fine-tuned adapter and generate sample text to verify quality.

Run this on Colab (same session as training, or a new one):
    !python training/sample.py
    !python training/sample.py --adapter model/adapter --samples 5

What this does:
  1. Loads Qwen2.5-1.5B base model in 4-bit
  2. Attaches your LoRA adapter
  3. Feeds it conversation prompts
  4. Generates continuations
  5. Prints them so you can judge quality

The key question: does the output sound like your group?
Not perfect English -- like your group.
"""

import argparse
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Test prompts
# Replace these with actual openers from your group's chat for best results.
# The model continues whatever text you give it -- so give it something
# that matches what your group actually talks about.
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    # Short opener -- tests if the model can initiate naturally
    "bro did you see that",
    # Reaction context -- tests emotional register
    "lmaooo no way",
    # Plans context -- tests topic handling
    "whose place are we doing it at this time",
    # Hype context -- tests energy matching
    "BRO I SWEAR",
    # Quiet opener -- tests if it can continue low-energy messages
    "guys",
]


# Model loading


def load_model(adapter_path, config_path="training/config.yaml"):
    from unsloth import FastLanguageModel

    # Read max_seq_length from config if available
    max_seq_length = 1024
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            max_seq_length = cfg.get("model", {}).get("max_seq_length", 1024)

    print("Loading base model + adapter from: {}".format(adapter_path))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,  # load adapter directly (unsloth merges automatically)
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    FastLanguageModel.for_inference(model)  # 2x faster inference mode

    print("Model ready.\n")
    return model, tokenizer


# Generation
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=120,
    temperature=0.85,
    top_p=0.92,
    repetition_penalty=1.1,
    num_samples=3,
    first_line_only=False,
):
    """
    Generate `num_samples` continuations for the given prompt.
    Returns a list of generated strings (prompt stripped out).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    results = []
    for _ in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt)

        new_tokens = output_ids[0][input_len:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if first_line_only:
            generated = generated.split("\n")[0].strip()
        results.append(generated)

    return results


# Display


def display_results(prompt, results):
    width = 60
    print("\n" + "=" * width)
    print('PROMPT: "{}"'.format(prompt))
    print("=" * width)
    for i, text in enumerate(results, 1):
        print("\n  [{}]  {}".format(i, text))
    print()


# Entry point


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from the fine-tuned BrosBot adapter."
    )
    parser.add_argument(
        "--adapter",
        default="model/adapter",
        help="Path to the LoRA adapter folder (default: model/adapter)",
    )
    parser.add_argument(
        "--config",
        default="training/config.yaml",
        help="Path to config.yaml (default: training/config.yaml)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of continuations to generate per prompt (default: 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="Max new tokens to generate per sample (default: 120)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.85,
        help="Sampling temperature -- higher = more random (default: 0.85)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run a single custom prompt instead of the default set",
    )
    parser.add_argument(
        "--firstline",
        action="store_true",  # no type=, no default= needed
        help="Return only the first line of each generated sample",
    )

    args = parser.parse_args()

    # Verify adapter exists
    if not Path(args.adapter).exists():
        print("ERROR: adapter not found at '{}'".format(args.adapter))
        print("  Make sure model/adapter/ contains adapter_model.safetensors")
        return

    # Load model
    model, tokenizer = load_model(args.adapter, args.config)

    # Prompts to test
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print("Generating {} sample(s) per prompt...\n".format(args.samples))
    print("Judge the output honestly:")
    print("  - Does it sound like your group's vocabulary?")
    print("  - Does it use the right energy / register?")
    print("  - Does it feel like something someone in the group would actually send?")
    print(
        "  - Ignore grammar and spelling -- your group probably doesn't care either\n"
    )

    # Generate
    for prompt in prompts:
        results = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            num_samples=args.samples,
            first_line_only=args.firstline,
        )
        display_results(prompt, results)

    print("\nDone.")
    print("If the outputs feel right: proceed to Step 5 (inference setup).")
    print("If they feel off: consider 2 more epochs or inspect the training data.")


if __name__ == "__main__":
    main()
