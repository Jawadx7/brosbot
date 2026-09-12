# CockBlockAI

A group-chat voice LLM, fine-tuned on a real WhatsApp group conversation and
deployed as an active participant in a custom real-time group chat interface.

CockBlockAI learns how *your* group talks — the slang, the energy, the
in-jokes — by fine-tuning a small open model (Qwen2.5-1.5B) on plain text
extracted from a WhatsApp group export. It then joins a live group chat as a
bot (internally referred to as **BrosBot**), deciding when to chime in, tag
along on topics, and even break silence on its own.

```
WhatsApp export ──> parse ──> format ──> LoRA fine-tune (Colab) ──> GGUF ──> Ollama
                                                                              │
                              chat app (Node/Socket.io)  <──────────────  bot agent (Python)
```

## How it works

1. **Extract** the group conversation from a WhatsApp `.txt` export.
2. **Clean & window** the messages into overlapping conversation windows —
   no speaker names, no timestamps, just raw group flow.
3. **Fine-tune** a small base model with LoRA on free Google Colab hardware to
   absorb the group's style.
4. **Serve** the tuned model locally with Ollama.
5. **Deploy** it as a chat participant: a Python agent watches the live chat,
   decides *when* to speak, and a Node/Socket.io server delivers it.

## Project layout

| Path           | Purpose                                                            |
| -------------- | ------------------------------------------------------------------ |
| `data/`        | WhatsApp parsing and training-data formatting                      |
| `training/`    | LoRA fine-tuning, sampling, and perplexity evaluation scripts      |
| `model/`       | Saved LoRA adapter weights and GGUF model files (not in git)      |
| `agent/`       | Python bot — trigger logic + local Ollama inference                |
| `app/`         | Group chat app — plain-JS frontend (`client/`) + Node server (`server/`) |
| `evaluation/`  | Blind human-eval and perplexity tooling                            |

## Prerequisites

- **Python 3.12+**
- **Node.js** (for the chat server)
- **Ollama** ([ollama.com](https://ollama.com)) for local inference
- Optional: **Google Colab** (or any CUDA GPU + ~12GB VRAM) for fine-tuning

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
cd app/server && npm install
```

Python deps are only `python-socketio` and `requests`; everything else needed
for training lives in the Colab notebooks/scripts.

### 2. Prepare the training data

Export the group chat in WhatsApp (`Group info → More → Export chat → Without
media`) and drop the resulting `.txt` into `data/raw/chat.txt`.

```bash
python data/parse.py                      # data/raw/chat.txt → data/processed/messages.json
python data/format.py --preview 3         # messages.json → data/processed/train.txt
```

`parse.py` strips system events, media placeholders, deleted/edit notices, and
short noise messages. `format.py` splits the chat into sessions (gap > 8h)
and slices each into sliding windows of 14 messages (step 7), separated by
`<|endoftext|>`. The output is exactly the format the model sees at inference.

### 3. Fine-tune (Google Colab)

Run `training/train.py` in a Colab cell with a free T4 GPU. All
hyperparameters live in `training/config.yaml`.

```python
!pip install unsloth
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install pyyaml datasets
!python training/train.py --config training/config.yaml
```

This trains a LoRA adapter on top of `Qwen/Qwen2.5-1.5B` (4-bit loaded), and
saves only the adapter weights (~50–150MB) to `model/adapter/`.

### 4. Verify the model

```python
!python training/sample.py --samples 3
```

Judge the output honestly — does it sound like the group? Run with
`--prompt "..."` to test your own openers.

### 5. Prepare the Ollama model

Convert the merged adapter to GGUF so Ollama can serve it locally. Place the
quantized GGUF at `model/gguf/brosbot-q4_k_m.gguf` (matching the `Modelfile`),
then register it:

```bash
ollama create brosbot -f Modelfile
ollama serve
```

### 6. Run the chat server

```bash
cd app/server
npm run dev
```

Open http://localhost:3000, enter a name, and you're in the chat. The server
keeps the last 100 messages in memory and exposes the HTTP endpoints the bot
uses (`/api/bot-message`, `/api/history`, `/api/health`).

### 7. Run the bot agent

In a separate terminal:

```bash
python agent/bot.py
```

The agent joins the chat as the bot, listens to every message, decides whether
to respond (`agent/trigger.py`), builds a plain-text context window from recent
history (`agent/prompt.py`), generates a single-line reply via Ollama
(`agent/inference.py`), and posts it back to the server.

To flip the trigger behavior from a random participant to "always responds /
change the name", set the env vars below before launching.

## Configuration

Copy `.env.example` to `.env` and adjust (or export the same variables):

| Variable                    | Default              | Description                                   |
| --------------------------- | -------------------- | --------------------------------------------- |
| `SERVER_URL`                | `http://localhost:3000` | Chat server base URL the bot posts to      |
| `SERVER_PORT`               | `3000`               | Port the chat server listens on               |
| `BOT_NAME`                  | `BrosBot`            | Display name of the bot                       |
| `TRIGGER_SPONTANEOUS_PROB`  | `0.20`               | Chance to chime in on any message             |
| `TRIGGER_KEYWORD_PROB`      | `0.60`               | Chance to reply when a keyword matches        |
| `TRIGGER_INACTIVITY_SECONDS`| `180`                | Seconds of silence before the bot initiates   |
| `BOT_COOLDOWN_MESSAGES`     | `2`                  | Min human messages before the bot can speak again |

## How the bot behaves

The bot's respond/don't-respond decision lives in `agent/trigger.py`:

- **Tagged** — `@BrosBot` (any casing/spacing) always forces a reply and
  bypasses cooldown.
- **Keyword** — topic words (anime, banter, university/exams, etc.) trigger a
  probabilistic reply. Keywords are grouped and easily editable at the top of
  the file.
- **Spontaneous** — a small random chance to reply to any message so it
  doesn't only react to keywords.
- **Inactivity** — if the chat has been silent past the timeout (and cooldown
  is clear), the bot begins a new message on its own.
- **Cooldown** — the bot never replies to its own messages and needs a minimum
  number of human messages between its turns.

## Evaluation

Two eval paths are included (used during the project's evaluation phase):

- **Perplexity** — `data/split_holdout.py` splits the data at the *session*
  level (so no message leaks between train/test), then
  `training/evaluate_perplexity.py` compares base vs. fine-tuned perplexity on
  the held-out set.
- **Blind human eval** — `evaluation/generate_eval_samples.py` produces
  randomized A/B pairs (base vs. fine-tuned) for raters plus a private answer
  key; `evaluation/analyze_eval_results.py` scores the exported Google Forms
  responses (tone, coherence, fluency, forced choice).

```bash
python data/split_holdout.py                  # → train_holdout.txt / test_holdout.txt
python evaluation/analyze_eval_results.py --responses evaluation/responses.csv
```

## License

Private/internal project.