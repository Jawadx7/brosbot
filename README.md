# CockBlockAI
 
A group-voice LLM fine-tuned on a WhatsApp group chat, deployed as
a participant in a custom real-time group chat interface.
 
## Project layout
- `data/`      WhatsApp parsing and training data formatting
- `training/`  LoRA fine-tuning scripts (designed to run on Google Colab)
- `model/`     Saved LoRA adapter weights (not committed to git)
- `agent/`     Python bot agent — trigger logic + local LLM inference
- `server/`    Next.js group chat app — frontend + Socket.io server
 
## Steps
1. Parse WhatsApp export        →  python data/parse.py
2. Format training data         →  python data/format.py
3. Fine-tune on Colab           →  training/train.py
4. Verify local inference       →  training/sample.py
5. Run bot agent                →  python agent/bot.py
6. Run chat server              →  cd server && npm run dev
