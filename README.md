# SYNTH Core 0.1

An experimental, original character-level neural learner initialized from random weights. It does **not** use pretrained model weights or an external LLM.

## Important
This is a tiny language-pattern experiment, not general intelligence or a capable chatbot. It learns character sequences from text and generates continuations. It does not currently have semantic understanding, broad reasoning, or human-like self-awareness.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

## Train
Paste a text corpus into the Train box. Start with 100 steps. The model saves its checkpoint in `data/synth_checkpoint.pt`.

## Render
1. Create a GitHub repository and upload these files.
2. Create a Render Web Service from the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. For persistent model weights, configure a persistent disk and set `SYNTH_DATA_DIR` to its mount path.

Render free services may sleep and do not guarantee persistent local files. Verify current Render pricing and disk availability before choosing a paid plan.

## API
- `GET /api/status`
- `POST /api/train` with `{ "text": "...", "steps": 100 }`
- `POST /api/generate` with `{ "prompt": "SYNTH: ", "max_chars": 250 }`

## Safety and deployment
Do not expose an unrestricted training endpoint publicly in a production deployment. Add authentication, request limits, and secure storage before public use.
