import os
import json
import threading
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("SYNTH_DATA_DIR", APP_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = DATA_DIR / "synth_checkpoint.pt"
CORPUS = DATA_DIR / "training.txt"
DEVICE = torch.device("cpu")
TRAIN_LOCK = threading.Lock()

app = FastAPI(title="SYNTH Core 0.1", description="Original randomly initialized character-level neural learner.")

class TinyLanguageNet(nn.Module):
    """Small character-level GRU. All weights are initialized from scratch."""
    def __init__(self, vocab_size: int, embedding_dim: int = 48, hidden_size: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.gru(embedded)
        return self.output(out)

class Synth:
    def __init__(self):
        self.model: Optional[TinyLanguageNet] = None
        self.optimizer = None
        self.vocab = []
        self.stoi = {}
        self.itos = {}
        self.steps = 0
        self.loss = None
        self.load_checkpoint()

    def prepare(self, text: str):
        chars = sorted(set(text))
        if len(chars) < 2:
            raise ValueError("Training text must contain at least two distinct characters.")
        self.vocab = chars
        self.stoi = {c:i for i,c in enumerate(chars)}
        self.itos = {i:c for c,i in self.stoi.items()}
        self.model = TinyLanguageNet(len(chars)).to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003)
        self.steps = 0
        self.loss = None

    def load_checkpoint(self):
        if not CHECKPOINT.exists():
            return
        try:
            ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
            self.vocab = ckpt["vocab"]
            self.stoi = {c:i for i,c in enumerate(self.vocab)}
            self.itos = {i:c for c,i in self.stoi.items()}
            self.model = TinyLanguageNet(len(self.vocab)).to(DEVICE)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003)
            if ckpt.get("optimizer"):
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self.steps = ckpt.get("steps", 0)
            self.loss = ckpt.get("loss")
        except Exception as exc:
            print("Checkpoint load failed:", exc)

    def save(self):
        if self.model is None:
            return
        torch.save({
            "vocab": self.vocab,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "loss": self.loss
        }, CHECKPOINT)

    def train(self, text: str, steps: int = 100, seq_len: int = 64, batch_size: int = 16):
        if self.model is None or sorted(set(text)) != self.vocab:
            self.prepare(text)
        encoded = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        if len(encoded) <= seq_len + 1:
            raise ValueError(f"Training corpus needs more than {seq_len+1} characters.")
        self.model.train()
        for _ in range(steps):
            starts = torch.randint(0, len(encoded)-seq_len-1, (batch_size,))
            x = torch.stack([encoded[i:i+seq_len] for i in starts])
            y = torch.stack([encoded[i+1:i+seq_len+1] for i in starts])
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, len(self.vocab)), y.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.steps += 1
            self.loss = float(loss.item())
        self.save()
        return self.loss

    @torch.no_grad()
    def generate(self, prompt: str, max_chars: int = 300, temperature: float = 0.8):
        if self.model is None:
            raise ValueError("SYNTH has not been trained yet.")
        self.model.eval()
        prompt = "".join(c for c in prompt if c in self.stoi)
        if not prompt:
            prompt = self.vocab[0]
        ids = torch.tensor([[self.stoi[c] for c in prompt]], dtype=torch.long)
        result = list(prompt)
        for _ in range(max_chars):
            logits = self.model(ids[:, -128:])[:, -1, :]
            logits = logits / max(temperature, 0.05)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            result.append(self.itos[next_id])
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
        return "".join(result)

synth = Synth()

class TrainRequest(BaseModel):
    text: Optional[str] = None
    steps: int = Field(default=100, ge=1, le=2000)

class GenerateRequest(BaseModel):
    prompt: str = Field(default="SYNTH: ")
    max_chars: int = Field(default=250, ge=1, le=1000)
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SYNTH Core</title><style>
body{font-family:system-ui;background:#0b1020;color:#eef2ff;max-width:850px;margin:30px auto;padding:18px}
.card{background:#151d32;border:1px solid #283553;border-radius:14px;padding:18px;margin:14px 0}
textarea,input{box-sizing:border-box;width:100%;background:#0b1020;color:#fff;border:1px solid #394967;border-radius:8px;padding:12px}
button{background:#4d8dff;color:white;border:0;border-radius:8px;padding:12px 18px;margin-top:10px;font-weight:700}
pre{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#aab8d4}
</style></head><body><h1>🧠 SYNTH Core 0.1</h1>
<p class="muted">Original character-level neural network. Random initialization; no pretrained model.</p>
<div class="card"><h2>Train</h2><p>Paste a text corpus (at least a few hundred characters).</p>
<textarea id="corpus" rows="8" placeholder="Training text goes here..."></textarea>
<label>Training steps (100–500 is a good first test)</label><input id="steps" type="number" value="100" min="1" max="2000">
<button onclick="train()">Train SYNTH</button><pre id="trainout"></pre></div>
<div class="card"><h2>Generate</h2><input id="prompt" value="SYNTH: "><button onclick="generate()">Generate text</button><pre id="out"></pre></div>
<div class="card"><h2>Status</h2><button onclick="status()">Refresh status</button><pre id="status"></pre></div>
<script>
async function train(){let t=document.getElementById('corpus').value;let s=Number(document.getElementById('steps').value);
document.getElementById('trainout').textContent='Training…';
let r=await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t,steps:s})});
document.getElementById('trainout').textContent=await r.text();}
async function generate(){let p=document.getElementById('prompt').value;let r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p,max_chars:300})});let j=await r.json();document.getElementById('out').textContent=j.text||j.detail;}
async function status(){let r=await fetch('/api/status');document.getElementById('status').textContent=JSON.stringify(await r.json(),null,2);}
status();
</script></body></html>
""")

@app.get("/api/status")
def status():
    return {"name":"SYNTH Core 0.1","model_initialized":synth.model is not None,
            "vocabulary_size":len(synth.vocab),"training_steps":synth.steps,
            "last_loss":synth.loss,"device":"CPU","pretrained_weights":False}

@app.post("/api/train")
def train(req: TrainRequest):
    text = req.text
    if not text and CORPUS.exists():
        text = CORPUS.read_text(encoding="utf-8")
    if not text:
        raise HTTPException(400, "Provide training text in the request.")
    with TRAIN_LOCK:
        try:
            loss = synth.train(text, steps=req.steps)
            CORPUS.write_text(text, encoding="utf-8")
            return {"status":"trained","steps_total":synth.steps,"last_loss":loss,"checkpoint_saved":True}
        except Exception as exc:
            raise HTTPException(400, str(exc))

@app.post("/api/generate")
def generate(req: GenerateRequest):
    try:
        return {"text":synth.generate(req.prompt, req.max_chars, req.temperature)}
    except Exception as exc:
        raise HTTPException(400, str(exc))
