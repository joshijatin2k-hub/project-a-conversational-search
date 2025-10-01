# Conversational AI + Intelligent Search (single-file FastAPI)

Tiny FastAPI app that ingests `.txt/.md` docs and answers questions using **hybrid retrieval** (embeddings + keyword), with **citations** and an **"I don't know"** fallback.

## Quickstart
```
python -m venv .venv
# Windows: . .venv\Scripts\Activate.ps1
pip install -r requirements.txt

mkdir -p data/source
echo "BMW internship notes. Conversational AI, intelligent search, cloud, APIs, data governance." > data/source/notes.txt

uvicorn app:app --reload --port 8000
# Open http://127.0.0.1:8000/docs```

## Endpoints
- `POST /ingest` — index docs in `data/source/`
- `POST /search` — hybrid retrieval (alpha = embedding weight)
- `POST /chat` — concise answer + citations + confidence

## Design choices
- Chunk ≈ 180 words (30 overlap), `k=5`
- Hybrid score = `alpha * cosine(embeddings) + (1 - alpha) * keyword_score`
- Guardrail: if top score < 0.25 → “I don’t know”
- Typical latency on laptop: ~0.8–1.2s
