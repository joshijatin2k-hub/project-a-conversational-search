# Conversational AI + Intelligent Search (single-file FastAPI)

Tiny FastAPI app that ingests `.txt/.md` docs and answers questions using **hybrid retrieval** (embeddings + keyword), with **citations** and an **"I don't know"** fallback.

## Quickstart
```bash
python -m venv .venv
# Windows: . .venv\Scripts\Activate.ps1
pip install -r requirements.txt

mkdir -p data/source
echo "BMW internship notes. Conversational AI, intelligent search, cloud, APIs, data governance." > data/source/notes.txt

uvicorn app:app --reload --port 8000
# Open http://127.0.0.1:8000/docs
