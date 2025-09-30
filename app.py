from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, glob, re, time
import numpy as np

# ---- minimal, single-file Project A ----
# Features:
# - /ingest : load .txt / .md into memory, chunked
# - /search : hybrid score = alpha*embedding + (1-alpha)*keyword
# - /chat   : compose concise answer with citations + "I don't know" fallback
#
# No FAISS, no PDFs, no external LLMs. Pure Python + sentence-transformers.

app = FastAPI(title="Project A — Conversational AI + Intelligent Search (single-file)")

# ---- in-memory "DB" ----
DOCS: List[Dict[str, Any]] = []     # each: {"source": path, "chunk": i, "text": "..."}
EMB_MATRIX: Optional[np.ndarray] = None
EMB_READY = False
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast

# lazy import to speed startup if user only hits /health
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder

# ---------------- utils ----------------
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _chunk(text: str, words=180, overlap=30) -> List[str]:
    tokens = re.split(r"\s+", text)
    chunks = []
    i = 0
    while i < len(tokens):
        ch = " ".join(tokens[i:i+words]).strip()
        if ch:
            chunks.append(ch)
        i += max(1, words - overlap)
    return chunks

def _keyword_score(text: str, q_terms: List[str]) -> float:
    t = text.lower()
    return float(sum(1 for tt in q_terms if tt in t)) / max(1, len(q_terms))

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

def _hybrid_search(query: str, k: int = 5, alpha: float = 0.7):
    # alpha: weight for embeddings (0..1). 1.0 = embeddings only; 0.0 = keyword only
    global EMB_MATRIX, DOCS, EMB_READY
    if not DOCS:
        return []

    # keyword part
    q_terms = [t for t in re.split(r"\W+", query.lower()) if t]
    kw_scores = np.zeros(len(DOCS), dtype=np.float32)
    if q_terms:
        for i, d in enumerate(DOCS):
            kw_scores[i] = _keyword_score(d["text"], q_terms)

    # embeddings part
    if EMB_READY and alpha > 0:
        emb = get_embedder().encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        # cosine sim (matrix already normalized)
        sims = EMB_MATRIX @ emb
    else:
        sims = np.zeros(len(DOCS), dtype=np.float32)

    # hybrid
    scores = alpha * sims + (1.0 - alpha) * kw_scores
    idxs = np.argsort(-scores)[:k]
    results = []
    for i in idxs:
        results.append({
            "source": f"{DOCS[i]['source']}#chunk{DOCS[i]['chunk']}",
            "score": float(scores[i]),
            "text": DOCS[i]["text"]
        })
    return results

def _extractive_answer(query: str, hits: List[Dict[str, Any]], min_conf: float = 0.25, max_sentences: int = 4):
    # confidence: require top score >= min_conf
    if not hits or hits[0]["score"] < min_conf:
        return {
            "answer": "I don't know from the provided documents.",
            "citations": [],
            "confidence": hits[0]["score"] if hits else 0.0
        }
    # pick sentences containing query terms from top chunks
    q_terms = set([t for t in re.split(r"\W+", query.lower()) if t])
    selected = []
    cites = []
    for h in hits:
        sentences = re.split(r"(?<=[.!?])\s+", h["text"])
        for s in sentences:
            if any(t in s.lower() for t in q_terms):
                selected.append(s.strip())
                cites.append(h["source"])
                if len(selected) >= max_sentences:
                    break
        if len(selected) >= max_sentences:
            break
    if not selected:
        # fallback: take first 2 sentences of top hit
        sentences = re.split(r"(?<=[.!?])\s+", hits[0]["text"])
        selected = [s.strip() for s in sentences[:2] if s.strip()]
        cites = [hits[0]["source"]]*len(selected)
    # de-dup citations, keep order
    seen=set(); ordered=[]
    for c in cites:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return {
        "answer": " ".join(selected),
        "citations": ordered[:3],
        "confidence": hits[0]["score"]
    }

# -------------- API schemas --------------
class IngestRequest(BaseModel):
    source_dir: Optional[str] = "data/source"

class IngestResponse(BaseModel):
    files: int
    chunks: int
    embedding_ready: bool

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    alpha: float = 0.7  # 0..1 (embedding weight)

class Hit(BaseModel):
    source: str
    score: float
    text: str

class SearchResponse(BaseModel):
    hits: List[Hit]
    latency_ms: float

class ChatRequest(BaseModel):
    query: str
    k: int = 5
    alpha: float = 0.7
    min_conf: float = 0.25

class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    latency_ms: float

# --------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"status":"ok", "docs": len(DOCS), "embeddings": EMB_READY, "model": MODEL_NAME}

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    global DOCS, EMB_MATRIX, EMB_READY
    DOCS.clear()
    files = []
    for ext in ("*.txt","*.md"):
        files += glob.glob(os.path.join(req.source_dir, "**", ext), recursive=True)
    for f in files:
        text = _read_text(f)
        for i, ch in enumerate(_chunk(text, words=180, overlap=30)):
            DOCS.append({"source": f, "chunk": i, "text": ch})

    # build embeddings matrix
    EMB_READY = False
    EMB_MATRIX = None
    if DOCS:
        texts = [d["text"] for d in DOCS]
        embs = get_embedder().encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        EMB_MATRIX = _normalize(embs)
        EMB_READY = True

    return IngestResponse(files=len(files), chunks=len(DOCS), embedding_ready=EMB_READY)

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.time()
    hits = _hybrid_search(req.query, k=req.k, alpha=max(0.0, min(1.0, req.alpha)))
    return SearchResponse(hits=[Hit(**h) for h in hits], latency_ms=round((time.time()-t0)*1000,1))

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.time()
    hits = _hybrid_search(req.query, k=req.k, alpha=max(0.0, min(1.0, req.alpha)))
    summary = _extractive_answer(req.query, hits, min_conf=req.min_conf)
    return ChatResponse(
        answer=summary["answer"],
        citations=summary["citations"],
        confidence=float(summary["confidence"]),
        latency_ms=round((time.time()-t0)*1000,1),
    )
