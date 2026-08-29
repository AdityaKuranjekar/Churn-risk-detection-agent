import os
import numpy as np
import httpx

class EmbeddingUnavailable(Exception):
    pass

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
EMBED_DIM_EXPECTED = 768

def backend_info() -> dict:
    if EMBED_BACKEND == "gemini":
        return {"backend": "gemini", "model": GEMINI_EMBED_MODEL, "dim": EMBED_DIM_EXPECTED}
    else:
        return {"backend": "ollama", "model": OLLAMA_MODEL, "dim": EMBED_DIM_EXPECTED}

def _normalize(vec: list) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    if v.shape[-1] != EMBED_DIM_EXPECTED:
        raise ValueError(f"Expected embedding dim {EMBED_DIM_EXPECTED}, got {v.shape[-1]}")
    v /= (np.linalg.norm(v) + 1e-12)
    return v

def _embed_ollama(text: str) -> list:
    try:
        r = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": text},
            timeout=30.0
        )
        r.raise_for_status()
        return r.json()["embedding"]
    except Exception as e:
        raise EmbeddingUnavailable(f"Ollama embedding failed: {e}")

def _embed_gemini(text: str, kind: str) -> list:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        task_type = "retrieval_document" if kind == "doc" else "retrieval_query"
        resp = genai.embed_content(
            model=GEMINI_EMBED_MODEL,
            content=text,
            task_type=task_type
        )
        return resp["embedding"]
    except Exception as e:
        raise EmbeddingUnavailable(f"Gemini embedding failed: {e}")

def embed_one(text: str, kind: str = "doc") -> np.ndarray:
    if EMBED_BACKEND == "gemini":
        vec = _embed_gemini(text, kind)
    else:
        vec = _embed_ollama(text)
    return _normalize(vec)

def embed_many(texts: list[str], kind: str = "doc") -> np.ndarray:
    vectors = []
    for text in texts:
        vectors.append(embed_one(text, kind=kind))
    if not vectors:
        return np.empty((0, EMBED_DIM_EXPECTED), dtype=np.float32)
    return np.vstack(vectors)
