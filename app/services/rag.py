import json
import os
import hashlib
import pickle
import datetime
import numpy as np
import logging
from collections import Counter

from app.services.embeddings import embed_one, embed_many, backend_info, EmbeddingUnavailable, EMBED_BACKEND
from app.services.action_catalog import CATALOG

PLAYBOOK_PATH = "data/playbook.json"
CACHE_PATH = f"data/playbook_embeddings.{EMBED_BACKEND}.pkl"
TOP_K_DEFAULT = 1
MIN_SIM_WARN = 0.20

logger = logging.getLogger(__name__)
_CORPUS = None

def load_playbook() -> list[dict]:
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
        
    with open(PLAYBOOK_PATH, "rb") as f:
        data = f.read()
        corpus = json.loads(data.decode("utf-8"))
        
    valid_ids = {c["id"] for c in CATALOG}
    seen_ids = set()
    for s in corpus:
        if s["id"] in seen_ids:
            logger.warning(f"Duplicate playbook id: {s['id']}")
        seen_ids.add(s["id"])
        
        if s["recommended_action"] not in valid_ids:
            logger.warning(f"Playbook {s['id']} recommended_action '{s['recommended_action']}' not in CATALOG")
            
    _CORPUS = corpus
    return _CORPUS

def corpus_size() -> int:
    return len(load_playbook())

def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()

def get_matrix():
    corpus = load_playbook()
    curr_hash = _file_hash(PLAYBOOK_PATH)
    
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
            
        if cache.get("corpus_hash") == curr_hash and cache.get("backend") == EMBED_BACKEND:
            return cache["ids"], cache["matrix"]
            
    # Need to build cache
    texts = [s["text"] for s in corpus]
    ids = [s["id"] for s in corpus]
    
    matrix = embed_many(texts, kind="doc")
    
    cache = {
        "backend": EMBED_BACKEND,
        "model": backend_info()["model"],
        "dim": backend_info()["dim"],
        "ids": ids,
        "matrix": matrix,
        "corpus_hash": curr_hash,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
        
    return ids, matrix

def warm():
    get_matrix()
    info = backend_info()
    print(f"RAG warm up complete. Backend: {info['backend']} ({info['model']})")

def _tokenize(text: str) -> set:
    text = text.lower()
    for p in ".,!?;:()[]{}":
        text = text.replace(p, " ")
    tokens = text.split()
    stopwords = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "is", "are", "of"}
    return set(tokens) - set(stopwords)

def _lexical_scores(query_text: str, corpus: list[dict]) -> np.ndarray:
    q_tokens = _tokenize(query_text)
    scores = []
    
    for s in corpus:
        text_to_token = f"{s['title']} {' '.join(s['tags'])} {s['applies_when']} {s['text']}"
        s_tokens = _tokenize(text_to_token)
        
        intersection = q_tokens.intersection(s_tokens)
        union = q_tokens.union(s_tokens)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Tag boost
        s_tags_tokens = _tokenize(" ".join(s['tags']))
        tag_hits = len(q_tokens.intersection(s_tags_tokens))
        
        score = jaccard + (0.5 * tag_hits)
        scores.append(score)
        
    return np.array(scores, dtype=np.float32)

def retrieve(query_text: str, top_k: int = TOP_K_DEFAULT, _force_lexical: bool = False) -> list[dict]:
    corpus = load_playbook()
    
    try:
        if _force_lexical:
            raise EmbeddingUnavailable("Forced lexical")
        
        ids, M = get_matrix()
        q = embed_one(query_text, kind="query")
        sims = M @ q
        method = "embedding"
    except EmbeddingUnavailable as e:
        logger.warning(f"Falling back to lexical search. Cause: {e}")
        sims = _lexical_scores(query_text, corpus)
        method = "lexical_fallback"
        
    order = np.argsort(-sims)[:top_k]
    results = []
    
    for rank, i in enumerate(order):
        s = dict(corpus[i])
        s["score"] = float(sims[i])
        s["rank"] = rank
        s["retrieval_method"] = method
        s["low_confidence"] = bool(sims[i] < MIN_SIM_WARN)
        results.append(s)
        
    return results

def retrieve_best(query_text: str) -> dict:
    return retrieve(query_text, top_k=1)[0]

def build_query(signals: dict, base_action: dict) -> str:
    bits = [base_action.get("playbook_query", "")]
    if signals.get("payment_failures"):
        bits.append(f"{signals['payment_failures']} payment failures")
    if signals.get("usage_trend_pct", 0) < -0.15:
        bits.append(f"usage down {abs(signals['usage_trend_pct']):.0%}")
    if signals.get("days_to_renewal", 999) <= 45:
        bits.append(f"{signals['days_to_renewal']} days to renewal")
    if (signals.get("avg_sentiment") or 0) < -0.3:
        bits.append("negative customer sentiment")
    return "; ".join([b for b in bits if b])

def action_matches(snippet: dict, base_action: dict) -> bool:
    return snippet.get("recommended_action") == base_action.get("action")
