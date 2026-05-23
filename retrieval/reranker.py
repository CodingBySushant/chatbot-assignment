"""
Cross-encoder reranker — ms-marco-MiniLM-L-6-v2.
Optimized for CPU: reduced max_length, truncated input, cached model.
"""

import logging
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker: CrossEncoder | None = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker: {model_name}")
        _reranker = CrossEncoder(
            model_name,
            max_length=256,   # reduced from 512 — biggest CPU speed win
        )
        logger.info("Reranker loaded.")
    return _reranker


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 3,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict[str, Any]]:
    """
    Re-score candidates with cross-encoder and return top_k sorted by score.
    Truncates chunk text to 500 chars for faster inference on CPU.
    """
    if not candidates:
        return []

    reranker = get_reranker(model_name)

    # Truncate chunk text — less tokens = faster, quality barely affected
    pairs = [(query, c["text"][:500]) for c in candidates]

    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    results = []
    for doc, score in ranked[:top_k]:
        d = doc.copy()
        d["rerank_score"] = float(score)
        results.append(d)

    return results
