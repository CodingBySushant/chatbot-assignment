"""
Embedder — wraps all-MiniLM-L6-v2 (384-dim, Apache 2.0).
Provides batch encoding and singleton model loading.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(
    texts: List[str],
    batch_size: int = 64,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Returns shape (N, 384) float32 array.
    Normalizes embeddings for cosine similarity via dot product.
    """
    model = get_embedder(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(
    query: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Returns shape (384,) normalized float32 vector."""
    model = get_embedder(model_name)
    vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec[0].astype(np.float32)
