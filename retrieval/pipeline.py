"""
End-to-end RAG pipeline:
  query → embed → ANN search → rerank → generate
"""

import logging
import os
import time
from typing import Dict, Any

from ingestion.embedder import embed_query
from ingestion.vector_store import VectorStore
from retrieval.reranker import rerank
from retrieval.generator import generate_answer

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.store = VectorStore(
            path=os.getenv("QDRANT_PATH", "./data/qdrant_db"),
            collection_name=os.getenv("COLLECTION_NAME", "rag_corpus"),
        )
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.reranker_model = os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.llm_model = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
        self.top_k_retrieve = int(os.getenv("TOP_K_RETRIEVE", 20))
        self.top_k_rerank = int(os.getenv("TOP_K_RERANK", 5))

    def query(self, question: str) -> Dict[str, Any]:
        timings = {}
        t_total = time.perf_counter()

        # 1. Embed query
        t = time.perf_counter()
        q_vec = embed_query(question, model_name=self.embedding_model)
        timings["embed_ms"] = round((time.perf_counter() - t) * 1000)

        # 2. ANN retrieval
        t = time.perf_counter()
        candidates = self.store.search(q_vec, top_k=self.top_k_retrieve)
        timings["retrieve_ms"] = round((time.perf_counter() - t) * 1000)

        if not candidates:
            return {
                "answer": "No relevant documents found. Please check that PDFs have been ingested.",
                "sources": [],
                "retrieved_chunks": [],
                "timings": timings,
                "total_ms": round((time.perf_counter() - t_total) * 1000),
            }

        # 3. Rerank
        t = time.perf_counter()
        reranked = rerank(
            question,
            candidates,
            top_k=self.top_k_rerank,
            model_name=self.reranker_model,
        )
        timings["rerank_ms"] = round((time.perf_counter() - t) * 1000)

        # 4. Generate
        t = time.perf_counter()
        result = generate_answer(question, reranked, model=self.llm_model)
        timings["generate_ms"] = round((time.perf_counter() - t) * 1000)

        timings["total_ms"] = round((time.perf_counter() - t_total) * 1000)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": [
                {
                    "text": c["text"],
                    "filename": c["filename"],
                    "page_number": c["page_number"],
                    "ann_score": round(c.get("score", 0), 4),
                    "rerank_score": round(c.get("rerank_score", 0), 4),
                }
                for c in reranked
            ],
            "timings": timings,
            "usage": result.get("usage", {}),
        }
