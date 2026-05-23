"""
Qdrant vector store — supports both local and cloud mode.

Local mode  : set QDRANT_PATH in .env, leave QDRANT_URL empty
Cloud mode  : set QDRANT_URL and QDRANT_API_KEY in .env
"""

import logging
import os
import time

from typing import List, Dict, Any, Set

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
    Filter,
    FieldCondition,
    MatchValue,
)

from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

VECTOR_DIM = 384


class VectorStore:
    def __init__(
        self,
        path: str = None,
        collection_name: str = "rag_corpus",
    ):
        self.collection_name = collection_name

        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()

        if qdrant_url:
            # Cloud mode
            logger.info(f"Connecting to Qdrant Cloud: {qdrant_url}")
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=60,
            )
        else:
            # Local mode (fallback for dev)
            local_path = path or os.getenv("QDRANT_PATH", "./data/qdrant_db")
            logger.info(f"Using local Qdrant at: {local_path}")
            self.client = QdrantClient(path=local_path)

        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10_000,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
        else:
            logger.info(f"Using existing collection: {self.collection_name}")

    def upsert_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Bulk upsert chunks with their embeddings."""
        points = []
        for chunk, vec in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=abs(hash(chunk.chunk_id)) % (2**63),
                    vector=vec.tolist(),
                    payload={
                        "chunk_id":          chunk.chunk_id,
                        "pdf_id":            chunk.pdf_id,
                        "filename":          chunk.filename,
                        "page_number":       chunk.page_number,
                        "chunk_index":       chunk.chunk_index,
                        "text":              chunk.text,
                        "token_count":       chunk.token_count,
                        "extraction_method": chunk.extraction_method,
                    },
                )
            )

        # Small batches + retry for Qdrant Cloud free tier
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            retries = 3
            for attempt in range(retries):
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True,
                    )
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        logger.warning(
                            f"Batch {i//batch_size + 1} failed "
                            f"(attempt {attempt+1}), retrying... {e}"
                        )
                        time.sleep(3)  # fixed E702 — separate line
                    else:
                        raise
            logger.debug(
                f"Upserted batch {i//batch_size + 1}/"
                f"{(len(points)-1)//batch_size + 1}"
            )

        logger.info(f"Upserted {len(points)} chunks to Qdrant.")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """ANN search — returns list of payload dicts with score."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        hits = []
        for r in results:
            payload = r.payload.copy()
            payload["score"] = r.score
            hits.append(payload)
        return hits

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

    def collection_exists_and_populated(self) -> bool:
        return self.count() > 0

    def get_ingested_files(self) -> Set[str]:
        """Return set of filenames already ingested into the collection."""
        ingested = set()
        limit = 100
        offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=["filename"],
                with_vectors=False,
            )
            for r in results:
                if r.payload and "filename" in r.payload:
                    ingested.add(r.payload["filename"])
            if next_offset is None:
                break
            offset = next_offset

        return ingested

    def delete_file(self, filename: str):
        """Delete all vectors for a specific PDF filename."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename),
                    )
                ]
            ),
        )
        logger.info(f"Deleted all vectors for: {filename}")