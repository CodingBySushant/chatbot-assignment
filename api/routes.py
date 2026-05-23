"""
FastAPI routes — chat endpoint + status + doc list.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from retrieval.pipeline import RAGPipeline
from ingestion.pdf_extractor import extract_pdf
from ingestion.chunker import chunk_page
from ingestion.embedder import embed_texts
from ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton pipeline (models loaded once)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ── Request / Response schemas ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class SourceRef(BaseModel):
    filename: str
    page_number: int


class ChunkResult(BaseModel):
    text: str
    filename: str
    page_number: int
    ann_score: float
    rerank_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceRef]
    retrieved_chunks: List[ChunkResult]
    timings: dict
    usage: dict


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/status")
async def status():
    """Check pipeline health and corpus size."""
    from ingestion.vector_store import VectorStore
    store = VectorStore(
        path=os.getenv("QDRANT_PATH", "./data/qdrant_db"),
        collection_name=os.getenv("COLLECTION_NAME", "rag_corpus"),
    )
    count = store.count()
    return {
        "status": "ready" if count > 0 else "empty",
        "vector_count": count,
        "collection": os.getenv("COLLECTION_NAME", "rag_corpus"),
    }


@router.get("/documents")
async def list_documents():
    """List ingested PDFs with page count info."""
    pdf_dir = Path("./data/pdfs")
    files = []
    for p in sorted(pdf_dir.glob("*.pdf")):
        files.append({"filename": p.name, "size_mb": round(p.stat().st_size / 1e6, 2)})
    return {"documents": files, "count": len(files)}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main RAG chat endpoint."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    pipeline = get_pipeline()
    try:
        result = pipeline.query(query)
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))

    return result


@router.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF and trigger background ingestion."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = Path("./data/pdfs") / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    background_tasks.add_task(_ingest_single, str(dest))
    return {"message": f"Uploaded {file.filename}. Ingestion started in background."}


def _ingest_single(pdf_path: str):
    """Background ingestion of a single uploaded PDF."""
    from dotenv import load_dotenv
    load_dotenv()
    store = VectorStore(
        path=os.getenv("QDRANT_PATH", "./data/qdrant_db"),
        collection_name=os.getenv("COLLECTION_NAME", "rag_corpus"),
    )
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = int(os.getenv("CHUNK_SIZE", 800))
    overlap = int(os.getenv("CHUNK_OVERLAP", 150))

    all_chunks = []
    for page in extract_pdf(pdf_path):
        all_chunks.extend(chunk_page(page, chunk_size=chunk_size, overlap=overlap))

    if all_chunks:
        embeddings = embed_texts([c.text for c in all_chunks], model_name=model)
        store.upsert_chunks(all_chunks, embeddings)
        logger.info(f"Background ingest done: {Path(pdf_path).name} → {len(all_chunks)} chunks")
