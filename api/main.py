"""
FastAPI application entry point.
Supports both local dev and GCP Cloud Run deployment.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import router  # moved to top to fix E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("main")

# Add file handler only if logs dir is writable (not always true on Cloud Run)
try:
    Path("logs").mkdir(exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler("logs/server.log"))
except Exception:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-loading embedding model and reranker...")
    try:
        from ingestion.embedder import get_embedder
        from retrieval.reranker import get_reranker
        get_embedder(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        get_reranker(os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        logger.info("Models ready. Server accepting requests.")
    except Exception as e:
        logger.error(f"Model pre-load failed: {e}")
    yield


app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")