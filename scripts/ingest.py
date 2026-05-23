"""
Ingestion pipeline entry point.

Usage:
    python -m scripts.ingest                              # skip already ingested
    python -m scripts.ingest --force                      # re-ingest ALL PDFs
    python -m scripts.ingest --reprocess filename.pdf     # re-ingest one file
"""

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# noqa: E402 — path must be set before local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pdf_extractor import extract_pdf  # noqa: E402
from ingestion.chunker import chunk_page  # noqa: E402
from ingestion.embedder import embed_texts  # noqa: E402
from ingestion.vector_store import VectorStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ingest.log"),
    ],
)
logger = logging.getLogger("ingest")

CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 150))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_PATH     = os.getenv("QDRANT_PATH", "./data/qdrant_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_corpus")


def ingest_single(pdf_path: Path, store: VectorStore, force: bool = False) -> int:
    """Ingest one PDF. Returns number of chunks ingested."""
    filename = pdf_path.name

    if force:
        logger.info(f"  Deleting existing vectors for: {filename}")
        store.delete_file(filename)

    all_chunks = []
    for page in extract_pdf(str(pdf_path)):
        chunks = chunk_page(page, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning(f"  No chunks extracted from {filename} -- skipping.")
        return 0

    logger.info(f"  {filename}: {len(all_chunks)} chunks -- embedding...")
    embeddings = embed_texts([c.text for c in all_chunks], model_name=EMBEDDING_MODEL)
    store.upsert_chunks(all_chunks, embeddings)
    logger.info(f"  {filename}: ingested successfully.")
    return len(all_chunks)


def ingest_directory(pdf_dir: str, force: bool = False, reprocess: str = None):
    store = VectorStore(path=QDRANT_PATH, collection_name=COLLECTION_NAME)

    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {pdf_dir}")
        return

    # Single file reprocess mode
    if reprocess:
        target = Path(pdf_dir) / reprocess
        if not target.exists():
            logger.error(f"File not found: {target}")
            return
        logger.info(f"Reprocessing single file: {reprocess}")
        count = ingest_single(target, store, force=True)
        logger.info(f"Done. Chunks ingested: {count} | Qdrant total: {store.count()} vectors")
        return

    # Determine which files are already ingested
    if force:
        already_ingested = set()
        logger.info("Force mode enabled -- re-ingesting all PDFs.")
    else:
        already_ingested = store.get_ingested_files()

    to_ingest = [p for p in pdf_files if p.name not in already_ingested]
    skipped   = [p for p in pdf_files if p.name in already_ingested]

    logger.info("-- Ingestion Summary --")
    logger.info(f"  Total PDFs found : {len(pdf_files)}")
    logger.info(f"  Already ingested : {len(skipped)}  (will skip)")
    logger.info(f"  New to ingest    : {len(to_ingest)}")

    if skipped:
        logger.info("  Skipping (already in Qdrant):")
        for p in skipped:
            logger.info(f"    OK  {p.name}")

    if not to_ingest:
        logger.info(f"  Nothing new to ingest. Qdrant total: {store.count()} vectors.")
        logger.info("  To re-ingest a single file : python -m scripts.ingest --reprocess filename.pdf")
        logger.info("  To re-ingest everything    : python -m scripts.ingest --force")
        return

    logger.info(f"  Ingesting {len(to_ingest)} new PDF(s):")
    for p in to_ingest:
        logger.info(f"    >>  {p.name}")

    total_chunks = 0
    t0 = time.time()

    for pdf_path in tqdm(to_ingest, desc="Ingesting PDFs"):
        total_chunks += ingest_single(pdf_path, store, force=False)

    elapsed = time.time() - t0
    logger.info("-- Done --")
    logger.info(f"  New chunks ingested : {total_chunks}")
    logger.info(f"  PDFs processed      : {len(to_ingest)}")
    logger.info(f"  Time taken          : {elapsed:.1f}s")
    logger.info(f"  Qdrant total        : {store.count()} vectors")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir",   default="./data/pdfs")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--reprocess", type=str, default=None)
    args = parser.parse_args()
    ingest_directory(args.pdf_dir, force=args.force, reprocess=args.reprocess)