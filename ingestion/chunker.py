"""
Text chunker — splits page text into overlapping passages.
Uses tiktoken for accurate token counting (cl100k_base tokenizer).
"""

import re
import logging
from dataclasses import dataclass
from typing import List

import tiktoken

from ingestion.pdf_extractor import PageContent

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (rough heuristic, fast)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Chunk:
    chunk_id: str          # unique: pdf_id_page_chunkIdx
    pdf_id: str
    filename: str
    page_number: int
    chunk_index: int       # within the page
    text: str
    token_count: int
    extraction_method: str


def chunk_page(
    page: PageContent,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Slide a window of `chunk_size` tokens over the page text,
    stepping by (chunk_size - overlap) tokens each iteration.
    Tries to respect sentence boundaries.
    """
    sentences = _split_sentences(page.text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    buffer: List[str] = []
    buffer_tokens = 0
    chunk_idx = 0

    def flush(buf: List[str]) -> None:
        nonlocal chunk_idx
        text = " ".join(buf).strip()
        if not text:
            return
        tc = _count_tokens(text)
        chunks.append(
            Chunk(
                chunk_id=f"{page.pdf_id}_p{page.page_number}_c{chunk_idx}",
                pdf_id=page.pdf_id,
                filename=page.filename,
                page_number=page.page_number,
                chunk_index=chunk_idx,
                text=text,
                token_count=tc,
                extraction_method=page.extraction_method,
            )
        )
        chunk_idx += 1

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        if buffer_tokens + sent_tokens > chunk_size and buffer:
            flush(buffer)
            # Keep overlap: walk back from end of buffer until we have ~overlap tokens
            overlap_buf: List[str] = []
            overlap_tokens = 0
            for s in reversed(buffer):
                st = _count_tokens(s)
                if overlap_tokens + st <= overlap:
                    overlap_buf.insert(0, s)
                    overlap_tokens += st
                else:
                    break
            buffer = overlap_buf
            buffer_tokens = overlap_tokens

        buffer.append(sent)
        buffer_tokens += sent_tokens

    if buffer:
        flush(buffer)

    return chunks
