"""
PDF Extractor — native text + OCR fallback for scanned pages.
Primary: PyMuPDF  |  Fallback: Tesseract via pytesseract
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

NATIVE_TEXT_THRESHOLD = 50   # chars — below this we treat page as scanned


@dataclass
class PageContent:
    pdf_id: str
    filename: str
    page_number: int
    text: str
    extraction_method: str
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())


def _clean_text(raw: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", raw)
    text = re.sub(r"(?m)^\s*[-–]?\s*\d{1,4}\s*[-–]?\s*$", "", text)
    text = re.sub(r" {2,}", " ", text)
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if not re.match(r"^[=\-_*#~]{4,}$", l)]
    return "\n".join(lines).strip()


def _ocr_page(pdf_path: str, page_index: int) -> str:
    try:
        images = convert_from_path(
            pdf_path, first_page=page_index + 1, last_page=page_index + 1, dpi=200
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0], lang="eng")
    except Exception as e:
        logger.warning(f"OCR failed page {page_index+1}: {e}")
        return ""


def extract_pdf(pdf_path: str) -> Generator[PageContent, None, None]:
    pdf_path = str(pdf_path)
    filename = Path(pdf_path).name
    pdf_id = Path(pdf_path).stem

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Cannot open {pdf_path}: {e}")
        return

    logger.info(f"Extracting {filename} ({len(doc)} pages)…")

    for i in range(len(doc)):
        page = doc[i]
        raw = page.get_text("text")
        method = "native"

        if len(raw.strip()) < NATIVE_TEXT_THRESHOLD:
            raw = _ocr_page(pdf_path, i)
            method = "ocr"

        cleaned = _clean_text(raw)
        if cleaned:
            yield PageContent(
                pdf_id=pdf_id,
                filename=filename,
                page_number=i + 1,
                text=cleaned,
                extraction_method=method,
            )

    doc.close()
