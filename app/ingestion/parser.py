from __future__ import annotations

"""
VectorMind - PDF Parser
========================
Extracts text from PDF files page-by-page using pdfplumber.
Each page's text is returned alongside its page number and source filename.

Key design decision:
  - We process the PDF from an in-memory byte stream (no disk storage needed).
  - Every extracted page carries its 1-indexed page number and the original filename
    so downstream components always have full provenance metadata.
"""

import io
import logging
from dataclasses import dataclass

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Represents the extracted text content of a single PDF page."""
    text: str
    page_number: int       # 1-indexed
    filename: str


def extract_pages_from_pdf(file_bytes: bytes, filename: str) -> list[PageContent]:
    """
    Extract text from every page of a PDF file.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename:   Original filename (preserved as metadata).

    Returns:
        A list of PageContent objects, one per non-empty page.

    Raises:
        ValueError: If the PDF contains no extractable text at all.
    """
    pages: list[PageContent] = []

    # Open PDF from in-memory bytes — no temporary file written to disk
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        logger.info(f"Parsing '{filename}' — {len(pdf.pages)} page(s) detected.")

        for page in pdf.pages:
            text = page.extract_text()

            # Skip pages that yielded no text (e.g. scanned images without OCR)
            if not text or not text.strip():
                logger.debug(f"Page {page.page_number} of '{filename}' has no text — skipped.")
                continue

            pages.append(
                PageContent(
                    text=text.strip(),
                    page_number=page.page_number,   # pdfplumber uses 1-indexed pages
                    filename=filename,
                )
            )

    if not pages:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "The PDF may contain only scanned images."
        )

    logger.info(f"Extracted text from {len(pages)}/{len(pdf.pages)} page(s) of '{filename}'.")
    return pages
