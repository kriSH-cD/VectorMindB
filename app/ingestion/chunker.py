from __future__ import annotations

"""
VectorMind - Text Chunker
===========================
Splits extracted page text into overlapping chunks suitable for embedding.

Chunking strategy:
  - We use tiktoken (cl100k_base) for accurate token counting, which matches
    the tokenizer used by OpenAI models and is a good proxy for sentence-transformers.
  - Each chunk targets CHUNK_SIZE tokens (default 400) with CHUNK_OVERLAP tokens
    of overlap (default 50) to preserve context across chunk boundaries.
  - Chunks are split on sentence boundaries where possible to maintain coherence.
  - Every chunk inherits the page number and filename from its source page.
  - If a page's text is shorter than CHUNK_SIZE, the entire page becomes one chunk.
"""

import logging
import re
from dataclasses import dataclass

import tiktoken

from app.config import get_settings
from app.ingestion.parser import PageContent

logger = logging.getLogger(__name__)

# Use cl100k_base — the tokenizer for GPT-4 / GPT-3.5-turbo family
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class TextChunk:
    """A single chunk of text with full provenance metadata."""
    text: str
    page_number: int
    filename: str
    chunk_index: int          # Position of this chunk within its source page


def _count_tokens(text: str) -> int:
    """Return the number of tokens in the text using tiktoken."""
    return len(_ENCODER.encode(text))


def _split_into_sentences(text: str) -> list[str]:
    """
    Naive but effective sentence splitter.
    Splits on period / question-mark / exclamation followed by whitespace.
    Keeps the delimiter attached to the preceding sentence.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_page(page: PageContent) -> list[TextChunk]:
    """
    Split a single page's text into overlapping token-bounded chunks.

    Algorithm:
      1. Split the page text into sentences.
      2. Accumulate sentences into a chunk until adding the next sentence
         would exceed CHUNK_SIZE tokens.
      3. Emit the chunk, then roll back by CHUNK_OVERLAP tokens worth of
         sentences to create the overlap window for the next chunk.

    Args:
        page: A PageContent object (text + metadata).

    Returns:
        A list of TextChunk objects derived from this page.
    """
    settings = get_settings()
    sentences = _split_into_sentences(page.text)

    # If the whole page fits in one chunk, return it directly
    if _count_tokens(page.text) <= settings.chunk_size:
        return [
            TextChunk(
                text=page.text,
                page_number=page.page_number,
                filename=page.filename,
                chunk_index=0,
            )
        ]

    chunks: list[TextChunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_idx = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)

        # If adding this sentence would bust the limit, emit the current chunk
        if current_tokens + sentence_tokens > settings.chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    page_number=page.page_number,
                    filename=page.filename,
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1

            # ── Overlap: keep trailing sentences that fit within CHUNK_OVERLAP ──
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tokens = _count_tokens(s)
                if overlap_tokens + s_tokens > settings.chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Emit the final chunk (if any remaining text)
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(
            TextChunk(
                text=chunk_text,
                page_number=page.page_number,
                filename=page.filename,
                chunk_index=chunk_idx,
            )
        )

    logger.info(
        f"Page {page.page_number} of '{page.filename}' → {len(chunks)} chunk(s)."
    )
    return chunks


def chunk_pages(pages: list[PageContent]) -> list[TextChunk]:
    """
    Chunk all pages of a document.

    Args:
        pages: List of PageContent objects from the parser.

    Returns:
        A flat list of TextChunk objects across all pages.
    """
    all_chunks: list[TextChunk] = []
    for page in pages:
        all_chunks.extend(chunk_page(page))

    logger.info(
        f"Total chunks generated: {len(all_chunks)} from {len(pages)} page(s)."
    )
    return all_chunks
