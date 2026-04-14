from __future__ import annotations

"""
VectorMind - Ingestion Pipeline
=================================
Orchestrates the full document ingestion flow:
  PDF bytes → page extraction → chunking → embedding → storage in ChromaDB.

This module ties together the parser, chunker, embedder, and vector store
into a single, clean entry point used by the /upload API endpoint.
"""

import logging

from app.ingestion.parser import extract_pages_from_pdf
from app.ingestion.chunker import chunk_pages
from app.embedding.embedder import get_embedder
from app.db.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    """
    Full ingestion pipeline for a single PDF file.

    Steps:
      1. Extract text from each page (preserving page numbers).
      2. Chunk the text into overlapping segments.
      3. Generate dense embeddings for each chunk.
      4. Store chunks + embeddings + metadata in ChromaDB.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        filename:   Original filename for metadata tracking.

    Returns:
        A summary dict with ingestion statistics.
    """
    # Step 1: Parse PDF → list of PageContent
    logger.info(f"[Pipeline] Starting ingestion for '{filename}'.")
    pages = extract_pages_from_pdf(file_bytes, filename)

    # Step 2: Chunk pages → list of TextChunk
    chunks = chunk_pages(pages)
    logger.info(f"[Pipeline] Generated {len(chunks)} chunks from {len(pages)} pages.")

    # Step 3: Generate embeddings for all chunk texts
    embedder = get_embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed(texts)
    logger.info(f"[Pipeline] Generated {len(embeddings)} embeddings.")

    # Step 4: Store in ChromaDB with metadata
    vector_store = get_vector_store()
    vector_store.add_documents(chunks, embeddings)
    logger.info(f"[Pipeline] Stored {len(chunks)} chunks in ChromaDB.")

    return {
        "filename": filename,
        "pages_extracted": len(pages),
        "chunks_created": len(chunks),
        "status": "success",
    }
