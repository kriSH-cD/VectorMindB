from __future__ import annotations

"""
VectorMind - Ingestion Pipeline
=================================
Orchestrates the full document ingestion flow:
  PDF bytes → page extraction → chunking → embedding → storage in ChromaDB.

Supports chat-scoped storage via chat_id parameter.
Each chat session stores its embeddings in an isolated ChromaDB collection.
"""

import logging

from app.ingestion.parser import extract_pages_from_pdf, PageContent
from app.ingestion.chunker import chunk_pages
from app.embedding.embedder import get_embedder
from app.db.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def _get_collection_name(chat_id: str | None) -> str | None:
    """Derive the ChromaDB collection name from a chat_id."""
    if chat_id:
        return f"chat_{chat_id}"
    return None  # Falls back to global default


def ingest_pdf(file_bytes: bytes, filename: str, chat_id: str | None = None) -> dict:
    """
    Full ingestion pipeline for a single PDF file.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        filename:   Original filename for metadata tracking.
        chat_id:    Chat session ID — embeddings are stored in chat-specific collection.

    Returns:
        A summary dict with ingestion statistics.
    """
    # Step 1: Parse PDF → list of PageContent
    logger.info(f"[Pipeline] Starting ingestion for '{filename}' (chat={chat_id or 'GLOBAL'}).")
    pages = extract_pages_from_pdf(file_bytes, filename)

    # Step 2: Chunk pages → list of TextChunk
    chunks = chunk_pages(pages)
    logger.info(f"[Pipeline] Generated {len(chunks)} chunks from {len(pages)} pages.")

    # Step 3: Generate embeddings for all chunk texts
    embedder = get_embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed(texts)
    logger.info(f"[Pipeline] Generated {len(embeddings)} embeddings.")

    # Step 4: Store in chat-scoped ChromaDB collection
    collection_name = _get_collection_name(chat_id)
    vector_store = get_vector_store(collection_name)
    vector_store.add_documents(chunks, embeddings)
    logger.info(f"[Pipeline] Stored {len(chunks)} chunks in collection '{collection_name or 'default'}'.")

    return {
        "filename": filename,
        "pages_extracted": len(pages),
        "chunks_created": len(chunks),
        "status": "success",
    }


def ingest_text(text: str, filename: str, chat_id: str | None = None) -> dict:
    """
    Ingestion pipeline for raw text.

    Args:
        text:     The raw text to ingest.
        filename: A label for this text.
        chat_id:  Chat session ID — embeddings are stored in chat-specific collection.

    Returns:
        A summary dict with ingestion statistics.
    """
    logger.info(f"[Pipeline] Starting ingestion for raw text: '{filename}' (chat={chat_id or 'GLOBAL'}).")

    # Step 1: Create a single PageContent object
    pages = [PageContent(text=text, page_number=1, filename=filename)]

    # Step 2: Chunk pages → list of TextChunk
    chunks = chunk_pages(pages)
    logger.info(f"[Pipeline] Generated {len(chunks)} chunks from raw text.")

    # Step 3: Generate embeddings
    embedder = get_embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed(texts)

    # Step 4: Store in chat-scoped ChromaDB collection
    collection_name = _get_collection_name(chat_id)
    vector_store = get_vector_store(collection_name)
    vector_store.add_documents(chunks, embeddings)

    return {
        "filename": filename,
        "pages_extracted": 1,
        "chunks_created": len(chunks),
        "status": "success",
    }
