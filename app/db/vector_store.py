from __future__ import annotations

"""
VectorMind - Vector Store
===========================
High-level abstraction over ChromaDB for storing and querying document chunks.

Responsibilities:
  - Adding document chunks with their embeddings and metadata to ChromaDB.
  - Querying by embedding vector to retrieve similar chunks.
  - Fetching all stored documents (used to rebuild the BM25 index on startup).
  - Generating unique, deterministic IDs for each chunk to prevent duplicates
    when the same PDF is uploaded multiple times.
"""

import hashlib
import logging
from functools import lru_cache

import chromadb

from app.config import get_settings
from app.db.chroma_client import get_chroma_client
from app.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document storage and retrieval in ChromaDB."""

    def __init__(self):
        settings = get_settings()
        client = get_chroma_client()

        # Get or create the collection (idempotent)
        self.collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        logger.info(
            f"ChromaDB collection '{settings.chroma_collection_name}' ready "
            f"({self.collection.count()} documents currently stored)."
        )

    def _generate_chunk_id(self, chunk: TextChunk) -> str:
        """
        Generate a deterministic ID for a chunk based on its content and metadata.
        This prevents duplicate entries if the same PDF is uploaded again.
        """
        raw = f"{chunk.filename}::page{chunk.page_number}::idx{chunk.chunk_index}::{chunk.text[:100]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def add_documents(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store document chunks with their embeddings and metadata in ChromaDB.

        Args:
            chunks:     List of TextChunk objects.
            embeddings: Corresponding embedding vectors (same order as chunks).
        """
        if not chunks:
            return

        ids = [self._generate_chunk_id(c) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "page_number": c.page_number,
                "filename": c.filename,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        # Upsert to handle re-uploads gracefully (no duplicates)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB.")

    def query_by_embedding(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> dict:
        """
        Query ChromaDB for the most similar chunks to the given embedding.

        Args:
            query_embedding: The query's dense embedding vector.
            n_results:       Number of results to retrieve.
            where:           Optional metadata filter dict for ChromaDB.
                             Example: {"filename": "report.pdf"}

        Returns:
            Raw ChromaDB query results dict with keys:
            'ids', 'documents', 'metadatas', 'distances'.
        """
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        # Apply metadata filter if provided (e.g., filter by filename)
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)
        return results

    def get_all_documents(self) -> dict:
        """
        Retrieve ALL documents and metadata from the collection.

        This is used on application startup to reconstruct the in-memory
        BM25 index from the persistent ChromaDB store. Since BM25 is a
        pure keyword-based method, it doesn't need embeddings — just text
        and metadata.

        Returns:
            ChromaDB get() result dict with 'ids', 'documents', 'metadatas'.
        """
        count = self.collection.count()
        if count == 0:
            return {"ids": [], "documents": [], "metadatas": []}

        results = self.collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )
        logger.info(f"Retrieved {len(results['ids'])} documents for BM25 index rebuild.")
        return results

    def get_document_count(self) -> int:
        """Return the total number of chunks stored in the collection."""
        return self.collection.count()

    def clear_all(self) -> None:
        """
        Delete all documents from the collection.
        This provides a secure way to wipe the memory on an explicit user request
        or when a fresh session starts.
        """
        try:
            results = self.collection.get()
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            logger.info("ChromaDB collection cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {e}")


@lru_cache()
def get_vector_store() -> VectorStore:
    """Return a cached singleton VectorStore instance."""
    return VectorStore()
