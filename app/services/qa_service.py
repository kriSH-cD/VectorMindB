from __future__ import annotations

"""
VectorMind - QA Service
=========================
Orchestration layer that ties together all components for the two main operations:
  1. upload_document — Ingest a PDF into the system.
  2. ask_question    — Run the full RAG pipeline to answer a query.

This service acts as the bridge between the API endpoints and the core modules,
keeping the FastAPI route handlers thin and focused on HTTP concerns.
"""

import logging

from app.ingestion.pipeline import ingest_pdf
from app.retrieval.retriever import get_retriever
from app.llm.generator import get_generator
from app.llm.prompt import FALLBACK_RESPONSE

logger = logging.getLogger(__name__)


class QAService:
    """High-level service orchestrating document upload and question answering."""

    def __init__(self):
        self.retriever = get_retriever()
        self.generator = get_generator()

    def upload_document(self, file_bytes: bytes, filename: str) -> dict:
        """
        Process and ingest a PDF document.

        Steps:
          1. Run the ingestion pipeline (parse → chunk → embed → store).
          2. Rebuild the BM25 index to include the new document.

        Args:
            file_bytes: Raw bytes of the uploaded PDF.
            filename:   Original filename.

        Returns:
            Summary dict with ingestion statistics.
        """
        logger.info(f"[QAService] Uploading document: '{filename}'")

        # Run the full ingestion pipeline
        result = ingest_pdf(file_bytes, filename)

        # Rebuild BM25 index to include newly added chunks
        self.retriever._rebuild_bm25_index()
        logger.info("[QAService] BM25 index rebuilt after ingestion.")

        return result

    def ask_question(self, query: str, filename: str | None = None, chat_history: list[dict] | None = None) -> dict:
        """
        Answer a user's question using the full RAG pipeline and conversation history.

        Steps:
          1. Hybrid retrieval (vector + BM25) → reranking.
          2. LLM generation with retrieved context and history.
          3. Format the response with answer + source citations.

        Args:
            query:        The user's natural language question.
            filename:     Optional — restrict retrieval to a specific uploaded PDF.
                          If None, searches across ALL uploaded documents.
            chat_history: Optional list of previous chat messages.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of source dicts).
        """
        logger.info(f"[QAService] Answering query: '{query}' | File: {filename or 'ALL'}")

        # Stage 1: Retrieve relevant chunks (with optional file filter)
        retrieved_chunks = self.retriever.retrieve(query, filename=filename)

        if not retrieved_chunks:
            return {
                "answer": FALLBACK_RESPONSE,
                "sources": [],
            }

        # Stage 2: Generate answer with LLM
        answer = self.generator.generate(query, retrieved_chunks, chat_history=chat_history)

        # Stage 3: Format source citations
        # Only attach sources if the LLM actually found an answer using them.
        is_fallback = (
            answer == FALLBACK_RESPONSE or 
            "not found" in answer.lower() or 
            "not available" in answer.lower()
        )

        if is_fallback:
            sources = []
        else:
            sources = [
                {
                    "page": chunk["page"],
                    "text": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
                    "file": chunk["file"],
                    "relevance_score": round(chunk.get("rerank_score", 0.0), 4),
                }
                for chunk in retrieved_chunks
            ]

        logger.info(
            f"[QAService] Answer generated with {len(sources)} source citations."
        )

        return {
            "answer": answer,
            "sources": sources,
        }


# ── Singleton ──
_qa_service: QAService | None = None


def get_qa_service() -> QAService:
    """Return a singleton QAService instance (lazy-initialized)."""
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService()
    return _qa_service
