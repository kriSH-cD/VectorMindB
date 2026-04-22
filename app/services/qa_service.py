from __future__ import annotations

"""
VectorMind - QA Service (Chat-Scoped)
=======================================
Orchestration layer with per-chat session isolation.

All operations are scoped to a chat_id:
  - upload_document → stores in chat-specific ChromaDB collection
  - ingest_text     → stores in chat-specific ChromaDB collection
  - ask_question    → retrieves ONLY from that chat's collection
"""

import logging

from app.ingestion.pipeline import ingest_pdf, ingest_text as pipeline_ingest_text
from app.retrieval.retriever import get_retriever
from app.llm.generator import get_generator
from app.llm.prompt import FALLBACK_RESPONSE

logger = logging.getLogger(__name__)

# No-document fallback — used when querying a chat with no uploaded content
NO_DOCUMENTS_RESPONSE = (
    "No documents have been uploaded to this chat session yet. "
    "Please upload a PDF or paste text before asking questions."
)


class QAService:
    """High-level service orchestrating document upload and question answering."""

    def __init__(self):
        self.generator = get_generator()

    def upload_document(self, file_bytes: bytes, filename: str, chat_id: str | None = None) -> dict:
        """
        Process and ingest a PDF document into a chat-specific collection.

        Args:
            file_bytes: Raw bytes of the uploaded PDF.
            filename:   Original filename.
            chat_id:    Chat session ID for collection scoping.

        Returns:
            Summary dict with ingestion statistics.
        """
        logger.info(f"[QAService] Uploading document: '{filename}' (chat={chat_id or 'GLOBAL'})")

        # Run the full ingestion pipeline — scoped to this chat
        result = ingest_pdf(file_bytes, filename, chat_id=chat_id)

        return result

    def ingest_text(self, text: str, filename: str, chat_id: str | None = None) -> dict:
        """
        Process and ingest raw text into a chat-specific collection.

        Args:
            text:     The raw text string to ingest.
            filename: A label for this text.
            chat_id:  Chat session ID for collection scoping.

        Returns:
            Summary dict with ingestion statistics.
        """
        logger.info(f"[QAService] Ingesting text: '{filename}' (chat={chat_id or 'GLOBAL'})")

        result = pipeline_ingest_text(text, filename, chat_id=chat_id)

        return result

    def ask_question(
        self,
        query: str,
        filename: str | None = None,
        chat_history: list[dict] | None = None,
        chat_id: str | None = None,
    ) -> dict:
        """
        Answer a user's question using the full RAG pipeline.
        
        Retrieval is scoped to this chat's collection only.
        Only the provided chat_history is sent to the LLM — no cross-chat mixing.

        Args:
            query:        The user's natural language question.
            filename:     Optional — restrict retrieval to a specific uploaded PDF.
            chat_history: This chat's conversation history only.
            chat_id:      Chat session ID for collection scoping.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of source dicts).
        """
        logger.info(f"[QAService] Answering query: '{query}' | Chat: {chat_id or 'GLOBAL'} | File: {filename or 'ALL'}")

        # Create a chat-scoped retriever (its own collection + BM25 index)
        retriever = get_retriever(chat_id=chat_id)

        # Check if this chat has any documents at all
        doc_count = retriever.vector_store.get_document_count()
        if doc_count == 0:
            return {
                "answer": NO_DOCUMENTS_RESPONSE,
                "sources": [],
            }

        # Stage 1: Retrieve relevant chunks from THIS chat only
        retrieved_chunks = retriever.retrieve(query, filename=filename)

        if not retrieved_chunks:
            return {
                "answer": FALLBACK_RESPONSE,
                "sources": [],
            }

        # Stage 2: Generate answer with LLM — only this chat's history
        answer = self.generator.generate(query, retrieved_chunks, chat_history=chat_history)

        # Stage 3: Detect fallback — strip sources when LLM couldn't answer
        answer_lower = answer.lower()
        is_fallback = (
            answer == FALLBACK_RESPONSE or 
            answer == NO_DOCUMENTS_RESPONSE or
            "not found in the uploaded" in answer_lower or
            "not found in the provided" in answer_lower or
            "not available in the provided" in answer_lower or
            "not available in the uploaded" in answer_lower or
            "not mentioned in the" in answer_lower or
            "does not contain" in answer_lower or
            "do not contain" in answer_lower or
            "no information" in answer_lower or
            "not in the provided" in answer_lower or
            "not present in" in answer_lower
        )

        if is_fallback:
            # Strip any "Sources:" text block the LLM may have included
            # and return a clean, source-free fallback answer
            clean_answer = answer
            if "sources:" in answer_lower:
                # Cut everything from "Sources:" onward
                idx = answer_lower.index("sources:")
                clean_answer = answer[:idx].strip()
            
            # Remove "Answer:" prefix if present
            if clean_answer.lower().startswith("answer:"):
                clean_answer = clean_answer[7:].strip()
            
            # If after cleanup nothing meaningful remains, use standard fallback
            if not clean_answer or len(clean_answer) < 10:
                clean_answer = "This information is not available in the provided documents."
            
            sources = []
            answer = clean_answer
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
