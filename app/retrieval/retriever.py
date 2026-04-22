from __future__ import annotations

"""
VectorMind - Hybrid Retriever (Chat-Scoped)
====================================================
Two-stage retrieval pipeline with per-chat isolation:

  Stage 1 — Hybrid Search (Vector + BM25):
    - Vector search via chat-scoped ChromaDB collection.
    - BM25 keyword search built on-the-fly from the same collection.
    - Hybrid fusion with normalized score weighting.

  Stage 2 — Top-K selection from hybrid candidates.

Chat Isolation:
  - Each chat_id maps to a dedicated ChromaDB collection (chat_{chat_id}).
  - BM25 index is rebuilt per-request from that collection only.
  - No cross-chat data leakage is possible.
"""

import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.db.vector_store import get_vector_store
from app.embedding.embedder import get_embedder

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """
    Tokenize text using word-boundary regex.
    Used for BOTH corpus indexing and query tokenization (consistency is critical).
    """
    return re.findall(r"\b\w+\b", text.lower())


def _min_max_normalize(scores: list[float]) -> list[float]:
    """
    Normalize a list of scores to [0, 1] using min-max normalization.
    Returns uniform 0.5 if all scores are identical (avoids division by zero).
    """
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)
    range_s = max_s - min_s

    if range_s == 0:
        return [0.5] * len(scores)

    return [(s - min_s) / (range_s + 1e-8) for s in scores]


# ── Module-level BM25 cache ──
# Keyed by (collection_name, doc_count) — if doc count hasn't changed, reuse.
_bm25_cache: dict[str, tuple[int, BM25Okapi | None, list[dict], list[list[str]]]] = {}


class HybridRetriever:
    """
    Two-stage hybrid retriever with per-chat isolation.
    
    Unlike the old singleton design, this retriever is instantiated per-request
    with a specific chat_id, ensuring complete data isolation between sessions.
    BM25 index is cached per-collection and only rebuilt when doc count changes.
    """

    def __init__(self, chat_id: str | None = None):
        self.settings = get_settings()
        self.embedder = get_embedder()
        self.chat_id = chat_id

        # Resolve the collection name for this chat
        collection_name = f"chat_{chat_id}" if chat_id else None
        self.vector_store = get_vector_store(collection_name)
        self._collection_key = collection_name or "default"

        # BM25 index — built from this chat's collection only (cached)
        self.bm25_index: BM25Okapi | None = None
        self.bm25_corpus: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []

        # Build or reuse cached BM25
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        """
        Build or reuse cached BM25 index.
        Only fetches all documents if the doc count changed since last build.
        """
        current_count = self.vector_store.get_document_count()

        # Check cache — if doc count unchanged, reuse
        cached = _bm25_cache.get(self._collection_key)
        if cached and cached[0] == current_count and current_count > 0:
            self.bm25_index = cached[1]
            self.bm25_corpus = cached[2]
            self._tokenized_corpus = cached[3]
            logger.info(f"[Retriever] BM25 cache HIT ({current_count} docs, chat={self.chat_id or 'GLOBAL'})")
            return

        if current_count == 0:
            logger.info(f"[Retriever] No documents in collection (chat={self.chat_id or 'GLOBAL'}) — BM25 empty.")
            self.bm25_index = None
            self.bm25_corpus = []
            self._tokenized_corpus = []
            _bm25_cache[self._collection_key] = (0, None, [], [])
            return

        # Cache miss — rebuild
        all_docs = self.vector_store.get_all_documents()

        self.bm25_corpus = [
            {
                "id": doc_id,
                "text": doc_text,
                "metadata": doc_meta,
            }
            for doc_id, doc_text, doc_meta in zip(
                all_docs["ids"],
                all_docs["documents"],
                all_docs["metadatas"],
            )
        ]

        self._tokenized_corpus = [
            tokenize(doc["text"]) for doc in self.bm25_corpus
        ]

        self.bm25_index = BM25Okapi(self._tokenized_corpus)

        # Store in cache
        _bm25_cache[self._collection_key] = (
            current_count, self.bm25_index, self.bm25_corpus, self._tokenized_corpus
        )
        logger.info(f"[Retriever] BM25 cache MISS — rebuilt with {len(self.bm25_corpus)} docs (chat={self.chat_id or 'GLOBAL'}).")

    # ────────────────────────────────────────────────────────────
    # STAGE 1A: Vector (Semantic) Search
    # ────────────────────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        k: int,
        filename: str | None = None,
    ) -> list[dict]:
        """Perform vector (semantic) search using this chat's ChromaDB collection."""
        query_embedding = self.embedder.embed_query(query)

        where_filter = {"filename": filename} if filename else None
        results = self.vector_store.query_by_embedding(
            query_embedding, n_results=k, where=where_filter
        )

        # Minimum relevance threshold — chunks below this are considered
        # unrelated to the query and are discarded. This prevents the LLM
        # from fabricating answers from irrelevant content.
        MIN_VECTOR_SCORE = 0.3

        candidates = []
        if results["ids"] and results["ids"][0]:
            for doc_id, text, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                vector_score = max(0.0, 1.0 - distance)

                # Skip chunks that are too dissimilar to the query
                if vector_score < MIN_VECTOR_SCORE:
                    logger.debug(f"Skipping chunk (score={vector_score:.3f} < {MIN_VECTOR_SCORE}): {text[:60]}...")
                    continue

                candidates.append({
                    "id": doc_id,
                    "text": text,
                    "page": metadata["page_number"],
                    "file": metadata["filename"],
                    "vector_score": float(vector_score),
                })

        logger.debug(f"Vector search returned {len(candidates)} candidates.")
        return candidates

    # ────────────────────────────────────────────────────────────
    # STAGE 1B: BM25 (Keyword) Search
    # ────────────────────────────────────────────────────────────

    def _bm25_search(
        self,
        query: str,
        k: int,
        filename: str | None = None,
    ) -> list[dict]:
        """Perform BM25 (keyword) search over this chat's in-memory index."""
        if self.bm25_index is None or not self.bm25_corpus:
            logger.warning("BM25 index is empty — skipping keyword search.")
            return []

        tokenized_query = tokenize(query)

        if not tokenized_query:
            logger.warning("Query produced no tokens after tokenization — skipping BM25.")
            return []

        raw_scores = self.bm25_index.get_scores(tokenized_query)

        scored_docs = []
        for i, (score, doc) in enumerate(zip(raw_scores, self.bm25_corpus)):
            if filename and doc["metadata"]["filename"] != filename:
                continue
            scored_docs.append((float(score), doc))

        if not scored_docs:
            return []

        bm25_raw = [s for s, _ in scored_docs]
        bm25_normalized = _min_max_normalize(bm25_raw)

        normalized_docs = [
            (norm_score, doc) for norm_score, (_, doc) in zip(bm25_normalized, scored_docs)
        ]
        normalized_docs.sort(key=lambda x: x[0], reverse=True)

        candidates = []
        for score, doc in normalized_docs[:k]:
            candidates.append({
                "id": doc["id"],
                "text": doc["text"],
                "page": doc["metadata"]["page_number"],
                "file": doc["metadata"]["filename"],
                "bm25_score": score,
            })

        logger.debug(f"BM25 search returned {len(candidates)} candidates.")
        return candidates

    # ────────────────────────────────────────────────────────────
    # STAGE 1C: Hybrid Fusion
    # ────────────────────────────────────────────────────────────

    def _hybrid_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """Fuse vector search and BM25 results into a single ranked list."""
        if vector_results:
            raw_vec_scores = [d["vector_score"] for d in vector_results]
            norm_vec_scores = _min_max_normalize(raw_vec_scores)
            for doc, norm_score in zip(vector_results, norm_vec_scores):
                doc["vector_score_normalized"] = norm_score

        merged: dict[str, dict] = {}

        for doc in vector_results:
            doc_id = doc["id"]
            merged[doc_id] = {
                "id": doc_id,
                "text": doc["text"],
                "page": doc["page"],
                "file": doc["file"],
                "vector_score": doc.get("vector_score_normalized", 0.0),
                "bm25_score": 0.0,
            }

        for doc in bm25_results:
            doc_id = doc["id"]
            if doc_id in merged:
                merged[doc_id]["bm25_score"] = doc["bm25_score"]
            else:
                merged[doc_id] = {
                    "id": doc_id,
                    "text": doc["text"],
                    "page": doc["page"],
                    "file": doc["file"],
                    "vector_score": 0.0,
                    "bm25_score": doc["bm25_score"],
                }

        w_vec = self.settings.vector_weight
        w_bm25 = self.settings.bm25_weight

        for doc in merged.values():
            doc["hybrid_score"] = (
                w_vec * doc["vector_score"] + w_bm25 * doc["bm25_score"]
            )

        fused = sorted(merged.values(), key=lambda d: d["hybrid_score"], reverse=True)

        logger.info(
            f"Hybrid fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 "
            f"→ {len(fused)} unique candidates."
        )
        return fused

    # ────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, filename: str | None = None) -> list[dict]:
        """
        Full two-stage retrieval pipeline, scoped to this chat's collection.

        Args:
            query:    The user's natural language question.
            filename: Optional — restrict retrieval to a specific uploaded PDF.

        Returns:
            List of the top reranked documents.
        """
        if not query or not query.strip():
            logger.warning("[Retriever] Empty query received — returning empty results.")
            return []

        k = self.settings.vector_search_k

        logger.info(f"[Retriever] Query: '{query}' | Chat: {self.chat_id or 'GLOBAL'} | File: {filename or 'ALL'}")

        vector_results = self._vector_search(query, k, filename=filename)
        bm25_results = self._bm25_search(query, k, filename=filename)

        hybrid_candidates = self._hybrid_fusion(vector_results, bm25_results)

        candidates_for_rerank = hybrid_candidates[:k]

        if not candidates_for_rerank:
            logger.warning("No candidates found — returning empty results.")
            return []

        final_results = candidates_for_rerank[:self.settings.final_top_k]

        logger.info(
            f"[Retriever] Final results: {len(final_results)} chunks "
            f"(pages: {[d['page'] for d in final_results]})"
        )
        for res in final_results:
            res['rerank_score'] = res.get('hybrid_score', 0.0)

        return final_results


def get_retriever(chat_id: str | None = None) -> HybridRetriever:
    """
    Create a HybridRetriever scoped to a specific chat session.
    
    NOT cached — each call creates a fresh retriever with its own
    BM25 index built from the chat's collection only.
    """
    return HybridRetriever(chat_id=chat_id)
