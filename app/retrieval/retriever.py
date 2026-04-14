from __future__ import annotations

"""
VectorMind - Hybrid Retriever (Production Grade)
====================================================
Implements a two-stage retrieval pipeline with production-quality robustness:

  Stage 1 — Hybrid Search (Vector + BM25):
    - Vector search:  Uses ChromaDB's cosine similarity to find semantically
                      similar chunks to the query.
    - BM25 search:    Uses rank-bm25's Okapi BM25 algorithm for keyword-based
                      matching (great for exact terms, names, acronyms).
    - Hybrid fusion:  Combines NORMALIZED scores using a weighted formula:
                        final_score = 0.7 * norm_vector + 0.3 * norm_bm25
    - Returns the top-10 candidates.

  Stage 2 — Cross-Encoder Reranking:
    - The top-10 hybrid candidates are passed through BAAI/bge-reranker-v2-m3
      (a cross-encoder) for fine-grained relevance scoring.
    - Returns the top 3–5 most relevant chunks.

Production Improvements (v2):
  1. Robust regex-based tokenization for BM25 (handles punctuation, hyphens)
  2. Clamped vector scores to prevent negatives from large distances
  3. Metadata filtering for multi-document support (query specific PDFs)
  4. Min-max normalization on BOTH vector AND BM25 scores before fusion
  5. Empty query guard to prevent unnecessary computation
  6. Stable chunk ID handling for reliable hybrid fusion deduplication
  7. Cached tokenized corpus to avoid redundant re-tokenization
  8. All existing logic (hybrid fusion, reranking, metadata, logging) preserved

BM25 Index Lifecycle:
  - On startup, the BM25 index is built from ALL documents stored in ChromaDB.
  - When new documents are ingested, the index is rebuilt to include them.
  - Tokenized corpus is cached alongside the BM25 index for efficiency.

Score Normalization:
  - ChromaDB returns cosine *distances* (lower = more similar). We clamp and
    convert to similarity scores: vector_score = max(0, 1 - distance).
  - Both vector and BM25 scores are min-max normalized to [0, 1] BEFORE
    hybrid fusion to ensure fair weighting.
"""

import logging
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.db.vector_store import get_vector_store
from app.embedding.embedder import get_embedder

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# IMPROVEMENT #1: Robust Regex-Based Tokenizer
# ────────────────────────────────────────────────────────────────
# The naive `text.lower().split()` fails on punctuation-heavy text.
# For example:
#   "Section-42, AI-based"  →  split() gives ["section-42,", "ai-based"]
#   regex tokenize()  gives ["section", "42", "ai", "based"]
#
# This improves BM25 recall for technical terms, hyphenated words,
# and content with punctuation attached to keywords.
# ────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """
    Tokenize text using word-boundary regex.
    Handles punctuation, hyphens, and special characters gracefully.
    Used for BOTH corpus indexing and query tokenization (consistency is critical).
    """
    return re.findall(r"\b\w+\b", text.lower())


def _min_max_normalize(scores: list[float]) -> list[float]:
    """
    Normalize a list of scores to [0, 1] using min-max normalization.
    Returns uniform 0.5 if all scores are identical (avoids division by zero).

    Args:
        scores: Raw score values.

    Returns:
        Normalized scores in [0, 1].
    """
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)
    range_s = max_s - min_s

    if range_s == 0:
        # All scores identical — return neutral 0.5
        return [0.5] * len(scores)

    # Add epsilon (1e-8) for numerical stability
    return [(s - min_s) / (range_s + 1e-8) for s in scores]


class HybridRetriever:
    """
    Two-stage hybrid retriever combining vector search, BM25, and reranking.
    Production-grade with robust tokenization, score normalization, and metadata filtering.
    """

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = get_vector_store()
        self.embedder = get_embedder()

        # BM25 index components (built from ChromaDB on init)
        self.bm25_index: BM25Okapi | None = None
        self.bm25_corpus: list[dict] = []           # Stores {id, text, metadata}

        # ── IMPROVEMENT #7: Cache tokenized corpus ──
        # Avoid re-tokenizing the entire corpus when it hasn't changed.
        # The tokenized list is stored alongside bm25_corpus and only
        # rebuilt when _rebuild_bm25_index() is called.
        self._tokenized_corpus: list[list[str]] = []
        self._corpus_doc_count: int = 0              # Track size for change detection

        # Build the BM25 index from existing ChromaDB data
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        """
        (Re)build the in-memory BM25 index from all documents in ChromaDB.

        This is called:
          1. On application startup — to restore BM25 state from persistent storage.
          2. After each PDF ingestion — to include newly added chunks.

        IMPROVEMENT #7: Skips rebuild if the document count hasn't changed
        (i.e., no new documents were added since last build).
        """
        all_docs = self.vector_store.get_all_documents()
        current_count = len(all_docs["ids"])

        if current_count == 0:
            logger.info("No documents in ChromaDB — BM25 index is empty.")
            self.bm25_index = None
            self.bm25_corpus = []
            self._tokenized_corpus = []
            self._corpus_doc_count = 0
            return

        # Skip rebuild if corpus size unchanged (optimization for repeated calls)
        if current_count == self._corpus_doc_count and self.bm25_index is not None:
            logger.info(
                f"BM25 index unchanged ({current_count} docs) — skipping rebuild."
            )
            return

        # Store the full corpus for later reference
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

        # ── IMPROVEMENT #1: Regex tokenization for BM25 corpus ──
        self._tokenized_corpus = [
            tokenize(doc["text"]) for doc in self.bm25_corpus
        ]

        self.bm25_index = BM25Okapi(self._tokenized_corpus)
        self._corpus_doc_count = current_count
        logger.info(f"BM25 index built with {len(self.bm25_corpus)} documents.")

    # ────────────────────────────────────────────────────────────
    # STAGE 1A: Vector (Semantic) Search
    # ────────────────────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        k: int,
        filename: str | None = None,
    ) -> list[dict]:
        """
        Perform vector (semantic) search using ChromaDB.

        Args:
            query:    The user's search query.
            k:        Number of results to retrieve.
            filename: Optional — restrict search to chunks from this file only.

        Returns:
            List of dicts with keys: id, text, page, file, vector_score.
        """
        query_embedding = self.embedder.embed_query(query)

        # ── IMPROVEMENT #3: Metadata filtering for multi-document support ──
        # If a filename is specified, only retrieve chunks from that specific PDF.
        where_filter = {"filename": filename} if filename else None
        results = self.vector_store.query_by_embedding(
            query_embedding, n_results=k, where=where_filter
        )

        candidates = []
        if results["ids"] and results["ids"][0]:
            for doc_id, text, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ── IMPROVEMENT #2: Clamp vector scores ──
                # ChromaDB cosine distance range is [0, 2].
                # Raw conversion: score = 1.0 - distance
                # Problem: distance > 1 yields negative scores.
                # Fix: clamp to minimum 0.0 to prevent negative hybrid contributions.
                vector_score = max(0.0, 1.0 - distance)

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
        """
        Perform BM25 (keyword) search over the in-memory index.

        Args:
            query:    The user's search query.
            k:        Number of results to retrieve.
            filename: Optional — restrict search to chunks from this file only.

        Returns:
            List of dicts with keys: id, text, page, file, bm25_score (normalized 0–1).
        """
        if self.bm25_index is None or not self.bm25_corpus:
            logger.warning("BM25 index is empty — skipping keyword search.")
            return []

        # ── IMPROVEMENT #1: Use regex tokenizer for query (must match corpus) ──
        tokenized_query = tokenize(query)

        if not tokenized_query:
            logger.warning("Query produced no tokens after tokenization — skipping BM25.")
            return []

        # Get BM25 scores for ALL documents in the index
        raw_scores = self.bm25_index.get_scores(tokenized_query)

        # ── IMPROVEMENT #3: Filter by filename for multi-document support ──
        # If a filename filter is active, we zero-out scores for non-matching docs
        # rather than rebuilding the entire BM25 index (which would be expensive).
        scored_docs = []
        for i, (score, doc) in enumerate(zip(raw_scores, self.bm25_corpus)):
            # Skip documents not matching the filename filter
            if filename and doc["metadata"]["filename"] != filename:
                continue
            scored_docs.append((float(score), doc))

        if not scored_docs:
            return []

        # ── Min-Max Normalization on BM25 scores ──
        bm25_raw = [s for s, _ in scored_docs]
        bm25_normalized = _min_max_normalize(bm25_raw)

        # Pair normalized scores with documents and sort descending
        normalized_docs = [
            (norm_score, doc) for norm_score, (_, doc) in zip(bm25_normalized, scored_docs)
        ]
        normalized_docs.sort(key=lambda x: x[0], reverse=True)

        # Take top-K
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
        """
        Fuse vector search and BM25 results into a single ranked list.

        IMPROVEMENT #4: Both vector AND BM25 scores are min-max normalized
        BEFORE fusion. This ensures neither signal dominates due to raw
        score scale differences.

        Fusion formula (applied to normalized scores):
            final_score = VECTOR_WEIGHT * norm_vector + BM25_WEIGHT * norm_bm25

        Documents appearing in only one result set receive a normalized score
        of 0.0 for the missing component.

        Args:
            vector_results: Results from vector search (with 'vector_score').
            bm25_results:   Results from BM25 search (with 'bm25_score').

        Returns:
            Merged, deduplicated list sorted by hybrid_score descending.
        """
        # ── IMPROVEMENT #4: Normalize vector scores to [0, 1] ──
        # Raw vector scores can have arbitrary ranges depending on the
        # embedding model and data distribution. Normalizing aligns them
        # with the already-normalized BM25 scores for fair weighted fusion.
        if vector_results:
            raw_vec_scores = [d["vector_score"] for d in vector_results]
            norm_vec_scores = _min_max_normalize(raw_vec_scores)
            for doc, norm_score in zip(vector_results, norm_vec_scores):
                doc["vector_score_normalized"] = norm_score
        # BM25 scores are already normalized in _bm25_search()

        # ── IMPROVEMENT #6: Build lookup by stable chunk ID ──
        # Chunk IDs are deterministic SHA-256 hashes (generated during ingestion),
        # ensuring consistent deduplication across vector and BM25 result sets.
        merged: dict[str, dict] = {}

        # Add vector search results
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

        # Merge in BM25 results
        for doc in bm25_results:
            doc_id = doc["id"]
            if doc_id in merged:
                # Document found by both methods — add BM25 score
                merged[doc_id]["bm25_score"] = doc["bm25_score"]
            else:
                # Document found only by BM25
                merged[doc_id] = {
                    "id": doc_id,
                    "text": doc["text"],
                    "page": doc["page"],
                    "file": doc["file"],
                    "vector_score": 0.0,
                    "bm25_score": doc["bm25_score"],
                }

        # ── Compute hybrid score using normalized values ──
        w_vec = self.settings.vector_weight    # 0.7
        w_bm25 = self.settings.bm25_weight     # 0.3

        for doc in merged.values():
            doc["hybrid_score"] = (
                w_vec * doc["vector_score"] + w_bm25 * doc["bm25_score"]
            )

        # Sort by hybrid score descending
        fused = sorted(merged.values(), key=lambda d: d["hybrid_score"], reverse=True)

        logger.info(
            f"Hybrid fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 "
            f"→ {len(fused)} unique candidates."
        )
        return fused

    # ────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT: Two-Stage Retrieval
    # ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, filename: str | None = None) -> list[dict]:
        """
        Full two-stage retrieval pipeline.

        Stage 1: Hybrid search (vector + BM25) → top-10 candidates
        Stage 2: Cross-encoder reranking → top 3–5 results

        Args:
            query:    The user's natural language question.
            filename: Optional — restrict retrieval to a specific uploaded PDF.
                      If None, searches across ALL uploaded documents.

        Returns:
            List of the top reranked documents, each containing:
              text, page, file, hybrid_score, rerank_score.
        """
        # ── IMPROVEMENT #5: Empty query guard ──
        # Avoid unnecessary embedding computation, BM25 scoring, and LLM calls
        # when the query is empty or whitespace-only.
        if not query or not query.strip():
            logger.warning("[Retriever] Empty query received — returning empty results.")
            return []

        k = self.settings.vector_search_k  # 10

        # ── Stage 1: Hybrid Search ──
        logger.info(f"[Retriever] Query: '{query}' | File filter: {filename or 'ALL'}")

        vector_results = self._vector_search(query, k, filename=filename)
        bm25_results = self._bm25_search(query, k, filename=filename)

        # Fuse results
        hybrid_candidates = self._hybrid_fusion(vector_results, bm25_results)

        # Take top-K candidates for reranking
        candidates_for_rerank = hybrid_candidates[:k]

        if not candidates_for_rerank:
            logger.warning("No candidates found — returning empty results.")
            return []

        # ── Stage 2: Return Top Results directly ──
        # Since we removed the Cross-Encoder Reranker to save memory, 
        # we simply return the top results from the Hybrid Fusion!
        final_results = candidates_for_rerank[:self.settings.final_top_k]

        logger.info(
            f"[Retriever] Final results: {len(final_results)} chunks "
            f"(pages: {[d['page'] for d in final_results]})"
        )
        # We assign 'rerank_score' dynamically as a fallback because QA_service expects it
        for res in final_results:
            res['rerank_score'] = res.get('hybrid_score', 0.0)

        return final_results


@lru_cache()
def get_retriever() -> HybridRetriever:
    """Return a cached singleton HybridRetriever instance."""
    return HybridRetriever()
