from __future__ import annotations

"""
VectorMind - Configuration Module
==================================
Centralized configuration using pydantic-settings.
All values are loaded from .env file or environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── Groq LLM ───────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # ── Jina Embeddings ────────────────────────────────────
    jina_api_key: str = ""
    jina_model: str = "jina-embeddings-v2-base-en"

    # ── ChromaDB ────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "vectormind_docs"

    # ── Chunking Parameters ────────────────────────────────
    chunk_size: int = 300          # Target tokens per chunk (smaller = faster LLM)
    chunk_overlap: int = 40        # Overlap tokens between chunks

    # ── Retrieval Parameters ───────────────────────────────
    vector_search_k: int = 5      # Candidates from vector search (was 10)
    final_top_k: int = 3          # Final chunks sent to LLM (was 5)
    vector_weight: float = 0.7    # Weight for vector similarity
    bm25_weight: float = 0.3      # Weight for BM25 keyword match

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton of the application settings.
    Using lru_cache ensures the .env file is read only once.
    """
    return Settings()
