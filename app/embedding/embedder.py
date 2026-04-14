from __future__ import annotations

"""
VectorMind - Embedder (Jina AI Lightweight)
===========================================
Generates dense vector embeddings using Jina AI's REST API.

Replaces local HuggingFace/PyTorch models to push memory consumption
even lower, removing heavy library dependencies entirely.
"""

import logging
import requests
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """Generates dense embeddings using Jina's API over REST."""

    def __init__(self):
        settings = get_settings()
        logger.info(f"Initialized Jina Embedder with model: {settings.jina_model}")
        
        self.api_key = settings.jina_api_key
        if not self.api_key:
            logger.warning("JINA_API_KEY is not set! Embedding calls will fail.")
            
        self.model_name = settings.jina_model
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of text chunks.
        """
        if not texts:
            return []

        response = requests.post(
            self.url,
            headers=self.headers,
            json={"model": self.model_name, "input": texts},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        # Jina returns { "data": [ { "embedding": [...] }, ... ] }
        return [item["embedding"] for item in data.get("data", [])]

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a simple query string.
        """
        embeddings = self.embed([query])
        if not embeddings:
            raise RuntimeError("Jina API returned empty embeddings for query.")
        return embeddings[0]


@lru_cache()
def get_embedder() -> Embedder:
    """Return a cached singleton Embedder instance."""
    return Embedder()

