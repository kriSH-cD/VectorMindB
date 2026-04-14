"""
VectorMind - ChromaDB Client
==============================
Initializes and manages the persistent ChromaDB client.

ChromaDB is used as our vector database with local persistence.
The client is created once and reused across the application lifecycle
via a singleton pattern to avoid multiple database connections.
"""

import logging
from functools import lru_cache

import chromadb

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_chroma_client() -> chromadb.ClientAPI:
    """
    Return a cached persistent ChromaDB client.
    Data is stored at the path specified by CHROMA_PERSIST_DIR in settings.
    """
    settings = get_settings()
    logger.info(f"Initializing ChromaDB client (persist_dir={settings.chroma_persist_dir})")

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    logger.info("ChromaDB client initialized successfully.")
    return client
