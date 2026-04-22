
"""
VectorMind - FastAPI Application (Chat-Scoped)
================================================
Main entry point for the Document QA API.

All endpoints now accept a `chat_id` parameter to ensure complete
session isolation. Each chat gets its own ChromaDB collection and
BM25 index — zero cross-chat data leakage.

Endpoints:
  POST /upload      — Upload multiple PDF documents for a specific chat.
  POST /ingest-text — Ingest raw text for a specific chat.
  POST /query       — Ask a question scoped to a specific chat's documents.
  GET  /health      — Health check endpoint.
  GET  /stats       — Get system statistics (doc count, etc.).
"""

import asyncio
import logging
import time

from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.qa_service import get_qa_service
from app.db.vector_store import get_vector_store

# ── Logging Configuration ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── FastAPI App ──
app = FastAPI(
    title="VectorMind",
    description="Document Question Answering system powered by Hybrid RAG",
    version="2.0.0",
)

# ── CORS — allow React frontend ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "https://vector-mind-gamma.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ──
class QueryRequest(BaseModel):
    """Schema for the /query endpoint request body."""
    query: str
    filename: Optional[str] = None      # Optional: restrict search to a specific PDF
    chat_history: list[dict] = []       # This chat's conversation history ONLY
    chat_id: Optional[str] = None       # Chat session ID for collection scoping


class SourceInfo(BaseModel):
    """Schema for a single source citation."""
    page: int
    text: str
    file: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Schema for the /query endpoint response."""
    answer: str
    sources: list[SourceInfo]


class UploadResponse(BaseModel):
    """Schema for the /upload endpoint response for a single file."""
    message: str
    filename: str
    pages_extracted: int
    chunks_created: int


class MultiUploadResponse(BaseModel):
    """Schema for the /upload endpoint response with multiple files."""
    results: list[UploadResponse]
    total_chunks: int


class TextIngestRequest(BaseModel):
    """Schema for the /ingest-text endpoint request body."""
    text: str
    filename: Optional[str] = "Raw Text"
    chat_id: Optional[str] = None       # Chat session ID for collection scoping


class StatsResponse(BaseModel):
    """Schema for the /stats endpoint response."""
    total_chunks: int
    status: str


# ── Endpoints ──

@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "VectorMind"}


@app.get("/stats", response_model=StatsResponse)
def get_stats(chat_id: Optional[str] = Query(None)):
    """Get stats for a specific chat's collection, or global if no chat_id."""
    try:
        collection_name = f"chat_{chat_id}" if chat_id else None
        store = get_vector_store(collection_name)
        count = store.get_document_count()
        return StatsResponse(total_chunks=count, status="operational")
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
def clear_memory(chat_id: Optional[str] = Query(None)):
    """Wipe stored documents for a specific chat, or all if no chat_id."""
    try:
        collection_name = f"chat_{chat_id}" if chat_id else None
        store = get_vector_store(collection_name)
        store.clear_all()
        
        logger.info(f"[API] Successfully cleared memory for chat={chat_id or 'GLOBAL'}.")
        return {"message": f"Memory cleared for chat {chat_id or 'all'}."}
    except Exception as e:
        logger.error(f"[API] Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=MultiUploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    chat_id: Optional[str] = Query(None),
):
    """
    Upload multiple PDF documents for processing.
    Embeddings are stored in a chat-specific ChromaDB collection.
    """
    results = []
    total_chunks = 0
    qa_service = get_qa_service()
    
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            logger.warning(f"[API] Skipping non-PDF file: {file.filename}")
            continue

        try:
            start_time = time.time()
            file_bytes = await file.read()
            
            if len(file_bytes) == 0:
                logger.warning(f"[API] Skipping empty file: {file.filename}")
                continue
                
            # Run ingestion scoped to this chat's collection
            result = await asyncio.to_thread(
                qa_service.upload_document, file_bytes, file.filename, chat_id
            )
            
            upload_res = UploadResponse(
                message=f"Successfully processed '{file.filename}'.",
                filename=result["filename"],
                pages_extracted=result["pages_extracted"],
                chunks_created=result["chunks_created"],
            )
            results.append(upload_res)
            total_chunks += result["chunks_created"]
            
            elapsed = time.time() - start_time
            logger.info(f"[API] Processed {file.filename} in {elapsed:.2f}s (chat={chat_id or 'GLOBAL'})")
            
        except Exception as e:
            logger.error(f"[API] Failed to process {file.filename}: {e}", exc_info=True)
            continue

    if not results:
        raise HTTPException(status_code=400, detail="No valid PDF files were uploaded or processed.")

    return MultiUploadResponse(results=results, total_chunks=total_chunks)


@app.post("/ingest-text", response_model=UploadResponse)
async def ingest_text_endpoint(request: TextIngestRequest):
    """
    Ingest raw text into a chat-specific collection.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        qa_service = get_qa_service()
        result = await asyncio.to_thread(
            qa_service.ingest_text, request.text, request.filename, request.chat_id
        )
        
        return UploadResponse(
            message=f"Successfully processed text as '{request.filename}'.",
            filename=result["filename"],
            pages_extracted=result["pages_extracted"],
            chunks_created=result["chunks_created"],
        )
    except Exception as e:
        logger.error(f"[API] Text ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Ask a question scoped to a specific chat's documents only.
    Only the provided chat_history is sent to the LLM.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    MAX_QUERY_LENGTH = 1000
    if len(request.query) > MAX_QUERY_LENGTH:
        logger.warning(f"[API] Query rejected: too long ({len(request.query)} chars).")
        raise HTTPException(
            status_code=400, 
            detail=f"Query exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters."
        )

    logger.info(
        f"[API] Query received: '{request.query}' | "
        f"Chat: {request.chat_id or 'GLOBAL'} | "
        f"File: {request.filename or 'ALL'} | "
        f"History: {len(request.chat_history)} messages"
    )

    try:
        start_time = time.time()
        qa_service = get_qa_service()
        
        result = await asyncio.to_thread(
            qa_service.ask_question,
            request.query.strip(),
            request.filename,
            request.chat_history,
            request.chat_id,
        )
        
        formatted_sources = []
        for s in result["sources"]:
            text = s["text"]
            truncated_text = text[:300] + "..." if len(text) > 300 else text
            formatted_sources.append(SourceInfo(
                page=s["page"],
                text=truncated_text,
                file=s["file"],
                relevance_score=s["relevance_score"]
            ))

        elapsed = time.time() - start_time
        logger.info(f"[API] Query processed in {elapsed:.2f}s (chat={request.chat_id or 'GLOBAL'})")

        return QueryResponse(
            answer=result["answer"],
            sources=formatted_sources,
        )

    except RuntimeError as e:
        logger.error(f"[API] Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[API] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 60)
    logger.info("  VectorMind - Document QA System Starting...")
    logger.info("  Session isolation: ENABLED (per-chat ChromaDB collections)")
    logger.info("=" * 60)
