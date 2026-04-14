
"""
VectorMind - FastAPI Application
===================================
Main entry point for the Document QA API.

Endpoints:
  POST /upload  — Upload a PDF document for ingestion.
  POST /query   — Ask a question against uploaded documents.
  GET  /health  — Health check endpoint.
  GET  /stats   — Get system statistics (doc count, etc.).

CORS is enabled for the React frontend running on localhost:3000.
"""

import asyncio
import logging
import time

from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.qa_service import get_qa_service
from app.db.vector_store import get_vector_store
from app.retrieval.retriever import get_retriever

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
    version="1.0.0",
)

# ── CORS — allow React frontend ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ──
class QueryRequest(BaseModel):
    """Schema for the /query endpoint request body."""
    query: str
    filename: Optional[str] = None   # Optional: restrict search to a specific PDF
    chat_history: list[dict] = []    # Optional: previous chat history for context


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
    """Schema for the /upload endpoint response."""
    message: str
    filename: str
    pages_extracted: int
    chunks_created: int


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
def get_stats():
    """Get system statistics — total chunks stored in the vector DB."""
    try:
        store = get_vector_store()
        count = store.get_document_count()
        return StatsResponse(total_chunks=count, status="operational")
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
def clear_memory():
    """Wipe all stored documents from ChromaDB and BM25 to start a fresh RAG session."""
    try:
        # Clear global persistent vector store
        store = get_vector_store()
        store.clear_all()
        
        # Explicitly empty out the BM25 dictionary tracker
        retriever = get_retriever()
        retriever._rebuild_bm25_index()
        
        logger.info("[API] Successfully cleared all RAG memory.")
        return {"message": "Memory cleared completely."}
    except Exception as e:
        logger.error(f"[API] Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for processing and storage.

    The file is:
      1. Validated (must be a PDF).
      2. Read into memory (no disk storage).
      3. Processed through the ingestion pipeline:
         parse → chunk → embed → store in ChromaDB.
      4. BM25 index is rebuilt to include new chunks.
    """
    # ── Validation ──
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    logger.info(f"[API] Upload request received: {file.filename}")

    try:
        start_time = time.time()
        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)

        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        # ── Configuration: File Size Validation ──
        MAX_FILE_SIZE_MB = 10
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"[API] Upload rejected: File too large ({file_size_mb:.2f}MB).")
            raise HTTPException(
                status_code=413, 
                detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB."
            )

        logger.info(f"[API] Processing {file.filename} ({file_size_mb:.2f}MB)")

        # Run the ingestion pipeline in a background thread to prevent blocking the event loop
        qa_service = get_qa_service()
        result = await asyncio.to_thread(qa_service.upload_document, file_bytes, file.filename)
        
        elapsed = time.time() - start_time
        logger.info(f"[API] Upload completed in {elapsed:.2f}s: {result['chunks_created']} chunks created.")

        return UploadResponse(
            message=f"Successfully processed '{file.filename}'.",
            filename=result["filename"],
            pages_extracted=result["pages_extracted"],
            chunks_created=result["chunks_created"],
        )

    except ValueError as e:
        # Raised when PDF has no extractable text
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[API] Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Ask a question against uploaded documents.

    The query goes through the full RAG pipeline:
      1. Hybrid retrieval (vector + BM25).
      2. Cross-encoder reranking.
      3. LLM generation with strict citation rules.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    # ── Configuration: Query Length Validation ──
    MAX_QUERY_LENGTH = 1000
    if len(request.query) > MAX_QUERY_LENGTH:
        logger.warning(f"[API] Query rejected: too long ({len(request.query)} chars).")
        raise HTTPException(
            status_code=400, 
            detail=f"Query exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters."
        )

    logger.info(f"[API] Query received: '{request.query}' | File: {request.filename or 'ALL'}")

    try:
        start_time = time.time()
        qa_service = get_qa_service()
        
        # Run QA retrieval/generation in a thread to prevent blocking other requests
        result = await asyncio.to_thread(
            qa_service.ask_question,
            request.query.strip(),
            request.filename,
            request.chat_history,
        )
        
        # ── Configuration: Source Sequence Truncation ──
        # Ensure returned sources explicitly do not bloat the API response
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
        logger.info(f"[API] Query processed in {elapsed:.2f}s")

        return QueryResponse(
            answer=result["answer"],
            sources=formatted_sources,
        )

    except RuntimeError as e:
        # LLM or retrieval errors
        logger.error(f"[API] Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[API] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# ── Startup Event ──
@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 60)
    logger.info("  VectorMind - Document QA System Starting...")
    logger.info("=" * 60)
    
app.add_middleware(
    CORSMiddleware,
    # Replace "*" with actual URL for strict security!
    allow_origins=[
        "https://vector-mind-gamma.vercel.app",  # Production Vercel App
        "http://localhost:5173",                 # Local Vite Standard
        "http://localhost:5174"                  # Local Vite Fallback
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
