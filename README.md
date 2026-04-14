# VectorMind — Backend ⚙️

This directory contains the FastAPI Application Layer, Vector Database, and LLM orchestration logic serving the Document QA system.

## Architecture Highlights
- **FastAPI Core (`main.py`)**: Asynchronous REST framework utilizing endpoints for `/upload`, `/query`, and `/health`. Pydantic models validate constraints (e.g. 10MB limits, 1,000 char queries).
- **Hybrid Search Engine (`vector_store.py`)**: 
  - *Dense*: Uses `sentence-transformers` and `chromadb`.
  - *Sparse*: Built on `rank_bm25` directly executed from tokenized text.
- **Cross-Encoder Reranking (`reranker.py`)**: All retrieved chunks from the hybrid index are mapped through a massive `bge-reranker-base` cross-encoder to guarantee precision before being allowed into the LLM context.
- **Defensive LLM (`generator.py`)**: Employs heavily structured OpenAI/OpenRouter Prompts requiring explicit citations strings. Any query resulting in 0 relevant contexts falls back to a standardized refusal mechanism.

## Quick Start
1. Ensure your IDE is targeting Python 3.9+ and your virtual environment is active.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your variables! Check `config.py` and drop your `.env` into this folder:
   ```env
   OPENAI_API_KEY="sk-or-v1-..."
   ```
4. Start the Application:
   ```bash
   uvicorn app.main:app --reload
   ```

*The frontend layer requires this backend to be operating cleanly on PORT 8000.*
