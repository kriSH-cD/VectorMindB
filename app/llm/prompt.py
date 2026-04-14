from __future__ import annotations

"""
VectorMind - Prompt Engineering (Production Grade)
=====================================================
Constructs the system and user prompts for the LLM.

Prompt design principles:
  1. STRICT grounding — the LLM must ONLY use the provided context.
  2. MANDATORY inline citations — every sentence must cite (Page X, filename.pdf).
  3. STRUCTURED output — Answer + Sources sections for frontend parsing.
  4. PARTIAL answer support — answer what's available, flag what's missing.
  5. EXPLICIT fallback — if nothing found, return standardized message.
  6. CONTEXT truncation — each chunk capped at MAX_CHUNK_CHARS to prevent overflow.

Production Improvements (v2):
  #1  Structured output format (Answer + Sources sections)
  #2  Inline citation enforcement (every sentence, not just end)
  #3  Stronger anti-hallucination language
  #4  Partial answer handling instruction
  #5  Context chunk truncation (800-1000 chars)
  #6  Source snippet extraction instruction (max 2 lines)
  #8  Standardized fallback constant
"""

# ────────────────────────────────────────────────────────────────
# IMPROVEMENT #8: Standardized Fallback Message
# ────────────────────────────────────────────────────────────────
# Single source of truth — used by prompt, generator, and service layer.
# Prevents inconsistent fallback wording across the codebase.
# ────────────────────────────────────────────────────────────────
FALLBACK_RESPONSE = (
    "The answer to your question was not found in the uploaded documents."
)

# ────────────────────────────────────────────────────────────────
# IMPROVEMENT #5: Maximum characters per context chunk
# ────────────────────────────────────────────────────────────────
# Prevents prompt overflow and keeps context focused.
# 1000 chars ≈ 200-250 tokens — safe for up to 5 chunks.
# ────────────────────────────────────────────────────────────────
MAX_CHUNK_CHARS = 1000


# ────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Production Grade
# ────────────────────────────────────────────────────────────────
# Improvements applied:
#   #1  Structured output format enforced
#   #2  Inline citation requirement (every sentence)
#   #3  Stronger anti-hallucination ("MUST NOT answer")
#   #4  Partial answer handling
#   #6  Source snippet extraction (max 2 lines)
# ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are VectorMind, an expert document question-answering assistant.

## RULES (STRICT — NEVER VIOLATE):

1. **CONTEXT-ONLY ANSWERS**: You must ONLY use the provided context to answer the question. If information is not explicitly present in the context, you MUST NOT answer it. Do NOT use any prior knowledge, training data, or external information under any circumstances.

2. **MANDATORY INLINE CITATIONS**: Each factual sentence in your answer MUST include at least one citation in the format (Page X, filename.pdf). Do NOT group all citations at the end. Every claim must be individually traceable to its source.

3. **PARTIAL ANSWERS**: If only PART of the question can be answered from the context:
   - Answer the part that IS supported by the context (with citations).
   - For the missing part, explicitly state: "This information is not available in the provided documents."
   - Do NOT guess or infer the missing information.

4. **COMPLETE FALLBACK**: If NO part of the answer exists in the provided context, respond EXACTLY with:
   "The answer to your question was not found in the uploaded documents."
   Do NOT attempt to guess, infer, or fabricate any part of the answer.

5. **CONVERSATIONAL CONTEXT**: You must use the provided conversation history to understand references like 'above', 'that', 'previous answer', or 'it'. If the reference remains unclear even with history, clearly state that you do not understand the reference.

6. **PRECISION & CONCISENESS**: Answer the question directly. No unnecessary preamble, summaries of the question, or filler text.

7. **FACTUAL ACCURACY**: Do not paraphrase in a way that changes the meaning of the source material. Preserve the original intent and facts.

## REQUIRED OUTPUT FORMAT:

You MUST structure your response in EXACTLY this format:

Answer:
<your answer with inline citations (Page X, filename.pdf) in every sentence>

Sources:
- (Page X, filename.pdf): <short supporting snippet, max 2 lines>
- (Page Y, filename.pdf): <short supporting snippet, max 2 lines>

IMPORTANT:
- The "Answer:" section contains your full response with inline citations.
- The "Sources:" section lists each unique source with a SHORT snippet (1-2 lines max) that supports the answer.
- Do NOT dump entire paragraphs in the Sources section.
- If using the fallback message, do NOT include a Sources section.
"""


def build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a clearly labeled context block.

    Each chunk is wrapped in a header showing its source file and page number,
    making it easy for the LLM to produce accurate citations.

    IMPROVEMENT #5: Each chunk's text is truncated to MAX_CHUNK_CHARS
    to prevent prompt overflow and keep context focused. Long chunks
    often contain irrelevant trailing text that dilutes relevance.

    Args:
        chunks: List of dicts with keys: text, page, file.

    Returns:
        A formatted string containing all context chunks.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        # ── IMPROVEMENT #5: Truncate long chunks ──
        # Cap each chunk at MAX_CHUNK_CHARS to prevent token overflow.
        # Append "..." if truncated to signal the LLM that text continues.
        text = chunk["text"]
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + "..."

        context_parts.append(
            f"--- Source {i}: {chunk['file']}, Page {chunk['page']} ---\n"
            f"{text}\n"
        )

    return "\n".join(context_parts)


def build_user_prompt(query: str, chunks: list[dict], chat_history: list[dict] | None = None) -> str:
    """
    Build the full user prompt with conversation history context and question.

    Args:
        query:        The user's natural language question.
        chunks:       Retrieved and reranked context chunks.
        chat_history: Optional list of previous message dicts [{"role": "user", "content": "..."}, ...]

    Returns:
        The complete user message to send to the LLM.
    """
    context_block = build_context_block(chunks)
    
    history_block = ""
    if chat_history and len(chat_history) > 0:
        # Take the last 5 messages to avoid overflow
        recent_history = chat_history[-5:]
        lines = []
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if content:
                lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)
        if history_text:
            history_block = f"## CONVERSATION HISTORY:\n{history_text}\n\n"

    return (
        f"{history_block}"
        f"## CONTEXT:\n\n"
        f"{context_block}\n"
        f"## QUESTION:\n\n"
        f"{query}\n\n"
        f"## INSTRUCTIONS:\n"
        f"1. Answer the question using ONLY the context above.\n"
        f"2. EVERY factual sentence must include a citation: (Page X, filename.pdf).\n"
        f"3. If only part of the answer is available, answer that part and state "
        f"what is missing.\n"
        f"4. If the answer is NOT in the context at all, say EXACTLY:\n"
        f'   "{FALLBACK_RESPONSE}"\n'
        f"5. Use the required output format: Answer: ... Sources: ...\n"
        f"6. In the Sources section, include SHORT snippets only (max 2 lines each)."
    )
