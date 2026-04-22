from __future__ import annotations

"""
VectorMind - LLM Generator (Production Grade)
================================================
Connects to the OpenAI-compatible API (GPT-4o via OpenRouter) to generate
grounded answers from retrieved document chunks.

The generator:
  1. Receives reranked context chunks from the retriever.
  2. Constructs a system prompt (strict grounding rules) + user prompt
     (context + question) using the prompt module.
  3. Calls the Chat Completions API.
  4. Validates the response (citation check, empty check).
  5. Returns the validated answer text.

Production Improvements (v2):
  #7  Defensive post-processing (citation validation, empty check)
  #8  Standardized fallback message from prompt.py constant
  #9  Empty context guard (no LLM call if no context)
  #10 Low temperature (0.1) + controlled penalties
  #11 All existing architecture preserved
"""

import logging
import re
from functools import lru_cache

from openai import OpenAI

from app.config import get_settings
from app.llm.prompt import SYSTEM_PROMPT, FALLBACK_RESPONSE, build_user_prompt

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Generates answers from context using OpenAI's Chat Completions API.
    Includes defensive post-processing to catch hallucinations and missing citations.
    """

    def __init__(self):
        settings = get_settings()

        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )

        self.client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = settings.groq_model
        logger.info(f"LLM Generator initialized (model={self.model}).")

    def _validate_response(self, answer: str) -> str:
        """
        Defensive post-processing to validate the LLM's response.

        IMPROVEMENT #7: Three validation checks:
          A. Empty answer           → return fallback
          B. Missing citations      → return fallback (likely hallucination)
          C. Fallback already given → pass through as-is

        Args:
            answer: Raw LLM response string.

        Returns:
            Validated answer string, or fallback message if validation fails.
        """
        # ── Check A: Empty or whitespace-only response ──
        if not answer or not answer.strip():
            logger.warning("[LLM] Empty response received — returning fallback.")
            return FALLBACK_RESPONSE

        # ── Check C: LLM already returned the fallback message ──
        # If the LLM correctly determined there's no answer, pass it through.
        # Use case-insensitive partial match to handle slight wording variations.
        if "not found in the uploaded documents" in answer.lower() or \
           "not available in the provided documents" in answer.lower():
            logger.info("[LLM] LLM returned fallback/partial-fallback — passing through.")
            return answer

        # ── Check B: Citation validation ──
        # A properly grounded answer MUST contain at least one "(Page" citation.
        # If no citation pattern is found, the LLM likely hallucinated or ignored
        # the grounding instructions — reject the response.
        citation_pattern = re.compile(r"\(Page\s+\d+", re.IGNORECASE)
        if not citation_pattern.search(answer):
            logger.warning(
                "[LLM] Response contains no citations — suspected hallucination. "
                "Returning fallback."
            )
            logger.debug(f"[LLM] Rejected response (no citations): {answer[:200]}...")
            return FALLBACK_RESPONSE

        return answer

    def generate(self, query: str, context_chunks: list[dict], chat_history: list[dict] | None = None) -> str:
        """
        Generate a grounded answer for the user's query, with conversation history.

        Flow:
          1. Guard: return fallback if no context chunks.
          2. Build prompt with context, question, and history.
          3. Call LLM API with low temperature.
          4. Validate response (citations, emptiness).
          5. Return validated answer.

        Args:
            query:          The user's natural language question.
            context_chunks: Reranked chunks with keys: text, page, file.
            chat_history:   Optional list of message dicts (role, content) for multi-turn.

        Returns:
            The LLM's validated answer string with citations,
            or the FALLBACK_RESPONSE if validation fails.
        """
        # ── IMPROVEMENT #9: Empty context guard ──
        # Never call the LLM if there's no context — it can only hallucinate.
        if not context_chunks:
            logger.info("[LLM] No context chunks provided — returning fallback.")
            return FALLBACK_RESPONSE

        # Build the user prompt with context and history
        user_prompt = build_user_prompt(query, context_chunks, chat_history=chat_history)

        logger.info(
            f"[LLM] Sending query to {self.model} "
            f"with {len(context_chunks)} context chunks."
        )

        try:
            # ── IMPROVEMENT #10: Controlled generation parameters ──
            # temperature=0.1  → Highly deterministic for factual accuracy
            # top_p=0.9        → Slight nucleus sampling for natural language
            # frequency_penalty=0.0 → No penalty (we want precise repetition of facts)
            # presence_penalty=0.0  → No penalty (citations may repeat page numbers)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=512,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            raw_answer = response.choices[0].message.content.strip()
            logger.info(f"[LLM] Raw answer received ({len(raw_answer)} chars).")

            # ── IMPROVEMENT #7: Defensive post-processing ──
            validated_answer = self._validate_response(raw_answer)

            if validated_answer != raw_answer:
                logger.warning("[LLM] Response was rejected by validation — fallback used.")
            else:
                logger.info("[LLM] Response passed validation.")

            return validated_answer

        except Exception as e:
            logger.error(f"[LLM] Error during generation: {e}")
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e


@lru_cache()
def get_generator() -> LLMGenerator:
    """Return a cached singleton LLMGenerator instance."""
    return LLMGenerator()
