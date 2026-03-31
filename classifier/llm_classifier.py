"""
classifier/llm_classifier.py
=============================
Step 6 (Phase 1): LLM-based query classifier using Ollama.

Called when Layer 1 (preprocessor) is not confident enough
to classify alone. Uses a locally-running Llama 3 model via
Ollama to reason about query intent.

WHY LLM FIRST:
  - Works with zero training data on day 1
  - Handles ambiguous and unusual queries well
  - Produces human-readable reasoning for debugging
  - Every call is logged → builds the dataset for XGBoost later

HOW TO RUN OLLAMA:
  1. Install Ollama: https://ollama.ai
  2. Pull model: ollama pull llama3
  3. Start server: ollama serve
  4. Verify: curl http://localhost:11434/api/tags
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import requests

import config as C
from shared.models import PreprocessorOutput

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# The prompt is stored as a constant so it is easy to version and audit.
# Every field in {curly_braces} is filled in at call time.

_SYSTEM_PROMPT = """You are a precise query classifier for a Retrieval-Augmented Generation (RAG) system.
Your classification determines which retrieval technique is used to answer the user's question.
Accuracy is critical — a wrong classification means the wrong documents are retrieved.

CATEGORIES AND THEIR RETRIEVAL TECHNIQUES:

factual
  The query asks for a specific fact, definition, date, name, or discrete piece of information.
  The answer is a single, concrete piece of knowledge.
  Retrieval technique: semantic vector search.
  Examples:
    "What is the boiling point of water?"
    "When was the Eiffel Tower built?"
    "Who invented the telephone?"
    "What does mRNA stand for?"

relational
  The query asks how entities relate to, influence, interact with, or affect each other.
  Answering requires understanding connections between concepts or entities.
  Retrieval technique: knowledge graph traversal.
  Examples:
    "How did colonialism influence India's economy?"
    "What is the relationship between cortisol and stress?"
    "How does deforestation affect rainfall patterns?"
    "What role did antibiotics play in the rise of drug resistance?"

comparative
  The query explicitly compares two or more things, asks for differences, similarities,
  advantages, or trade-offs between options.
  Retrieval technique: hybrid vector + BM25 retrieval.
  Examples:
    "Compare supervised and unsupervised learning"
    "What are the differences between RAM and ROM?"
    "Pros and cons of solar energy vs nuclear energy"
    "Which is better — Python or Java for machine learning?"

exploratory
  The query seeks a broad understanding, overview, or detailed explanation of a topic.
  There is no single specific fact being requested — the user wants to understand a subject.
  Retrieval technique: hierarchical document retrieval.
  Examples:
    "Explain how the immune system works"
    "What are the main causes of climate change?"
    "Give me an overview of quantum computing"
    "How does machine learning work?"

no_retrieval
  The query is simple enough to answer from general knowledge.
  No documents need to be retrieved.
  Examples:
    "What does RAM stand for?"
    "Define photosynthesis in one sentence"
    "What is 15 percent of 200?"
    "What does HTTP stand for?"
"""

_USER_PROMPT_TEMPLATE = """LINGUISTIC SIGNALS ALREADY DETECTED BY THE PREPROCESSOR (use as hints):
  Question word detected : {question_word}
  Question word signal   : {question_word_signal}
  Comparative markers    : {comparative_markers}
  Relational markers     : {relational_markers}
  Named entities         : {named_entities}
  Root verb              : {root_verb}
  Layer 1 scores         : {layer1_scores}
  Token count            : {token_count}

USER QUERY (original, unmodified):
{raw_text}

INSTRUCTIONS:
1. Read the query carefully.
2. Consider the preprocessor signals as hints — use your reasoning to confirm or override them.
3. Select exactly one category from: factual, relational, comparative, exploratory, no_retrieval
4. Assign a confidence value between 0.0 and 1.0 based on how certain you are.
   - 0.95+ : completely unambiguous
   - 0.80–0.94 : clear, minor ambiguity
   - 0.60–0.79 : some ambiguity, best guess
   - below 0.60: genuinely uncertain
5. Write exactly one sentence explaining your choice.
6. Determine if the query contains multiple independent sub-questions that should be answered separately.
   If yes, list them in sub_queries. If no, sub_queries should be an empty list [].

Respond with valid JSON only. No explanation outside the JSON. No markdown code blocks.
{{
    "query_type": "<category>",
    "confidence": <number between 0.0 and 1.0>,
    "reasoning": "<one sentence>",
    "needs_decomposition": <true or false>,
    "sub_queries": ["<sub-query 1>", "<sub-query 2>"]
}}"""


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def classify_with_llm(preprocessor: PreprocessorOutput) -> dict:
    """
    Call the LLM to classify a query that Layer 1 could not confidently classify.

    Parameters
    ----------
    preprocessor : PreprocessorOutput
        Full output from the preprocessor, containing all extracted signals.

    Returns
    -------
    dict with keys:
        query_type       : str
        confidence       : float
        reasoning        : str
        needs_decomposition : bool
        sub_queries      : list[str]
        classified_by    : str  ("layer2_llm")
        llm_raw_response : str  (for logging)
        error            : str  (empty string if no error)
    """
    t_start = time.perf_counter()

    # Build the filled-in user prompt
    user_prompt = _build_user_prompt(preprocessor)

    # Attempt LLM call with one automatic retry on failure
    raw_response = None
    last_error   = ""

    for attempt in range(2):
        try:
            raw_response = _call_ollama(user_prompt)
            break
        except Exception as exc:
            last_error = str(exc)
            log.warning(
                "LLM call attempt %d failed for query_id=%s: %s",
                attempt + 1, preprocessor.query_id, exc
            )
            time.sleep(0.5)

    if raw_response is None:
        log.error(
            "LLM classifier failed after 2 attempts for query_id=%s. "
            "Falling back to Layer 1 result.",
            preprocessor.query_id
        )
        return _build_fallback_result(preprocessor, last_error)

    # Parse the JSON response
    parsed = _parse_llm_response(raw_response, preprocessor)

    t_end = time.perf_counter()
    log.info(
        "LLM classified query_id=%s as '%s' (confidence=%.2f) in %.0fms",
        preprocessor.query_id,
        parsed["query_type"],
        parsed["confidence"],
        (t_end - t_start) * 1000,
    )

    return parsed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(preprocessor: PreprocessorOutput) -> str:
    """Fill the user prompt template with extracted signals."""

    # Format named entities as a readable string
    entities_str = ", ".join(
        f"{e.text} ({e.label})" for e in preprocessor.named_entities
    ) or "none detected"

    return _USER_PROMPT_TEMPLATE.format(
        question_word=preprocessor.question_word or "none",
        question_word_signal=preprocessor.question_word_signal or "{}",
        comparative_markers=preprocessor.comparative_markers_found or "none",
        relational_markers=preprocessor.relational_markers_found or "none",
        named_entities=entities_str,
        root_verb=preprocessor.root_verb or "none",
        layer1_scores=preprocessor.layer1_all_scores,
        token_count=preprocessor.token_count,
        raw_text=preprocessor.raw_text,
    )


def _call_ollama(user_prompt: str) -> str:
    """
    Send a request to the Ollama API and return the raw response string.

    Raises
    ------
    requests.Timeout
        If the LLM does not respond within LLM_TIMEOUT_SEC seconds.
    requests.ConnectionError
        If Ollama is not running.
    ValueError
        If the response has an unexpected structure.
    """
    payload = {
        "model":  C.LLM_MODEL,
        "format": C.LLM_FORMAT,   # Forces JSON output mode in Ollama
        "stream": False,
        "options": {
            "temperature": C.LLM_TEMPERATURE,
            "num_predict": C.LLM_MAX_TOKENS,
        },
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }

    response = requests.post(
        f"{C.LLM_HOST}/api/chat",
        json=payload,
        timeout=C.LLM_TIMEOUT_SEC,
    )
    response.raise_for_status()

    data = response.json()

    # Ollama's /api/chat endpoint returns content in this path
    try:
        content = data["message"]["content"]
    except KeyError as exc:
        raise ValueError(f"Unexpected Ollama response structure: {data}") from exc

    return content


def _parse_llm_response(raw_response: str, preprocessor: PreprocessorOutput) -> dict:
    """
    Parse the LLM's JSON response into a structured dict.

    If parsing fails (malformed JSON, missing fields, invalid values),
    falls back to the Layer 1 result with a logged warning.

    Returns
    -------
    dict with all required fields guaranteed to be present.
    """
    try:
        data = json.loads(raw_response)

        # Validate required fields
        query_type  = data["query_type"]
        confidence  = float(data["confidence"])
        reasoning   = str(data.get("reasoning", ""))
        needs_decomp = bool(data.get("needs_decomposition", False))
        sub_queries  = list(data.get("sub_queries", []))

        # Validate values are within expected ranges
        if query_type not in C.VALID_QUERY_TYPES:
            raise ValueError(f"Invalid query_type from LLM: '{query_type}'")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence out of range: {confidence}")

        # Filter out empty sub-query strings
        sub_queries = [s.strip() for s in sub_queries if s.strip()]

        return {
            "query_type":          query_type,
            "confidence":          round(confidence, 4),
            "reasoning":           reasoning,
            "needs_decomposition": needs_decomp,
            "sub_queries":         sub_queries,
            "classified_by":       "layer2_llm",
            "llm_raw_response":    raw_response,
            "error":               "",
        }

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        log.warning(
            "Failed to parse LLM response for query_id=%s: %s. "
            "Raw response: %s. Using Layer 1 fallback.",
            preprocessor.query_id, exc, raw_response[:200]
        )
        return _build_fallback_result(preprocessor, str(exc))


def _build_fallback_result(
    preprocessor: PreprocessorOutput,
    error_message: str
) -> dict:
    """
    Fallback when LLM fails or returns unparseable output.
    Uses the Layer 1 result with reduced confidence.
    """
    # Reduce confidence to signal this is a fallback result
    fallback_confidence = max(0.0, preprocessor.layer1_confidence - 0.20)

    return {
        "query_type":          preprocessor.layer1_predicted_type,
        "confidence":          round(fallback_confidence, 4),
        "reasoning":           f"LLM fallback — using Layer 1 result. Error: {error_message}",
        "needs_decomposition": preprocessor.needs_decomposition,
        "sub_queries":         preprocessor.sub_queries,
        "classified_by":       "layer2_llm_fallback",
        "llm_raw_response":    "",
        "error":               error_message,
    }
