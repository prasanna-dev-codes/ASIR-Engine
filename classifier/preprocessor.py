"""
classifier/preprocessor.py  (FIXED VERSION)
============================================
Steps 2–5 of the query classifier pipeline.

FIXES IN THIS VERSION vs ORIGINAL:
───────────────────────────────────────────────────────────────────────
FIX 1 — No-Retrieval Gate: Definition Starters Removed
  PROBLEM:  "what is X?" and "define X" queries were triggering the
            no-retrieval gate because they are short and entity-free.
            This incorrectly classified "What is photosynthesis?",
            "What is RAM?", "Define machine learning" as no_retrieval.

  ROOT CAUSE: The gate assumed short + no-entities = general knowledge.
              But "What is photosynthesis?" is ALWAYS a domain query
              that needs document retrieval — it is a factual query,
              not a no-retrieval query.

  FIX:      Removed definition starters ("what is", "define", etc.)
            from the no-retrieval gate entirely.
            These phrases always indicate factual queries.
            Only genuinely answerable-without-documents queries remain:
              - Arithmetic expressions (2+2, 10% of 200)
              - Abbreviation expansions (what does X stand for)
              - Unit conversions (5 km to meters)
              - System meta-queries (what can you do)

FIX 2 — No-Retrieval Gate: Math Detection Improved
  PROBLEM:  "What is 2+2?", "What is 7 times 8?", "What is 10% of 200?"
            were NOT caught by the gate because:
            (a) spaCy tokenises "2+2" as 3 tokens → total > 5 token limit
            (b) the math terms list had no arithmetic operators or
                natural language math phrases

  FIX:      Added regex-based arithmetic detection:
              - Detects digit+operator+digit patterns: 2+2, 10%200, 15-3
              - Detects "X times Y", "X divided by Y", "X squared"
              - Detects "X% of Y" percentage queries
            Token count threshold removed for math detection —
            math is detected by content, not length.

FIX 3 — No-Retrieval Gate: Abbreviation Queries
  PROBLEM:  "What does CPU stand for?" (6 tokens > 5 threshold) was
            not caught.

  FIX:      Added "stand for", "stands for", "full form of",
            "abbreviation of" as explicit no-retrieval triggers,
            applied regardless of token count.
───────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import time
import logging
from typing import Optional

import spacy

import config as C
from classifier.text_cleaner import clean_query
from shared.models import NamedEntity, PreprocessorOutput, RawQuery

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Load spaCy model once — reused for every query
# ─────────────────────────────────────────────────────────────────────
try:
    _NLP = spacy.load("en_core_web_sm")
    log.info("spaCy model loaded: en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found.\n"
        "Run: python -m spacy download en_core_web_sm"
    )

# ─────────────────────────────────────────────────────────────────────
# FIX 2: Compiled regex patterns for math detection.
# Applied in _check_no_retrieval() regardless of token count.
# ─────────────────────────────────────────────────────────────────────

# Matches: "2+2", "10-5", "15*3", "100/4", "10%200"
_ARITHMETIC_PATTERN = re.compile(
    r'\d+\s*[\+\-\*\/\%×÷]\s*\d+'
)

# Matches: "10 percent of 200", "15% of 300", "50 percent of 100"
_PERCENTAGE_OF_PATTERN = re.compile(
    r'\d+\s*(?:percent|%)\s+of\s+\d+'
)

# Matches: "7 times 8", "5 times 9", "3 times 4"
_TIMES_PATTERN = re.compile(
    r'\d+\s+times\s+\d+'
)

# Matches: "100 divided by 4", "50 divided by 5"
_DIVIDED_BY_PATTERN = re.compile(
    r'\d+\s+divided\s+by\s+\d+'
)

# Matches: "square of 5", "square root of 16", "cube of 3"
_POWER_PATTERN = re.compile(
    r'(?:square(?:\s+root)?|cube(?:\s+root)?|sqrt)\s+of\s+\d+'
)

# Matches: "5 squared", "3 cubed"
_SQUARED_PATTERN = re.compile(
    r'\d+\s+(?:squared|cubed)'
)

# FIX 3: Abbreviation expansion phrases — no-retrieval regardless of length
_ABBREVIATION_PATTERNS = [
    "stand for",
    "stands for",
    "full form of",
    "full form",
    "abbreviation of",
    "abbreviation for",
    "short form of",
    "acronym for",
]

# Natural-language math trigger words (token-count independent)
_MATH_NATURAL_TERMS = [
    "plus", "minus", "multiplied by", "divided by", "times",
    "to the power", "squared", "cubed", "factorial",
    "square root of", "cube root of",
]


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def preprocess(raw_query: RawQuery) -> PreprocessorOutput:
    """
    Run all preprocessing steps on a raw query.

    This is the only function called from outside this module.

    Parameters
    ----------
    raw_query : RawQuery

    Returns
    -------
    PreprocessorOutput
    """
    t_start = time.perf_counter()

    # Step 1: Clean text
    cleaned_text = clean_query(raw_query.raw_text)

    # Step 2: spaCy analysis
    doc = _NLP(cleaned_text)

    structural              = _extract_structural_features(doc, cleaned_text)
    q_word, q_sig           = _extract_question_word(doc, cleaned_text)
    comp_found, comp_sig    = _extract_comparative_markers(cleaned_text)
    rel_found,  rel_sig     = _extract_relational_markers(cleaned_text)
    entities                = _extract_named_entities(doc)
    ent_sig                 = _compute_entity_signal(entities)
    root_verb, action_verbs, verb_sig = _extract_verb_signals(doc)
    len_cat, len_sig        = _compute_length_signal(structural["token_count"])

    # Step 3: No-retrieval gate (FIXED — see module docstring)
    no_ret, no_ret_reason = _check_no_retrieval(cleaned_text, structural["token_count"])

    # Step 4: Signal aggregation → Layer 1 scores
    all_signals = {
        "question_word":      q_sig,
        "comparative_marker": comp_sig,
        "relational_marker":  rel_sig,
        "entity_type":        ent_sig,
        "verb_type":          verb_sig,
        "query_length":       len_sig,
    }
    layer1_scores             = _aggregate_signals(all_signals)
    l1_type, l1_conf, l1_ok  = _evaluate_layer1(layer1_scores)

    # Step 5: Decomposition check
    needs_decomp, sub_queries = _check_decomposition(cleaned_text, doc, structural)

    t_end = time.perf_counter()

    return PreprocessorOutput(
        query_id=raw_query.query_id,
        session_id=raw_query.session_id,
        raw_text=raw_query.raw_text,
        timestamp=raw_query.timestamp,
        cleaned_text=cleaned_text,

        token_count=structural["token_count"],
        sentence_count=structural["sentence_count"],
        is_question=structural["is_question"],
        is_multi_part=structural["is_multi_part"],

        question_word=q_word,
        question_word_signal=q_sig,

        comparative_markers_found=comp_found,
        comparative_marker_signal=comp_sig,
        relational_markers_found=rel_found,
        relational_marker_signal=rel_sig,

        named_entities=entities,
        entity_type_signal=ent_sig,

        root_verb=root_verb,
        action_verbs_found=action_verbs,
        verb_type_signal=verb_sig,

        length_category=len_cat,
        length_signal=len_sig,

        no_retrieval_triggered=no_ret,
        no_retrieval_reason=no_ret_reason,

        needs_decomposition=needs_decomp,
        sub_queries=sub_queries,

        layer1_predicted_type=l1_type,
        layer1_confidence=l1_conf,
        layer1_all_scores=layer1_scores,
        layer1_sufficient=l1_ok,

        preprocessing_time_ms=round((t_end - t_start) * 1000, 2),
    )


# ─────────────────────────────────────────────────────────────────────
# Step 2 helpers — linguistic feature extraction
# ─────────────────────────────────────────────────────────────────────

def _extract_structural_features(doc, cleaned_text: str) -> dict:
    token_count    = len([t for t in doc if not t.is_space])
    sentence_count = len(list(doc.sents))

    first_token   = doc[0].text.lower() if len(doc) > 0 else ""
    ends_with_q   = cleaned_text.rstrip().endswith("?")
    starts_with_q = first_token in C.QUESTION_WORD_SIGNALS

    is_question  = ends_with_q or starts_with_q

    is_multi_part = False
    if sentence_count >= C.DECOMPOSITION_MIN_SENTENCES:
        root_verbs = [
            t for sent in doc.sents
            for t in sent
            if t.dep_ == "ROOT" and t.pos_ == "VERB"
        ]
        is_multi_part = len(root_verbs) >= 2

    return {
        "token_count":    token_count,
        "sentence_count": sentence_count,
        "is_question":    is_question,
        "is_multi_part":  is_multi_part,
    }


def _extract_question_word(doc, cleaned_text: str) -> tuple[str, dict]:
    if len(doc) == 0:
        return "", {}

    first_token = doc[0].text.lower()
    if first_token in C.QUESTION_WORD_SIGNALS:
        return first_token, C.QUESTION_WORD_SIGNALS[first_token]

    if len(doc) > 1:
        second_token = doc[1].text.lower()
        if second_token in C.QUESTION_WORD_SIGNALS:
            return second_token, C.QUESTION_WORD_SIGNALS[second_token]

    return "", {}


def _extract_comparative_markers(cleaned_text: str) -> tuple[list, dict]:
    found    = []
    max_conf = 0.0

    for marker, confidence in C.COMPARATIVE_MARKERS.items():
        if marker in cleaned_text:
            found.append(marker)
            max_conf = max(max_conf, confidence)

    if not found:
        return [], {}
    return found, {"comparative": max_conf}


def _extract_relational_markers(cleaned_text: str) -> tuple[list, dict]:
    found    = []
    max_conf = 0.0

    for marker, confidence in C.RELATIONAL_MARKERS.items():
        if marker in cleaned_text:
            found.append(marker)
            max_conf = max(max_conf, confidence)

    if not found:
        return [], {}
    return found, {"relational": max_conf}


def _extract_named_entities(doc) -> list[NamedEntity]:
    return [
        NamedEntity(
            text=ent.text,
            label=ent.label_,
            start_char=ent.start_char,
            end_char=ent.end_char,
        )
        for ent in doc.ents
    ]


def _compute_entity_signal(entities: list[NamedEntity]) -> dict:
    if not entities:
        return {}

    type_counts: dict[str, int] = {}
    for ent in entities:
        type_counts[ent.label] = type_counts.get(ent.label, 0) + 1

    dominant_type  = max(type_counts, key=type_counts.get)
    dominant_count = type_counts[dominant_type]

    if dominant_count >= 2 and dominant_type in C.ENTITY_TYPE_SIGNALS_MULTIPLE:
        return C.ENTITY_TYPE_SIGNALS_MULTIPLE[dominant_type]
    elif dominant_type in C.ENTITY_TYPE_SIGNALS_SINGLE:
        return C.ENTITY_TYPE_SIGNALS_SINGLE[dominant_type]

    return {}


def _extract_verb_signals(doc) -> tuple[str, list, dict]:
    root_verb    = ""
    action_verbs = []

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root_verb = token.lemma_.lower()
            break

    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() in C.VERB_TYPE_SIGNALS:
            action_verbs.append(token.lemma_.lower())

    signal = {}
    if root_verb in C.VERB_TYPE_SIGNALS:
        signal = C.VERB_TYPE_SIGNALS[root_verb]
    elif action_verbs:
        signal = C.VERB_TYPE_SIGNALS.get(action_verbs[0], {})

    return root_verb, action_verbs, signal


def _compute_length_signal(token_count: int) -> tuple[str, dict]:
    if token_count <= C.LENGTH_KEYWORD_MAX:
        return "keyword", {"factual": 0.55, "exploratory": 0.45}
    elif token_count <= C.LENGTH_SHORT_MAX:
        return "short", {"factual": 0.55, "exploratory": 0.45}
    elif token_count <= C.LENGTH_STANDARD_MAX:
        return "standard", {"factual": 0.45, "exploratory": 0.55}
    else:
        return "long", {"exploratory": 0.65, "relational": 0.35}


# ─────────────────────────────────────────────────────────────────────
# Step 3: No-retrieval gate (FIXED)
# ─────────────────────────────────────────────────────────────────────

def _check_no_retrieval(
    cleaned_text: str,
    token_count:  int,
) -> tuple[bool, str]:
    """
    Determine if this query needs no document retrieval.

    IMPORTANT DESIGN DECISION (fixes the factual/no_retrieval confusion):
    ─────────────────────────────────────────────────────────────────────
    "What is X?" and "Define X" queries are REMOVED from this gate.

    Reason: these phrases always indicate factual queries that benefit
    from document retrieval. "What is photosynthesis?" and
    "What is RAM?" both need a knowledge base to answer correctly in
    the context of a RAG system.

    The no-retrieval gate now only catches queries that are PROVABLY
    answerable without any documents:
      1. Arithmetic expressions (2+2, 10% of 200, 7 times 8)
      2. Abbreviation expansions (what does X stand for?)
      3. Unit conversions (convert 5 km to meters)
      4. System meta-queries (what can you do?)

    Parameters
    ----------
    cleaned_text : str   — lowercased, normalised query text
    token_count  : int   — spaCy token count (not used as gate for math)

    Returns
    -------
    (triggered: bool, reason: str)
    """

    # ── Condition 1: Arithmetic expression ───────────────────────────
    # Detected by regex — no token count limit needed.
    # Examples: "What is 2+2?", "What is 10% of 200?", "15-3"
    if _ARITHMETIC_PATTERN.search(cleaned_text):
        return True, "arithmetic_expression"

    if _PERCENTAGE_OF_PATTERN.search(cleaned_text):
        return True, "percentage_calculation"

    if _TIMES_PATTERN.search(cleaned_text):
        return True, "multiplication"

    if _DIVIDED_BY_PATTERN.search(cleaned_text):
        return True, "division"

    if _POWER_PATTERN.search(cleaned_text):
        return True, "power_or_root"

    if _SQUARED_PATTERN.search(cleaned_text):
        return True, "power_expression"

    # ── Condition 2: Natural language math terms ──────────────────────
    # Only when digits are also present — prevents "times" in non-math
    # contexts from triggering (e.g. "How many times did India win?")
    has_digit = bool(re.search(r'\d', cleaned_text))
    if has_digit:
        for term in _MATH_NATURAL_TERMS:
            if term in cleaned_text:
                return True, f"math_natural_language:{term}"

    # ── Condition 3: Explicit math keyword queries (no digit needed) ──
    # These are unambiguously math regardless of digits
    explicit_math = [
        "what is the formula",
        "what is the value of pi",
        "value of pi",
        "what is pi",
    ]
    for phrase in explicit_math:
        if phrase in cleaned_text:
            return True, f"explicit_math:{phrase}"

    # ── Condition 4: Abbreviation / acronym expansion ─────────────────
    # "What does CPU stand for?", "What does HTTP stand for?"
    # Token count NOT checked — these can be 6+ tokens
    for phrase in _ABBREVIATION_PATTERNS:
        if phrase in cleaned_text:
            return True, f"abbreviation:{phrase}"

    # ── Condition 5: Unit conversion ──────────────────────────────────
    # "Convert 5 km to meters", "Convert 1 hour to minutes"
    # Only when digits are present AND "convert" is the verb
    if "convert" in cleaned_text and has_digit:
        return True, "unit_conversion"

    # ── Condition 6: System meta-queries ─────────────────────────────
    for phrase in C.NO_RETRIEVAL_SYSTEM_PHRASES:
        if phrase in cleaned_text:
            return True, "system_meta"

    return False, ""


# ─────────────────────────────────────────────────────────────────────
# Step 4: Signal aggregation → Layer 1 scores
# ─────────────────────────────────────────────────────────────────────

def _aggregate_signals(all_signals: dict) -> dict:
    scores = {qt: 0.0 for qt in C.VALID_QUERY_TYPES if qt != "no_retrieval"}

    for signal_name, signal_scores in all_signals.items():
        if not signal_scores:
            continue

        weight = C.SIGNAL_WEIGHTS.get(signal_name, 1.0)

        for query_type, value in signal_scores.items():
            if query_type in scores:
                scores[query_type] += value * weight

    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 4) for k, v in scores.items()}
    else:
        n      = len(scores)
        scores = {k: round(1.0 / n, 4) for k in scores}

    return scores


def _evaluate_layer1(layer1_scores: dict) -> tuple[str, float, bool]:
    if not layer1_scores:
        return "exploratory", 0.0, False

    sorted_scores = sorted(
        layer1_scores.items(), key=lambda x: x[1], reverse=True
    )
    top_type, top_conf   = sorted_scores[0]
    second_conf          = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    gap                  = top_conf - second_conf

    sufficient = (
        top_conf >= C.LAYER1_CONFIDENCE_THRESHOLD
        and gap  >= C.LAYER1_GAP_THRESHOLD
    )

    return top_type, round(top_conf, 4), sufficient


# ─────────────────────────────────────────────────────────────────────
# Step 5: Multi-part query decomposition
# ─────────────────────────────────────────────────────────────────────

def _check_decomposition(
    cleaned_text: str,
    doc,
    structural: dict,
) -> tuple[bool, list[str]]:
    token_count    = structural["token_count"]
    sentence_count = structural["sentence_count"]

    if token_count < C.DECOMPOSITION_MIN_TOKENS:
        return False, []

    # Trigger a: multiple sentences with distinct root verbs
    if sentence_count >= C.DECOMPOSITION_MIN_SENTENCES:
        sentences       = list(doc.sents)
        root_verb_sents = [
            s for s in sentences
            if any(t.dep_ == "ROOT" and t.pos_ == "VERB" for t in s)
        ]
        if len(root_verb_sents) >= 2:
            sub_queries = [s.text.strip() for s in sentences if s.text.strip()]
            if len(sub_queries) >= 2:
                return True, sub_queries

    # Trigger b: two distinct question words in a long query
    found_q_words = [
        w for w in C.QUESTION_WORD_SIGNALS
        if w in cleaned_text.split()
    ]
    if len(set(found_q_words)) >= 2:
        parts = _split_on_conjunction(cleaned_text)
        if len(parts) >= 2:
            return True, parts

    # Trigger c: explicit decomposition conjunction phrase
    for conj in C.DECOMPOSITION_CONJUNCTIONS:
        if conj in cleaned_text:
            parts = cleaned_text.split(conj, 1)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) == 2:
                return True, parts

    return False, []


def _split_on_conjunction(text: str) -> list[str]:
    if " and " not in text:
        return [text]

    parts = text.split(" and ", 1)
    parts = [p.strip() for p in parts]

    if all(len(p.split()) >= 4 for p in parts):
        return parts

    return [text]
