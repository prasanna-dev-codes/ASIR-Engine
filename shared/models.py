"""
shared/models.py
================
All shared dataclasses for the ASIR-RAG query classifier pipeline.

These are the data contracts between every module.
Every module imports from here — nothing is redefined elsewhere.

Import example:
    from shared.models import RawQuery, ClassifierOutput
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Stage 0 — Raw Query (created at UI submission)
# ---------------------------------------------------------------------------

@dataclass
class RawQuery:
    """
    Created the moment the user submits a query.
    Contains only what the user provided — nothing computed yet.
    """
    query_id:       str   # UUID string, unique per query
    session_id:     str   # identifies the active user session
    raw_text:       str   # exactly what the user typed, unmodified
    timestamp:      str   # ISO 8601 UTC: "2025-03-26T14:32:01Z"
    character_count: int  # len(raw_text)
    word_count:     int   # len(raw_text.split())


# ---------------------------------------------------------------------------
# Stage 1 — Preprocessor Output (after spaCy analysis)
# ---------------------------------------------------------------------------

@dataclass
class NamedEntity:
    """One named entity detected by spaCy."""
    text:        str   # e.g. "India"
    label:       str   # e.g. "GPE", "PERSON", "ORG"
    start_char:  int
    end_char:    int


@dataclass
class PreprocessorOutput:
    """
    Produced by the Preprocessor after full linguistic analysis.
    Contains all extracted signals and the Layer 1 classification result.
    """

    # ── Carried forward ─────────────────────────────────────────────────
    query_id:   str
    session_id: str
    raw_text:   str
    timestamp:  str

    # ── Cleaned text ────────────────────────────────────────────────────
    cleaned_text: str   # normalised, lowercased for analysis

    # ── Structural features ─────────────────────────────────────────────
    token_count:    int
    sentence_count: int
    is_question:    bool   # ends with ? or starts with question word
    is_multi_part:  bool   # multiple independent clauses

    # ── Question word ───────────────────────────────────────────────────
    question_word:            str    # "what", "who", "how", ... or ""
    question_word_signal:     dict   # {"factual": 0.65, "exploratory": 0.35}

    # ── Detected markers ────────────────────────────────────────────────
    comparative_markers_found: list[str]   # e.g. ["vs", "compared to"]
    comparative_marker_signal: dict        # {"comparative": 0.95}
    relational_markers_found:  list[str]   # e.g. ["influence of"]
    relational_marker_signal:  dict        # {"relational": 0.85}

    # ── Named entities ──────────────────────────────────────────────────
    named_entities:    list[NamedEntity]
    entity_type_signal: dict   # {"factual": 0.60, "relational": 0.40}

    # ── Verb analysis ───────────────────────────────────────────────────
    root_verb:         str    # lemma of the root verb, e.g. "compare"
    action_verbs_found: list[str]
    verb_type_signal:  dict   # {"comparative": 0.90}

    # ── Query length signal ─────────────────────────────────────────────
    length_category:   str    # "keyword" | "short" | "standard" | "long"
    length_signal:     dict   # {"factual": 0.60} etc.

    # ── No-retrieval gate ───────────────────────────────────────────────
    no_retrieval_triggered: bool
    no_retrieval_reason:    str   # "simple_definition" | "mathematical" |
                                  # "system_meta" | ""

    # ── Decomposition ───────────────────────────────────────────────────
    needs_decomposition: bool
    sub_queries:         list[str]   # populated when needs_decomposition=True

    # ── Layer 1 result ──────────────────────────────────────────────────
    layer1_predicted_type: str    # best guess from signals
    layer1_confidence:     float  # 0.0 to 1.0
    layer1_all_scores:     dict   # {"factual":0.72,"relational":0.18,...}
    layer1_sufficient:     bool   # True if confidence >= threshold AND gap >= threshold

    # ── Timing ──────────────────────────────────────────────────────────
    preprocessing_time_ms: float


# ---------------------------------------------------------------------------
# Stage 2 — Classifier Output (final product consumed by Router)
# ---------------------------------------------------------------------------

@dataclass
class ClassifierOutput:
    """
    The final output of the query classifier.
    This is the ONLY object the Router receives.
    Whether classification was done by Layer 1, LLM, or XGBoost,
    the Router always gets exactly this structure.
    """

    # ── Identity ────────────────────────────────────────────────────────
    query_id:   str
    session_id: str
    raw_text:   str
    cleaned_text: str
    timestamp:  str

    # ── Core decision ───────────────────────────────────────────────────
    query_type: str     # "factual" | "relational" | "comparative" |
                        # "exploratory" | "no_retrieval"
    confidence: float   # 0.0 to 1.0

    # ── Which layer made the decision ───────────────────────────────────
    classified_by: str  # "layer1_preprocessor" | "layer1_no_retrieval_gate" |
                        # "layer2_llm" | "layer2_xgboost" | "layer2_hybrid"

    # ── All category scores ─────────────────────────────────────────────
    # Used by Router for hybrid routing decisions
    all_scores: dict    # {"factual": 0.72, "relational": 0.18, ...}

    # ── LLM-specific fields (None when LLM was not used) ─────────────────
    llm_reasoning:      Optional[str] = None   # one-sentence explanation
    llm_raw_response:   Optional[str] = None   # full raw JSON string

    # ── XGBoost-specific fields (None in Phase 1) ────────────────────────
    xgboost_confidence:     Optional[float]  = None
    xgboost_feature_vector: Optional[list]   = None

    # ── Decomposition ───────────────────────────────────────────────────
    needs_decomposition:        bool       = False
    sub_queries:                list[str]  = field(default_factory=list)
    sub_query_classifications:  list       = field(default_factory=list)
    # Each element is a ClassifierOutput for the corresponding sub-query

    # ── Timing (milliseconds) ───────────────────────────────────────────
    preprocessing_time_ms:  float = 0.0
    classification_time_ms: float = 0.0
    total_time_ms:          float = 0.0

    # ── Phase tracking ──────────────────────────────────────────────────
    phase: str = "phase1_llm"
    # "phase1_llm" | "phase2_xgboost" | "phase2_hybrid"


# ---------------------------------------------------------------------------
# Feedback Record (written to SQLite after user rates a response)
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    """
    One complete interaction record stored in the feedback database.
    The first half (classification fields) is written immediately.
    The second half (feedback fields) is written when the user rates.
    """

    # Written immediately at classification time
    log_id:                  str
    query_id:                str
    session_id:              str
    raw_text:                str
    cleaned_text:            str
    query_type:              str
    confidence:              float
    classified_by:           str
    all_scores:              str    # JSON string
    llm_reasoning:           str
    preprocessing_ms:        float
    classification_ms:       float
    total_ms:                float
    needs_decomposition:     int    # 0 or 1 (SQLite has no bool)
    sub_queries:             str    # JSON array string
    timestamp:               str
    phase:                   str
    retrieval_strategy_used: str    # set by Router after routing

    # Written later when user provides feedback
    user_rating:          Optional[int]  = None   # 1 (bad) to 5 (excellent)
    answer_was_correct:   Optional[int]  = None   # 0 or 1
    feedback_timestamp:   Optional[str]  = None


# ---------------------------------------------------------------------------
# XGBoost Feature Vector (stored separately for training)
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """
    Numeric representation of a query for XGBoost training and inference.
    All fields must be numeric — no strings.
    """

    query_id:                str

    # Structural
    token_count:             int
    sentence_count:          int
    is_question:             int    # 0 or 1
    is_multi_part:           int    # 0 or 1

    # Marker signals
    has_comparative:         int    # 0 or 1
    comparative_strength:    float  # max confidence of found comparative markers
    has_relational:          int    # 0 or 1
    relational_strength:     float

    # Entity signals
    entity_count:            int
    has_multiple_entities:   int    # 0 or 1
    dominant_entity_type:    int    # encoded using ENTITY_TYPE_ENCODING

    # Question word (encoded)
    question_word_code:      int

    # Verb type (encoded)
    verb_type_code:          int

    # Layer 1 scores (continuous — carry real signal)
    layer1_factual:          float
    layer1_relational:       float
    layer1_comparative:      float
    layer1_exploratory:      float

    # PCA-compressed embedding (32 dimensions)
    # Stored as a flat list of 32 floats
    embedding_dims:          list[float]

    # Label — filled in from feedback after user rates
    true_label:              Optional[str] = None
