"""
config.py
=========
Single source of truth for every tunable parameter in ASIR-RAG.

RULES:
  - No hardcoded values anywhere else in the codebase.
  - Every module imports its constants from here.
  - To change any behaviour, change it here only.
"""

# ---------------------------------------------------------------------------
# Project Identity
# ---------------------------------------------------------------------------
PROJECT_NAME = "ASIR-RAG"
VERSION      = "1.0.0"

# ---------------------------------------------------------------------------
# Database Paths
# ---------------------------------------------------------------------------
FEEDBACK_DB_PATH    = "./database/feedback.db"   # SQLite feedback log
VECTOR_DB_PATH      = "./database/chroma_db"     # ChromaDB vector store
XGBOOST_MODEL_PATH  = "./models/xgb_classifier.pkl"
LABEL_ENCODER_PATH  = "./models/xgb_label_encoder.pkl"
FEATURE_SCALER_PATH = "./models/xgb_scaler.pkl"

# ---------------------------------------------------------------------------
# Embedding Model (used for query representation in XGBoost feature vector)
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_FULL_DIM    = 384   # raw output dimension of all-MiniLM-L6-v2
EMBEDDING_PCA_DIM     = 32    # compressed dimension fed into XGBoost

# ---------------------------------------------------------------------------
# Layer 1 — Preprocessor Thresholds
# ---------------------------------------------------------------------------

# Minimum confidence for Layer 1 to classify without calling LLM/XGBoost
LAYER1_CONFIDENCE_THRESHOLD  = 0.85

# Minimum gap between top-1 and top-2 scores (prevents routing on thin margin)
LAYER1_GAP_THRESHOLD         = 0.30

# Signal weights — higher = more trusted signal type
SIGNAL_WEIGHTS = {
    "comparative_marker": 1.4,
    "relational_marker":  1.3,
    "verb_type":          1.1,
    "question_word":      1.0,
    "entity_type":        0.9,
    "query_length":       0.5,
}

# ---------------------------------------------------------------------------
# Layer 2 — LLM Classifier (Phase 1)
# ---------------------------------------------------------------------------
LLM_MODEL       = "llama3"          # Ollama model name
LLM_HOST        = "http://localhost:11434"
LLM_TEMPERATURE = 0.1               # Low = consistent, not creative
LLM_MAX_TOKENS  = 300
LLM_FORMAT      = "json"            # Force Ollama JSON mode
LLM_TIMEOUT_SEC = 30                # Max seconds to wait for LLM response

# ---------------------------------------------------------------------------
# Layer 2 — XGBoost Classifier (Phase 2)
# ---------------------------------------------------------------------------

# If XGBoost confidence is below this, fall back to LLM
XGBOOST_FALLBACK_THRESHOLD = 0.65

# XGBoost training hyperparameters
XGB_N_ESTIMATORS      = 200
XGB_MAX_DEPTH         = 6
XGB_LEARNING_RATE     = 0.1
XGB_SUBSAMPLE         = 0.8
XGB_COLSAMPLE_BYTREE  = 0.8
XGB_RANDOM_STATE      = 42
XGB_EVAL_METRIC       = "mlogloss"
XGB_TRAIN_TEST_SPLIT  = 0.2   # 80% train, 20% test

# ---------------------------------------------------------------------------
# Phase Transition — When to switch from LLM to XGBoost
# ---------------------------------------------------------------------------
TRANSITION_MIN_TOTAL_SAMPLES    = 300   # total labelled rows needed
TRANSITION_MIN_PER_CLASS        = 50    # min samples per query type
TRANSITION_MAX_CLASS_DOMINANCE  = 0.60  # no single class > 60% of data

# ---------------------------------------------------------------------------
# Valid Query Types
# ---------------------------------------------------------------------------
VALID_QUERY_TYPES = [
    "factual",
    "relational",
    "comparative",
    "exploratory",
    "no_retrieval",
]

# Strategy each query type maps to (used by Router)
STRATEGY_MAP = {
    "factual":      "vector",
    "relational":   "graph",
    "comparative":  "hybrid",
    "exploratory":  "hierarchical",
    "no_retrieval": "none",
}

# ---------------------------------------------------------------------------
# No-Retrieval Gate — Trigger phrases
# ---------------------------------------------------------------------------
NO_RETRIEVAL_DEFINITION_STARTERS = [
    "what is", "what are", "what does", "what do",
    "define", "who is", "who are", "what was",
]
NO_RETRIEVAL_MAX_TOKENS          = 5    # only short queries qualify

NO_RETRIEVAL_MATH_TERMS = [
    "calculate", "compute", "convert", "percentage of",
    "how many", "what is the formula", "solve", "evaluate",
    "what is the value", "square root", "factorial",
]

NO_RETRIEVAL_SYSTEM_PHRASES = [
    "what can you do", "how do you work",
    "help me", "what are your capabilities",
    "who are you", "what are you",
]

# ---------------------------------------------------------------------------
# Comparative Markers — phrase → confidence weight
# ---------------------------------------------------------------------------
COMPARATIVE_MARKERS = {
    "vs":                   0.95,
    "versus":               0.95,
    "compare":              0.90,
    "compared to":          0.90,
    "difference between":   0.90,
    "differences between":  0.90,
    "similarities between": 0.85,
    "advantages of":        0.80,
    "disadvantages of":     0.80,
    "pros and cons":        0.85,
    "better than":          0.85,
    "worse than":           0.85,
    "contrast":             0.75,
    "unlike":               0.80,
    "whereas":              0.70,
    "in comparison":        0.80,
    "relative to":          0.70,
}

# ---------------------------------------------------------------------------
# Relational Markers — phrase → confidence weight
# ---------------------------------------------------------------------------
RELATIONAL_MARKERS = {
    "relationship between":  0.92,
    "connection between":    0.90,
    "how does":              0.60,   # "how does X affect Y"
    "affect":                0.75,
    "influence of":          0.85,
    "impact of":             0.82,
    "effect of":             0.82,
    "caused by":             0.80,
    "leads to":              0.78,
    "linked to":             0.85,
    "associated with":       0.80,
    "role of":               0.70,
    "interaction between":   0.88,
    "relate to":             0.82,
    "depends on":            0.75,
    "contributes to":        0.75,
}

# ---------------------------------------------------------------------------
# Question Word Signal Map — word → {query_type: weight}
# ---------------------------------------------------------------------------
QUESTION_WORD_SIGNALS = {
    "what":  {"factual": 0.65, "exploratory": 0.35},
    "who":   {"factual": 0.90, "relational":  0.10},
    "where": {"factual": 0.80, "relational":  0.20},
    "when":  {"factual": 0.95},
    "why":   {"exploratory": 0.60, "factual": 0.40},
    "how":   {"exploratory": 0.50, "factual": 0.30, "relational": 0.20},
    "which": {"comparative": 0.55, "factual": 0.45},
    "is":    {"factual": 0.70, "comparative": 0.30},
    "are":   {"factual": 0.70, "comparative": 0.30},
    "does":  {"factual": 0.60, "relational":  0.40},
    "can":   {"factual": 0.50, "exploratory": 0.50},
}

# ---------------------------------------------------------------------------
# Named Entity Type Signal Map — spaCy label → {query_type: weight}
# ---------------------------------------------------------------------------
ENTITY_TYPE_SIGNALS_SINGLE = {
    "PERSON":   {"factual": 0.70, "relational": 0.30},
    "ORG":      {"factual": 0.60, "relational": 0.40},
    "GPE":      {"factual": 0.80, "relational": 0.20},
    "DATE":     {"factual": 0.90},
    "TIME":     {"factual": 0.90},
    "PRODUCT":  {"factual": 0.55, "comparative": 0.45},
    "EVENT":    {"factual": 0.65, "exploratory": 0.35},
    "WORK_OF_ART": {"factual": 0.70, "exploratory": 0.30},
    "LAW":      {"factual": 0.75, "relational": 0.25},
    "NORP":     {"factual": 0.60, "relational": 0.40},
}

# When 2+ entities of the same type appear, use these weights instead
ENTITY_TYPE_SIGNALS_MULTIPLE = {
    "PERSON":   {"relational": 0.75, "factual": 0.25},
    "ORG":      {"relational": 0.65, "comparative": 0.35},
    "PRODUCT":  {"comparative": 0.70, "factual": 0.30},
    "GPE":      {"comparative": 0.55, "relational": 0.45},
}

# ---------------------------------------------------------------------------
# Verb Type Signal Map — lemma → {query_type: weight}
# ---------------------------------------------------------------------------
VERB_TYPE_SIGNALS = {
    "define":    {"factual": 0.85, "exploratory": 0.15},
    "explain":   {"exploratory": 0.65, "factual": 0.35},
    "describe":  {"exploratory": 0.60, "factual": 0.40},
    "compare":   {"comparative": 0.90},
    "contrast":  {"comparative": 0.90},
    "list":      {"exploratory": 0.65, "factual": 0.35},
    "analyse":   {"exploratory": 0.70, "relational": 0.30},
    "analyze":   {"exploratory": 0.70, "relational": 0.30},
    "relate":    {"relational": 0.85},
    "connect":   {"relational": 0.80},
    "affect":    {"relational": 0.80},
    "impact":    {"relational": 0.75},
    "cause":     {"relational": 0.80},
    "summarise": {"exploratory": 0.75},
    "summarize": {"exploratory": 0.75},
    "outline":   {"exploratory": 0.70},
    "identify":  {"factual": 0.70, "exploratory": 0.30},
    "calculate": {"factual": 0.80},
    "find":      {"factual": 0.75},
}

# ---------------------------------------------------------------------------
# Query Length Thresholds
# ---------------------------------------------------------------------------
LENGTH_KEYWORD_MAX        = 3    # <= 3 tokens → likely keyword query
LENGTH_SHORT_MAX          = 8    # 4-8 tokens → short natural language
LENGTH_STANDARD_MAX       = 20   # 9-20 tokens → standard question
LENGTH_LONG_MIN           = 21   # >= 21 tokens → long, check decomposition

# ---------------------------------------------------------------------------
# Decomposition Triggers
# ---------------------------------------------------------------------------
DECOMPOSITION_CONJUNCTIONS = [
    "and also", "as well as", "additionally",
    "furthermore", "also explain", "and how",
    "in addition", "moreover",
]
DECOMPOSITION_MIN_TOKENS   = 20   # don't decompose short queries
DECOMPOSITION_MIN_SENTENCES = 2   # must have 2+ sentences to consider split

# ---------------------------------------------------------------------------
# Question word encoding for XGBoost feature vector
# ---------------------------------------------------------------------------
QUESTION_WORD_ENCODING = {
    "what": 0, "who": 1, "where": 2, "when": 3, "why": 4,
    "how": 5, "which": 6, "is": 7, "are": 8, "does": 9,
    "can": 10, "other": 11,
}

# Verb type encoding for XGBoost feature vector
VERB_ENCODING = {
    "define": 0, "explain": 1, "describe": 2, "compare": 3,
    "contrast": 4, "list": 5, "analyse": 6, "relate": 7,
    "affect": 8, "cause": 9, "summarise": 10, "other": 11,
}
