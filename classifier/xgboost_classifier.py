"""
classifier/xgboost_classifier.py
==================================
Step 6 (Phase 2): XGBoost-based query classifier.

WHEN THIS IS USED:
  Phase 2 begins after ~300 labelled feedback interactions are collected.
  XGBoost replaces the LLM for medium-confidence cases.
  The LLM remains as a fallback for very uncertain queries.

TWO PARTS IN THIS FILE:
  1. classify_with_xgboost()  — inference (used at runtime)
  2. train_xgboost()          — training (run once when data is ready)

FEATURE VECTOR:
  The feature vector is constructed from preprocessor signals.
  It matches exactly what was stored in the feature_vectors table
  during Phase 1, so training and inference use identical features.

INSTALL:
  pip install xgboost scikit-learn joblib
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Optional

import numpy as np

import config as C
from shared.models import FeatureVector, PreprocessorOutput

log = logging.getLogger(__name__)

# Lazy imports — only loaded when actually needed
# This prevents ImportError at startup if xgboost/sklearn are not installed
_xgb_model      = None
_label_encoder  = None
_feature_scaler = None


# ---------------------------------------------------------------------------
# Inference — called at runtime during Phase 2
# ---------------------------------------------------------------------------

def classify_with_xgboost(preprocessor: PreprocessorOutput) -> dict:
    """
    Classify a query using the trained XGBoost model.

    If XGBoost confidence is below XGBOOST_FALLBACK_THRESHOLD,
    returns a low-confidence result and the orchestrator will
    fall back to the LLM.

    Parameters
    ----------
    preprocessor : PreprocessorOutput
        Full output from the preprocessor.

    Returns
    -------
    dict with keys:
        query_type              : str
        confidence              : float
        all_scores              : dict
        xgboost_confidence      : float
        xgboost_feature_vector  : list
        classified_by           : str
        error                   : str
    """
    global _xgb_model, _label_encoder, _feature_scaler

    # Load models lazily on first call
    if _xgb_model is None:
        _load_models()

    if _xgb_model is None:
        return {
            "query_type":             preprocessor.layer1_predicted_type,
            "confidence":             0.0,
            "all_scores":             preprocessor.layer1_all_scores,
            "xgboost_confidence":     0.0,
            "xgboost_feature_vector": [],
            "classified_by":          "xgboost_not_loaded",
            "error":                  "XGBoost model not found. Is Phase 2 ready?",
        }

    t_start = time.perf_counter()

    # Build feature vector
    fv = build_feature_vector(preprocessor)
    feature_array = _feature_vector_to_array(fv)

    # Scale features (same scaler used during training)
    if _feature_scaler is not None:
        feature_array_scaled = _feature_scaler.transform([feature_array])[0]
    else:
        feature_array_scaled = feature_array

    # Predict
    proba     = _xgb_model.predict_proba([feature_array_scaled])[0]
    classes   = _label_encoder.classes_
    top_idx   = int(np.argmax(proba))
    top_type  = classes[top_idx]
    top_conf  = float(proba[top_idx])

    all_scores = {
        classes[i]: round(float(proba[i]), 4)
        for i in range(len(classes))
    }

    t_end = time.perf_counter()
    log.debug(
        "XGBoost classified query_id=%s as '%s' (confidence=%.2f) in %.1fms",
        preprocessor.query_id, top_type, top_conf,
        (t_end - t_start) * 1000
    )

    return {
        "query_type":             top_type,
        "confidence":             round(top_conf, 4),
        "all_scores":             all_scores,
        "xgboost_confidence":     round(top_conf, 4),
        "xgboost_feature_vector": feature_array.tolist(),
        "classified_by":          "layer2_xgboost",
        "error":                  "",
    }


# ---------------------------------------------------------------------------
# Feature vector construction — used both at inference and training time
# ---------------------------------------------------------------------------

def build_feature_vector(preprocessor: PreprocessorOutput) -> FeatureVector:
    """
    Convert a PreprocessorOutput into a numeric FeatureVector.

    This function is the single source of truth for feature construction.
    It is called:
      - At inference time in classify_with_xgboost()
      - At training time when building the training matrix
      - After Phase 1 classification, to save features to the DB for training

    Returns
    -------
    FeatureVector dataclass with all numeric fields populated.
    """

    # ── Marker signals ────────────────────────────────────────────────
    has_comparative     = int(bool(preprocessor.comparative_markers_found))
    comp_strength       = preprocessor.comparative_marker_signal.get("comparative", 0.0)
    has_relational      = int(bool(preprocessor.relational_markers_found))
    rel_strength        = preprocessor.relational_marker_signal.get("relational", 0.0)

    # ── Entity signals ────────────────────────────────────────────────
    entity_count        = len(preprocessor.named_entities)
    has_multiple_ent    = int(entity_count >= 2)

    # Encode dominant entity type to integer
    entity_type_enc     = _encode_entity_type(preprocessor.named_entities)

    # ── Question word encoding ────────────────────────────────────────
    qw_code = C.QUESTION_WORD_ENCODING.get(
        preprocessor.question_word,
        C.QUESTION_WORD_ENCODING["other"]
    )

    # ── Verb encoding ─────────────────────────────────────────────────
    verb_code = C.VERB_ENCODING.get(
        preprocessor.root_verb,
        C.VERB_ENCODING["other"]
    )

    # ── Layer 1 scores ────────────────────────────────────────────────
    l1 = preprocessor.layer1_all_scores
    layer1_factual      = l1.get("factual",      0.0)
    layer1_relational   = l1.get("relational",   0.0)
    layer1_comparative  = l1.get("comparative",  0.0)
    layer1_exploratory  = l1.get("exploratory",  0.0)

    # ── Query embedding (PCA-compressed) ─────────────────────────────
    embedding_dims = _get_query_embedding(preprocessor.cleaned_text)

    return FeatureVector(
        query_id=preprocessor.query_id,

        # Structural
        token_count=preprocessor.token_count,
        sentence_count=preprocessor.sentence_count,
        is_question=int(preprocessor.is_question),
        is_multi_part=int(preprocessor.is_multi_part),

        # Marker signals
        has_comparative=has_comparative,
        comparative_strength=round(comp_strength, 4),
        has_relational=has_relational,
        relational_strength=round(rel_strength, 4),

        # Entity signals
        entity_count=entity_count,
        has_multiple_entities=has_multiple_ent,
        dominant_entity_type=entity_type_enc,

        # Encoded categoricals
        question_word_code=qw_code,
        verb_type_code=verb_code,

        # Layer 1 scores
        layer1_factual=round(layer1_factual, 4),
        layer1_relational=round(layer1_relational, 4),
        layer1_comparative=round(layer1_comparative, 4),
        layer1_exploratory=round(layer1_exploratory, 4),

        # Embedding
        embedding_dims=embedding_dims,
    )


def _feature_vector_to_array(fv: FeatureVector) -> np.ndarray:
    """Flatten a FeatureVector into a 1D numpy array for XGBoost."""
    fixed_features = [
        fv.token_count,
        fv.sentence_count,
        fv.is_question,
        fv.is_multi_part,
        fv.has_comparative,
        fv.comparative_strength,
        fv.has_relational,
        fv.relational_strength,
        fv.entity_count,
        fv.has_multiple_entities,
        fv.dominant_entity_type,
        fv.question_word_code,
        fv.verb_type_code,
        fv.layer1_factual,
        fv.layer1_relational,
        fv.layer1_comparative,
        fv.layer1_exploratory,
    ]
    return np.array(fixed_features + fv.embedding_dims, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_xgboost(
    db_path: str = C.FEEDBACK_DB_PATH,
    model_save_path: str = C.XGBOOST_MODEL_PATH,
    encoder_save_path: str = C.LABEL_ENCODER_PATH,
    scaler_save_path: str = C.FEATURE_SCALER_PATH,
) -> dict:
    """
    Train the XGBoost classifier on accumulated feedback data.

    Run this function once when the transition conditions are met
    (see feedback/transition.py for when to call this).

    Parameters
    ----------
    db_path : str
        Path to the SQLite feedback database.
    model_save_path : str
        Where to save the trained XGBoost model (.pkl).
    encoder_save_path : str
        Where to save the LabelEncoder (.pkl).
    scaler_save_path : str
        Where to save the StandardScaler (.pkl).

    Returns
    -------
    dict with training results:
        accuracy, per_class_report, n_train, n_test
    """
    # Import here to avoid hard dependency at startup
    import xgboost as xgb
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, accuracy_score

    log.info("Starting XGBoost training from database: %s", db_path)

    # ── Step 1: Load data ─────────────────────────────────────────────
    X, y, query_ids = _load_training_data(db_path)

    if len(X) < C.TRANSITION_MIN_TOTAL_SAMPLES:
        raise ValueError(
            f"Not enough training data: {len(X)} samples. "
            f"Need at least {C.TRANSITION_MIN_TOTAL_SAMPLES}."
        )

    log.info("Loaded %d labelled training samples.", len(X))

    # ── Step 2: Encode labels ─────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    log.info("Label classes: %s", list(le.classes_))

    # ── Step 3: Scale features ────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Step 4: Train/test split — stratified to preserve class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=C.XGB_TRAIN_TEST_SPLIT,
        stratify=y_encoded,
        random_state=C.XGB_RANDOM_STATE,
    )

    log.info("Training: %d samples | Test: %d samples", len(X_train), len(X_test))

    # ── Step 5: Train XGBoost ─────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=C.XGB_N_ESTIMATORS,
        max_depth=C.XGB_MAX_DEPTH,
        learning_rate=C.XGB_LEARNING_RATE,
        subsample=C.XGB_SUBSAMPLE,
        colsample_bytree=C.XGB_COLSAMPLE_BYTREE,
        use_label_encoder=False,
        eval_metric=C.XGB_EVAL_METRIC,
        random_state=C.XGB_RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Step 6: Evaluate ─────────────────────────────────────────────
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    report    = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )

    log.info("XGBoost training complete. Test accuracy: %.4f", accuracy)
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Step 7: Save model and encoder ───────────────────────────────
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    joblib.dump(model,  model_save_path)
    joblib.dump(le,     encoder_save_path)
    joblib.dump(scaler, scaler_save_path)

    log.info("Model saved to %s", model_save_path)
    log.info("Encoder saved to %s", encoder_save_path)
    log.info("Scaler saved to %s", scaler_save_path)

    return {
        "accuracy":         round(accuracy, 4),
        "per_class_report": report,
        "n_train":          len(X_train),
        "n_test":           len(X_test),
        "classes":          list(le.classes_),
    }


def _load_training_data(db_path: str):
    """
    Load feature vectors and labels from the SQLite database.
    Only includes rows where true_label is not NULL
    (i.e. the user has provided feedback).

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : list of label strings
    query_ids : list of query_id strings
    """
    conn = sqlite3.connect(db_path)

    try:
        # Build the column list for embedding dims
        embedding_cols = ", ".join(
            f"embedding_dim_{i}" for i in range(C.EMBEDDING_PCA_DIM)
        )

        rows = conn.execute(f"""
            SELECT
                query_id,
                token_count,
                sentence_count,
                is_question,
                is_multi_part,
                has_comparative,
                comparative_strength,
                has_relational,
                relational_strength,
                entity_count,
                has_multiple_entities,
                dominant_entity_type,
                question_word_code,
                verb_type_code,
                layer1_factual,
                layer1_relational,
                layer1_comparative,
                layer1_exploratory,
                {embedding_cols},
                true_label
            FROM feature_vectors
            WHERE true_label IS NOT NULL
        """).fetchall()

    finally:
        conn.close()

    query_ids = [row[0] for row in rows]
    y         = [row[-1] for row in rows]
    X_list    = [list(row[1:-1]) for row in rows]
    X         = np.array(X_list, dtype=np.float32)

    return X, y, query_ids


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models() -> None:
    """Load XGBoost model, label encoder, and scaler from disk."""
    global _xgb_model, _label_encoder, _feature_scaler

    try:
        import joblib

        if not os.path.exists(C.XGBOOST_MODEL_PATH):
            log.info(
                "XGBoost model not found at %s. "
                "Phase 2 not yet active.",
                C.XGBOOST_MODEL_PATH
            )
            return

        _xgb_model      = joblib.load(C.XGBOOST_MODEL_PATH)
        _label_encoder  = joblib.load(C.LABEL_ENCODER_PATH)

        if os.path.exists(C.FEATURE_SCALER_PATH):
            _feature_scaler = joblib.load(C.FEATURE_SCALER_PATH)

        log.info("XGBoost model loaded from %s", C.XGBOOST_MODEL_PATH)

    except Exception as exc:
        log.error("Failed to load XGBoost model: %s", exc)


def is_model_loaded() -> bool:
    """Return True if the XGBoost model is loaded and ready."""
    return _xgb_model is not None


# ---------------------------------------------------------------------------
# Helper: query embedding (PCA-compressed)
# ---------------------------------------------------------------------------

_embed_model = None
_pca_model   = None

def _get_query_embedding(cleaned_text: str) -> list[float]:
    """
    Compute a PCA-compressed embedding for the query.
    Returns a list of EMBEDDING_PCA_DIM floats.
    Falls back to zero vector if embedding fails.
    """
    global _embed_model, _pca_model

    try:
        from sentence_transformers import SentenceTransformer

        if _embed_model is None:
            _embed_model = SentenceTransformer(C.EMBED_MODEL_NAME)
            log.info("Embedding model loaded for XGBoost features.")

        embedding = _embed_model.encode([cleaned_text])[0]  # shape: (384,)

        # Apply PCA compression if a fitted PCA model exists
        pca_path = C.FEATURE_SCALER_PATH.replace("scaler", "pca")
        if _pca_model is None and os.path.exists(pca_path):
            import joblib
            _pca_model = joblib.load(pca_path)

        if _pca_model is not None:
            compressed = _pca_model.transform([embedding])[0]
            return [round(float(v), 6) for v in compressed]

        # If no PCA model yet (Phase 1), truncate to first PCA_DIM dims
        # This is a placeholder — will be replaced by proper PCA in Phase 2
        truncated = embedding[:C.EMBEDDING_PCA_DIM]
        return [round(float(v), 6) for v in truncated]

    except Exception as exc:
        log.warning("Embedding failed for query: %s. Using zero vector.", exc)
        return [0.0] * C.EMBEDDING_PCA_DIM


# ---------------------------------------------------------------------------
# Helper: entity type encoding
# ---------------------------------------------------------------------------

# Mapping of spaCy entity labels to integer codes for XGBoost
_ENTITY_TYPE_ENCODING = {
    "PERSON": 1, "ORG": 2, "GPE": 3, "DATE": 4, "TIME": 5,
    "PRODUCT": 6, "EVENT": 7, "WORK_OF_ART": 8, "LAW": 9,
    "NORP": 10, "LOC": 11, "MONEY": 12, "PERCENT": 13,
    "QUANTITY": 14, "ORDINAL": 15, "CARDINAL": 16,
}

def _encode_entity_type(entities: list) -> int:
    """
    Return integer code of the most frequent entity type.
    Returns 0 if no entities found.
    """
    if not entities:
        return 0

    type_counts: dict[str, int] = {}
    for ent in entities:
        type_counts[ent.label] = type_counts.get(ent.label, 0) + 1

    dominant = max(type_counts, key=type_counts.get)
    return _ENTITY_TYPE_ENCODING.get(dominant, 0)
