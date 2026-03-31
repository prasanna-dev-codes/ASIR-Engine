"""
feedback/logger.py  (FIXED VERSION)
=====================================
Writes every classification decision and its feature vector to SQLite.

FIX IN THIS VERSION:
─────────────────────────────────────────────────────────────────────
BUG:  "49 values for 50 columns" error in save_feature_vector()

CAUSE:
  placeholders = ", ".join("?" * (17 + C.EMBEDDING_PCA_DIM))
                                   ^^
  This calculated 17 + 32 = 49 placeholders.
  But the INSERT column list has 18 fixed fields + 32 embedding dims
  = 50 total values.

  The 18 fixed fields are:
    query_id, token_count, sentence_count,
    is_question, is_multi_part,
    has_comparative, comparative_strength,
    has_relational, relational_strength,
    entity_count, has_multiple_entities,
    dominant_entity_type,
    question_word_code, verb_type_code,
    layer1_factual, layer1_relational,
    layer1_comparative, layer1_exploratory
    = 18 fields (the original code counted 17, missing one)

FIX:
  Changed 17 → 18:
  placeholders = ", ".join("?" * (18 + C.EMBEDDING_PCA_DIM))
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

import config as C
from shared.models import ClassifierOutput, FeatureVector

log = logging.getLogger(__name__)


class FeedbackLogger:
    """
    Handles all database writes for the feedback system.
    Instantiate once in QueryClassifier and reuse.
    """

    def __init__(self, db_path: str = C.FEEDBACK_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialise_tables()
        log.info("FeedbackLogger ready. DB: %s", db_path)

    # ──────────────────────────────────────────────────────────────────
    # Table initialisation
    # ──────────────────────────────────────────────────────────────────

    def _initialise_tables(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(self._get_schema_sql())
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _get_schema_sql() -> str:
        embedding_cols = "\n".join(
            f"    embedding_dim_{i}     REAL DEFAULT 0.0,"
            for i in range(C.EMBEDDING_PCA_DIM)
        )

        return f"""
        CREATE TABLE IF NOT EXISTS classification_log (
            log_id                   TEXT PRIMARY KEY,
            query_id                 TEXT NOT NULL,
            session_id               TEXT,
            raw_text                 TEXT NOT NULL,
            cleaned_text             TEXT,
            query_type               TEXT NOT NULL,
            confidence               REAL,
            classified_by            TEXT,
            all_scores               TEXT,
            llm_reasoning            TEXT,
            llm_raw_response         TEXT,
            xgboost_confidence       REAL,
            preprocessing_ms         REAL,
            classification_ms        REAL,
            total_ms                 REAL,
            needs_decomposition      INTEGER DEFAULT 0,
            sub_queries              TEXT,
            retrieval_strategy_used  TEXT DEFAULT '',
            timestamp                TEXT,
            phase                    TEXT,

            user_rating              INTEGER,
            answer_was_correct       INTEGER,
            feedback_timestamp       TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_classification_log_query_id
            ON classification_log (query_id);
        CREATE INDEX IF NOT EXISTS idx_classification_log_query_type
            ON classification_log (query_type);
        CREATE INDEX IF NOT EXISTS idx_classification_log_timestamp
            ON classification_log (timestamp);

        CREATE TABLE IF NOT EXISTS feature_vectors (
            query_id                 TEXT PRIMARY KEY,
            token_count              INTEGER,
            sentence_count           INTEGER,
            is_question              INTEGER,
            is_multi_part            INTEGER,
            has_comparative          INTEGER,
            comparative_strength     REAL,
            has_relational           INTEGER,
            relational_strength      REAL,
            entity_count             INTEGER,
            has_multiple_entities    INTEGER,
            dominant_entity_type     INTEGER,
            question_word_code       INTEGER,
            verb_type_code           INTEGER,
            layer1_factual           REAL,
            layer1_relational        REAL,
            layer1_comparative       REAL,
            layer1_exploratory       REAL,
            {embedding_cols}
            true_label               TEXT
        );
        """

    # ──────────────────────────────────────────────────────────────────
    # Write: log a classification result
    # ──────────────────────────────────────────────────────────────────

    def log_classification(self, result: ClassifierOutput) -> str:
        log_id   = str(uuid.uuid4())
        strategy = C.STRATEGY_MAP.get(result.query_type, "unknown")

        conn = self._connect()
        try:
            conn.execute("""
                INSERT INTO classification_log (
                    log_id, query_id, session_id, raw_text, cleaned_text,
                    query_type, confidence, classified_by, all_scores,
                    llm_reasoning, llm_raw_response, xgboost_confidence,
                    preprocessing_ms, classification_ms, total_ms,
                    needs_decomposition, sub_queries,
                    retrieval_strategy_used, timestamp, phase
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                log_id,
                result.query_id,
                result.session_id,
                result.raw_text,
                result.cleaned_text,
                result.query_type,
                result.confidence,
                result.classified_by,
                json.dumps(result.all_scores),
                result.llm_reasoning or "",
                result.llm_raw_response or "",
                result.xgboost_confidence,
                result.preprocessing_time_ms,
                result.classification_time_ms,
                result.total_time_ms,
                int(result.needs_decomposition),
                json.dumps(result.sub_queries),
                strategy,
                result.timestamp,
                result.phase,
            ))
            conn.commit()
            log.debug("Logged classification for query_id=%s", result.query_id)
        except sqlite3.IntegrityError:
            log.warning(
                "Duplicate query_id=%s in classification_log. Skipping.",
                result.query_id
            )
        finally:
            conn.close()

        return log_id

    # ──────────────────────────────────────────────────────────────────
    # Write: save feature vector  ← BUG FIX HERE
    # ──────────────────────────────────────────────────────────────────

    def save_feature_vector(self, fv: FeatureVector) -> None:
        """
        Write a FeatureVector to the feature_vectors table.

        FIX: Changed placeholder count from (17 + PCA_DIM) to (18 + PCA_DIM).
        The fixed fields in the INSERT are:
          query_id, token_count, sentence_count,          ← 3
          is_question, is_multi_part,                     ← 2
          has_comparative, comparative_strength,           ← 2
          has_relational, relational_strength,             ← 2
          entity_count, has_multiple_entities,             ← 2
          dominant_entity_type,                            ← 1
          question_word_code, verb_type_code,              ← 2
          layer1_factual, layer1_relational,               ← 2
          layer1_comparative, layer1_exploratory           ← 2
                                                    TOTAL = 18 fixed fields
        + EMBEDDING_PCA_DIM (32) embedding dims
                                                    TOTAL = 50 placeholders
        """
        embedding_col_names = ", ".join(
            f"embedding_dim_{i}" for i in range(C.EMBEDDING_PCA_DIM)
        )

        # ✅ FIXED: was (17 + ...) which gave 49 — now correctly (18 + ...)
        placeholders = ", ".join(["?"] * (18 + C.EMBEDDING_PCA_DIM))

        # Pad/trim embedding to exactly PCA_DIM values
        embedding = (fv.embedding_dims or [])[:C.EMBEDDING_PCA_DIM]
        embedding += [0.0] * max(0, C.EMBEDDING_PCA_DIM - len(embedding))

        values = (
            fv.query_id,               # 1
            fv.token_count,            # 2
            fv.sentence_count,         # 3
            fv.is_question,            # 4
            fv.is_multi_part,          # 5
            fv.has_comparative,        # 6
            fv.comparative_strength,   # 7
            fv.has_relational,         # 8
            fv.relational_strength,    # 9
            fv.entity_count,           # 10
            fv.has_multiple_entities,  # 11
            fv.dominant_entity_type,   # 12
            fv.question_word_code,     # 13
            fv.verb_type_code,         # 14
            fv.layer1_factual,         # 15
            fv.layer1_relational,      # 16
            fv.layer1_comparative,     # 17
            fv.layer1_exploratory,     # 18  ← was being counted as 17
            *embedding,                # 19..50 (32 dims)
        )

        conn = self._connect()
        try:
            conn.execute(f"""
                INSERT OR IGNORE INTO feature_vectors (
                    query_id, token_count, sentence_count,
                    is_question, is_multi_part,
                    has_comparative, comparative_strength,
                    has_relational, relational_strength,
                    entity_count, has_multiple_entities,
                    dominant_entity_type,
                    question_word_code, verb_type_code,
                    layer1_factual, layer1_relational,
                    layer1_comparative, layer1_exploratory,
                    {embedding_col_names}
                ) VALUES ({placeholders})
            """, values)
            conn.commit()
        except Exception as exc:
            log.error(
                "Failed to save feature vector for query_id=%s: %s",
                fv.query_id, exc
            )
        finally:
            conn.close()

    # ──────────────────────────────────────────────────────────────────
    # Write: record user feedback
    # ──────────────────────────────────────────────────────────────────

    def record_feedback(
        self,
        query_id:           str,
        user_rating:        int,
        answer_was_correct: bool,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
            conn.execute("""
                UPDATE classification_log
                SET user_rating        = ?,
                    answer_was_correct = ?,
                    feedback_timestamp = ?
                WHERE query_id = ?
            """, (user_rating, int(answer_was_correct), now, query_id))

            if answer_was_correct:
                row = conn.execute("""
                    SELECT query_type FROM classification_log
                    WHERE query_id = ?
                """, (query_id,)).fetchone()

                if row:
                    conn.execute("""
                        UPDATE feature_vectors
                        SET true_label = ?
                        WHERE query_id = ?
                    """, (row[0], query_id))

            conn.commit()
            log.info(
                "Feedback recorded for query_id=%s | rating=%d | correct=%s",
                query_id, user_rating, answer_was_correct
            )
        finally:
            conn.close()

    # ──────────────────────────────────────────────────────────────────
    # Read: statistics for monitoring
    # ──────────────────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        conn = self._connect()
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM classification_log"
            ).fetchone()[0]

            labelled = conn.execute(
                "SELECT COUNT(*) FROM classification_log WHERE user_rating IS NOT NULL"
            ).fetchone()[0]

            per_type = conn.execute("""
                SELECT query_type, COUNT(*) as n
                FROM classification_log
                WHERE user_rating IS NOT NULL
                GROUP BY query_type
            """).fetchall()

            avg_rating = conn.execute(
                "SELECT AVG(user_rating) FROM classification_log "
                "WHERE user_rating IS NOT NULL"
            ).fetchone()[0]

            by_classifier = conn.execute("""
                SELECT classified_by, COUNT(*) as n
                FROM classification_log
                GROUP BY classified_by
            """).fetchall()

        finally:
            conn.close()

        return {
            "total_queries":    total,
            "labelled_queries": labelled,
            "per_type_counts":  {row[0]: row[1] for row in per_type},
            "average_rating":   round(avg_rating or 0.0, 2),
            "by_classifier":    {row[0]: row[1] for row in by_classifier},
        }

    # ──────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)
