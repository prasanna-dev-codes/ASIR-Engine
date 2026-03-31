"""
feedback/transition.py
======================
Determines whether enough labelled data has been collected
to transition from Phase 1 (LLM classifier) to Phase 2 (XGBoost).

THREE CONDITIONS MUST ALL BE TRUE:
  1. Total labelled samples >= TRANSITION_MIN_TOTAL_SAMPLES (300)
  2. Each query type has >= TRANSITION_MIN_PER_CLASS (50) samples
  3. No single type dominates > TRANSITION_MAX_CLASS_DOMINANCE (60%)

USAGE:
    from feedback.transition import should_transition_to_phase2
    if should_transition_to_phase2():
        train_xgboost()
"""

from __future__ import annotations

import logging
import sqlite3

import config as C

log = logging.getLogger(__name__)


def should_transition_to_phase2(db_path: str = C.FEEDBACK_DB_PATH) -> bool:
    """
    Check if all transition conditions are met for Phase 2.

    Parameters
    ----------
    db_path : str
        Path to the SQLite feedback database.

    Returns
    -------
    bool
        True if Phase 2 should be activated.
    """
    try:
        conn = sqlite3.connect(db_path)

        # ── Condition 1: Total labelled samples ──────────────────────
        total = conn.execute("""
            SELECT COUNT(*)
            FROM classification_log
            WHERE user_rating IS NOT NULL
        """).fetchone()[0]

        if total < C.TRANSITION_MIN_TOTAL_SAMPLES:
            log.debug(
                "Transition check: only %d labelled samples (need %d).",
                total, C.TRANSITION_MIN_TOTAL_SAMPLES
            )
            conn.close()
            return False

        # ── Condition 2: Per-class minimum ───────────────────────────
        per_type = conn.execute("""
            SELECT query_type, COUNT(*) as n
            FROM classification_log
            WHERE user_rating IS NOT NULL
            GROUP BY query_type
        """).fetchall()

        counts = {row[0]: row[1] for row in per_type}

        for qt in C.VALID_QUERY_TYPES:
            if counts.get(qt, 0) < C.TRANSITION_MIN_PER_CLASS:
                log.debug(
                    "Transition check: query_type '%s' only has %d samples (need %d).",
                    qt, counts.get(qt, 0), C.TRANSITION_MIN_PER_CLASS
                )
                conn.close()
                return False

        # ── Condition 3: No single class dominates ───────────────────
        max_count = max(counts.values()) if counts else 0
        dominance = max_count / total if total > 0 else 1.0

        if dominance > C.TRANSITION_MAX_CLASS_DOMINANCE:
            log.debug(
                "Transition check: class imbalance too high (%.1f%%). "
                "Collect more diverse query types.",
                dominance * 100
            )
            conn.close()
            return False

        conn.close()
        log.info(
            "Transition conditions met: %d total samples, "
            "per-class counts: %s, dominance: %.1f%%",
            total, counts, dominance * 100
        )
        return True

    except Exception as exc:
        log.warning("Transition check failed: %s", exc)
        return False


def get_transition_status(db_path: str = C.FEEDBACK_DB_PATH) -> dict:
    """
    Return a detailed status report on transition readiness.
    Useful for the UI analytics dashboard.

    Returns
    -------
    dict with keys:
        ready           : bool
        total_labelled  : int
        total_needed    : int
        per_type        : dict of {query_type: {"have": int, "need": int}}
        dominance       : float (0.0 to 1.0)
        missing_types   : list of query types below minimum
    """
    try:
        conn = sqlite3.connect(db_path)

        total = conn.execute("""
            SELECT COUNT(*)
            FROM classification_log
            WHERE user_rating IS NOT NULL
        """).fetchone()[0]

        per_type_rows = conn.execute("""
            SELECT query_type, COUNT(*) as n
            FROM classification_log
            WHERE user_rating IS NOT NULL
            GROUP BY query_type
        """).fetchall()

        conn.close()

        counts     = {row[0]: row[1] for row in per_type_rows}
        max_count  = max(counts.values()) if counts else 0
        dominance  = max_count / total if total > 0 else 0.0

        per_type_status = {}
        missing_types   = []

        for qt in C.VALID_QUERY_TYPES:
            have = counts.get(qt, 0)
            need = C.TRANSITION_MIN_PER_CLASS
            per_type_status[qt] = {"have": have, "need": need, "ready": have >= need}
            if have < need:
                missing_types.append(qt)

        ready = (
            total >= C.TRANSITION_MIN_TOTAL_SAMPLES
            and not missing_types
            and dominance <= C.TRANSITION_MAX_CLASS_DOMINANCE
        )

        return {
            "ready":          ready,
            "total_labelled": total,
            "total_needed":   C.TRANSITION_MIN_TOTAL_SAMPLES,
            "per_type":       per_type_status,
            "dominance":      round(dominance, 4),
            "missing_types":  missing_types,
        }

    except Exception as exc:
        log.warning("Could not get transition status: %s", exc)
        return {
            "ready": False,
            "error": str(exc),
        }
