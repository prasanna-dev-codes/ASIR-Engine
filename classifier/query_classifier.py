"""
classifier/query_classifier.py
================================
Main orchestrator for the ASIR-RAG query classifier pipeline.

This is the ONLY file that other modules (Router, UI, main.py) 
should call. It coordinates all internal components.

PUBLIC API:
    classifier = QueryClassifier()
    result     = classifier.classify(raw_query)

The result is always a ClassifierOutput — regardless of whether
the decision was made by Layer 1, the LLM, or XGBoost.

PIPELINE FLOW:
    RawQuery
      → TextCleaner     (Step 1)
      → Preprocessor    (Steps 2–5)
      → No-Retrieval Gate (Step 3, inside Preprocessor)
      → Layer 1 check   (Step 4, inside Preprocessor)
      → Layer 2 if needed:
            Phase 1: LLM Classifier
            Phase 2: XGBoost → LLM fallback if uncertain
      → ClassifierOutput assembled
      → FeedbackLogger  (async, non-blocking)
      → Return ClassifierOutput to Router
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import config as C
from classifier.preprocessor import preprocess
from classifier.llm_classifier import classify_with_llm
from classifier.xgboost_classifier import (
    classify_with_xgboost,
    build_feature_vector,
    is_model_loaded,
)
from feedback.logger import FeedbackLogger
from feedback.transition import should_transition_to_phase2
from shared.models import ClassifierOutput, RawQuery

log = logging.getLogger(__name__)


class QueryClassifier:
    """
    End-to-end query classifier for ASIR-RAG.

    Instantiate once and reuse for the lifetime of the application.
    Manages phase state (Phase 1 vs Phase 2) automatically.

    Example
    -------
    >>> classifier = QueryClassifier()
    >>> raw = RawQuery(
    ...     query_id=str(uuid.uuid4()),
    ...     session_id="session_001",
    ...     raw_text="How does insulin resistance lead to Type 2 diabetes?",
    ...     timestamp=datetime.utcnow().isoformat(),
    ...     character_count=51,
    ...     word_count=9,
    ... )
    >>> result = classifier.classify(raw)
    >>> print(result.query_type, result.confidence)
    relational 0.87
    """

    def __init__(
        self,
        feedback_logger: Optional[FeedbackLogger] = None,
        force_phase: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        feedback_logger : FeedbackLogger, optional
            Pre-initialised feedback logger. If None, creates a new one.
        force_phase : str, optional
            Force "phase1" or "phase2" regardless of data conditions.
            Useful for testing. If None, phase is determined automatically.
        """
        self._logger       = feedback_logger or FeedbackLogger()
        self._force_phase  = force_phase
        self._current_phase = None   # determined lazily on first classify()

        log.info("QueryClassifier initialised.")

    # ──────────────────────────────────────────────────────────────────────
    # Public: classify a single query
    # ──────────────────────────────────────────────────────────────────────

    def classify(self, raw_query: RawQuery) -> ClassifierOutput:
        """
        Run the full classification pipeline on a raw query.

        Parameters
        ----------
        raw_query : RawQuery
            Created at the UI layer the moment the user submits.

        Returns
        -------
        ClassifierOutput
            The complete classification result, ready for the Router.
        """
        t_pipeline_start = time.perf_counter()

        # Determine current phase (once per session start, rechecked periodically)
        phase = self._get_current_phase()

        # ── Step 1–5: Preprocessing ───────────────────────────────────
        preprocessor = preprocess(raw_query)

        # ── Step 3 (no-retrieval) was handled inside preprocess() ─────
        if preprocessor.no_retrieval_triggered:
            result = self._build_no_retrieval_output(
                raw_query, preprocessor, t_pipeline_start
            )
            self._log_and_store_features(result, preprocessor)
            return result

        # ── Step 5: Handle decomposition ─────────────────────────────
        if preprocessor.needs_decomposition and preprocessor.sub_queries:
            result = self._classify_decomposed(
                raw_query, preprocessor, phase, t_pipeline_start
            )
            self._log_and_store_features(result, preprocessor)
            return result

        # ── Layer 1 check ─────────────────────────────────────────────
        if preprocessor.layer1_sufficient:
            result = self._build_layer1_output(
                raw_query, preprocessor, t_pipeline_start
            )
            self._log_and_store_features(result, preprocessor)
            return result

        # ── Layer 2: route to LLM or XGBoost ─────────────────────────
        t_l2_start = time.perf_counter()

        if phase == "phase1":
            l2_result = classify_with_llm(preprocessor)
        else:
            l2_result = self._phase2_classify(preprocessor)

        t_l2_end = time.perf_counter()

        result = self._build_layer2_output(
            raw_query, preprocessor, l2_result,
            classification_ms=(t_l2_end - t_l2_start) * 1000,
            total_start=t_pipeline_start,
            phase=phase,
        )

        # Handle decomposition detected by LLM (not caught by preprocessor)
        if (
            l2_result.get("needs_decomposition")
            and l2_result.get("sub_queries")
            and not preprocessor.needs_decomposition
        ):
            result.needs_decomposition = True
            result.sub_queries = l2_result["sub_queries"]
            result = self._classify_sub_queries(result, phase)

        self._log_and_store_features(result, preprocessor)
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Convenience: classify from raw text string
    # ──────────────────────────────────────────────────────────────────────

    def classify_text(
        self,
        text: str,
        session_id: str = "default",
    ) -> ClassifierOutput:
        """
        Convenience method — wraps a plain string into a RawQuery and classifies.

        Parameters
        ----------
        text : str
            The user's query as a plain string.
        session_id : str
            Session identifier.

        Returns
        -------
        ClassifierOutput
        """
        now = datetime.now(timezone.utc).isoformat()
        raw = RawQuery(
            query_id=str(uuid.uuid4()),
            session_id=session_id,
            raw_text=text,
            timestamp=now,
            character_count=len(text),
            word_count=len(text.split()),
        )
        return self.classify(raw)

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2 dual-classifier logic
    # ──────────────────────────────────────────────────────────────────────

    def _phase2_classify(self, preprocessor) -> dict:
        """
        Phase 2 routing:
          1. Run XGBoost first (fast)
          2. If XGBoost confidence >= threshold → use XGBoost result
          3. If XGBoost confidence < threshold → fall back to LLM
             (pass XGBoost scores as context hints to LLM)
        """
        xgb_result = classify_with_xgboost(preprocessor)

        if xgb_result["xgboost_confidence"] >= C.XGBOOST_FALLBACK_THRESHOLD:
            # XGBoost is confident — use it directly
            return xgb_result
        else:
            # XGBoost is uncertain — use LLM with XGBoost scores as extra context
            log.debug(
                "XGBoost confidence %.2f below threshold %.2f for query_id=%s. "
                "Falling back to LLM.",
                xgb_result["xgboost_confidence"],
                C.XGBOOST_FALLBACK_THRESHOLD,
                preprocessor.query_id,
            )

            # Temporarily augment layer1 scores with XGBoost scores
            # so the LLM prompt includes them as additional signals
            original_l1_scores = preprocessor.layer1_all_scores.copy()
            preprocessor.layer1_all_scores = {
                **original_l1_scores,
                **{f"xgb_{k}": v for k, v in xgb_result["all_scores"].items()},
            }

            llm_result = classify_with_llm(preprocessor)

            # Restore original l1 scores
            preprocessor.layer1_all_scores = original_l1_scores

            # Merge XGBoost metadata into LLM result
            llm_result["xgboost_confidence"]    = xgb_result["xgboost_confidence"]
            llm_result["xgboost_feature_vector"] = xgb_result["xgboost_feature_vector"]
            llm_result["classified_by"]          = "layer2_hybrid"

            return llm_result

    # ──────────────────────────────────────────────────────────────────────
    # Decomposition handling
    # ──────────────────────────────────────────────────────────────────────

    def _classify_decomposed(
        self, raw_query: RawQuery, preprocessor, phase: str, t_start: float
    ) -> ClassifierOutput:
        """
        Classify each sub-query independently and assemble a parent result.

        The parent query_type is set to the most common sub-query type
        (or "hybrid" if all types differ). The Router handles decomposed
        queries by routing each sub-query to its own retriever.
        """
        sub_classifications = []

        for sq_text in preprocessor.sub_queries:
            sq_result = self.classify_text(sq_text, raw_query.session_id)
            sub_classifications.append(sq_result)

        # Parent type = most common sub-query type
        type_counts: dict[str, int] = {}
        for sc in sub_classifications:
            type_counts[sc.query_type] = type_counts.get(sc.query_type, 0) + 1
        parent_type = max(type_counts, key=type_counts.get)

        t_end = time.perf_counter()

        return ClassifierOutput(
            query_id=raw_query.query_id,
            session_id=raw_query.session_id,
            raw_text=raw_query.raw_text,
            cleaned_text=preprocessor.cleaned_text,
            timestamp=raw_query.timestamp,
            query_type=parent_type,
            confidence=min(sc.confidence for sc in sub_classifications),
            classified_by="decomposed",
            all_scores=preprocessor.layer1_all_scores,
            llm_reasoning=f"Decomposed into {len(sub_classifications)} sub-queries.",
            needs_decomposition=True,
            sub_queries=preprocessor.sub_queries,
            sub_query_classifications=sub_classifications,
            preprocessing_time_ms=preprocessor.preprocessing_time_ms,
            classification_time_ms=0.0,
            total_time_ms=round((t_end - t_start) * 1000, 2),
            phase=phase,
        )

    def _classify_sub_queries(
        self, parent_result: ClassifierOutput, phase: str
    ) -> ClassifierOutput:
        """Classify sub-queries that were detected by the LLM (not preprocessor)."""
        sub_classifications = []
        for sq_text in parent_result.sub_queries:
            sq = self.classify_text(sq_text, parent_result.session_id)
            sub_classifications.append(sq)
        parent_result.sub_query_classifications = sub_classifications
        return parent_result

    # ──────────────────────────────────────────────────────────────────────
    # Output builders
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_no_retrieval_output(
        raw_query: RawQuery, preprocessor, t_start: float
    ) -> ClassifierOutput:
        t_end = time.perf_counter()
        return ClassifierOutput(
            query_id=raw_query.query_id,
            session_id=raw_query.session_id,
            raw_text=raw_query.raw_text,
            cleaned_text=preprocessor.cleaned_text,
            timestamp=raw_query.timestamp,
            query_type="no_retrieval",
            confidence=0.98,
            classified_by="layer1_no_retrieval_gate",
            all_scores={"no_retrieval": 0.98},
            llm_reasoning=preprocessor.no_retrieval_reason,
            preprocessing_time_ms=preprocessor.preprocessing_time_ms,
            classification_time_ms=0.0,
            total_time_ms=round((t_end - t_start) * 1000, 2),
            phase="no_retrieval",
        )

    @staticmethod
    def _build_layer1_output(
        raw_query: RawQuery, preprocessor, t_start: float
    ) -> ClassifierOutput:
        t_end = time.perf_counter()
        return ClassifierOutput(
            query_id=raw_query.query_id,
            session_id=raw_query.session_id,
            raw_text=raw_query.raw_text,
            cleaned_text=preprocessor.cleaned_text,
            timestamp=raw_query.timestamp,
            query_type=preprocessor.layer1_predicted_type,
            confidence=preprocessor.layer1_confidence,
            classified_by="layer1_preprocessor",
            all_scores=preprocessor.layer1_all_scores,
            llm_reasoning=f"Layer 1 confidence: {preprocessor.layer1_confidence:.2f}",
            preprocessing_time_ms=preprocessor.preprocessing_time_ms,
            classification_time_ms=0.0,
            total_time_ms=round((t_end - t_start) * 1000, 2),
            phase="layer1",
        )

    @staticmethod
    def _build_layer2_output(
        raw_query: RawQuery,
        preprocessor,
        l2_result: dict,
        classification_ms: float,
        total_start: float,
        phase: str,
    ) -> ClassifierOutput:
        t_end = time.perf_counter()
        return ClassifierOutput(
            query_id=raw_query.query_id,
            session_id=raw_query.session_id,
            raw_text=raw_query.raw_text,
            cleaned_text=preprocessor.cleaned_text,
            timestamp=raw_query.timestamp,
            query_type=l2_result["query_type"],
            confidence=l2_result["confidence"],
            classified_by=l2_result.get("classified_by", "layer2"),
            all_scores=l2_result.get("all_scores", preprocessor.layer1_all_scores),
            llm_reasoning=l2_result.get("reasoning", ""),
            llm_raw_response=l2_result.get("llm_raw_response", ""),
            xgboost_confidence=l2_result.get("xgboost_confidence"),
            xgboost_feature_vector=l2_result.get("xgboost_feature_vector"),
            needs_decomposition=l2_result.get("needs_decomposition", False),
            sub_queries=l2_result.get("sub_queries", []),
            preprocessing_time_ms=preprocessor.preprocessing_time_ms,
            classification_time_ms=round(classification_ms, 2),
            total_time_ms=round((t_end - total_start) * 1000, 2),
            phase=f"phase1_llm" if phase == "phase1" else f"phase2",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Phase management
    # ──────────────────────────────────────────────────────────────────────

    def _get_current_phase(self) -> str:
        """
        Determine whether we are in Phase 1 (LLM) or Phase 2 (XGBoost).

        Phase 2 activates when:
          1. Transition conditions are met (enough labelled data)
          2. XGBoost model file exists on disk
        """
        if self._force_phase:
            return self._force_phase

        # Re-check periodically to detect transition without restart
        if self._current_phase is None:
            self._recheck_phase()

        return self._current_phase

    def _recheck_phase(self) -> None:
        """Check transition conditions and update phase state."""
        try:
            if is_model_loaded() and should_transition_to_phase2():
                self._current_phase = "phase2"
                log.info("Phase 2 active — using XGBoost classifier.")
            else:
                self._current_phase = "phase1"
                log.info("Phase 1 active — using LLM classifier.")
        except Exception as exc:
            log.warning("Phase check failed: %s. Defaulting to Phase 1.", exc)
            self._current_phase = "phase1"

    # ──────────────────────────────────────────────────────────────────────
    # Logging and feature storage
    # ──────────────────────────────────────────────────────────────────────

    def _log_and_store_features(
        self, result: ClassifierOutput, preprocessor
    ) -> None:
        """
        Log the classification result and save feature vector to the database.
        This is the data collection step that enables Phase 2 training.
        Runs asynchronously — does not block the return to the caller.
        """
        try:
            # Log classification to feedback DB
            self._logger.log_classification(result)

            # Save feature vector for future XGBoost training
            if result.query_type != "no_retrieval":
                fv = build_feature_vector(preprocessor)
                self._logger.save_feature_vector(fv)

        except Exception as exc:
            # Logging failure must never crash the classification pipeline
            log.warning(
                "Failed to log classification for query_id=%s: %s",
                result.query_id, exc
            )
