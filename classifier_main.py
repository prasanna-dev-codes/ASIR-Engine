"""
main.py
=======
Entry point demonstrating the complete ASIR-RAG query classifier pipeline.

Run this file to verify the entire pipeline works end to end:
    python main.py

PREREQUISITES:
    pip install spacy requests xgboost scikit-learn sentence-transformers joblib
    python -m spacy download en_core_web_sm
    ollama pull llama3
    ollama serve          (run in a separate terminal)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

# Configure logging before any imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import config as C
from classifier.query_classifier import QueryClassifier
from feedback.logger import FeedbackLogger
from feedback.transition import get_transition_status, should_transition_to_phase2
from shared.models import RawQuery


# ---------------------------------------------------------------------------
# Demo queries — one of each type
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    # (raw_text,                                          expected_type)
    ("What is the boiling point of water?",              "factual"),
    ("When was the Eiffel Tower built?",                 "factual"),
    ("What does mRNA stand for?",                        "no_retrieval"),
    ("How did colonialism influence India's economy?",   "relational"),
    ("What is the relationship between cortisol and stress response?", "relational"),
    ("Compare supervised and unsupervised learning",     "comparative"),
    ("What are the differences between RAM and ROM?",    "comparative"),
    ("Explain how the immune system works",              "exploratory"),
    ("Give me an overview of quantum computing",         "exploratory"),
    ("What causes the sky to be blue?",                  "factual"),
]


def run_demo():
    """
    Classify all demo queries and print a summary table.
    Shows the pipeline working end to end.
    """
    print("\n" + "=" * 80)
    print("  ASIR-RAG Query Classifier — End-to-End Demo")
    print("=" * 80 + "\n")

    # Instantiate once — reuse for all queries
    classifier = QueryClassifier()

    results = []

    for raw_text, expected in DEMO_QUERIES:
        result = classifier.classify_text(raw_text, session_id="demo_session")

        correct = "✓" if result.query_type == expected else "✗"
        results.append({
            "query":     raw_text[:55] + ("..." if len(raw_text) > 55 else ""),
            "predicted": result.query_type,
            "expected":  expected,
            "confidence": result.confidence,
            "by":        result.classified_by,
            "time_ms":   result.total_time_ms,
            "correct":   correct,
        })

        print(
            f"  {correct}  [{result.query_type:<14}] "
            f"conf={result.confidence:.2f}  "
            f"by={result.classified_by:<22}  "
            f"{result.total_time_ms:>7.1f}ms  "
            f"{raw_text[:50]}"
        )

    # Summary
    n_correct = sum(1 for r in results if r["correct"] == "✓")
    print(f"\n  Accuracy: {n_correct}/{len(results)} "
          f"({n_correct/len(results)*100:.0f}%)\n")

    # Transition status
    print("─" * 80)
    status = get_transition_status()
    print(f"  Phase 2 readiness: {'READY ✓' if status['ready'] else 'NOT YET'}")
    print(f"  Labelled samples:  {status['total_labelled']} / {status['total_needed']}")
    if not status["ready"] and "per_type" in status:
        print("  Per-type counts:")
        for qt, info in status["per_type"].items():
            bar = "█" * min(info["have"] // 5, 20)
            print(f"    {qt:<15} {info['have']:>3}/{info['need']}  {bar}")
    print()


def demonstrate_feedback():
    """
    Show how to record user feedback after a response is shown.
    This is called from the Streamlit UI after the user rates an answer.
    """
    classifier = QueryClassifier()
    logger     = FeedbackLogger()

    # Classify a query
    result = classifier.classify_text(
        "How does insulin resistance lead to Type 2 diabetes?",
        session_id="feedback_demo"
    )

    print(f"\nClassified as: {result.query_type} (confidence: {result.confidence:.2f})")
    print(f"Query ID: {result.query_id}")

    # Simulate: user saw the answer and rated it 4/5 and found it correct
    logger.record_feedback(
        query_id=result.query_id,
        user_rating=4,
        answer_was_correct=True,
    )
    print("Feedback recorded: rating=4, correct=True")


def demonstrate_xgboost_training():
    """
    Show how to trigger XGBoost training once enough data is collected.
    Run this after collecting 300+ labelled interactions.
    """
    from classifier.xgboost_classifier import train_xgboost

    if should_transition_to_phase2():
        print("\nTransition conditions met. Training XGBoost...")
        results = train_xgboost()
        print(f"Training complete. Accuracy: {results['accuracy']:.4f}")
        print(f"Training samples: {results['n_train']}")
        print(f"Test samples: {results['n_test']}")
    else:
        status = get_transition_status()
        print(f"\nNot enough data for Phase 2 yet.")
        print(f"Labelled samples: {status['total_labelled']} / {status['total_needed']}")


if __name__ == "__main__":
    # Run the full demo
    run_demo()

    # Demonstrate feedback recording
    demonstrate_feedback()

    # Demonstrate how training would be triggered
    demonstrate_xgboost_training()
