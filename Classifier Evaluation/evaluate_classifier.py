import uuid
from datetime import datetime
from collections import Counter

from classifier.query_classifier import QueryClassifier
from shared.models import RawQuery
from test_data import test_queries


def evaluate():
    print("\n🚀 Running Query Classifier Evaluation...\n")

    classifier = QueryClassifier()

    y_true = []
    y_pred = []
    correct = 0

    for i, (query, true_label) in enumerate(test_queries, 1):

        # ✅ Convert string → RawQuery object (CRITICAL FIX)
        raw_query = RawQuery(
            query_id=str(uuid.uuid4()),
            session_id="test_session",
            raw_text=query,
            timestamp=datetime.utcnow().isoformat(),
            character_count=len(query),
            word_count=len(query.split())
        )

        try:
            result = classifier.classify(raw_query)
            predicted = result.query_type
        except Exception as e:
            print(f"❌ Error processing query: {query}")
            print(f"Error: {e}")
            predicted = "error"

        y_true.append(true_label)
        y_pred.append(predicted)

        is_correct = predicted == true_label
        if is_correct:
            correct += 1

        print(f"{i}. Query: {query}")
        print(f"   ➤ Predicted: {predicted}")
        print(f"   ➤ Actual:    {true_label}")
        print(f"   ➤ Result:    {'✅ Correct' if is_correct else '❌ Wrong'}")
        print("-" * 60)

    # ✅ Overall accuracy
    total = len(test_queries)
    accuracy = correct / total if total > 0 else 0

    print("\n📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Total Queries: {total}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {accuracy:.2%}")

    # ✅ Per-class accuracy
    print("\n📌 Per-Class Accuracy:")
    class_stats = {}

    for label in set(y_true):
        total_label = sum(1 for y in y_true if y == label)
        correct_label = sum(
            1 for yt, yp in zip(y_true, y_pred)
            if yt == label and yp == label
        )
        class_stats[label] = (correct_label, total_label)

    for label, (c, t) in class_stats.items():
        if t > 0:
            print(f"{label:12} → {c}/{t} = {c/t:.2%}")

    # ✅ Prediction distribution
    print("\n📌 Prediction Distribution:")
    pred_counts = Counter(y_pred)
    for label, count in pred_counts.items():
        print(f"{label:12} → {count}")

    print("\n✅ Evaluation Complete!\n")


if __name__ == "__main__":
    evaluate()