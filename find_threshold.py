import json
import numpy as np
import laya


MODEL = "convaiinnovations/laya-typed-decisions"
DATASET = "threshold_dataset.jsonl"


def load_dataset():
    rows = []

    with open(DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            rows.append(json.loads(line))

    return rows


def get_probability(agent, question, context):
    state = {
        "question": question,
        "context": context,
    }

    questions = {
        "context_sufficient": {
            "type": "noul",
            "instructions": (
                "Determine whether the retrieved context contains enough "
                "explicit factual information to answer the user's question "
                "without using outside knowledge."
            ),
            "criteria": {
                "false": (
                    "The context is insufficient, unrelated, or missing "
                    "information needed to answer the question."
                ),
                "true": (
                    "The context contains enough factual information to "
                    "produce a grounded answer to the question."
                ),
            },
        }
    }

    result = agent.predict(state, questions)

    return float(
        result["answers"]["context_sufficient"]["noul"]
    )


def metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)

    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    accuracy = (tp + tn) / len(y_true)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def main():
    data = load_dataset()

    print(f"Loaded {len(data)} examples")

    print("\nLoading Laya...")
    agent = laya.load(
        MODEL,
        device="cpu",
        fast=False,
        compile=False,
    )

    probabilities = []
    expected = []

    for i, row in enumerate(data, start=1):

        probability = get_probability(
            agent,
            row["question"],
            row["context"],
        )

        probabilities.append(probability)
        expected.append(bool(row["expected"]))

        print(
            f"{i:03d} | "
            f"P={probability:.4f} | "
            f"expected={row['expected']}"
        )

    print("\n")
    print("=" * 90)
    print("THRESHOLD SWEEP")
    print("=" * 90)

    results = []

    for threshold in np.arange(0.10, 0.71, 0.01):

        predictions = [
            p >= threshold
            for p in probabilities
        ]

        m = metrics(expected, predictions)

        results.append({
            "threshold": round(float(threshold), 2),
            **m,
        })

    # Highest F1
    best_f1 = max(
        results,
        key=lambda x: x["f1"]
    )

    # Lowest threshold achieving >=95% precision
    high_precision = [
        r for r in results
        if r["precision"] >= 0.95
    ]

    best_precision_threshold = (
        min(
            high_precision,
            key=lambda x: x["threshold"]
        )
        if high_precision
        else None
    )

    print(
        "\nBest F1 threshold:\n",
        best_f1
    )

    print(
        "\nLowest threshold with >=95% precision:\n",
        best_precision_threshold
    )

    print("\nAll thresholds:\n")

    print(
        f"{'Threshold':<10}"
        f"{'Accuracy':<10}"
        f"{'Precision':<10}"
        f"{'Recall':<10}"
        f"{'F1':<10}"
        f"{'FP':<6}"
        f"{'FN':<6}"
    )

    print("-" * 70)

    for r in results:

        print(
            f"{r['threshold']:<10.2f}"
            f"{r['accuracy']:<10.3f}"
            f"{r['precision']:<10.3f}"
            f"{r['recall']:<10.3f}"
            f"{r['f1']:<10.3f}"
            f"{r['fp']:<6}"
            f"{r['fn']:<6}"
        )


if __name__ == "__main__":
    main()