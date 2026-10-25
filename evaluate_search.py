
"""
NeuroRank Phase 7:
End-to-end retrieval and ranking evaluation.

Run:
    python evaluate_search.py

Uses existing:
    data/examples.csv
    data/products.csv
    artifacts/reliability_ranker.joblib

Outputs:
    results/end_to_end_metrics.json
    results/per_query_evaluation.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from production_api import get_system, rank_products


ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "data" / "examples.csv"
OUT = ROOT / "results"

GRADES = {
    "I": 0,
    "C": 1,
    "S": 2,
    "E": 3,
}

KS = (10, 50, 100)
MODES = ("bm25", "semantic", "rrf", "lambdamart")


def dcg(grades):
    """Discounted cumulative gain."""
    values = np.asarray(grades, dtype=float)

    if not len(values):
        return 0.0

    gains = np.power(2.0, values) - 1.0
    discounts = np.log2(
        np.arange(2, len(values) + 2)
    )

    return float(np.sum(gains / discounts))


def recall_at_k(ranked_ids, relevant_ids, k):
    """Recall relative to known relevant products."""
    if not relevant_ids:
        return np.nan

    top = set(ranked_ids[:k])

    return len(top & relevant_ids) / len(relevant_ids)


def ndcg_at_k(ranked_ids, judgments, k):
    """
    Treat unjudged products as grade zero.
    Ideal DCG uses all available judged grades.
    """
    actual = [
        judgments.get(product_id, 0)
        for product_id in ranked_ids[:k]
    ]

    actual += [0] * (k - len(actual))

    ideal = sorted(
        judgments.values(),
        reverse=True,
    )[:k]

    ideal += [0] * (k - len(ideal))

    denominator = dcg(ideal)

    if denominator == 0:
        return np.nan

    return dcg(actual) / denominator


def retrieved_oracle_ndcg(ranked_ids, judgments, k=10):
    """
    Upper bound if the retrieved candidate set
    were perfectly reranked using known labels.
    """
    available = [
        judgments.get(pid, 0)
        for pid in ranked_ids
    ]

    available = sorted(
        available,
        reverse=True,
    )[:k]

    available += [0] * (k - len(available))

    ideal = sorted(
        judgments.values(),
        reverse=True,
    )[:k]

    ideal += [0] * (k - len(ideal))

    denominator = dcg(ideal)

    if denominator == 0:
        return np.nan

    return dcg(available) / denominator


def safe_mean(values):
    values = np.asarray(values, dtype=float)

    valid = values[np.isfinite(values)]

    return (
        round(float(np.mean(valid)), 4)
        if len(valid)
        else None
    )


def main():

    print("\nLoading held-out Amazon queries...")

    examples = pd.read_csv(EXAMPLES)

    test = examples[
        examples["split"] == "test"
    ].copy()

    if test.empty:
        raise ValueError("No test queries found")

    test["product_id"] = (
        test["product_id"].astype(str)
    )

    if not set(test["label"]).issubset(GRADES):
        raise ValueError("Unexpected ESCI labels")

    print("Loading the trained search system...")

    # Reuses the saved LambdaMART model.
    # Builds the local retrieval indexes once.
    engine, bundle = get_system()

    catalog_size = len(engine.products)
    max_k = min(max(KS), catalog_size)

    print("Catalog products:", catalog_size)

    rows = []

    for query_id, group in test.groupby(
        "query_id",
        sort=False,
    ):

        query = str(group.iloc[0]["query"])

        judgments = {}

        for record in group.itertuples():
            pid = str(record.product_id)
            grade = GRADES[record.label]

            judgments[pid] = max(
                grade,
                judgments.get(pid, 0),
            )

        relevant = {
            pid
            for pid, grade in judgments.items()
            if grade > 0
        }

        if not relevant:
            continue

        # Evaluate identical query/candidate universe
        # for each retrieval and ranking method.
        for mode in MODES:

            results = rank_products(
                engine,
                bundle,
                query,
                k=max_k,
                mode=mode,
            )

            ranked_ids = [
                str(item["product_id"])
                for item in results
            ]

            if len(ranked_ids) != len(set(ranked_ids)):
                raise ValueError(
                    "Duplicate product IDs in retrieval"
                )

            row = {
                "query_id": str(query_id),
                "query": query,
                "mode": mode,
                "known_relevant": len(relevant),
                "retrieved_count": len(ranked_ids),
                "NDCG@10": ndcg_at_k(
                    ranked_ids,
                    judgments,
                    10,
                ),
                "Oracle_NDCG@10": (
                    retrieved_oracle_ndcg(
                        ranked_ids,
                        judgments,
                        k=10,
                    )
                ),
            }

            for k in KS:
                row[f"Recall@{k}"] = recall_at_k(
                    ranked_ids,
                    relevant,
                    k,
                )

            rows.append(row)

        print(f"Evaluated query ID: {query_id}")

    report = pd.DataFrame(rows)

    if report.empty:
        raise ValueError(
            "No test queries with relevant judgments"
        )

    summary = {}

    for mode in MODES:

        part = report[
            report["mode"] == mode
        ]

        summary[mode] = {
            "queries": int(
                part["query_id"].nunique()
            ),
            "NDCG@10": safe_mean(
                part["NDCG@10"]
            ),
            "Oracle_NDCG@10": safe_mean(
                part["Oracle_NDCG@10"]
            ),
        }

        for k in KS:

            summary[mode][f"Recall@{k}"] = (
                safe_mean(part[f"Recall@{k}"])
            )

    result = {
        "dataset": "Amazon ESCI sampled catalog",
        "evaluation": "End-to-end, judgment-based",
        "catalog_products": catalog_size,
        "total_evaluated_queries": int(
            report["query_id"].nunique()
        ),
        "relevance_definition": (
            "Exact, Substitute or Complement"
        ),
        "unjudged_policy": (
            "Treated as zero for NDCG"
        ),
        "recall_denominator": (
            "Known relevant products in the sampled catalog"
        ),
        "methods": summary,
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUT / "end_to_end_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    report.to_csv(
        OUT / "per_query_evaluation.csv",
        index=False,
    )

    print("\nPHASE 7 RESULTS")
    print(json.dumps(result, indent=2))

    print("\nReports saved in results/.")


if __name__ == "__main__":
    main()
