
"""Phase 8: Train-only retrieval and robustness diagnostics."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from production_api import get_system, rank_products


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "examples.csv"
OUT = ROOT / "results"

SEED = 42
N_QUERIES = 20
K = 10

GRADE = {
    "I": 0,
    "C": 1,
    "S": 2,
    "E": 3,
}


def dcg(values):
    a = np.asarray(values, dtype=float)
    discounts = np.log2(np.arange(2, len(a) + 2))
    return float(np.sum((2.0 ** a - 1.0) / discounts))


def ndcg(ids, judgments, k=10):
    actual = [
        judgments.get(str(pid), 0)
        for pid in ids[:k]
    ]
    actual += [0] * (k - len(actual))

    ideal = sorted(
        judgments.values(),
        reverse=True,
    )[:k]
    ideal += [0] * (k - len(ideal))

    denominator = dcg(ideal)

    return (
        float(dcg(actual) / denominator)
        if denominator
        else None
    )


def transform_query(query):
    return {
        "original": query,
        "uppercase": query.upper(),
        "trailing_punctuation": query + " !!!",
    }


def overlap_at_k(first, second, k=10):
    a = set(first[:k])
    b = set(second[:k])
    return len(a & b) / max(1, len(a))


def score_feature_shift(engine, group, query, retrieved_ids):

    ids = {
        str(pid): i
        for i, pid in enumerate(engine.products["product_id"])
    }

    judged_idx = [
        ids[str(pid)]
        for pid in group["product_id"]
        if str(pid) in ids
    ]

    retrieved_idx = [
        ids[str(pid)]
        for pid in retrieved_ids
        if str(pid) in ids
    ]

    if not judged_idx or not retrieved_idx:
        return None

    bm25 = np.asarray(
        engine.bm25.get_scores(query.lower().split()),
        dtype=float,
    )

    qvec = engine.model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    semantic = engine.embeddings @ qvec

    return {
        "judged_bm25_mean": float(
            np.mean(bm25[judged_idx])
        ),
        "retrieved_bm25_mean": float(
            np.mean(bm25[retrieved_idx])
        ),
        "judged_semantic_mean": float(
            np.mean(semantic[judged_idx])
        ),
        "retrieved_semantic_mean": float(
            np.mean(semantic[retrieved_idx])
        ),
    }


def main():

    examples = pd.read_csv(DATA).fillna("")

    # Diagnostics use training queries only.
    # The previously examined test split is not used.
    train = examples[
        examples["split"] == "train"
    ].copy()

    if train.empty:
        raise ValueError("No training queries available")

    print("Loading NeuroRank models and indexes...")

    engine, bundle = get_system()

    query_ids = np.asarray(
        sorted(train["query_id"].astype(str).unique())
    )

    sampled = np.random.default_rng(SEED).choice(
        query_ids,
        size=min(N_QUERIES, len(query_ids)),
        replace=False,
    )

    records = []
    perturbations = []

    for qid in sampled:

        group = train[
            train["query_id"].astype(str) == qid
        ]

        query = str(group.iloc[0]["query"])

        judgments = {
            str(row.product_id): GRADE[str(row.label)]
            for row in group.itertuples()
        }

        outputs = {}

        for mode in ("semantic", "lambdamart"):

            outputs[mode] = [
                item["product_id"]
                for item in rank_products(
                    engine,
                    bundle,
                    query,
                    K,
                    mode,
                )
            ]

        semantic_ndcg = ndcg(
            outputs["semantic"],
            judgments,
            K,
        )

        learned_ndcg = ndcg(
            outputs["lambdamart"],
            judgments,
            K,
        )

        shift = score_feature_shift(
            engine,
            group,
            query,
            outputs["semantic"],
        )

        difference = (
            learned_ndcg - semantic_ndcg
            if (
                learned_ndcg is not None
                and semantic_ndcg is not None
            )
            else None
        )

        records.append({
            "query_id": qid,
            "query": query,
            "semantic_ndcg10_proxy": semantic_ndcg,
            "lambdamart_ndcg10_proxy": learned_ndcg,
            "lambdamart_minus_semantic": difference,
            **(shift or {}),
        })

        for kind, modified in transform_query(query).items():

            modified_ids = [
                item["product_id"]
                for item in rank_products(
                    engine,
                    bundle,
                    modified,
                    K,
                    "semantic",
                )
            ]

            perturbations.append({
                "query_id": qid,
                "variant": kind,
                "overlap_with_original_at_10": overlap_at_k(
                    outputs["semantic"],
                    modified_ids,
                    K,
                ),
            })

        print("Analyzed query:", qid)

    OUT.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(records)
    robustness = pd.DataFrame(perturbations)

    report.to_csv(
        OUT / "phase8_query_diagnostics.csv",
        index=False,
    )

    robustness.to_csv(
        OUT / "phase8_robustness.csv",
        index=False,
    )

    summary = {
        "source_split": "train_only",
        "sampled_queries": len(report),
        "judgment_policy": (
            "Unjudged treated as zero for diagnostic NDCG"
        ),
        "semantic_mean_ndcg10_proxy": round(
            float(report["semantic_ndcg10_proxy"].mean()),
            4,
        ),
        "lambdamart_mean_ndcg10_proxy": round(
            float(report["lambdamart_ndcg10_proxy"].mean()),
            4,
        ),
        "lambdamart_wins": int(
            (report["lambdamart_minus_semantic"] > 0).sum()
        ),
        "semantic_wins": int(
            (report["lambdamart_minus_semantic"] < 0).sum()
        ),
        "ties": int(
            (report["lambdamart_minus_semantic"] == 0).sum()
        ),
        "mean_bm25_judged": round(
            float(report["judged_bm25_mean"].mean()), 4
        ),
        "mean_bm25_retrieved": round(
            float(report["retrieved_bm25_mean"].mean()), 4
        ),
        "mean_semantic_judged": round(
            float(report["judged_semantic_mean"].mean()), 4
        ),
        "mean_semantic_retrieved": round(
            float(report["retrieved_semantic_mean"].mean()), 4
        ),
        "query_perturbation_overlap": {
            str(kind): round(
                float(
                    values["overlap_with_original_at_10"].mean()
                ),
                4,
            )
            for kind, values in robustness.groupby("variant")
        },
    }

    (OUT / "phase8_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 8 DIAGNOSTICS")
    print(json.dumps(summary, indent=2))
    print("\nReports saved in results/")


if __name__ == "__main__":
    main()
