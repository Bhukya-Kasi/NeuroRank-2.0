
"""Phase 11E: Frozen-model held-out end-to-end evaluation."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from train_phase11b import (
    ROOT,
    FEATURES,
    K_RETRIEVE,
    GRADES,
    read_split,
    build_catalog,
)
from train_phase11c import query_features


DATA = ROOT / "data" / "phase11"
ART = ROOT / "artifacts" / "phase11"
OUT = ROOT / "results" / "phase11"

MODES = [
    "bm25",
    "semantic",
    "rrf",
    "lambdamart",
    "adaptive",
]

TOP_K = 10


def dcg(grades):
    values = np.asarray(grades, dtype=float)
    discounts = np.log2(
        np.arange(2, len(values) + 2)
    )
    return float(
        np.sum((2.0 ** values - 1.0) / discounts)
    )


def ndcg_at_k(product_ids, judgments, k=10):
    actual = [
        judgments.get(str(pid), 0)
        for pid in product_ids[:k]
    ]
    actual += [0] * (k - len(actual))

    ideal = sorted(
        judgments.values(),
        reverse=True,
    )[:k]
    ideal += [0] * (k - len(ideal))

    denominator = dcg(ideal)
    if denominator == 0:
        return None

    return float(dcg(actual) / denominator)


def recall_at_k(product_ids, judgments, k=10):
    relevant = {
        pid
        for pid, grade in judgments.items()
        if grade > 0
    }

    if not relevant:
        return None

    returned = set(
        map(str, product_ids[:k])
    )

    return len(relevant & returned) / len(relevant)


def normalized_queries(frame):
    return set(
        frame["query"].fillna("").astype(str)
        .str.casefold()
        .str.split()
        .str.join(" ")
    )


def check_splits(train, validation, test):
    for first_name, first, second_name, second in [
        ("train", train, "validation", validation),
        ("train", train, "test", test),
        ("validation", validation, "test", test),
    ]:
        if (
            set(first["query_id"])
            & set(second["query_id"])
        ):
            raise ValueError(
                f"Query ID overlap: {first_name}/{second_name}"
            )

        if normalized_queries(first) & normalized_queries(second):
            raise ValueError(
                f"Query text overlap: {first_name}/{second_name}"
            )


def sort_indices(scores, indices):
    indices = np.asarray(indices, dtype=int)
    values = np.asarray(scores, dtype=float)[indices]
    order = np.argsort(
        -values,
        kind="stable",
    )
    return indices[order].tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    test = read_split("test")
    train = read_split("train")
    validation = read_split("validation")

    check_splits(train, validation, test)

    ranker_path = ART / "retrieval_aware_ranker.joblib"
    gate_path = ART / "query_reliability_gate.joblib"

    if not ranker_path.exists():
        raise FileNotFoundError(
            "Missing Phase 11B ranker."
        )

    if not gate_path.exists():
        raise FileNotFoundError(
            "Missing Phase 11C selector."
        )

    print("Loading frozen models...")
    # Load only local model artifacts you created and trust.
    bundle = joblib.load(ranker_path)
    gate_bundle = joblib.load(gate_path)

    if bundle.get("features") != FEATURES:
        raise ValueError("Ranker feature mismatch.")

    if bundle.get("candidate_depth") != K_RETRIEVE:
        raise ValueError("Retrieval depth mismatch.")

    ranker = bundle["ranker"]
    gate = gate_bundle["gate"]

    print("Loading Phase 11 retrieval system...")

    (
        products,
        encoder,
        embeddings,
        index,
        bm25,
        id_to_idx,
    ) = build_catalog()

    if len(products) == 0:
        raise ValueError("Empty catalog.")

    product_ids = products["product_id"].astype(str).tolist()
    titles = products["title"].fillna("").astype(str).tolist()

    # Verify that all labeled test products exist in catalog.
    missing = set(test["product_id"]) - set(id_to_idx)
    if missing:
        raise ValueError(
            f"{len(missing)} judged test products missing "
            "from catalog."
        )

    records = []
    choices = []
    queries_evaluated = 0

    for qid, group in test.groupby("query_id", sort=True):
        if group["query"].nunique() != 1:
            raise ValueError(
                f"Conflicting query text for ID {qid}"
            )

        query = str(group.iloc[0]["query"])
        qtokens = query.lower().split()
        qset = set(qtokens)

        judgments = {
            str(row.product_id): GRADES[str(row.label)]
            for row in group.itertuples(index=False)
        }

        if not any(g > 0 for g in judgments.values()):
            continue

        bm25_scores = np.asarray(
            bm25.get_scores(qtokens),
            dtype=np.float32,
        )

        query_vector = np.asarray(
            encoder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

        similarities = embeddings @ query_vector[0]

        depth = min(K_RETRIEVE, len(products))

        top_bm25 = np.argsort(
            -bm25_scores,
            kind="stable",
        )[:depth].tolist()

        _, found = index.search(
            np.ascontiguousarray(query_vector),
            depth,
        )

        top_semantic = [
            int(i) for i in found[0]
            if i >= 0
        ]

        candidates = sorted(
            set(top_bm25) | set(top_semantic)
        )

        lexical_rank = {
            idx: rank + 1
            for rank, idx in enumerate(top_bm25)
        }

        semantic_rank = {
            idx: rank + 1
            for rank, idx in enumerate(top_semantic)
        }

        feature_rows = []
        rrf_scores = []

        for idx in candidates:
            title = titles[idx].lower()
            title_tokens = title.split()

            overlap = len(
                qset & set(title_tokens)
            ) / max(len(qset), 1)

            b_rank = lexical_rank.get(idx)
            s_rank = semantic_rank.get(idx)

            feature_rows.append([
                float(bm25_scores[idx]),
                float(similarities[idx]),
                float(overlap),
                float(query.lower() in title),
                1.0 / b_rank if b_rank else 0.0,
                1.0 / s_rank if s_rank else 0.0,
                float(len(title_tokens)),
                float(len(qtokens)),
            ])

            rrf_scores.append(
                (1.0 / (60 + b_rank) if b_rank else 0.0)
                + (1.0 / (60 + s_rank) if s_rank else 0.0)
            )

        X = np.asarray(feature_rows, dtype=np.float32)

        if X.shape[1] != len(FEATURES):
            raise ValueError("Serving feature shape mismatch.")

        learned = ranker.predict(X)
        semantic_values = X[:, 1]
        bm25_values = X[:, 0]

        order_bm25 = sort_indices(
            bm25_values,
            range(len(candidates)),
        )

        order_semantic = sort_indices(
            semantic_values,
            range(len(candidates)),
        )

        order_rrf = sort_indices(
            rrf_scores,
            range(len(candidates)),
        )

        order_learned = sort_indices(
            learned,
            range(len(candidates)),
        )

        p_lambda = float(
            gate.predict_proba(
                np.asarray(
                    [query_features(query)],
                    dtype=float,
                )
            )[0, 1]
        )

        use_lambda = p_lambda > 0.5

        rankings = {
            "bm25": order_bm25,
            "semantic": order_semantic,
            "rrf": order_rrf,
            "lambdamart": order_learned,
            "adaptive": (
                order_learned
                if use_lambda
                else order_semantic
            ),
        }

        candidate_ids = [
            product_ids[idx]
            for idx in candidates
        ]

        candidate_recall = recall_at_k(
            candidate_ids,
            judgments,
            len(candidate_ids),
        )

        choices.append({
            "query_id": qid,
            "query": query,
            "p_lambda": p_lambda,
            "selected": (
                "lambdamart"
                if use_lambda
                else "semantic"
            ),
            "candidate_pool_size": len(candidates),
            "candidate_recall": candidate_recall,
        })

        for mode, positions in rankings.items():
            ranked_ids = [
                product_ids[candidates[position]]
                for position in positions
            ]

            records.append({
                "query_id": qid,
                "mode": mode,
                "ndcg10": ndcg_at_k(
                    ranked_ids,
                    judgments,
                    TOP_K,
                ),
                "recall10": recall_at_k(
                    ranked_ids,
                    judgments,
                    10,
                ),
                "recall100": recall_at_k(
                    ranked_ids,
                    judgments,
                    100,
                ),
            })

        queries_evaluated += 1

        if queries_evaluated % 10 == 0:
            print(
                "Evaluated queries:",
                queries_evaluated,
            )

    if not records:
        raise ValueError(
            "No test queries with positive judgments."
        )

    per_query = pd.DataFrame(records)
    selection_df = pd.DataFrame(choices)

    per_query.to_csv(
        OUT / "phase11e_per_query.csv",
        index=False,
    )

    selection_df.to_csv(
        OUT / "phase11e_gate_choices.csv",
        index=False,
    )

    table = (
        per_query.groupby("mode", sort=False)[
            ["ndcg10", "recall10", "recall100"]
        ]
        .mean()
        .reindex(MODES)
    )

    table.to_csv(
        OUT / "phase11e_comparison.csv"
    )

    summary = {
        "evaluation": "held_out_end_to_end",
        "queries_evaluated": queries_evaluated,
        "catalog_products": int(len(products)),
        "candidate_depth_per_retriever": int(K_RETRIEVE),
        "mean_candidate_pool_recall": float(
            selection_df["candidate_recall"].mean()
        ),
        "lambda_selected_queries": int(
            (selection_df["selected"] == "lambdamart").sum()
        ),
        "metrics": {
            mode: {
                metric: round(float(table.loc[mode, metric]), 4)
                for metric in [
                    "ndcg10",
                    "recall10",
                    "recall100",
                ]
            }
            for mode in MODES
        },
        "judgment_policy": (
            "Unjudged returned products assigned zero "
            "gain for end-to-end evaluation."
        ),
        "selection_policy": (
            "Frozen Phase 11B ranker and Phase 11C "
            "gate; no test-based tuning."
        ),
    }

    (OUT / "phase11e_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL HELD-OUT RESULTS")
    print(table.round(4).to_string())
    print("\nCandidate recall:",
          round(summary["mean_candidate_pool_recall"], 4))
    print("\nSaved:", OUT / "phase11e_metrics.json")


if __name__ == "__main__":
    main()
