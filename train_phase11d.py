
"""Phase 11D: Feature-group ablation study for LambdaMART."""

from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_phase11b import (
    ROOT,
    FEATURES,
    K_EVAL,
    read_split,
    build_catalog,
    extract_rows,
    ndcg_judged,
)


OUT = ROOT / "results" / "phase11"
ART = ROOT / "artifacts" / "phase11"

SEED = 42

FEATURE_GROUPS = {
    "lexical": [
        "bm25",
        "bm25_rank",
    ],
    "semantic": [
        "semantic",
        "semantic_rank",
    ],
    "text_matching": [
        "title_overlap",
        "exact_phrase",
    ],
    "length": [
        "title_length",
        "query_length",
    ],
}


def make_model():

    return lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=5,
        label_gain=[0, 1, 3, 7],
        random_state=SEED,
        verbosity=-1,
        n_jobs=1,
    )


def evaluate_by_query(
    scores,
    labels,
    groups,
    query_ids,
):

    rows = []
    offset = 0

    for i, size in enumerate(groups):

        end = offset + size

        value = ndcg_judged(
            scores[offset:end],
            labels[offset:end],
            K_EVAL,
        )

        rows.append({
            "query_id": str(query_ids[i]),
            "ndcg10": value,
        })

        offset = end

    if offset != len(labels):
        raise ValueError(
            "Ranking group sizes do not match labels."
        )

    return pd.DataFrame(rows)


def train_experiment(
    experiment_name,
    selected_features,
    X_train,
    y_train,
    groups_train,
    X_valid,
    y_valid,
    groups_valid,
    query_ids,
):

    indices = [
        FEATURES.index(name)
        for name in selected_features
    ]

    train_matrix = X_train[:, indices]
    valid_matrix = X_valid[:, indices]

    model = make_model()

    model.fit(
        train_matrix,
        y_train,
        group=groups_train,
    )

    scores = model.predict(
        valid_matrix
    )

    per_query = evaluate_by_query(
        scores,
        y_valid,
        groups_valid,
        query_ids,
    )

    mean_ndcg = float(
        per_query["ndcg10"].mean()
    )

    importance = {
        name: int(value)
        for name, value in zip(
            selected_features,
            model.feature_importances_,
        )
    }

    return mean_ndcg, per_query, importance


def main():

    print("\nNEURORANK PHASE 11D")
    print("Feature Ablation Study\n")

    if len(FEATURES) != len(set(FEATURES)):
        raise ValueError(
            "Duplicate feature names."
        )

    all_group_features = [
        name
        for group in FEATURE_GROUPS.values()
        for name in group
    ]

    if set(all_group_features) != set(FEATURES):
        raise ValueError(
            "Feature groups do not cover all features."
        )

    train = read_split("train")
    validation = read_split("validation")

    # Prevent query leakage across train and validation.
    normalize = lambda q: " ".join(
        str(q).casefold().split()
    )

    if (
        set(train["query_id"])
        & set(validation["query_id"])
    ):
        raise ValueError(
            "Train and validation share query IDs."
        )

    if (
        set(train["query"].map(normalize))
        & set(validation["query"].map(normalize))
    ):
        raise ValueError(
            "Train and validation share query text."
        )

    print("Loading search catalog and embeddings...")

    (
        products,
        encoder,
        embeddings,
        index,
        bm25,
        id_to_idx,
    ) = build_catalog()

    print("\nExtracting training features...")

    (
        X_train,
        y_train,
        groups_train,
        train_details,
    ) = extract_rows(
        train,
        products,
        encoder,
        embeddings,
        index,
        bm25,
        id_to_idx,
    )

    print("\nExtracting validation features...")

    (
        X_valid,
        y_valid,
        groups_valid,
        valid_details,
    ) = extract_rows(
        validation,
        products,
        encoder,
        embeddings,
        index,
        bm25,
        id_to_idx,
    )

    # Each group is contiguous in extract_rows().
    query_ids = []
    offset = 0

    for size in groups_valid:

        block = valid_details.iloc[
            offset:offset + size
        ]

        query_ids.append(
            str(block.iloc[0]["query_id"])
        )

        offset += size

    if offset != len(valid_details):
        raise ValueError(
            "Validation group alignment failed."
        )

    experiments = {
        "full_model": list(FEATURES),
    }

    for group_name, removed in FEATURE_GROUPS.items():

        experiments[
            f"without_{group_name}"
        ] = [
            name
            for name in FEATURES
            if name not in removed
        ]

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ART.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    query_tables = []
    importances = {}

    for name, selected in experiments.items():

        print(f"\nTraining: {name}")

        mean_ndcg, per_query, importance = train_experiment(
            name,
            selected,
            X_train,
            y_train,
            groups_train,
            X_valid,
            y_valid,
            groups_valid,
            query_ids,
        )

        print(
            "Validation NDCG@10:",
            round(mean_ndcg, 4),
        )

        results.append({
            "experiment": name,
            "ndcg10": mean_ndcg,
            "feature_count": len(selected),
            "removed_features": (
                "none"
                if name == "full_model"
                else ", ".join(
                    feature
                    for feature in FEATURES
                    if feature not in selected
                )
            ),
        })

        per_query = per_query.rename(
            columns={
                "ndcg10": name,
            }
        )

        query_tables.append(per_query)

        importances[name] = importance

    summary = pd.DataFrame(results)

    full_score = float(
        summary.loc[
            summary["experiment"] == "full_model",
            "ndcg10",
        ].iloc[0]
    )

    summary["delta_vs_full"] = (
        summary["ndcg10"] - full_score
    )

    # Merge query-aligned scores for later paired analysis.
    query_comparison = query_tables[0]

    for table in query_tables[1:]:

        query_comparison = query_comparison.merge(
            table,
            on="query_id",
            how="inner",
            validate="one_to_one",
        )

    if len(query_comparison) != len(query_ids):
        raise ValueError(
            "Missing query scores after merging."
        )

    summary = summary.round(
        {
            "ndcg10": 4,
            "delta_vs_full": 4,
        }
    )

    summary.to_csv(
        OUT / "phase11d_ablation.csv",
        index=False,
    )

    query_comparison.to_csv(
        OUT / "phase11d_per_query.csv",
        index=False,
    )

    (OUT / "phase11d_importance.json").write_text(
        json.dumps(importances, indent=2),
        encoding="utf-8",
    )

    report = {
        "evaluation": (
            "validation_judged_retrieved_candidates"
        ),
        "validation_queries": len(query_ids),
        "train_ranking_groups": len(groups_train),
        "baseline_full_model_ndcg10": round(
            full_score, 4
        ),
        "experiments": summary.to_dict(
            orient="records"
        ),
        "final_test_used": False,
        "note": (
            "Ablations are development experiments. "
            "Unjudged retrieval results are excluded. "
            "This is not end-to-end NDCG."
        ),
    }

    (OUT / "phase11d_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 11D RESULTS\n")

    print(
        summary[
            [
                "experiment",
                "ndcg10",
                "delta_vs_full",
            ]
        ].to_string(index=False)
    )

    print("\nSaved results to:", OUT)


if __name__ == "__main__":
    main()
