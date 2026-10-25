
"""
NeuroRank 2.0 - Supervised Learning to Rank

Train LightGBM LambdaMART using Amazon ESCI relevance labels.

Inputs:
    data/products.csv
    data/examples.csv

Outputs:
    artifacts/lambdamart.joblib
    results/lambdamart_metrics.json

Run:
    python train_ranker.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRanker
from sklearn.metrics.pairwise import cosine_similarity

from neural_search import NeuralSearchEngine


# --------------------------------------------
# Configuration
# --------------------------------------------

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
RESULT_DIR = Path("results")

ARTIFACT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

LABEL_MAP = {
    "I": 0,
    "C": 1,
    "S": 2,
    "E": 3,
}

FEATURE_NAMES = [
    "bm25_score",
    "semantic_similarity",
    "query_title_overlap",
    "exact_phrase_match",
    "title_length",
    "query_length",
]

RANDOM_STATE = 42


# --------------------------------------------
# 1. Load dataset
# --------------------------------------------

def load_data():

    products = pd.read_csv(
        DATA_DIR / "products.csv"
    ).fillna("")

    examples = pd.read_csv(
        DATA_DIR / "examples.csv"
    ).fillna("")

    required_products = {
        "product_id",
        "title",
        "description",
    }

    required_examples = {
        "query_id",
        "query",
        "product_id",
        "label",
        "split",
    }

    if not required_products.issubset(products.columns):
        raise ValueError("Missing product columns")

    if not required_examples.issubset(examples.columns):
        raise ValueError("Missing example columns")

    if examples["product_id"].isna().any():
        raise ValueError("Missing product IDs")

    if examples.duplicated(
        subset=["query_id", "product_id"]
    ).any():
        raise ValueError("Duplicate query-product pairs")

    if not set(examples["label"]).issubset(LABEL_MAP):
        raise ValueError("Unexpected relevance labels")

    train_ids = set(
        examples.loc[
            examples["split"] == "train",
            "query_id"
        ]
    )

    test_ids = set(
        examples.loc[
            examples["split"] == "test",
            "query_id"
        ]
    )

    if train_ids & test_ids:
        raise ValueError("Train/test query ID overlap")

    if not train_ids or not test_ids:
        raise ValueError("Train and test data required")

    return products, examples


# --------------------------------------------
# 2. Feature extraction
# --------------------------------------------

def generate_features(products, examples, engine):

    product_indices = {
        str(product_id): index
        for index, product_id in enumerate(
            products["product_id"]
        )
    }

    all_features = []
    all_labels = []
    all_query_ids = []
    all_splits = []

    # One feature row per labeled query-product pair.
    # Group by query ID for ranking training.

    grouped = examples.groupby(
        ["split", "query_id"],
        sort=True,
    )

    for (split, query_id), group in grouped:

        query = str(group.iloc[0]["query"])

        tokens = set(query.lower().split())

        bm25_scores = engine.bm25.get_scores(
            query.lower().split()
        )

        query_embedding = engine.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        candidate_indices = np.asarray([
            product_indices[str(product_id)]
            for product_id in group["product_id"]
        ])

        semantic_scores = (
            engine.embeddings[candidate_indices]
            @ query_embedding[0]
        )

        for position, (_, row) in enumerate(
            group.iterrows()
        ):

            index = int(candidate_indices[position])

            title = str(
                products.iloc[index]["title"]
            ).lower()

            title_tokens = set(title.split())

            overlap = (
                len(tokens & title_tokens)
                / max(len(tokens), 1)
            )

            phrase_match = float(
                query.lower() in title
            )

            features = [
                float(bm25_scores[index]),
                float(semantic_scores[position]),
                float(overlap),
                phrase_match,
                float(np.log1p(len(title))),
                float(len(tokens)),
            ]

            all_features.append(features)
            all_labels.append(
                LABEL_MAP[row["label"]]
            )
            all_query_ids.append(str(query_id))
            all_splits.append(str(split))

    X = pd.DataFrame(
        all_features,
        columns=FEATURE_NAMES,
    )

    X["label"] = all_labels
    X["query_id"] = all_query_ids
    X["split"] = all_splits

    return X


# --------------------------------------------
# 3. NDCG evaluation
# --------------------------------------------

def ndcg_at_k(labels, scores, k=10):

    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)

    if len(labels) == 0:
        return 0.0

    order = np.argsort(
        -scores,
        kind="stable",
    )[:k]

    ideal_order = np.argsort(
        -labels,
        kind="stable",
    )[:k]

    def dcg(relevance):

        gains = 2 ** relevance - 1

        discounts = np.log2(
            np.arange(2, len(relevance) + 2)
        )

        return float(
            np.sum(gains / discounts)
        )

    ideal = dcg(labels[ideal_order])

    if ideal == 0:
        return 0.0

    return dcg(labels[order]) / ideal


def evaluate_ranking(frame, predictions):

    scores = []

    offset = 0

    for _, group in frame.groupby(
        "query_id",
        sort=False,
    ):

        size = len(group)

        labels = group["label"].to_numpy()

        query_predictions = predictions[
            offset:offset + size
        ]

        scores.append(
            ndcg_at_k(
                labels,
                query_predictions,
                k=10,
            )
        )

        offset += size

    return float(np.mean(scores))


# --------------------------------------------
# 4. Main training pipeline
# --------------------------------------------

def main():

    print("\nLoading Amazon ESCI dataset...")

    products, examples = load_data()

    print("Products:", len(products))
    print("Labeled pairs:", len(examples))

    print("\nInitializing neural search engine...")

    engine = NeuralSearchEngine(products)

    print("\nGenerating ranking features...")

    features = generate_features(
        products,
        examples,
        engine,
    )

    train = features[
        features["split"] == "train"
    ].copy()

    test = features[
        features["split"] == "test"
    ].copy()

    # Query groups must be contiguous for LambdaMART.
    train = train.sort_values(
        "query_id",
        kind="stable",
    ).reset_index(drop=True)

    test = test.sort_values(
        "query_id",
        kind="stable",
    ).reset_index(drop=True)

    train_group = (
        train.groupby("query_id", sort=False)
        .size()
        .to_numpy()
    )

    test_group = (
        test.groupby("query_id", sort=False)
        .size()
        .to_numpy()
    )

    X_train = train[FEATURE_NAMES]
    y_train = train["label"].astype(int)

    X_test = test[FEATURE_NAMES]
    y_test = test["label"].astype(int)

    assert train_group.sum() == len(train)
    assert test_group.sum() == len(test)

    print("\nTraining LightGBM LambdaMART...")

    ranker = LGBMRanker(
        objective="lambdarank",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=5,
        random_state=RANDOM_STATE,
        verbosity=-1,
        label_gain=[0, 1, 3, 7],
    )

    ranker.fit(
        X_train,
        y_train,
        group=train_group,
        eval_at=[10],
    )

    print("\nEvaluating on held-out test queries...")

    predictions = ranker.predict(X_test)

    lexical_predictions = test[
        "bm25_score"
    ].to_numpy()

    semantic_predictions = test[
        "semantic_similarity"
    ].to_numpy()

    lexical_ndcg = evaluate_ranking(
        test,
        lexical_predictions,
    )

    semantic_ndcg = evaluate_ranking(
        test,
        semantic_predictions,
    )

    lambdamart_ndcg = evaluate_ranking(
        test,
        predictions,
    )

    results = {
        "dataset": "Amazon ESCI subset",
        "test_queries": int(len(test_group)),
        "candidate_pairs": int(len(test)),
        "evaluation": "Labeled-candidate reranking",
        "BM25_NDCG@10": round(lexical_ndcg, 4),
        "MiniLM_NDCG@10": round(semantic_ndcg, 4),
        "LambdaMART_NDCG@10": round(
            lambdamart_ndcg,
            4,
        ),
    }

    print("\nRANKING RESULTS")
    print(json.dumps(results, indent=2))

    # Save only trained ranker and feature schema.
    # The neural encoder/index are rebuilt separately.

    joblib.dump(
        {
            "model": ranker,
            "feature_names": FEATURE_NAMES,
            "label_map": LABEL_MAP,
        },
        ARTIFACT_DIR / "lambdamart.joblib",
    )

    with open(
        RESULT_DIR / "lambdamart_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print("\nSaved ranking model and metrics.")

    print("\nFeature importance:")

    importance = sorted(
        zip(
            FEATURE_NAMES,
            ranker.feature_importances_,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature, value in importance:
        print(f"{feature}: {value}")


if __name__ == "__main__":
    main()
