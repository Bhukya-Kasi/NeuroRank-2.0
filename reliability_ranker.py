
"""NeuroRank Phase 5: Adaptive learning-to-rank."""

from pathlib import Path
import json

import joblib
import numpy as np

from lightgbm import LGBMRanker
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_ranker import (
    load_data,
    generate_features,
    NeuralSearchEngine,
    FEATURE_NAMES,
    evaluate_ranking,
    ndcg_at_k,
)


SEED = 42

OUTPUT = Path("results/reliability_metrics.json")
ARTIFACT = Path("artifacts/reliability_ranker.joblib")


# -------------------------------------------------
# 1. Query-level data splitting
# -------------------------------------------------

def order(frame):
    return frame.sort_values(
        "query_id",
        kind="stable",
    ).reset_index(drop=True)


def split_queries(frame, validation_fraction=0.25):

    train = frame[
        frame["split"] == "train"
    ].copy()

    test = frame[
        frame["split"] == "test"
    ].copy()

    query_ids = train["query_id"].astype(
        str
    ).unique()

    if len(query_ids) < 8:
        raise ValueError(
            "At least 8 training queries required"
        )

    if test["query_id"].nunique() == 0:
        raise ValueError("No test queries found")

    rng = np.random.default_rng(SEED)

    shuffled = rng.permutation(query_ids)

    n_val = max(
        2,
        int(round(
            len(shuffled) * validation_fraction
        )),
    )

    validation_ids = set(
        shuffled[:n_val]
    )

    val = train[
        train["query_id"].astype(str).isin(
            validation_ids
        )
    ].copy()

    fit = train[
        ~train["query_id"].astype(str).isin(
            validation_ids
        )
    ].copy()

    assert set(fit.query_id).isdisjoint(
        val.query_id
    )

    assert set(fit.query_id).isdisjoint(
        test.query_id
    )

    assert set(val.query_id).isdisjoint(
        test.query_id
    )

    return order(fit), order(val), order(test)


# -------------------------------------------------
# 2. LambdaMART predictions
# -------------------------------------------------

def model_scores(ranker, frame):

    return np.asarray(
        ranker.predict(
            frame[FEATURE_NAMES]
        ),
        dtype=float,
    )


# -------------------------------------------------
# 3. Reliability-related query features
# -------------------------------------------------

def query_signals(group, scores):

    bm25 = group[
        "bm25_score"
    ].to_numpy(dtype=float)

    semantic = group[
        "semantic_similarity"
    ].to_numpy(dtype=float)

    scores = np.asarray(
        scores,
        dtype=float,
    )

    ranked = np.sort(scores)[::-1]

    score_margin = (
        ranked[0] - ranked[1]
        if len(ranked) > 1
        else 0.0
    )

    return np.array([
        len(group),
        float(np.max(bm25)),
        float(np.max(semantic)),
        float(np.mean(semantic)),
        float(
            np.max(bm25) - np.median(bm25)
        ),
        float(score_margin),
        float(np.std(scores)),
        float(
            np.mean(
                group["query_title_overlap"]
            )
        ),
    ], dtype=float)


# -------------------------------------------------
# 4. Train adaptive strategy selector
# -------------------------------------------------

def fit_gate(validation, ranker):

    features = []
    targets = []

    for _, group in validation.groupby(
        "query_id",
        sort=False,
    ):

        labels = group[
            "label"
        ].to_numpy(dtype=int)

        bm25 = group[
            "bm25_score"
        ].to_numpy(dtype=float)

        learned = model_scores(
            ranker,
            group,
        )

        features.append(
            query_signals(
                group,
                learned,
            )
        )

        bm25_quality = ndcg_at_k(
            labels,
            bm25,
            k=10,
        )

        learned_quality = ndcg_at_k(
            labels,
            learned,
            k=10,
        )

        # Target 1: LambdaMART is better.
        # Target 0: BM25 is better or tied.

        targets.append(
            int(
                learned_quality > bm25_quality
            )
        )

    # If all validation queries select the same
    # method, use a constant policy.

    if len(set(targets)) == 1:

        return {
            "kind": "constant",
            "choice": targets[0],
        }

    gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=SEED,
            max_iter=1000,
        ),
    )

    gate.fit(
        np.asarray(features),
        targets,
    )

    return {
        "kind": "logistic",
        "model": gate,
    }


# -------------------------------------------------
# 5. Make adaptive predictions
# -------------------------------------------------

def adaptive_predictions(frame, ranker, gate):

    predictions = []

    for _, group in frame.groupby(
        "query_id",
        sort=False,
    ):

        learned = model_scores(
            ranker,
            group,
        )

        bm25 = group[
            "bm25_score"
        ].to_numpy(dtype=float)

        if gate["kind"] == "constant":

            use_learned = bool(
                gate["choice"]
            )

        else:

            signals = query_signals(
                group,
                learned,
            )

            use_learned = bool(
                gate["model"].predict(
                    signals.reshape(1, -1)
                )[0]
            )

        selected_scores = (
            learned
            if use_learned
            else bm25
        )

        predictions.append(
            selected_scores
        )

    return np.concatenate(predictions)


# -------------------------------------------------
# 6. Complete experiment
# -------------------------------------------------

def main():

    print("\nLoading Amazon ESCI data...")

    products, examples = load_data()

    print("Products:", len(products))
    print("Labeled pairs:", len(examples))

    print("\nLoading BM25 and MiniLM...")

    engine = NeuralSearchEngine(products)

    print("\nGenerating ranking features...")

    features = generate_features(
        products,
        examples,
        engine,
    )

    train, val, test = split_queries(
        features
    )

    print("\nQuery counts:")
    print("Training:", train.query_id.nunique())
    print("Validation:", val.query_id.nunique())
    print("Testing:", test.query_id.nunique())

    groups = (
        train.groupby(
            "query_id",
            sort=False,
        )
        .size()
        .to_numpy()
    )

    # Train LambdaMART only on training queries.

    ranker = LGBMRanker(
        objective="lambdarank",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=5,
        random_state=SEED,
        verbosity=-1,
        label_gain=[0, 1, 3, 7],
    )

    print("\nTraining LambdaMART...")

    ranker.fit(
        train[FEATURE_NAMES],
        train["label"].astype(int),
        group=groups,
    )

    print("\nTraining adaptive selector...")

    gate = fit_gate(
        val,
        ranker,
    )

    print("\nEvaluating held-out test queries...")

    bm25_predictions = test[
        "bm25_score"
    ].to_numpy()

    semantic_predictions = test[
        "semantic_similarity"
    ].to_numpy()

    learned_predictions = model_scores(
        ranker,
        test,
    )

    adaptive = adaptive_predictions(
        test,
        ranker,
        gate,
    )

    metrics = {
        "dataset": "Amazon ESCI subset",
        "evaluation": "Labeled-candidate reranking",
        "train_queries": int(
            train.query_id.nunique()
        ),
        "validation_queries": int(
            val.query_id.nunique()
        ),
        "test_queries": int(
            test.query_id.nunique()
        ),
        "test_pairs": int(len(test)),
        "gate_type": gate["kind"],
        "BM25_NDCG@10": round(
            evaluate_ranking(
                test,
                bm25_predictions,
            ),
            4,
        ),
        "MiniLM_NDCG@10": round(
            evaluate_ranking(
                test,
                semantic_predictions,
            ),
            4,
        ),
        "LambdaMART_NDCG@10": round(
            evaluate_ranking(
                test,
                learned_predictions,
            ),
            4,
        ),
        "Adaptive_NDCG@10": round(
            evaluate_ranking(
                test,
                adaptive,
            ),
            4,
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    joblib.dump(
        {
            "ranker": ranker,
            "gate": gate,
            "feature_names": FEATURE_NAMES,
        },
        ARTIFACT,
    )

    print("\nFINAL RANKING EVALUATION")
    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print("\nModel and metrics saved.")


if __name__ == "__main__":
    main()
