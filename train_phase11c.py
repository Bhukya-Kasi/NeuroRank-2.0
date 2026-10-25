
"""Phase 11C: query-dependent ranking selection; no final-test access."""

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_phase11b import (
    ROOT, FEATURES, K_EVAL, read_split,
    build_catalog, extract_rows, ndcg_judged,
)


ART = ROOT / "artifacts" / "phase11"
OUT = ROOT / "results" / "phase11"
SEED = 42


def query_features(query):

    tokens = re.findall(
        r"\w+",
        str(query).casefold(),
    )

    n = len(tokens)

    return [
        float(n),
        float(len(str(query))),
        float(len(set(tokens)) / max(n, 1)),
        float(
            sum(len(t) for t in tokens) / max(n, 1)
        ),
        float(
            sum(
                any(ch.isdigit() for ch in t)
                for t in tokens
            )
        ),
    ]


def model_factory():

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        ),
    )


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ART.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation = read_split("validation")

    bundle_path = (
        ART / "retrieval_aware_ranker.joblib"
    )

    if not bundle_path.exists():
        raise FileNotFoundError(
            "Run Phase 11B first: missing "
            + str(bundle_path)
        )

    # Load only model artifacts you created and trust.
    bundle = joblib.load(bundle_path)

    if bundle.get("features") != FEATURES:
        raise ValueError(
            "Phase 11B feature list is incompatible"
        )

    print("Loading Phase 11 search catalog...")

    (
        products,
        encoder,
        embeddings,
        index,
        bm25,
        id_to_idx,
    ) = build_catalog()

    print("Extracting validation candidates...")

    (
        X,
        y,
        groups,
        details,
    ) = extract_rows(
        validation,
        products,
        encoder,
        embeddings,
        index,
        bm25,
        id_to_idx,
    )

    print("Running LambdaMART inference...")

    pred = bundle["ranker"].predict(X)

    queries_by_id = (
        validation.groupby(
            "query_id",
            sort=True,
        )["query"].first().to_dict()
    )

    records = []
    offset = 0

    for size in groups:

        end = offset + size

        subset = details.iloc[offset:end]

        qid = str(
            subset.iloc[0]["query_id"]
        )

        query = str(
            queries_by_id[qid]
        )

        sem = ndcg_judged(
            X[offset:end, 1],
            y[offset:end],
            K_EVAL,
        )

        learned = ndcg_judged(
            pred[offset:end],
            y[offset:end],
            K_EVAL,
        )

        offset = end

        if sem is None or learned is None:
            continue

        records.append({
            "query_id": qid,
            "query": query,
            "semantic_ndcg": float(sem),
            "lambdamart_ndcg": float(learned),
            "lambda_better": int(
                learned > sem + 1e-12
            ),
        })

    report = pd.DataFrame(records)

    if len(report) < 10:
        raise ValueError(
            f"Only {len(report)} usable validation groups; "
            "need at least 10"
        )

    qX = np.asarray(
        [
            query_features(q)
            for q in report["query"]
        ],
        dtype=float,
    )

    labels = report[
        "lambda_better"
    ].to_numpy(dtype=int)

    classes, counts = np.unique(
        labels,
        return_counts=True,
    )

    folds = (
        min(5, int(counts.min()))
        if len(classes) == 2
        else 0
    )

    if folds < 2:
        raise ValueError(
            "Not enough examples of both LambdaMART "
            "wins and non-wins for stratified "
            "cross-validation. Report the "
            "baselines instead."
        )

    print(
        f"Starting {folds}-fold "
        "query-level cross-validation..."
    )

    probabilities = np.full(
        len(report),
        np.nan,
    )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=SEED,
    )

    for train_idx, held_idx in cv.split(
        qX,
        labels,
    ):

        gate = model_factory()

        gate.fit(
            qX[train_idx],
            labels[train_idx],
        )

        probabilities[held_idx] = (
            gate.predict_proba(
                qX[held_idx]
            )[:, 1]
        )

    report[
        "out_of_fold_p_lambda_better"
    ] = probabilities

    report["out_of_fold_selection"] = np.where(
        probabilities > 0.5,
        "lambdamart",
        "semantic",
    )

    report["selected_ndcg"] = np.where(
        probabilities > 0.5,
        report["lambdamart_ndcg"],
        report["semantic_ndcg"],
    )

    report.to_csv(
        OUT / "phase11c_out_of_fold.csv",
        index=False,
    )

    print("Training final selector...")

    final_gate = model_factory()

    final_gate.fit(
        qX,
        labels,
    )

    joblib.dump(
        {
            "gate": final_gate,
            "query_features": [
                "word_count",
                "character_count",
                "unique_word_ratio",
                "mean_word_length",
                "number_containing_word_count",
            ],
            "strategy": (
                "choose_lambdamart_when_probability_gt_0.5"
            ),
            "base_ranker_file": (
                "retrieval_aware_ranker.joblib"
            ),
            "trained_on": (
                "phase11_validation_judgments_only"
            ),
        },
        ART / "query_reliability_gate.joblib",
    )

    summary = {
        "evaluation": (
            "cross_validated_query_selector_on_"
            "judged_retrieved_candidates"
        ),
        "query_count": int(len(report)),
        "folds": int(folds),
        "lambda_better_queries": int(
            labels.sum()
        ),
        "semantic_ndcg_mean": round(
            float(
                report["semantic_ndcg"].mean()
            ),
            4,
        ),
        "lambdamart_ndcg_mean": round(
            float(
                report["lambdamart_ndcg"].mean()
            ),
            4,
        ),
        "adaptive_out_of_fold_ndcg_mean": round(
            float(
                report["selected_ndcg"].mean()
            ),
            4,
        ),
        "selected_lambdamart_queries": int(
            (probabilities > 0.5).sum()
        ),
        "final_test_evaluated": False,
        "warning": (
            "Not end-to-end NDCG. "
            "Gate feature quality may be weak."
        ),
    }

    (OUT / "phase11c_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 11C RESULTS")

    print(
        json.dumps(summary, indent=2)
    )

    print(
        "\nSaved:",
        ART / "query_reliability_gate.joblib",
    )


if __name__ == "__main__":
    main()
