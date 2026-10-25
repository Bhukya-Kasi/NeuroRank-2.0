
"""Phase 11B: Retrieval-Aware LambdaMART Training."""

from pathlib import Path
import hashlib
import json

import faiss
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "phase11"
ART = ROOT / "artifacts" / "phase11"
OUT = ROOT / "results" / "phase11"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

K_RETRIEVE = 100
K_EVAL = 10

GRADES = {
    "I": 0,
    "C": 1,
    "S": 2,
    "E": 3,
}

FEATURES = [
    "bm25",
    "semantic",
    "title_overlap",
    "exact_phrase",
    "bm25_rank",
    "semantic_rank",
    "title_length",
    "query_length",
]


def dcg(labels):

    labels = np.asarray(labels, dtype=float)

    discounts = np.log2(
        np.arange(2, len(labels) + 2)
    )

    gains = np.power(2.0, labels) - 1.0

    return float(np.sum(gains / discounts))


def ndcg_judged(scores, labels, k=10):

    labels = np.asarray(labels)

    if len(labels) == 0:
        return None

    if np.max(labels) == 0:
        return None

    ranking = np.argsort(
        -np.asarray(scores),
        kind="stable",
    )[:k]

    ideal = np.sort(labels)[::-1][:k]

    denominator = dcg(ideal)

    if denominator == 0:
        return None

    return float(
        dcg(labels[ranking]) / denominator
    )


def read_split(name):

    path = DATA / f"{name}_judgments.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run Phase 11A first."
        )

    df = pd.read_csv(
        path,
        dtype={
            "query_id": str,
            "product_id": str,
        },
    ).fillna("")

    required = {
        "query_id",
        "product_id",
        "query",
        "label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if not set(df["label"]).issubset(GRADES):
        raise ValueError(
            f"Unexpected relevance labels in {path}"
        )

    if df.duplicated(
        ["query_id", "product_id"]
    ).any():
        raise ValueError(
            f"Duplicate judgments in {path}"
        )

    return df


def build_catalog():

    path = DATA / "products.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Missing Phase 11 product catalog."
        )

    products = pd.read_csv(
        path,
        dtype={"product_id": str},
    ).fillna("")

    required = {
        "product_id",
        "title",
        "description",
    }

    missing = required - set(products.columns)

    if missing:
        raise ValueError(
            f"Missing catalog columns: {missing}"
        )

    products = products.drop_duplicates(
        "product_id"
    ).reset_index(drop=True)

    documents = (
        products["title"].astype(str)
        + " "
        + products["description"].astype(str)
    ).tolist()

    ART.mkdir(
        parents=True,
        exist_ok=True,
    )

    fingerprint_data = (
        MODEL_NAME
        + "\n"
        + "\n".join(
            str(pid) + "\t" + doc
            for pid, doc in zip(
                products["product_id"],
                documents,
            )
        )
    )

    fingerprint = hashlib.sha256(
        fingerprint_data.encode("utf-8")
    ).hexdigest()

    embeddings_path = ART / "embeddings.npy"
    metadata_path = ART / "embeddings_meta.json"

    print("Loading MiniLM model...")

    model = SentenceTransformer(MODEL_NAME)

    cache_valid = False

    if (
        embeddings_path.exists()
        and metadata_path.exists()
    ):

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        cache_valid = (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("model_name") == MODEL_NAME
        )

    if cache_valid:

        print("Loading cached embeddings...")

        embeddings = np.load(
            embeddings_path,
            allow_pickle=False,
        )

        if embeddings.shape[0] != len(products):
            raise ValueError(
                "Embedding cache size mismatch."
            )

    else:

        print("Generating product embeddings...")

        embeddings = model.encode(
            documents,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        np.save(
            embeddings_path,
            embeddings,
        )

        metadata_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "model_name": MODEL_NAME,
                    "product_count": len(products),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )

    print("Creating FAISS index...")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    print("Creating BM25 index...")

    tokenized = [
        document.lower().split()
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized)

    id_to_idx = {
        str(pid): idx
        for idx, pid in enumerate(
            products["product_id"]
        )
    }

    return (
        products,
        model,
        embeddings,
        index,
        bm25,
        id_to_idx,
    )


def extract_rows(
    split,
    products,
    model,
    embeddings,
    index,
    bm25,
    id_to_idx,
):

    X = []
    y = []
    groups = []
    details = []

    inspected = 0
    retrieved_judgments = 0

    for qid, group in split.groupby(
        "query_id",
        sort=True,
    ):

        query = str(
            group.iloc[0]["query"]
        )

        if not query.strip():
            continue

        if group["query"].nunique() != 1:
            raise ValueError(
                f"Multiple query texts for ID {qid}"
            )

        qtokens = query.lower().split()

        bm25_scores = np.asarray(
            bm25.get_scores(qtokens),
            dtype=np.float32,
        )

        query_embedding = np.asarray(
            model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

        similarities = (
            embeddings @ query_embedding[0]
        )

        top_bm25 = np.argsort(
            -bm25_scores,
            kind="stable",
        )[:K_RETRIEVE]

        _, semantic_indices = index.search(
            np.ascontiguousarray(
                query_embedding
            ),
            min(
                K_RETRIEVE,
                len(products),
            ),
        )

        top_semantic = semantic_indices[0]

        top_semantic = top_semantic[
            top_semantic >= 0
        ]

        candidates = (
            set(map(int, top_bm25))
            | set(map(int, top_semantic))
        )

        bm25_ranks = {
            int(idx): rank + 1
            for rank, idx in enumerate(top_bm25)
        }

        semantic_ranks = {
            int(idx): rank + 1
            for rank, idx in enumerate(top_semantic)
        }

        judged = []

        for record in group.itertuples(
            index=False
        ):

            idx = id_to_idx.get(
                str(record.product_id)
            )

            if (
                idx is not None
                and idx in candidates
            ):

                grade = GRADES[
                    str(record.label)
                ]

                judged.append(
                    (idx, grade)
                )

        inspected += 1
        retrieved_judgments += len(judged)

        # Only use groups with competing labels.
        if len(judged) < 2:
            continue

        if len({
            grade
            for _, grade in judged
        }) < 2:
            continue

        query_tokens = set(qtokens)

        for idx, grade in judged:

            title = str(
                products.iloc[idx]["title"]
            ).lower()

            title_tokens = title.split()

            overlap = len(
                query_tokens & set(title_tokens)
            ) / max(
                len(query_tokens),
                1,
            )

            features = [
                float(bm25_scores[idx]),
                float(similarities[idx]),
                float(overlap),
                float(query.lower() in title),
                (
                    1.0 / bm25_ranks[idx]
                    if idx in bm25_ranks
                    else 0.0
                ),
                (
                    1.0 / semantic_ranks[idx]
                    if idx in semantic_ranks
                    else 0.0
                ),
                float(len(title_tokens)),
                float(len(qtokens)),
            ]

            X.append(features)
            y.append(grade)

            details.append({
                "query_id": qid,
                "query": query,
                "product_id": str(
                    products.iloc[idx]["product_id"]
                ),
                "label": grade,
            })

        groups.append(
            len(judged)
        )

    print(
        f"Queries inspected: {inspected}"
    )

    print(
        f"Usable ranking groups: {len(groups)}"
    )

    print(
        f"Retrieved judged pairs: {retrieved_judgments}"
    )

    if not groups:
        raise ValueError(
            "No suitable ranking groups. "
            "Increase K_RETRIEVE or inspect data."
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        groups,
        pd.DataFrame(details),
    )


def evaluate_groups(
    X,
    y,
    groups,
    learned_scores,
):

    records = []
    offset = 0

    for size in groups:

        start = offset
        end = offset + size

        records.append({
            "bm25": ndcg_judged(
                X[start:end, 0],
                y[start:end],
                K_EVAL,
            ),
            "semantic": ndcg_judged(
                X[start:end, 1],
                y[start:end],
                K_EVAL,
            ),
            "lambdamart": ndcg_judged(
                learned_scores[start:end],
                y[start:end],
                K_EVAL,
            ),
        })

        offset = end

    return pd.DataFrame(records)


def main():

    print("\nNEURORANK PHASE 11B")
    print("Retrieval-Aware LambdaMART\n")

    train = read_split("train")
    validation = read_split("validation")

    if (
        set(train["query_id"])
        & set(validation["query_id"])
    ):
        raise ValueError(
            "Train/validation query ID leakage."
        )

    normalize = lambda s: " ".join(
        str(s).casefold().split()
    )

    train_queries = set(
        train["query"].map(normalize)
    )

    validation_queries = set(
        validation["query"].map(normalize)
    )

    if train_queries & validation_queries:
        raise ValueError(
            "Train/validation query text leakage."
        )

    (
        products,
        model,
        embeddings,
        index,
        bm25,
        id_to_idx,
    ) = build_catalog()

    print("\nPreparing training candidates...")

    (
        X_train,
        y_train,
        groups_train,
        _,
    ) = extract_rows(
        train,
        products,
        model,
        embeddings,
        index,
        bm25,
        id_to_idx,
    )

    print("\nPreparing validation candidates...")

    (
        X_valid,
        y_valid,
        groups_valid,
        validation_details,
    ) = extract_rows(
        validation,
        products,
        model,
        embeddings,
        index,
        bm25,
        id_to_idx,
    )

    print("\nTraining LambdaMART...")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=5,
        label_gain=[0, 1, 3, 7],
        random_state=42,
        verbosity=-1,
    )

    ranker.fit(
        X_train,
        y_train,
        group=groups_train,
        eval_set=[
            (X_valid, y_valid)
        ],
        eval_group=[
            groups_valid
        ],
        eval_at=[10],
    )

    print("\nEvaluating validation queries...")

    predictions = ranker.predict(
        X_valid
    )

    per_query = evaluate_groups(
        X_valid,
        y_valid,
        groups_valid,
        predictions,
    )

    metrics = {
        "bm25": round(
            float(per_query["bm25"].mean()),
            4,
        ),
        "semantic": round(
            float(per_query["semantic"].mean()),
            4,
        ),
        "lambdamart": round(
            float(per_query["lambdamart"].mean()),
            4,
        ),
        "train_queries": len(
            groups_train
        ),
        "validation_queries": len(
            groups_valid
        ),
        "scope": (
            "judged_retrieved_candidates_only"
        ),
        "candidate_pool": (
            "BM25 top100 union MiniLM top100"
        ),
        "test_split_used": False,
        "warning": (
            "Judged-subset NDCG is not "
            "end-to-end NDCG."
        ),
    }

    ART.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        ART / "retrieval_aware_ranker.joblib"
    )

    joblib.dump(
        {
            "ranker": ranker,
            "features": FEATURES,
            "model_name": MODEL_NAME,
            "candidate_depth": K_RETRIEVE,
        },
        model_path,
    )

    per_query.to_csv(
        OUT / "validation_per_query.csv",
        index=False,
    )

    validation_details.to_csv(
        OUT / "validation_judged_candidates.csv",
        index=False,
    )

    metrics_path = (
        OUT / "validation_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nPHASE 11B VALIDATION RESULTS")

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print("\nModel saved:", model_path)


if __name__ == "__main__":
    main()
  