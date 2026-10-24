
"""NeuroRank 2.0: Search and ranking ML prototype."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


DATA = Path("data")
MODEL = Path("artifacts/neurorank.joblib")

GRADES = {
    "I": 0,
    "C": 1,
    "S": 2,
    "E": 3,
}

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# 1. Demonstration dataset
# --------------------------------------------------

def make_demo():
    """Generate a synthetic dataset for software testing."""

    DATA.mkdir(exist_ok=True)

    categories = {
        "headphones": [
            "wireless bluetooth headphones",
            "noise cancelling headphones",
            "travel headphones",
        ],
        "laptops": [
            "lightweight student laptop",
            "gaming laptop",
            "business laptop",
        ],
        "chargers": [
            "usb c laptop charger",
            "fast phone charger",
            "portable charger",
        ],
        "shoes": [
            "running shoes",
            "walking shoes",
            "waterproof hiking shoes",
        ],
        "bags": [
            "laptop backpack",
            "travel backpack",
            "school bag",
        ],
    }

    products = []
    examples = []

    for category, titles in categories.items():
        for i, title in enumerate(titles):
            products.append({
                "product_id": f"{category}-{i}",
                "title": title,
                "description": (
                    f"{category} durable quality "
                    f"everyday comfort {title}"
                ),
            })

    for query_number, product in enumerate(products):

        query = product["title"]

        for candidate in products:

            if candidate["product_id"] == product["product_id"]:
                label = "E"

            elif (
                candidate["product_id"].split("-")[0]
                == product["product_id"].split("-")[0]
            ):
                label = "S"

            else:
                label = "I"

            examples.append({
                "query_id": f"q{query_number}",
                "query": query,
                "product_id": candidate["product_id"],
                "label": label,
                "split": (
                    "train"
                    if query_number % 5
                    else "test"
                ),
            })

    pd.DataFrame(products).to_csv(
        DATA / "products.csv",
        index=False,
    )

    pd.DataFrame(examples).to_csv(
        DATA / "examples.csv",
        index=False,
    )

    print("Synthetic demonstration dataset created.")


# --------------------------------------------------
# 2. Data validation and loading
# --------------------------------------------------

def load_tables():

    products = pd.read_csv(
        DATA / "products.csv"
    ).fillna("")

    examples = pd.read_csv(
        DATA / "examples.csv"
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
        raise ValueError("Invalid product columns")

    if not required_examples.issubset(examples.columns):
        raise ValueError("Invalid example columns")

    if products["product_id"].duplicated().any():
        raise ValueError("Duplicate product IDs")

    if not set(examples["label"]).issubset(GRADES):
        raise ValueError("Invalid relevance labels")

    if not set(examples["product_id"]).issubset(
        set(products["product_id"])
    ):
        raise ValueError("Some product IDs are missing")

    split_counts = examples.groupby(
        "query_id"
    )["split"].nunique()

    if (split_counts > 1).any():
        raise ValueError(
            "Query IDs cannot occur across different splits"
        )

    if not {"train", "test"}.issubset(
        set(examples["split"])
    ):
        raise ValueError("Train and test splits required")

    return products, examples


# --------------------------------------------------
# 3. Machine learning search engine
# --------------------------------------------------

class SearchEngine:

    def __init__(self, neural=False):

        self.neural = neural
        self.encoder = None

    def fit(self, products, examples):

        self.products = products.reset_index(
            drop=True
        )

        self.ids = {
            str(product_id): index
            for index, product_id
            in enumerate(self.products["product_id"])
        }

        documents = (
            self.products["title"].astype(str)
            + " "
            + self.products["description"].astype(str)
        ).tolist()

        # Word-level text representation
        self.word = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        # Character-level representation
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
        )

        self.word_docs = self.word.fit_transform(
            documents
        )

        self.char_docs = self.char.fit_transform(
            documents
        )

        # Optional neural embeddings
        if self.neural:

            from sentence_transformers import (
                SentenceTransformer
            )

            self.encoder = SentenceTransformer(
                ENCODER
            )

            self.embeddings = np.asarray(
                self.encoder.encode(
                    documents,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )

        else:
            self.embeddings = None

        train = examples[
            examples["split"] == "train"
        ]

        features = []
        targets = []

        for query, group in train.groupby(
            "query",
            sort=False,
        ):

            scores = self.raw_scores(
                str(query)
            )

            for row in group.itertuples():

                index = self.ids[
                    str(row.product_id)
                ]

                feature = self.feature_at(
                    str(query),
                    index,
                    scores,
                )

                features.append(feature)

                targets.append(
                    int(row.label == "E")
                )

        if len(set(targets)) != 2:
            raise ValueError(
                "Training requires positive and negative labels"
            )

        # Feature standardization
        self.scaler = StandardScaler().fit(
            features
        )

        # Supervised relevance classifier
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )

        self.model.fit(
            self.scaler.transform(features),
            targets,
        )

        return self

    # ----------------------------------------------
    # 4. Retrieval scores
    # ----------------------------------------------

    def raw_scores(self, query):

        word_scores = cosine_similarity(
            self.word.transform([query]),
            self.word_docs,
        )[0]

        char_scores = cosine_similarity(
            self.char.transform([query]),
            self.char_docs,
        )[0]

        if self.neural:

            vector = self.encoder.encode(
                [query],
                normalize_embeddings=True,
            )[0]

            neural_scores = (
                self.embeddings @ vector
            )

            neural_scores = (
                neural_scores + 1
            ) / 2

        else:

            neural_scores = np.zeros(
                len(word_scores)
            )

        return (
            word_scores,
            char_scores,
            neural_scores,
        )

    # ----------------------------------------------
    # 5. Feature engineering
    # ----------------------------------------------

    def feature_at(
        self,
        query,
        index,
        scores,
    ):

        word, char, neural = scores

        query_tokens = set(
            query.lower().split()
        )

        title = str(
            self.products.iloc[index].title
        ).lower()

        title_tokens = set(
            title.split()
        )

        overlap = (
            len(query_tokens & title_tokens)
            / max(len(query_tokens), 1)
        )

        return [
            float(word[index]),
            float(char[index]),
            float(neural[index]),
            float(overlap),
            float(np.log1p(len(title))),
        ]

    # ----------------------------------------------
    # 6. Hybrid and adaptive ranking
    # ----------------------------------------------

    def score_candidates(
        self,
        query,
        indices,
    ):

        raw = self.raw_scores(query)

        word, char, neural = raw

        if self.neural:

            hybrid = (
                0.55 * word
                + 0.25 * char
                + 0.20 * neural
            )

        else:

            hybrid = (
                0.70 * word
                + 0.30 * char
            )

        X = [
            self.feature_at(
                query,
                int(index),
                raw,
            )
            for index in indices
        ]

        learned = self.model.predict_proba(
            self.scaler.transform(X)
        )[:, 1]

        # This is a heuristic confidence proxy,
        # not statistically calibrated uncertainty.
        confidence = np.clip(
            2 * np.abs(learned - 0.5),
            0,
            1,
        )

        adaptive = (
            confidence * learned
            + (1 - confidence)
            * hybrid[indices]
        )

        return {
            "lexical": word[indices],
            "hybrid": hybrid[indices],
            "learned": learned,
            "adaptive": adaptive,
        }

    # ----------------------------------------------
    # 7. Search function
    # ----------------------------------------------

    def search(
        self,
        query,
        k=5,
        mode="adaptive",
    ):

        allowed_modes = {
            "lexical",
            "hybrid",
            "learned",
            "adaptive",
        }

        if mode not in allowed_modes:
            raise ValueError(
                "Invalid ranking mode"
            )

        if not 1 <= k <= 100:
            raise ValueError(
                "k must be between 1 and 100"
            )

        raw = self.raw_scores(query)

        if self.neural:

            base = (
                0.55 * raw[0]
                + 0.25 * raw[1]
                + 0.20 * raw[2]
            )

        else:

            base = (
                0.70 * raw[0]
                + 0.30 * raw[1]
            )

        candidates = np.argsort(
            -base
        )[:min(100, len(base))]

        scores = self.score_candidates(
            query,
            candidates,
        )[mode]

        selected = np.argsort(
            -scores
        )[:k]

        results = []

        for position in selected:

            index = int(
                candidates[position]
            )

            product = self.products.iloc[
                index
            ]

            results.append({
                "product_id": str(
                    product.product_id
                ),
                "title": str(
                    product.title
                ),
                "score": round(
                    float(scores[position]),
                    5,
                ),
            })

        return results

    # ----------------------------------------------
    # 8. Model saving and loading
    # ----------------------------------------------

    def save(self):

        MODEL.parent.mkdir(
            exist_ok=True
        )

        state = self.__dict__.copy()

        # Reload neural encoder when necessary
        state["encoder"] = None

        joblib.dump(
            state,
            MODEL,
        )

        cached.cache_clear()

    @classmethod
    def load(cls):

        instance = cls()

        instance.__dict__.update(
            joblib.load(MODEL)
        )

        if instance.neural:

            from sentence_transformers import (
                SentenceTransformer
            )

            instance.encoder = SentenceTransformer(
                ENCODER
            )

        return instance


# --------------------------------------------------
# 9. Ranking evaluation
# --------------------------------------------------

def ndcg(relevances, ideal, k=10):

    relevances = np.asarray(
        relevances[:k]
    )

    ideal = np.asarray(
        sorted(
            ideal,
            reverse=True,
        )[:k]
    )

    if len(relevances) == 0:
        return 0.0

    def gain(values):

        numerator = (
            2.0 ** values - 1
        )

        denominator = np.log2(
            np.arange(
                2,
                len(values) + 2,
            )
        )

        return float(
            np.sum(
                numerator / denominator
            )
        )

    ideal_gain = gain(ideal)

    if ideal_gain == 0:
        return 0.0

    return (
        gain(relevances)
        / ideal_gain
    )


def evaluate(engine, examples):

    modes = [
        "lexical",
        "hybrid",
        "learned",
        "adaptive",
    ]

    results = {
        mode: []
        for mode in modes
    }

    test = examples[
        examples["split"] == "test"
    ]

    for query_id, group in test.groupby(
        "query_id"
    ):

        query = str(
            group.iloc[0]["query"]
        )

        indices = np.asarray([
            engine.ids[str(product_id)]
            for product_id
            in group["product_id"]
        ])

        grades = np.asarray([
            GRADES[label]
            for label
            in group["label"]
        ])

        scores = engine.score_candidates(
            query,
            indices,
        )

        for mode, values in scores.items():

            order = np.argsort(
                -values
            )

            ranked_grades = grades[
                order
            ]

            metric = ndcg(
                ranked_grades,
                grades,
                10,
            )

            results[mode].append(
                metric
            )

    return {
        name: {
            "NDCG@10": round(
                float(np.mean(values)),
                4,
            ),
            "queries": len(values),
        }
        for name, values
        in results.items()
    }


# --------------------------------------------------
# 10. API model loader
# --------------------------------------------------

@lru_cache(maxsize=1)
def cached():

    if not MODEL.exists():
        raise FileNotFoundError(
            "Train the model first: python main.py train"
        )

    return SearchEngine.load()


# --------------------------------------------------
# 11. FastAPI
# --------------------------------------------------

app = FastAPI(
    title="NeuroRank 2.0",
    description="Machine learning search and ranking platform",
    version="0.1.0",
)


class SearchRequest(BaseModel):

    query: str = Field(
        min_length=2,
        max_length=300,
    )

    k: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    mode: str = "adaptive"


@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_available": MODEL.exists(),
    }


@app.post("/search")
def api_search(
    request: SearchRequest,
):

    try:

        results = cached().search(
            request.query,
            request.k,
            request.mode,
        )

        return {
            "query": request.query,
            "mode": request.mode,
            "results": results,
        }

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error


# --------------------------------------------------
# 12. Command-line interface
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "demo",
            "train",
            "evaluate",
            "search",
        ],
    )

    parser.add_argument(
        "query",
        nargs="?",
        default="wireless bluetooth headphones",
    )

    parser.add_argument(
        "--neural",
        action="store_true",
    )

    args = parser.parse_args()

    if args.command == "demo":

        make_demo()

    elif args.command == "train":

        products, examples = load_tables()

        engine = SearchEngine(
            neural=args.neural
        )

        engine.fit(
            products,
            examples,
        )

        engine.save()

        print(
            f"Model saved to {MODEL}"
        )

        print(
            f"Total products: {len(products)}"
        )

    elif args.command == "search":

        results = cached().search(
            args.query
        )

        print(
            json.dumps(
                results,
                indent=2,
            )
        )

    elif args.command == "evaluate":

        products, examples = load_tables()

        engine = cached()

        metrics = evaluate(
            engine,
            examples,
        )

        Path("results").mkdir(
            exist_ok=True
        )

        Path(
            "results/metrics.json"
        ).write_text(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
