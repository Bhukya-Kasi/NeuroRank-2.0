
"""NeuroRank: BM25 + MiniLM/FAISS retrieval and LambdaMART ranking."""

from functools import lru_cache
from pathlib import Path
from time import perf_counter

import faiss
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from cached_search import CachedNeuralSearchEngine


ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "artifacts" / "reliability_ranker.joblib"
PRODUCTS_PATH = ROOT / "data" / "products.csv"

FEATURES = [
    "bm25_score",
    "semantic_similarity",
    "query_title_overlap",
    "exact_phrase_match",
    "title_length",
    "query_length",
]

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_system()
    yield


app = FastAPI(
    title="NeuroRank 2.0",
    version="0.9.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(
        default="semantic",
        pattern="^(lambdamart|adaptive|bm25|semantic|rrf)$",
    )


@lru_cache(maxsize=1)
def get_system():

    if not PRODUCTS_PATH.is_file() or not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Missing data/products.csv or "
            "artifacts/reliability_ranker.joblib"
        )

    products = pd.read_csv(PRODUCTS_PATH).fillna("")

    engine = CachedNeuralSearchEngine(
    products,
    cache_dir=ROOT / "artifacts" / "search_cache",
)

    # Load only model files that you created and trust.
    bundle = joblib.load(MODEL_PATH)

    if list(bundle["feature_names"]) != FEATURES:
        raise ValueError(
            "Saved ranker feature schema does not match API features"
        )

    return engine, bundle


def rank_products(engine, bundle, query, k, mode):

    words = query.lower().split()

    bm25 = np.asarray(
        engine.bm25.get_scores(words),
        dtype=float,
    )

    qvec = engine.model.encode(
        [query],
        normalize_embeddings=True,
    ).astype("float32")

    n = len(engine.products)
    count = min(max(100, k), n)

    lexical_ids = np.argsort(
        -bm25,
        kind="stable",
    )[:count]

    _, vector_ids = engine.index.search(
        qvec,
        count,
    )

    semantic_ids = vector_ids[0]

    # Collect candidates from both retrieval methods.
    rrf = {}

    for ranking in (lexical_ids, semantic_ids):
        for rank, idx in enumerate(ranking, 1):

            idx = int(idx)

            if idx >= 0:
                rrf[idx] = (
                    rrf.get(idx, 0.0)
                    + 1.0 / (60 + rank)
                )

    candidates = np.asarray(
        list(rrf.keys()),
        dtype=int,
    )

    semantic = (
        engine.embeddings[candidates] @ qvec[0]
    )

    qtokens = set(words)

    features = []

    for idx, sim in zip(candidates, semantic):

        title = str(
            engine.products.iloc[idx]["title"]
        ).lower()

        overlap = (
            len(qtokens & set(title.split()))
            / max(1, len(qtokens))
        )

        features.append([
            float(bm25[idx]),
            float(sim),
            float(overlap),
            float(query.lower() in title),
            float(np.log1p(len(title))),
            float(len(qtokens)),
        ])

    frame = pd.DataFrame(
        features,
        columns=FEATURES,
    )

    learned = np.asarray(
        bundle["ranker"].predict(frame),
        dtype=float,
    )

    # Select the requested ranking method.
    if mode == "bm25":
        scores = bm25[candidates]

    elif mode == "semantic":
        scores = semantic

    elif mode == "rrf":
        scores = np.asarray(
            [rrf[idx] for idx in candidates]
        )

    elif mode == "lambdamart":
        scores = learned

    else:
        # Experimental adaptive selector
        gate = bundle["gate"]

        if gate["kind"] == "constant":
            choose_learned = bool(
                gate["choice"]
            )

        else:
            sorted_scores = np.sort(learned)[::-1]

            signals = np.asarray([[
                len(candidates),
                float(np.max(bm25[candidates])),
                float(np.max(semantic)),
                float(np.mean(semantic)),
                float(
                    np.max(bm25[candidates])
                    - np.median(bm25[candidates])
                ),
                float(
                    sorted_scores[0] - sorted_scores[1]
                ) if len(sorted_scores) > 1 else 0.0,
                float(np.std(learned)),
                float(
                    frame["query_title_overlap"].mean()
                ),
            ]])

            choose_learned = bool(
                gate["model"].predict(signals)[0]
            )

        scores = (
            learned
            if choose_learned
            else bm25[candidates]
        )

    chosen = np.argsort(
        -scores,
        kind="stable",
    )[:k]

    results = []

    for j in chosen:

        product = engine.products.iloc[
            candidates[j]
        ]

        results.append({
            "product_id": str(product["product_id"]),
            "title": str(product["title"]),
            "score": round(float(scores[j]), 6),
        })

    return results


@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_file_exists": MODEL_PATH.exists(),
        "catalog_file_exists": PRODUCTS_PATH.exists(),
    }


@app.post("/search")
def search(request: SearchRequest):

    start = perf_counter()

    try:
        engine, bundle = get_system()

        results = rank_products(
            engine,
            bundle,
            request.query,
            request.k,
            request.mode,
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except (ValueError, KeyError) as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    return {
        "query": request.query,
        "mode": request.mode,
        "candidate_scope": "local_sample_catalog",
        "latency_ms": round(
            (perf_counter() - start) * 1000,
            2,
        ),
        "results": results,
    }

from collections import Counter
from threading import Lock
from time import perf_counter

from fastapi import Request


REQUEST_COUNTS = Counter()
REQUEST_LATENCIES_MS = []
MONITOR_LOCK = Lock()


@app.middleware("http")
async def record_api_metrics(request: Request, call_next):

    start = perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response

    finally:
        elapsed = (perf_counter() - start) * 1000

        if request.url.path == "/search":

            with MONITOR_LOCK:
                REQUEST_COUNTS[str(status)] += 1

                REQUEST_LATENCIES_MS.append(
                    elapsed
                )

                # Retain at most 1000 recent observations.
                if len(REQUEST_LATENCIES_MS) > 1000:
                    del REQUEST_LATENCIES_MS[:-1000]


@app.get("/metrics")
def service_metrics():

    with MONITOR_LOCK:
        counts = dict(REQUEST_COUNTS)
        timings = list(REQUEST_LATENCIES_MS)

    if not timings:
        return {
            "requests_recorded": 0,
            "status_counts": counts,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }

    values = np.asarray(timings)

    return {
        "requests_recorded": len(timings),
        "status_counts": counts,
        "p50_ms": round(
            float(np.percentile(values, 50)), 2
        ),
        "p95_ms": round(
            float(np.percentile(values, 95)), 2
        ),
        "p99_ms": round(
            float(np.percentile(values, 99)), 2
        ),
    }
