
# NeuroRank 2.0
## Reliability-Aware Neural Search, Learning-to-Rank, and Continuous ML Optimization Platform

An end-to-end machine learning search system developed
using the Amazon ESCI product-search relevance dataset.

NeuroRank integrates lexical retrieval, transformer-based
semantic retrieval, FAISS vector search, supervised
learning-to-rank, and an experimental adaptive ranking
strategy.

## Problem Statement

Real-world e-commerce search systems must retrieve
relevant products from large catalogs and rank them
according to user intent.

A supervised ranker trained on predefined candidate
sets may perform differently when deployed behind a
live retrieval pipeline.

NeuroRank investigates this training-serving candidate
distribution mismatch.

## System Architecture

User Query
    |
    +-- BM25 Retrieval
    |
    +-- MiniLM Semantic Retrieval + FAISS
    |
    v
Candidate Fusion (RRF)
    |
    v
LambdaMART Ranking / Semantic Ranking
    |
    v
FastAPI Search Service
    |
    v
Interactive Dashboard + Monitoring

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Sentence Transformers
- BM25
- FAISS
- LightGBM LambdaMART
- FastAPI and Uvicorn
- Streamlit
- Pytest
- Git and GitHub

## Dataset

Amazon Shopping Queries Dataset (ESCI)

Source:
https://github.com/amazon-science/esci-data

The experimental catalog contains 2,936 products.
The reported initial evaluation uses 30 test queries.

The dataset is a sampled catalog, not the full
Amazon product inventory.

## Ranking Performance

### Labeled-candidate reranking

| Model | NDCG@10 |
|---|---:|
| BM25 | 0.7469 |
| MiniLM | 0.7743 |
| LambdaMART | 0.7802 |

### End-to-end sampled-catalog evaluation

| Model | NDCG@10 | Recall@100 |
|---|---:|---:|
| BM25 | 0.6134 | 0.7291 |
| MiniLM | 0.6722 | 0.8480 |
| RRF | 0.6717 | 0.8381 |
| LambdaMART | 0.5187 | 0.8119 |

End-to-end metrics use available ESCI judgments,
with unjudged results treated as relevance zero
for NDCG calculation.

These are preliminary results on a small sampled
catalog, not general performance guarantees.

## API Performance

A local sequential benchmark of 100 warmed requests
reported:

| Metric | Latency |
|---|---:|
| Mean | 34.82 ms |
| p50 | 33.69 ms |
| p95 | 40.00 ms |
| p99 | 58.44 ms |

Hardware, catalog size and concurrency affect latency.

## Run Locally

Create a Python 3.11 virtual environment.

Install the requirements:

```powershell
python -m pip install -r requirements.txt
```

Prepare the Amazon ESCI dataset as described in
prepare_amazon.py.

Train the baseline and ranking models:

```powershell
python main.py train
python train_ranker.py
python reliability_ranker.py
```

Start the API:

```powershell
python -m uvicorn production_api:app --reload
```

Swagger documentation:

http://127.0.0.1:8000/docs

Start the dashboard in another terminal:

```powershell
python -m streamlit run dashboard.py
```

## Experimental Findings

LambdaMART performs better on predefined labeled
candidates than in the complete retrieval pipeline.

Feature-distribution diagnostics suggest a
candidate-distribution mismatch.

The tested adaptive selector did not outperform the
strongest baseline on the existing held-out subset.

This motivates further retrieval-aware training
and validation on new held-out queries.

## Limitations

- Sampled catalog of 2,936 products
- Limited relevance annotations
- Only 30 queries in the initial test subset
- Full production-scale concurrency untested
- Adaptive method improvement not established
- Fresh independent research validation pending

## Research Deliverables

- IEEE conference-format paper: paper/
- LaTeX Beamer presentation: slides/

## Project Status

Functional ML research prototype with working API,
baseline experiments, ranking evaluations and
local inference benchmarks.

Not yet validated as a production-scale search system.
