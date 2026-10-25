
import pandas as pd
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class NeuralSearchEngine:

    def __init__(self, products):

        self.products = products.reset_index(drop=True)

        self.documents = (
            self.products["title"].fillna("")
            + " "
            + self.products["description"].fillna("")
        ).tolist()

        self.tokenized_docs = [
            document.lower().split()
            for document in self.documents
        ]

        print("Building BM25 index...")

        self.bm25 = BM25Okapi(self.tokenized_docs)

        print("Loading Sentence Transformer...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Generating product embeddings...")

        self.embeddings = self.model.encode(
            self.documents,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")

        print("Building FAISS vector index...")

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def search(self, query, k=5):

        if not query.strip():
            raise ValueError("Query cannot be empty")

        k = min(k, len(self.products))

        query_tokens = query.lower().split()

        bm25_scores = np.asarray(
            self.bm25.get_scores(query_tokens),
            dtype=float
        )

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        semantic_scores, semantic_indices = (
            self.index.search(
                query_embedding,
                k
            )
        )

        semantic_scores = semantic_scores[0]
        semantic_indices = semantic_indices[0]

        bm25_top = np.argsort(-bm25_scores)[:k]

        # Reciprocal rank fusion combines rankings
        # without assuming that raw model scores
        # are on comparable scales.

        fused = {}

        for rank, idx in enumerate(bm25_top, start=1):
            idx = int(idx)
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)

        for rank, idx in enumerate(semantic_indices, start=1):
            idx = int(idx)
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)

        ranked = sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True
        )[:k]

        results = []

        for idx, score in ranked:

            product = self.products.iloc[idx]

            results.append({
                "product_id": str(product["product_id"]),
                "title": str(product["title"]),
                "fusion_score": round(float(score), 6)
            })

        return results


if __name__ == "__main__":

    products = pd.read_csv(
        "data/products.csv"
    )

    engine = NeuralSearchEngine(products)

    query = "wireless bluetooth headphones"

    results = engine.search(query, k=5)

    print("\nNEURAL SEARCH RESULTS\n")

    for result in results:
        print(result)
