
"""Persistent BM25, MiniLM and FAISS search indexes."""

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from neural_search import NeuralSearchEngine


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class CachedNeuralSearchEngine(NeuralSearchEngine):

    def __init__(self, products, cache_dir="artifacts/search_cache"):

        self.products = products.reset_index(drop=True).copy()

        self.documents = (
            self.products["title"].fillna("").astype(str)
            + " "
            + self.products["description"].fillna("").astype(str)
        ).tolist()

        self.tokenized_docs = [
            document.lower().split()
            for document in self.documents
        ]

        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_docs)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        embeddings_path = self.cache_dir / "embeddings.npy"
        index_path = self.cache_dir / "products.faiss"
        metadata_path = self.cache_dir / "metadata.json"

        fingerprint = self._fingerprint()

        cache_valid = False

        if all(path.exists() for path in (
            embeddings_path,
            index_path,
            metadata_path,
        )):
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )

                cache_valid = (
                    metadata.get("fingerprint") == fingerprint
                    and metadata.get("model_name") == MODEL_NAME
                    and metadata.get("product_count") == len(self.products)
                )
            except (ValueError, OSError):
                cache_valid = False

        print("Loading MiniLM...")
        self.model = SentenceTransformer(MODEL_NAME)

        if cache_valid:
            try:
                print("Loading cached embeddings and FAISS index...")

                self.embeddings = np.load(
                    embeddings_path,
                    allow_pickle=False,
                )

                self.index = faiss.read_index(
                    str(index_path)
                )

                if (
                    self.embeddings.ndim != 2
                    or self.embeddings.shape[0] != len(self.products)
                    or self.index.ntotal != len(self.products)
                    or self.index.d != self.embeddings.shape[1]
                ):
                    raise ValueError("Cached index dimensions mismatch")

                print("Search cache loaded.")
                return

            except (ValueError, OSError, RuntimeError) as error:
                print("Cache unusable; rebuilding:", error)

        print("Generating MiniLM product embeddings...")

        self.embeddings = np.asarray(
            self.model.encode(
                self.documents,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

        self.index = faiss.IndexFlatIP(
            self.embeddings.shape[1]
        )

        self.index.add(
            np.ascontiguousarray(self.embeddings)
        )

        np.save(
            embeddings_path,
            self.embeddings,
            allow_pickle=False,
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        metadata_path.write_text(
            json.dumps({
                "fingerprint": fingerprint,
                "model_name": MODEL_NAME,
                "product_count": len(self.products),
                "embedding_dimension": int(
                    self.embeddings.shape[1]
                ),
            }, indent=2),
            encoding="utf-8",
        )

        print("Persistent search cache saved.")

    def _fingerprint(self):

        digest = hashlib.sha256()
        digest.update(MODEL_NAME.encode("utf-8"))

        for product_id, document in zip(
            self.products["product_id"].astype(str),
            self.documents,
        ):
            digest.update(product_id.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(document.encode("utf-8"))
            digest.update(b"\x00")

        return digest.hexdigest()


if __name__ == "__main__":

    import pandas as pd

    products = pd.read_csv(
        "data/products.csv"
    ).fillna("")

    engine = CachedNeuralSearchEngine(products)

    print(
        engine.search(
            "wireless bluetooth headphones",
            k=5,
        )
    )
