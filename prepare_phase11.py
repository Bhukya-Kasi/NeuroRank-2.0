
"""Prepare fresh Amazon ESCI research data for Phase 11."""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "phase11"

SEED = 202611
TARGETS = {
    "train": 400,
    "validation": 100,
    "test": 100,
}


def normalize(text):
    return re.sub(
        r"\s+", " ", str(text).casefold()
    ).strip()


def main():

    examples_file = (
        RAW / "shopping_queries_dataset_examples.parquet"
    )

    products_file = (
        RAW / "shopping_queries_dataset_products.parquet"
    )

    for path in [examples_file, products_file]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {path}"
            )

    print("Reading Amazon ESCI examples...")

    examples = pd.read_parquet(examples_file)

    examples = examples[
        (examples["product_locale"] == "us")
        & (examples["small_version"] == 1)
        & (examples["split"] == "train")
    ].copy()

    examples["query_id"] = (
        examples["query_id"].astype(str)
    )

    examples["product_id"] = (
        examples["product_id"].astype(str)
    )

    examples["query_key"] = (
        examples["query"].fillna("").map(normalize)
    )

    examples = examples[
        examples["esci_label"].isin(
            ["E", "S", "C", "I"]
        )
        & examples["query_key"].ne("")
    ].copy()

    # Exclude queries already used in earlier phases.
    old_file = ROOT / "data" / "examples.csv"

    if not old_file.exists():
        raise FileNotFoundError(
            "Missing data/examples.csv"
        )

    old = pd.read_csv(old_file)

    old_ids = set(
        old["query_id"].astype(str)
    )

    old_queries = set(
        old["query"].map(normalize)
    )

    examples = examples[
        ~examples["query_id"].isin(old_ids)
        & ~examples["query_key"].isin(old_queries)
    ].copy()

    # Ensure query IDs have consistent query text.
    id_counts = (
        examples.groupby("query_id")["query_key"]
        .nunique()
    )

    valid_ids = id_counts[
        id_counts == 1
    ].index

    examples = examples[
        examples["query_id"].isin(valid_ids)
    ].copy()

    # Exclude query text shared by different query IDs.
    key_counts = (
        examples.groupby("query_key")["query_id"]
        .nunique()
    )

    valid_keys = key_counts[
        key_counts == 1
    ].index

    examples = examples[
        examples["query_key"].isin(valid_keys)
    ].copy()

    print("Checking eligible queries...")

    grouped = examples.groupby("query_id")

    counts = grouped.size()

    has_positive = grouped["esci_label"].agg(
        lambda x: x.isin(["E", "S", "C"]).any()
    )

    has_negative = grouped["esci_label"].agg(
        lambda x: (x == "I").any()
    )

    eligible = sorted(
        set(counts[counts >= 5].index)
        & set(has_positive[has_positive].index)
        & set(has_negative[has_negative].index)
    )

    required = sum(TARGETS.values())

    print("Eligible queries:", len(eligible))

    if len(eligible) < required:
        raise ValueError(
            f"Need {required} queries, "
            f"but found only {len(eligible)}."
        )

    rng = np.random.default_rng(SEED)

    chosen = rng.choice(
        eligible,
        size=required,
        replace=False,
    )

    assignment = {}

    start = 0

    for split, count in TARGETS.items():

        selected_ids = chosen[
            start:start + count
        ]

        for qid in selected_ids:
            assignment[str(qid)] = split

        start += count

    selected = examples[
        examples["query_id"].isin(assignment)
    ].copy()

    selected["research_split"] = (
        selected["query_id"].map(assignment)
    )

    selected = selected.rename(
        columns={"esci_label": "label"}
    )

    selected = selected.drop(
        columns=["query_key"]
    )

    print("Reading product catalog...")

    products = pd.read_parquet(
        products_file,
        columns=[
            "product_id",
            "product_locale",
            "product_title",
            "product_description",
        ],
    )

    products = products[
        products["product_locale"] == "us"
    ].copy()

    products["product_id"] = (
        products["product_id"].astype(str)
    )

    products = products[
        products["product_id"].isin(
            selected["product_id"]
        )
    ].drop_duplicates("product_id")

    products = products.rename(
        columns={
            "product_title": "title",
            "product_description": "description",
        }
    )

    products["title"] = (
        products["title"].fillna("")
    )

    products["description"] = (
        products["description"].fillna("")
    )

    selected = selected[
        selected["product_id"].isin(
            products["product_id"]
        )
    ].copy()

    final_counts = (
        selected.groupby("research_split")["query_id"]
        .nunique()
        .to_dict()
    )

    if final_counts != TARGETS:
        raise ValueError(
            f"Split counts changed: {final_counts}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in TARGETS:

        subset = selected[
            selected["research_split"] == split
        ]

        subset.to_csv(
            OUT / f"{split}_judgments.csv",
            index=False,
        )

    products.to_csv(
        OUT / "products.csv",
        index=False,
    )

    manifest = {
        "seed": SEED,
        "query_counts": final_counts,
        "total_judgments": int(len(selected)),
        "catalog_products": int(len(products)),
        "previous_queries_excluded": True,
        "label_mapping": {
            "I": 0,
            "C": 1,
            "S": 2,
            "E": 3,
        },
        "warning": (
            "Unjudged products are not verified negatives."
        ),
    }

    with open(
        OUT / "manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("\nPHASE 11 DATASET READY")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
