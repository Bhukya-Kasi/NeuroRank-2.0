
"""Prepare the real Amazon ESCI data for NeuroRank."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"

EXAMPLES_FILE = RAW / "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = RAW / "shopping_queries_dataset_products.parquet"

# Start with a manageable subset on a laptop.
TRAIN_QUERIES = 120
TEST_QUERIES = 30

SEED = 42


def main():
    for path in [EXAMPLES_FILE, PRODUCTS_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {path}"
            )

    print("Loading Amazon ESCI examples...")

    examples = pd.read_parquet(
        EXAMPLES_FILE,
        columns=[
            "query_id",
            "query",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "split",
        ],
    )

    # Use the established Task 1 dataset and US locale.
    examples = examples[
        (examples["small_version"] == 1)
        & (examples["product_locale"] == "us")
        & (examples["split"].isin(["train", "test"]))
    ].copy()

    examples = examples.dropna(
        subset=["query_id", "query", "product_id", "esci_label"]
    )

    # Select distinct queries within each official split.
    selected = []

    for split, n in [
        ("train", TRAIN_QUERIES),
        ("test", TEST_QUERIES),
    ]:
        part = examples[examples["split"] == split]

        query_ids = part["query_id"].drop_duplicates()

        if len(query_ids) < n:
            raise ValueError(
                f"Only {len(query_ids)} queries in {split}"
            )

        sampled_ids = query_ids.sample(
            n=n,
            random_state=SEED,
        )

        selected.append(
            part[part["query_id"].isin(sampled_ids)]
        )

    examples = pd.concat(
        selected,
        ignore_index=True,
    )

    # Avoid repeated identical query strings across splits.
    train_texts = set(
        examples.loc[examples["split"] == "train", "query"]
        .str.strip().str.lower()
    )

    examples = examples[
        ~(
            (examples["split"] == "test")
            & (
                examples["query"].str.strip().str.lower()
                .isin(train_texts)
            )
        )
    ].copy()

    # The current SearchEngine maps product_id to one product.
    # Restricting to "us" avoids cross-locale ID ambiguity.
    examples = examples.rename(
        columns={"esci_label": "label"}
    )

    print("Loading product metadata...")

    products = pd.read_parquet(
        PRODUCTS_FILE,
        columns=[
            "product_id",
            "product_locale",
            "product_title",
            "product_description",
        ],
    )

    required_ids = set(examples["product_id"])

    products = products[
        (products["product_locale"] == "us")
        & (products["product_id"].isin(required_ids))
    ].copy()

    products = products.drop_duplicates(
        subset=["product_id"]
    )

    products = products.rename(
        columns={
            "product_title": "title",
            "product_description": "description",
        }
    )

    products["title"] = products["title"].fillna("")
    products["description"] = products[
        "description"
    ].fillna("")

    valid_ids = set(products["product_id"])

    examples = examples[
        examples["product_id"].isin(valid_ids)
    ].copy()

    examples = examples[
        examples["label"].isin(["E", "S", "C", "I"])
    ].copy()

    # Prevent ambiguity in the current evaluation function.
    examples = examples.drop_duplicates(
        subset=["query_id", "product_id"]
    )

    output_examples = examples[
        [
            "query_id",
            "query",
            "product_id",
            "label",
            "split",
        ]
    ]

    output_products = products[
        [
            "product_id",
            "title",
            "description",
        ]
    ]

    if output_examples.empty or output_products.empty:
        raise ValueError("Dataset preparation produced no rows")

    train_labels = set(
        output_examples.loc[
            output_examples["split"] == "train",
            "label",
        ]
    )

    if "E" not in train_labels or len(train_labels) == 1:
        raise ValueError(
            "Training subset needs both exact and non-exact labels"
        )

    if (
        output_examples.loc[
            output_examples["split"] == "test",
            "query_id",
        ].nunique() == 0
    ):
        raise ValueError("No test queries remain")

    DATA.mkdir(exist_ok=True)

    output_examples.to_csv(
        DATA / "examples.csv",
        index=False,
    )

    output_products.to_csv(
        DATA / "products.csv",
        index=False,
    )

    print("\nDataset preparation successful!")
    print("Products:", len(output_products))
    print("Examples:", len(output_examples))

    print("\nQuery counts by split:")
    print(
        output_examples.groupby("split")[
            "query_id"
        ].nunique()
    )

    print("\nRelevance label counts:")
    print(output_examples["label"].value_counts())


if __name__ == "__main__":
    main()
