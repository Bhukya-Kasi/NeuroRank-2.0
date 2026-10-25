
"""Phase 11F: Paired statistical analysis of frozen test results.

Reads Phase 11E per-query metrics. Does not train or tune models.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "phase11"
INPUT = RESULTS / "phase11e_per_query.csv"

SEED = 202611
BOOTSTRAP_SAMPLES = 10000
PERMUTATIONS = 20000
ALPHA = 0.05

MODELS = [
    "bm25",
    "semantic",
    "rrf",
    "lambdamart",
    "adaptive",
]

METRICS = ["ndcg10", "recall10", "recall100"]

# These comparisons are defined before running this script.
COMPARISONS = [
    ("rrf", "semantic"),
    ("rrf", "bm25"),
    ("rrf", "lambdamart"),
    ("rrf", "adaptive"),
    ("adaptive", "semantic"),
]


def load_results():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run Phase 11E first."
        )

    data = pd.read_csv(
        INPUT,
        dtype={
            "query_id": str,
            "mode": str,
        },
    )

    required = {
        "query_id",
        "mode",
        *METRICS,
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if data.duplicated(["query_id", "mode"]).any():
        raise ValueError(
            "Duplicate query_id/mode combinations detected."
        )

    if not set(data["mode"]).issubset(MODELS):
        raise ValueError("Unexpected ranking mode found.")

    expected_modes = set(MODELS)

    for qid, group in data.groupby("query_id"):

        if set(group["mode"]) != expected_modes:
            raise ValueError(
                f"Missing model results for query {qid}"
            )

    for metric in METRICS:

        data[metric] = pd.to_numeric(
            data[metric],
            errors="raise",
        )

        if not np.isfinite(data[metric]).all():
            raise ValueError(
                f"Invalid values in {metric}"
            )

        if (
            (data[metric] < 0)
            | (data[metric] > 1)
        ).any():
            raise ValueError(
                f"Out-of-range values in {metric}"
            )

    return data


def metric_matrix(data, metric):

    wide = data.pivot(
        index="query_id",
        columns="mode",
        values=metric,
    )

    return wide[MODELS].sort_index()


def bootstrap_means(values, rng):

    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        raise ValueError("At least two queries required.")

    means = np.empty(BOOTSTRAP_SAMPLES)

    for start in range(0, BOOTSTRAP_SAMPLES, 500):

        count = min(
            500,
            BOOTSTRAP_SAMPLES - start,
        )

        indices = rng.integers(
            0,
            n,
            size=(count, n),
        )

        means[start:start + count] = (
            values[indices].mean(axis=1)
        )

    return [
        float(np.quantile(means, ALPHA / 2)),
        float(np.quantile(means, 1 - ALPHA / 2)),
    ]


def paired_permutation_pvalue(differences, rng):

    differences = np.asarray(
        differences,
        dtype=float,
    )

    observed = abs(
        float(np.mean(differences))
    )

    exceedances = 0

    for start in range(0, PERMUTATIONS, 500):

        count = min(
            500,
            PERMUTATIONS - start,
        )

        signs = rng.choice(
            [-1.0, 1.0],
            size=(count, len(differences)),
        )

        simulated = abs(
            (signs * differences).mean(axis=1)
        )

        exceedances += int(
            np.count_nonzero(
                simulated >= observed - 1e-15
            )
        )

    return float(
        (exceedances + 1) / (PERMUTATIONS + 1)
    )


def holm_adjust(pvalues):

    values = np.asarray(pvalues, dtype=float)
    m = len(values)

    order = np.argsort(values)
    adjusted = np.empty(m)

    running = 0.0

    for position, idx in enumerate(order):

        candidate = (
            (m - position) * values[idx]
        )

        running = max(running, candidate)

        adjusted[idx] = min(1.0, running)

    return adjusted


def main():

    print("\nNEURORANK PHASE 11F")
    print("Statistical analysis of held-out rankings\n")

    data = load_results()

    query_count = data["query_id"].nunique()

    print("Queries:", query_count)

    if query_count < 20:
        print(
            "Warning: small test sample; "
            "confidence intervals may be unstable."
        )

    rng = np.random.default_rng(SEED)

    baseline_rows = []
    comparisons = []

    # Compute bootstrap CIs for the model means.
    for metric in METRICS:

        wide = metric_matrix(data, metric)

        for model in MODELS:

            values = wide[model].to_numpy()
            low, high = bootstrap_means(values, rng)

            baseline_rows.append({
                "metric": metric,
                "model": model,
                "mean": float(values.mean()),
                "ci95_lower": low,
                "ci95_upper": high,
            })

    # Hypothesis tests are limited to NDCG@10.
    wide = metric_matrix(data, "ndcg10")

    for first, second in COMPARISONS:

        difference = (
            wide[first] - wide[second]
        ).to_numpy(dtype=float)

        low, high = bootstrap_means(
            difference,
            rng,
        )

        pvalue = paired_permutation_pvalue(
            difference,
            rng,
        )

        tolerance = 1e-12

        comparisons.append({
            "first_model": first,
            "second_model": second,
            "mean_difference": float(
                difference.mean()
            ),
            "ci95_lower": low,
            "ci95_upper": high,
            "pvalue_unadjusted": pvalue,
            "first_wins": int(
                np.sum(difference > tolerance)
            ),
            "second_wins": int(
                np.sum(difference < -tolerance)
            ),
            "ties": int(
                np.sum(abs(difference) <= tolerance)
            ),
        })

    baseline_df = pd.DataFrame(baseline_rows)
    comparison_df = pd.DataFrame(comparisons)

    comparison_df["pvalue_holm"] = holm_adjust(
        comparison_df["pvalue_unadjusted"]
    )

    comparison_df["significant_holm_05"] = (
        comparison_df["pvalue_holm"] < ALPHA
    )

    RESULTS.mkdir(parents=True, exist_ok=True)

    baseline_df.to_csv(
        RESULTS / "phase11f_model_intervals.csv",
        index=False,
    )

    comparison_df.to_csv(
        RESULTS / "phase11f_paired_comparisons.csv",
        index=False,
    )

    summary = {
        "phase": "11F",
        "queries": int(query_count),
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAP_SAMPLES,
        "permutation_resamples": PERMUTATIONS,
        "confidence_level": 0.95,
        "pvalue_adjustment": "Holm",
        "primary_metric": "ndcg10",
        "model_intervals": baseline_df.round(
            6
        ).to_dict(orient="records"),
        "paired_comparisons": comparison_df.round(
            6
        ).to_dict(orient="records"),
        "methodology": (
            "Query-level paired bootstrap percentile "
            "confidence intervals and two-sided "
            "paired sign-flip permutation tests."
        ),
        "limitations": [
            "Unjudged products treated as zero relevance.",
            "Queries treated as independent sampling units.",
            "Results apply to the sampled ESCI catalog.",
            "Statistical tests do not establish calibration.",
            "Models are not retrained or tuned here.",
        ],
    }

    (RESULTS / "phase11f_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nNDCG@10 MODEL INTERVALS\n")

    display = baseline_df[
        baseline_df["metric"] == "ndcg10"
    ].copy()

    print(
        display[
            [
                "model",
                "mean",
                "ci95_lower",
                "ci95_upper",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nPAIRED NDCG@10 COMPARISONS\n")

    print(
        comparison_df[
            [
                "first_model",
                "second_model",
                "mean_difference",
                "ci95_lower",
                "ci95_upper",
                "pvalue_holm",
                "significant_holm_05",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nFiles saved to:", RESULTS)


if __name__ == "__main__":
    main()
