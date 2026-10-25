
"""NeuroRank 2.0 - Interactive ML Search Dashboard."""

import os
import time

import pandas as pd
import requests
import streamlit as st


API_URL = os.environ.get(
    "NEURORANK_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

MODES = [
    "semantic",
    "bm25",
    "rrf",
    "lambdamart",
    "adaptive",
]

MODE_LABELS = {
    "semantic": "MiniLM Semantic",
    "bm25": "BM25 Lexical",
    "rrf": "Reciprocal Rank Fusion",
    "lambdamart": "LambdaMART",
    "adaptive": "Adaptive Ranking",
}


st.set_page_config(
    page_title="NeuroRank 2.0",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 NeuroRank 2.0")

st.caption(
    "Neural Search | Learning to Rank | "
    "Search Reliability | ML Engineering"
)


def api_get(path):
    response = requests.get(
        API_URL + path,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def api_search(query, mode, k):
    payload = {
        "query": query,
        "mode": mode,
        "k": k,
    }

    started = time.perf_counter()

    response = requests.post(
        API_URL + "/search",
        json=payload,
        timeout=120,
    )

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError(
            "Unexpected search API response."
        )

    if not isinstance(
        data.get("results"),
        list,
    ):
        raise ValueError(
            "API response has no results list."
        )

    return data, elapsed_ms


def normalize_results(data):
    rows = []

    for rank, item in enumerate(
        data.get("results", []),
        start=1,
    ):
        if not isinstance(item, dict):
            continue

        rows.append({
            "Rank": rank,
            "Product ID": str(
                item.get("product_id", "")
            ),
            "Product": str(
                item.get("title", "")
            ),
            "Score": item.get("score"),
        })

    return pd.DataFrame(
        rows,
        columns=[
            "Rank",
            "Product ID",
            "Product",
            "Score",
        ],
    )


def display_search(data, elapsed_ms):
    results = normalize_results(data)

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Products returned",
            len(results),
        )

    with b:
        latency = data.get("latency_ms")

        if isinstance(latency, (int, float)):
            st.metric(
                "API inference latency",
                f"{latency:.1f} ms",
            )
        else:
            st.metric(
                "API inference latency",
                "Unavailable",
            )

    with c:
        st.metric(
            "End-to-end HTTP latency",
            f"{elapsed_ms:.1f} ms",
        )

    if results.empty:
        st.warning("No products found.")
    else:
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

    scope = data.get(
        "candidate_scope",
        "not specified",
    )

    st.caption(
        f"Candidate scope: {scope}. "
        "Ranking scores may use different scales "
        "across models."
    )


with st.sidebar:
    st.header("System Configuration")

    st.text_input(
        "FastAPI server",
        value=API_URL,
        disabled=True,
    )

    st.markdown("### API Status")

    try:
        health = api_get("/health")
        st.success("API connected")
        st.json(health)

    except (
        requests.RequestException,
        ValueError,
    ):
        st.error("API unavailable")
        st.info(
            "Start the API using: "
            "python -m uvicorn "
            "production_api:app --reload"
        )

    st.divider()

    st.markdown("### Monitoring")

    try:
        metrics = api_get("/metrics")
        st.json(metrics)
    except (
        requests.RequestException,
        ValueError,
    ):
        st.caption(
            "Metrics endpoint not available."
        )


tab_search, tab_compare, tab_about = st.tabs(
    [
        "Product Search",
        "Model Comparison",
        "About NeuroRank",
    ]
)


with tab_search:
    st.subheader("Search Products")

    query = st.text_input(
        "Enter product query",
        value="wireless bluetooth headphones",
        key="search_query",
    )

    col_mode, col_k = st.columns(2)

    with col_mode:
        mode = st.selectbox(
            "Ranking method",
            MODES,
            format_func=lambda m: MODE_LABELS[m],
        )

    with col_k:
        k = st.slider(
            "Number of products",
            min_value=1,
            max_value=30,
            value=10,
        )

    if st.button(
        "Search Products",
        type="primary",
    ):
        if not query.strip():
            st.warning("Enter a search query.")

        else:
            with st.spinner(
                "Retrieving and ranking products..."
            ):
                try:
                    data, latency = api_search(
                        query,
                        mode,
                        k,
                    )

                    st.session_state[
                        "latest_search"
                    ] = (
                        data,
                        latency,
                        query,
                        mode,
                    )

                except (
                    requests.RequestException,
                    ValueError,
                ) as error:
                    st.error(
                        f"Search failed: {error}"
                    )

    latest = st.session_state.get(
        "latest_search"
    )

    if latest:
        data, latency, old_query, old_mode = latest

        st.caption(
            f"Results for '{old_query}' "
            f"using {MODE_LABELS[old_mode]}"
        )

        display_search(data, latency)


with tab_compare:
    st.subheader("Compare Ranking Algorithms")

    compare_query = st.text_input(
        "Product query",
        value="wireless gaming headset",
        key="compare_query",
    )

    left, right = st.columns(2)

    with left:
        first_mode = st.selectbox(
            "First model",
            MODES,
            index=0,
            format_func=lambda m: MODE_LABELS[m],
        )

    with right:
        second_mode = st.selectbox(
            "Second model",
            MODES,
            index=2,
            format_func=lambda m: MODE_LABELS[m],
        )

    compare_k = st.slider(
        "Comparison depth",
        min_value=5,
        max_value=30,
        value=10,
    )

    if st.button(
        "Compare Models",
        type="primary",
    ):
        if not compare_query.strip():
            st.warning("Enter a query.")

        elif first_mode == second_mode:
            st.warning(
                "Select two different models."
            )

        else:
            try:
                with st.spinner(
                    "Running both ranking methods..."
                ):
                    first_data, first_latency = api_search(
                        compare_query,
                        first_mode,
                        compare_k,
                    )

                    second_data, second_latency = api_search(
                        compare_query,
                        second_mode,
                        compare_k,
                    )

                    st.session_state[
                        "comparison"
                    ] = (
                        first_data,
                        first_latency,
                        second_data,
                        second_latency,
                        first_mode,
                        second_mode,
                        compare_query,
                    )

            except (
                requests.RequestException,
                ValueError,
            ) as error:
                st.error(
                    f"Comparison failed: {error}"
                )

    comparison = st.session_state.get(
        "comparison"
    )

    if comparison:
        (
            first_data,
            first_latency,
            second_data,
            second_latency,
            mode_a,
            mode_b,
            comparison_query,
        ) = comparison

        st.caption(
            f"Comparison for: {comparison_query}"
        )

        left, right = st.columns(2)

        with left:
            st.markdown(
                f"### {MODE_LABELS[mode_a]}"
            )
            display_search(
                first_data,
                first_latency,
            )

        with right:
            st.markdown(
                f"### {MODE_LABELS[mode_b]}"
            )
            display_search(
                second_data,
                second_latency,
            )

        ids_a = [
            str(item.get("product_id"))
            for item in first_data["results"]
            if isinstance(item, dict)
        ]

        ids_b = [
            str(item.get("product_id"))
            for item in second_data["results"]
            if isinstance(item, dict)
        ]

        common = len(
            set(ids_a) & set(ids_b)
        )

        denominator = max(
            1,
            min(len(ids_a), len(ids_b)),
        )

        st.metric(
            "Top-K product overlap",
            f"{100 * common / denominator:.1f}%",
        )

        st.caption(
            "Overlap measures shared product IDs, "
            "not relevance or ranking quality. "
            "API calls are sequential, so their "
            "latencies are not a controlled benchmark."
        )


with tab_about:
    st.subheader("About NeuroRank 2.0")

    st.markdown(
        """
        NeuroRank 2.0 is a multi-stage
        e-commerce search research prototype.

        **Core technology**

        - BM25 lexical retrieval
        - MiniLM semantic embeddings
        - FAISS vector search
        - LightGBM LambdaMART
        - Experimental adaptive ranking
        - FastAPI inference
        - Streamlit interactive interface

        **Research question**

        How does a learned ranker behave
        when the candidate distribution at
        inference differs from the
        supervised training candidates?

        **Evaluation limitations**

        - Performance depends on catalog size
        - ESCI relevance judgments are incomplete
        - Unjudged results are not verified negatives
        - The adaptive model has not demonstrated
          superiority over the strongest baseline
        """
    )

    st.info(
        "The Phase 11 research models are separate "
        "from the current production API. "
        "This dashboard invokes the models exposed "
        "by production_api.py."
    )
