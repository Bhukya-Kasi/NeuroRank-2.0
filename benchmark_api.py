
import time
import numpy as np
import requests

URL = "http://127.0.0.1:8000/search"

QUERIES = [
    "wireless bluetooth headphones",
    "laptop backpack",
    "running shoes",
    "portable charger",
    "noise cancelling headset",
]

latencies = []
errors = 0

# Warm up the server
requests.post(
    URL,
    json={
        "query": QUERIES[0],
        "k": 5,
        "mode": "lambdamart",
    },
    timeout=120,
).raise_for_status()

for i in range(100):

    query = QUERIES[i % len(QUERIES)]

    start = time.perf_counter()

    try:
        response = requests.post(
            URL,
            json={
                "query": query,
                "k": 5,
                "mode": "lambdamart",
            },
            timeout=30,
        )

        response.raise_for_status()

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed)

    except requests.RequestException as error:
        errors += 1
        print("Request failed:", error)

if latencies:
    print("\nNeuroRank API Benchmark")
    print("Successful requests:", len(latencies))
    print("Failed requests:", errors)
    print("Mean:", round(np.mean(latencies), 2), "ms")
    print("p50:", round(np.percentile(latencies, 50), 2), "ms")
    print("p95:", round(np.percentile(latencies, 95), 2), "ms")
    print("p99:", round(np.percentile(latencies, 99), 2), "ms")
