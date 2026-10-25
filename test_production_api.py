
from fastapi.testclient import TestClient

import production_api


def test_health_endpoint(monkeypatch):

    monkeypatch.setattr(
        production_api,
        "get_system",
        lambda: (None, None),
    )

    with TestClient(production_api.app) as client:

        response = client.get("/health")

        assert response.status_code == 200

        body = response.json()

        assert body["status"] == "ok"


def test_invalid_search_request(monkeypatch):

    monkeypatch.setattr(
        production_api,
        "get_system",
        lambda: (None, None),
    )

    with TestClient(production_api.app) as client:

        response = client.post(
            "/search",
            json={
                "query": "",
                "k": 5,
                "mode": "semantic",
            },
        )

        assert response.status_code == 422


def test_metrics_endpoint(monkeypatch):

    monkeypatch.setattr(
        production_api,
        "get_system",
        lambda: (None, None),
    )

    with TestClient(production_api.app) as client:

        response = client.get("/metrics")

        assert response.status_code == 200

        body = response.json()

        assert "p95_ms" in body
        assert "requests_recorded" in body
