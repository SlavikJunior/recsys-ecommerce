"""Тесты HTTP-контракта без реальных артефактов и Redis."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from recsys.serving import api as api_module
from recsys.serving.registry import Recommendation


class FakeService:
    n_items = 3
    user_ids = ["u1"]
    item_index = {"i1": 0, "i2": 1, "i3": 2}
    manifest = {"created_at": "2026-01-01T00:00:00Z"}
    redis = None

    def __init__(self) -> None:
        self.impressions = 0
        self.clicks = 0
        self.calls: list[tuple[str, int]] = []

    def recommend(self, user_id: str, k: int = 10) -> tuple[list[Recommendation], str]:
        self.calls.append((user_id, k))
        recs = [
            Recommendation(
                item_id=f"i{i}", score=1.0 / (i + 1), title=f"t{i}", brand="b", price=1.0
            )
            for i in range(min(k, 3))
        ]
        return recs, "model"

    def similar(self, item_id: str, k: int = 10) -> list[Recommendation]:
        if item_id not in self.item_index:
            return []
        return [Recommendation(item_id="i2", score=0.9, title="t2", brand="b", price=2.0)]

    def record_impressions(self, item_ids: list[str]) -> None:
        self.impressions += len(item_ids)

    def record_click(self) -> None:
        self.clicks += 1

    @property
    def ctr_proxy(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake = FakeService()
    monkeypatch.setattr(api_module, "service", fake)
    monkeypatch.setattr(api_module, "drift", None)
    return TestClient(api_module.app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["n_items"] == 3


def test_recommend_returns_k_items(client: TestClient) -> None:
    response = client.get("/recommend/u1?k=2")
    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    assert payload["source"] == "model"
    assert [item["item_id"] for item in payload["items"]] == ["i0", "i1"]


def test_similar_unknown_item(client: TestClient) -> None:
    assert client.get("/similar/nope").status_code == 404


def test_feedback_updates_ctr(client: TestClient) -> None:
    client.get("/recommend/u1?k=2")
    response = client.post("/feedback", json={"user_id": "u1", "item_id": "i0", "event": "click"})
    assert response.status_code == 200


def test_metrics_endpoint(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "recsys_requests_total" in response.text
