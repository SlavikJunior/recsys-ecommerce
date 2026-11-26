"""HTTP загружает реальный checkpoint и явно сообщает fallback неизвестному user."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

from recsys.data.dataset import InteractionData
from recsys.models.onetrans import OneTrans, OneTransConfig
from recsys.models.onetrans_features import OneTransFeatures
from recsys.models.onetrans_recommender import dataset_fingerprint, save_checkpoint
from recsys.serving.onetrans_api import create_app


def test_real_checkpoint_http_contract(toy_data: InteractionData, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    features = OneTransFeatures.from_data(toy_data)
    model = OneTrans(
        OneTransConfig(width=8, heads=2),
        features.torch_table(),
        features.feature_sizes,
        toy_data.n_users,
    )
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, dataset_fingerprint(toy_data), False, {"epoch": 0})
    app = create_app(checkpoint, data=toy_data)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model"] == "onetrans"
        user = str(toy_data.users[0])
        response = client.get("/recommend", params={"user_id": user, "k": 2})
        assert response.status_code == 200
        payload = response.json()
        assert payload["checkpoint_sha256"] == health.json()["checkpoint_sha256"]
        assert payload["model"] == "onetrans" and payload["fallback"] is False
        ids = [row["item_id"] for row in payload["recommendations"]]
        seen = set(toy_data.items[toy_data.train_csr[0].indices].tolist())
        assert len(ids) == len(set(ids)) == 2
        assert not set(ids) & seen
        repeat = client.get("/recommend", params={"user_id": user, "k": 2})
        assert repeat.json() == payload
        unknown = client.get("/recommend", params={"user_id": "unknown", "k": 2}).json()
        assert unknown["fallback"] is True and unknown["model"] == "popularity"
        assert client.get("/recommend", params={"user_id": user, "k": 0}).status_code == 422
        assert client.get("/recommend", params={"k": 2}).status_code == 422
        metrics = client.get("/metrics").text
        assert 'onetrans_cache_events_total{kind="exact_hits"}' in metrics
        assert "onetrans_request_seconds_count" in metrics
        expected = app.state.model.recommend(np.array([0]), k=2)[0]
        assert ids == [str(toy_data.items[item]) for item in expected]
