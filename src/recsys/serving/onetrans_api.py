"""Отдельный HTTP deployment настоящего OneTrans checkpoint, порт по умолчанию 8002."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
from fastapi import FastAPI, Query, Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

from recsys.data.dataset import InteractionData, load_dataset
from recsys.models.onetrans_recommender import OneTransRecommender, dataset_fingerprint
from recsys.models.popularity import PopularityRecommender

LOGGER = logging.getLogger("recsys.onetrans")


def create_app(
    checkpoint: Path,
    category: str = "Video_Games",
    device: str = "cpu",
    candidate_batch: int = 2048,
    data: InteractionData | None = None,
) -> FastAPI:
    registry = CollectorRegistry()
    requests = Counter(
        "onetrans_requests_total", "HTTP requests", ["path", "status"], registry=registry
    )
    latency = Histogram("onetrans_request_seconds", "HTTP latency", ["path"], registry=registry)
    cache_events = Counter(
        "onetrans_cache_events_total", "Cache events", ["kind"], registry=registry
    )
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        dataset = data if data is not None else load_dataset(category)
        app.state.data = dataset
        app.state.model = OneTransRecommender(checkpoint, device, candidate_batch).fit(dataset)
        app.state.fallback = PopularityRecommender().fit(dataset)
        app.state.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        app.state.dataset_fingerprint = dataset_fingerprint(dataset)
        yield

    app = FastAPI(title="OneTrans ranker — block 2", version="1.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        # Неизвестные URL не создают бесконечную кардинальность labels.
        path = (
            request.url.path
            if request.url.path in {"/recommend", "/health", "/metrics"}
            else "other"
        )
        status = 500
        try:
            response: Response = await call_next(request)
            status = response.status_code
            return response
        finally:
            requests.labels(path, str(status)).inc()
            latency.labels(path).observe(time.perf_counter() - started)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": "onetrans",
            "device": device,
            "checkpoint_sha256": app.state.checkpoint_sha256,
            "dataset_fingerprint": app.state.dataset_fingerprint,
            "catalog_items": app.state.data.n_items,
        }

    @app.get("/recommend")
    def recommend(
        user_id: Annotated[str, Query(min_length=1, max_length=256)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> dict[str, Any]:
        dataset: InteractionData = app.state.data
        model: OneTransRecommender = app.state.model
        started = time.perf_counter()
        user = int(dataset.users.get_indexer([user_id])[0])
        with lock:
            before = dict(model.cache_counters)
            if user < 0:
                # У неизвестного user нет обученного embedding: используем явный fallback.
                ids = np.argsort(-app.state.fallback.scores, kind="stable")[
                    : min(k, dataset.n_items)
                ]
                used = "popularity"
            else:
                count = min(k, dataset.n_items - dataset.train_csr[user].nnz)
                ids = (
                    model.recommend(np.array([user]), k=count)[0]
                    if count
                    else np.array([], dtype=np.int64)
                )
                used = "onetrans"
            for kind, value in model.cache_counters.items():
                cache_events.labels(kind).inc(value - before[kind])
        elapsed = time.perf_counter() - started
        LOGGER.info(
            json.dumps(
                {
                    "event": "recommend",
                    "model": used,
                    "k": len(ids),
                    "seconds": elapsed,
                    "checkpoint_sha256": app.state.checkpoint_sha256,
                }
            )
        )
        return {
            "user_id": user_id,
            "model": used,
            "fallback": user < 0,
            "checkpoint_sha256": app.state.checkpoint_sha256,
            "recommendations": [
                {"item_id": str(dataset.items[int(item)]), "rank": rank + 1}
                for rank, item in enumerate(ids)
            ],
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--candidate-batch", type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(4)
    import uvicorn

    uvicorn.run(
        create_app(args.checkpoint, args.category, args.device, args.candidate_batch),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
