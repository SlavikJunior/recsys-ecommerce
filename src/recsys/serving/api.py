"""FastAPI-сервис рекомендаций.

Контракт:
- GET /recommend/{user_id} — персональные рекомендации или популярное;
- GET /similar/{item_id} — «с этим товаром также смотрят»;
- POST /feedback — событие пользователя (impression/click/purchase);
- GET /metrics — Prometheus;
- GET /health — готовность и версия модели.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from recsys.monitoring import metrics as m
from recsys.serving.registry import ArtifactError, RecommenderService

CATEGORY = os.environ.get("RECSYS_CATEGORY", "Video_Games")
REDIS_URL = os.environ.get("RECSYS_REDIS_URL", "redis://127.0.0.1:6379/0")
# Chaos-переключатели нужны только для демонстрации алертов, по умолчанию выключены.
CHAOS = os.environ.get("RECSYS_CHAOS")
CHAOS_DELAY_MS = int(os.environ.get("RECSYS_CHAOS_DELAY_MS", "0"))

service: RecommenderService | None = None
drift: m.DriftMonitor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global service, drift
    try:
        service = RecommenderService(category=CATEGORY, redis_url=REDIS_URL, chaos=CHAOS)
    except ArtifactError as exc:
        # Сервис не должен молча стартовать без модели: /health вернёт 503.
        print(f"[startup] {exc}")
        service = None
        yield
        return
    drift = m.DriftMonitor(reference=service.drift_reference, popularity=service.reference_counts)
    m.MODEL_INFO.info(
        {
            "category": CATEGORY,
            "n_items": str(service.n_items),
            "created_at": str(service.manifest.get("created_at", "")),
        }
    )
    yield


app = FastAPI(title="recsys-ecommerce", version="0.1.0", lifespan=lifespan)


@contextmanager
def timed(endpoint: str) -> Iterator[None]:
    started = time.perf_counter()
    if CHAOS_DELAY_MS:
        time.sleep(CHAOS_DELAY_MS / 1000)
    try:
        yield
        m.REQUEST_COUNT.labels(endpoint=endpoint, status="ok").inc()
    except Exception:
        m.REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        raise
    finally:
        m.REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - started)


class FeedbackEvent(BaseModel):
    user_id: str
    item_id: str
    event: Literal["impression", "click", "purchase"]
    position: int = Field(ge=0, le=100, default=0)


class ItemOut(BaseModel):
    item_id: str
    title: str
    brand: str
    price: float | None
    score: float


class RecommendResponse(BaseModel):
    user_id: str
    source: str
    items: list[ItemOut]


def _service() -> RecommenderService:
    if service is None:
        raise HTTPException(status_code=503, detail="модель не загружена")
    return service


@app.get("/health")
def health() -> dict[str, Any]:
    if service is None:
        raise HTTPException(status_code=503, detail="no artifacts")
    return {
        "status": "ok",
        "category": CATEGORY,
        "n_items": service.n_items,
        "n_users": len(service.user_ids),
        "cache": service.redis is not None,
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: str, k: int = Query(default=10, ge=1, le=50)) -> RecommendResponse:
    svc = _service()
    with timed("recommend"):
        recs, source = svc.recommend(user_id, k=k)
        m.CACHE_EVENTS.labels(result="hit" if source == "cache" else "miss").inc()
        if source == "popularity":
            m.FALLBACK_COUNT.inc()
        indices = [svc.item_index[r.item_id] for r in recs if r.item_id in svc.item_index]
        if drift is not None:
            drift.observe(indices)
            m.COVERAGE_GAUGE.set(drift.coverage())
            m.PSI_GAUGE.set(drift.psi())
        for item in recs:
            m.IMPRESSIONS.labels(item_id=item.item_id).inc()
        svc.record_impressions([r.item_id for r in recs])
    return RecommendResponse(
        user_id=user_id,
        source=source,
        items=[
            ItemOut(
                item_id=r.item_id,
                title=r.title,
                brand=r.brand,
                price=r.price,
                score=r.score,
            )
            for r in recs
        ],
    )


@app.get("/similar/{item_id}", response_model=RecommendResponse)
def similar(item_id: str, k: int = Query(default=10, ge=1, le=50)) -> RecommendResponse:
    svc = _service()
    with timed("similar"):
        recs = svc.similar(item_id, k=k)
    if not recs:
        raise HTTPException(status_code=404, detail="неизвестный item_id")
    return RecommendResponse(
        user_id="",
        source="item_cf",
        items=[
            ItemOut(item_id=r.item_id, title=r.title, brand=r.brand, price=r.price, score=r.score)
            for r in recs
        ],
    )


@app.post("/feedback")
def feedback(event: FeedbackEvent) -> dict[str, str]:
    svc = _service()
    with timed("feedback"):
        m.FEEDBACK_COUNT.labels(event=event.event).inc()
        if event.event in {"click", "purchase"}:
            svc.record_click()
        m.CTR_GAUGE.set(svc.ctr_proxy)
        # Кэш инвалидируем на purchase: история пользователя изменилась.
        if event.event == "purchase" and svc.redis is not None:
            svc.redis.delete(f"rec:{event.user_id}:10")
    return {"status": "accepted"}


@app.get("/metrics")
def prometheus_metrics() -> Response:
    m.update_process_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
