"""Сервисный слой: загрузка артефактов и онлайн-ранжирование.

В проде дорогие части (обучение, эмбеддинги, матрица соседей) считаются
офлайн, а онлайн остаётся только скалярное произведение и маскирование
просмотренного. Здесь этот контракт и зафиксирован.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.config import ARTIFACTS_DIR

SERVING_DIR = ARTIFACTS_DIR / "serving"


@dataclass
class Recommendation:
    item_id: str
    score: float
    title: str
    brand: str
    price: float | None


class ArtifactError(RuntimeError):
    pass


class RecommenderService:
    """Инференс без LLM: эмбеддинги + ANN по всему каталогу."""

    def __init__(
        self,
        category: str = "Video_Games",
        artifacts_dir: Path | None = None,
        redis_url: str | None = None,
        cache_ttl_seconds: int = 60,
        chaos: str | None = None,
    ) -> None:
        self.category = category
        # Режим деградации для проверки алертов: tail — отдаёт только хвост
        # каталога (имитация сломанного релиза), latency — добавляет задержку.
        self.chaos = chaos
        self.dir = (artifacts_dir or SERVING_DIR) / category
        self.cache_ttl = cache_ttl_seconds
        self._load()
        self.redis = self._connect_redis(redis_url)
        self.impressions = 0
        self.clicks = 0

    def _load(self) -> None:
        if not self.dir.exists():
            raise ArtifactError(
                f"нет артефактов в {self.dir}: сначала запусти python -m recsys.serving.build"
            )
        manifest_path = self.dir / "manifest.json"
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        self.item_ids = pd.read_parquet(self.dir / "item_ids.parquet")["item_id"].to_numpy()
        self.user_ids = pd.read_parquet(self.dir / "user_ids.parquet")["user_id"].to_numpy()
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {it: i for i, it in enumerate(self.item_ids)}
        self.item_meta = pd.read_parquet(self.dir / "item_meta.parquet").set_index("item_id")
        self.user_embeddings = np.load(self.dir / "user_embeddings.npy")
        self.item_embeddings = np.load(self.dir / "item_embeddings.npy")
        self.popularity = np.load(self.dir / "popularity.npy")
        self.seen = sp.load_npz(self.dir / "seen_csr.npz").tocsr()
        self.item_similarity = sp.load_npz(self.dir / "item_similarity.npz").tocsr()
        self.reference_counts = np.load(self.dir / "item_counts.npy")

    @staticmethod
    def _connect_redis(url: str | None) -> Any:
        if url is None:
            return None
        try:
            import redis

            client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=0.05)
            client.ping()
            return client
        except Exception:  # Redis недоступен — это допустимый режим, а не падение
            return None

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def _describe(self, item_idx: int, score: float) -> Recommendation:
        item_id = str(self.item_ids[item_idx])
        meta = self.item_meta.loc[item_id] if item_id in self.item_meta.index else None
        price = None if meta is None or pd.isna(meta["price"]) else float(meta["price"])
        return Recommendation(
            item_id=item_id,
            score=float(score),
            title="" if meta is None else str(meta["title"]),
            brand="" if meta is None else str(meta["brand"]),
            price=price,
        )

    def recommend(self, user_id: str, k: int = 10) -> tuple[list[Recommendation], str]:
        """Возвращает (рекомендации, источник). Источник важен для мониторинга."""
        if self.chaos == "tail":
            tail = np.argsort(self.reference_counts)[: max(1000, k * 100)]
            pick = np.random.default_rng().choice(tail, size=k, replace=False)
            return [self._describe(int(i), 1.0) for i in pick], "chaos_tail"
        if self.redis is not None:
            cached = self.redis.get(f"rec:{user_id}:{k}")
            if cached:
                payload = json.loads(cached)
                return self._from_cache(payload), "cache"

        idx = self.user_index.get(user_id)
        if idx is None:
            return self._popular(k), "popularity"

        scores = self.user_embeddings[idx] @ self.item_embeddings.T
        scores = scores.copy()
        seen = self.seen.indices[self.seen.indptr[idx] : self.seen.indptr[idx + 1]]
        scores[seen] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        recs = [self._describe(int(i), float(scores[i])) for i in top]

        if self.redis is not None:
            self.redis.setex(
                f"rec:{user_id}:{k}",
                self.cache_ttl,
                json.dumps([{"item_id": r.item_id, "score": r.score} for r in recs]),
            )
        return recs, "model"

    def _from_cache(self, payload: list[dict[str, Any]]) -> list[Recommendation]:
        out = []
        for row in payload:
            idx = self.item_index.get(row["item_id"])
            if idx is None:
                continue
            out.append(self._describe(int(idx), float(row["score"])))
        return out

    def _popular(self, k: int) -> list[Recommendation]:
        top = np.argsort(-self.popularity)[:k]
        return [self._describe(int(i), float(self.popularity[i])) for i in top]

    def similar(self, item_id: str, k: int = 10) -> list[Recommendation]:
        idx = self.item_index.get(item_id)
        if idx is None:
            return []
        row = self.item_similarity.getrow(idx)
        scores = np.asarray(row.toarray()).ravel()
        scores[idx] = -np.inf
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [self._describe(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def record_impressions(self, item_ids: list[str]) -> None:
        self.impressions += len(item_ids)

    def record_click(self) -> None:
        self.clicks += 1

    @property
    def ctr_proxy(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0

    def latency_benchmark(self, n_requests: int = 200, k: int = 10) -> dict[str, float]:
        """Локальный бенчмарк без HTTP: чистое время ранжирования."""
        rng = np.random.default_rng(0)
        users = rng.choice(self.user_ids, size=n_requests, replace=True)
        latencies = []
        for user in users:
            started = time.perf_counter()
            self.recommend(str(user), k=k)
            latencies.append((time.perf_counter() - started) * 1000)
        arr = np.array(latencies)
        return {
            "n": float(len(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
        }
