"""Prometheus-метрики сервиса и PSI-дрейф распределения показов.

Специфика RecSys-мониторинга: latency и ошибки мало что говорят о качестве,
поэтому рядом с инфраструктурными метриками живут ML-прокси (CTR,
покрытие каталога) и дрейф популярности. PSI считаем по распределению
показанных товаров относительно обучающего распределения.
"""

from __future__ import annotations

import threading

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Info

REQUEST_COUNT = Counter("recsys_requests_total", "Запросы к API", ["endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "recsys_request_latency_seconds",
    "Латентность обработки запроса",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
CACHE_EVENTS = Counter("recsys_cache_events_total", "Кэш рекомендаций", ["result"])
FALLBACK_COUNT = Counter(
    "recsys_fallback_total", "Запросы, обслуженные популярностью (холодный старт)"
)
IMPRESSIONS = Counter("recsys_recommended_items_total", "Показы товаров", ["item_id"])
FEEDBACK_COUNT = Counter("recsys_feedback_total", "События обратной связи", ["event"])
CTR_GAUGE = Gauge("recsys_ctr_proxy", "CTR-прокси по событиям feedback")
COVERAGE_GAUGE = Gauge("recsys_catalog_coverage", "Доля каталога в показах")
PSI_GAUGE = Gauge("recsys_popularity_psi", "PSI показов относительно train-распределения")
MODEL_INFO = Info("recsys_model", "Информация о загруженной модели")


class DriftMonitor:
    """Считает PSI (Population Stability Index) по потоку показов.

    PSI < 0.1 — распределение стабильно, 0.1..0.25 — умеренный сдвиг,
    > 0.25 — заметный дрейф, повод для алерта и переобучения.
    """

    def __init__(self, reference: np.ndarray, n_bins: int = 10) -> None:
        self.reference = reference.astype(np.float64)
        self.n_bins = n_bins
        self._counts = np.zeros(len(self.reference), dtype=np.float64)
        self._total = 0.0
        self._lock = threading.Lock()

    def observe(self, item_idx: np.ndarray | list[int]) -> None:
        indices = np.asarray(item_idx, dtype=np.int64)
        with self._lock:
            np.add.at(self._counts, indices, 1.0)
            self._total += len(indices)

    def psi(self) -> float:
        with self._lock:
            if self._total == 0:
                return 0.0
            actual = self._counts / self._total
        ref = self.reference / max(self.reference.sum(), 1e-12)
        eps = 1e-6
        return float(np.sum((actual - ref) * np.log((actual + eps) / (ref + eps))))

    def coverage(self) -> float:
        with self._lock:
            return float((self._counts > 0).mean())

    def reset(self) -> None:
        with self._lock:
            self._counts[:] = 0.0
            self._total = 0.0
