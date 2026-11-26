"""Prometheus-метрики сервиса и PSI-дрейф распределения показов.

Специфика RecSys-мониторинга: latency и ошибки мало что говорят о качестве,
поэтому рядом с инфраструктурными метриками живут ML-прокси (CTR,
покрытие каталога) и дрейф популярности.

PSI считается по децилям популярности, а не по отдельным товарам: на 26 тыс.
товаров и нескольких тысячах показов почти все бины пустые, и «попервотоварный»
PSI всегда огромный. Децили дают устойчивую метрику «сдвинулись ли показы в
хвост или в голову каталога».
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
PSI_GAUGE = Gauge("recsys_popularity_psi", "PSI показов по децилям популярности")
MODEL_INFO = Info("recsys_model", "Информация о загруженной модели")
# На macOS prometheus-client не отдаёт process_* (нужен /proc), поэтому
# ресурсы процесса публикуем сами.
PROCESS_RSS_MB = Gauge("recsys_process_rss_mb", "RSS процесса API, МБ")
PROCESS_CPU_SECONDS = Gauge(
    "recsys_process_cpu_seconds", "Пользовательское CPU-время процесса API, c"
)


def update_process_metrics() -> None:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss на macOS в байтах, на Linux — в килобайтах.
    scale = 1024 * 1024 if usage.ru_maxrss > 10_000_000 else 1024
    PROCESS_RSS_MB.set(usage.ru_maxrss / scale)
    PROCESS_CPU_SECONDS.set(usage.ru_utime + usage.ru_stime)


def decile_bins(reference: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Раскладывает товары по децилям популярности и считает эталонные доли.

    0 — самый популярный дециль. Возвращает (bin_of_item, reference_share).
    """
    order = np.argsort(-reference, kind="stable")
    bin_of_item = np.empty(len(reference), dtype=np.int64)
    edges = np.linspace(0, len(order), n_bins + 1).astype(int)
    for bin_idx in range(n_bins):
        bin_of_item[order[edges[bin_idx] : edges[bin_idx + 1]]] = bin_idx
    shares = np.bincount(bin_of_item, weights=reference, minlength=n_bins)
    shares = shares / max(shares.sum(), 1e-12)
    return bin_of_item, shares


class DriftMonitor:
    """PSI по децилям популярности.

    PSI < 0.1 — распределение стабильно, 0.1..0.25 — умеренный сдвиг,
    > 0.25 — заметный дрейф, повод для алерта и переобучения.
    """

    def __init__(
        self,
        reference: np.ndarray,
        popularity: np.ndarray | None = None,
        n_bins: int = 10,
    ) -> None:
        # Децили считаем по популярности товаров в train, а доли — по эталонным
        # показам. Если считать децили по самим показам, товары, которые модель
        # ни разу не рекомендовала, попадают в «пустые» бины и PSI раздувается.
        self.bin_of_item, _ = decile_bins(
            popularity if popularity is not None else reference, n_bins
        )
        shares = np.bincount(self.bin_of_item, weights=reference, minlength=n_bins)
        self.reference_shares = shares / max(shares.sum(), 1e-12)
        self.n_items = len(reference)
        self._bin_counts = np.zeros(n_bins, dtype=np.float64)
        self._seen_items = np.zeros(self.n_items, dtype=bool)
        self._total = 0.0
        self._lock = threading.Lock()

    def observe(self, item_idx: np.ndarray | list[int]) -> None:
        indices = np.asarray(item_idx, dtype=np.int64)
        if indices.size == 0:
            return
        with self._lock:
            np.add.at(self._bin_counts, self.bin_of_item[indices], 1.0)
            self._seen_items[indices] = True
            self._total += len(indices)

    def psi(self) -> float:
        with self._lock:
            if self._total == 0:
                return 0.0
            actual = self._bin_counts / self._total
        eps = 1e-6
        return float(
            np.sum(
                (actual - self.reference_shares)
                * np.log((actual + eps) / (self.reference_shares + eps))
            )
        )

    def coverage(self) -> float:
        with self._lock:
            return float(self._seen_items.mean())

    def reset(self) -> None:
        with self._lock:
            self._bin_counts[:] = 0.0
            self._seen_items[:] = False
            self._total = 0.0
