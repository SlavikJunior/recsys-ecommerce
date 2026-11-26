"""Генератор дашбордов Grafana: дашборды как код, а не клики в UI.

Два дашборда:
1. «RecSys — сервис и качество»: RPS, латентность, ошибки, кэш, CTR-прокси,
   покрытие каталога, PSI-дрейф, доля fallback.
2. «RecSys — инфраструктура»: CPU, память, load average, диск.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "deploy" / "grafana" / "dashboards"
DS = {"type": "prometheus", "uid": "prometheus"}


def ts(
    title: str,
    expr: str,
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    unit: str = "short",
    legend: str = "",
) -> dict[str, Any]:
    return {
        "type": "timeseries",
        "title": title,
        "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {
            "defaults": {"unit": unit, "custom": {"fillOpacity": 8, "lineWidth": 2}},
            "overrides": [],
        },
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "multi"},
        },
        "targets": [{"expr": expr, "refId": "A", "legendFormat": legend or "__auto"}],
    }


def stat(
    title: str,
    expr: str,
    x: int,
    y: int,
    w: int = 4,
    h: int = 4,
    unit: str = "percentunit",
    thresholds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "type": "stat",
        "title": title,
        "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "thresholds": {
                    "mode": "absolute",
                    "steps": thresholds
                    or [
                        {"color": "red", "value": None},
                        {"color": "orange", "value": 0.1},
                        {"color": "green", "value": 0.25},
                    ],
                },
            },
            "overrides": [],
        },
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value"},
        "targets": [{"expr": expr, "refId": "A"}],
    }


def service_dashboard() -> dict[str, Any]:
    panels = [
        stat(
            "RPS",
            'sum(rate(recsys_requests_total{endpoint="recommend"}[1m]))',
            0,
            0,
            4,
            4,
            unit="reqps",
        ),
        stat(
            "p95 latency",
            'histogram_quantile(0.95, sum by (le) (rate(recsys_request_latency_seconds_bucket{endpoint="recommend"}[1m])))',
            4,
            0,
            4,
            4,
            unit="s",
            thresholds=[
                {"color": "green", "value": None},
                {"color": "orange", "value": 0.15},
                {"color": "red", "value": 0.3},
            ],
        ),
        stat(
            "Кэш: hit rate",
            'sum(rate(recsys_cache_events_total{result="hit"}[1m])) / clamp_min(sum(rate(recsys_cache_events_total[1m])), 1e-9)',
            8,
            0,
            4,
            4,
            thresholds=[
                {"color": "red", "value": None},
                {"color": "orange", "value": 0.2},
                {"color": "green", "value": 0.5},
            ],
        ),
        stat(
            "CTR-прокси",
            "recsys_ctr_proxy",
            12,
            0,
            4,
            4,
            thresholds=[
                {"color": "red", "value": None},
                {"color": "orange", "value": 0.05},
                {"color": "green", "value": 0.15},
            ],
        ),
        stat(
            "Покрытие каталога",
            "recsys_catalog_coverage",
            16,
            0,
            4,
            4,
            thresholds=[
                {"color": "red", "value": None},
                {"color": "orange", "value": 0.3},
                {"color": "green", "value": 0.6},
            ],
        ),
        stat(
            "PSI дрейфа",
            "recsys_popularity_psi",
            20,
            0,
            4,
            4,
            unit="none",
            thresholds=[
                {"color": "green", "value": None},
                {"color": "orange", "value": 0.1},
                {"color": "red", "value": 0.25},
            ],
        ),
        ts(
            "Латентность по эндпоинтам (p50/p95/p99)",
            "histogram_quantile(0.5, sum by (le, endpoint) (rate(recsys_request_latency_seconds_bucket[1m])))",
            0,
            4,
            12,
            8,
            unit="s",
            legend="p50 {{endpoint}}",
        ),
        ts(
            "RPS по эндпоинтам",
            "sum by (endpoint) (rate(recsys_requests_total[1m]))",
            12,
            4,
            12,
            8,
            legend="{{endpoint}}",
        ),
        ts(
            "Доля fallback (холодный старт)",
            'sum(rate(recsys_fallback_total[1m])) / clamp_min(sum(rate(recsys_requests_total{endpoint="recommend"}[1m])), 1e-9)',
            0,
            12,
            12,
            8,
            unit="percentunit",
            legend="fallback",
        ),
        ts(
            "Ошибки API",
            'sum by (status) (rate(recsys_requests_total{status="error"}[1m]))',
            12,
            12,
            12,
            8,
            legend="{{status}}",
        ),
        ts(
            "Топ-10 показываемых товаров",
            "topk(10, sum by (item_id) (rate(recsys_recommended_items_total[5m])))",
            0,
            20,
            24,
            8,
            legend="{{item_id}}",
        ),
    ]
    return {
        "uid": "recsys-service",
        "title": "RecSys — сервис и качество",
        "schemaVersion": 39,
        "refresh": "5s",
        "time": {"from": "now-30m", "to": "now"},
        "panels": panels,
    }


def infra_dashboard() -> dict[str, Any]:
    panels = [
        ts(
            "CPU, %",
            '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)',
            0,
            0,
            12,
            8,
            unit="percent",
            legend="cpu",
        ),
        ts(
            "Память, %",
            "100 * (node_memory_active_bytes + node_memory_compressed_bytes) / "
            "(node_memory_active_bytes + node_memory_inactive_bytes + "
            "node_memory_free_bytes + node_memory_compressed_bytes)",
            12,
            0,
            12,
            8,
            unit="percent",
            legend="mem",
        ),
        ts(
            "Load average (1m)",
            "node_load1",
            0,
            8,
            12,
            8,
            legend="load1",
        ),
        ts(
            "Свободно на диске, %",
            'node_filesystem_avail_bytes{mountpoint="/System/Volumes/Data"} / node_filesystem_size_bytes{mountpoint="/System/Volumes/Data"} * 100',
            12,
            8,
            12,
            8,
            unit="percent",
            legend="disk free",
        ),
        ts(
            "CPU-время API, c",
            "rate(recsys_process_cpu_seconds[1m])",
            0,
            16,
            12,
            8,
            unit="s",
            legend="cpu seconds/s",
        ),
        ts(
            "RSS процесса API, МБ",
            "recsys_process_rss_mb",
            12,
            16,
            12,
            8,
            unit="decmbytes",
            legend="rss",
        ),
    ]
    return {
        "uid": "recsys-infra",
        "title": "RecSys — инфраструктура",
        "schemaVersion": 39,
        "refresh": "5s",
        "time": {"from": "now-30m", "to": "now"},
        "panels": panels,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, dashboard in (
        ("recsys-service.json", service_dashboard()),
        ("recsys-infra.json", infra_dashboard()),
    ):
        (OUT / name).write_text(json.dumps(dashboard, indent=2, ensure_ascii=False))
        print(OUT / name)


if __name__ == "__main__":
    main()
