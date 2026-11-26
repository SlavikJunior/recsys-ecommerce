"""Сохранить фактическое HTTP deployment/load evidence без замены модели заглушкой."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from recsys.data.dataset import load_dataset


def fetch(base: str, path: str) -> tuple[int, bytes, float]:
    started = time.perf_counter()
    try:
        with urlopen(base + path, timeout=30) as response:
            return response.status, response.read(), time.perf_counter() - started
    except HTTPError as error:
        return error.code, error.read(), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("requests должен быть положительным")
    data = load_dataset(args.category)
    selected = np.load(args.output / "selection.npz", allow_pickle=False)["test_users"]
    status, body, _ = fetch(args.base_url, "/health")
    if status != 200:
        raise RuntimeError(f"health: HTTP {status}")
    health = json.loads(body)
    timings = []
    responses = []
    started = time.perf_counter()
    for index in range(args.requests):
        uid = int(selected[index % min(10, len(selected))])
        user_id = str(data.users[uid])
        status, body, elapsed = fetch(
            args.base_url, "/recommend?" + urlencode({"user_id": user_id, "k": 20})
        )
        if status != 200:
            raise RuntimeError(f"recommend: HTTP {status}")
        payload = json.loads(body)
        if (
            payload["model"] != "onetrans"
            or payload["checkpoint_sha256"] != health["checkpoint_sha256"]
        ):
            raise RuntimeError("сервер не использует ожидаемый OneTrans checkpoint")
        ids = data.items.get_indexer([row["item_id"] for row in payload["recommendations"]])
        if len(ids) != 20 or len(set(ids)) != 20 or np.any(ids < 0):
            raise RuntimeError("невалидное ранжирование HTTP")
        if np.intersect1d(ids, data.train_csr[uid].indices).size:
            raise RuntimeError("HTTP выдал seen items")
        timings.append(elapsed)
        if index < 2:
            responses.append(payload)
    total_seconds = time.perf_counter() - started
    unknown_status, unknown_body, _ = fetch(
        args.base_url, "/recommend?user_id=__unknown_smoke__&k=10"
    )
    unknown = json.loads(unknown_body)
    invalid_status, _, _ = fetch(args.base_url, "/recommend?user_id=x&k=0")
    if unknown_status != 200 or not unknown["fallback"] or invalid_status != 422:
        raise RuntimeError("fallback/validation contract нарушен")
    _, metrics, _ = fetch(args.base_url, "/metrics")
    (args.output / "http_metrics.prom").write_bytes(metrics)
    record = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "health": health,
        "requests": len(timings),
        "serial": True,
        "seconds": total_seconds,
        "qps": len(timings) / total_seconds,
        "latency_seconds": {
            "p50": float(np.median(timings)),
            "p95": float(np.quantile(timings, 0.95)),
            "max": float(np.max(timings)),
        },
        "sample_responses": responses,
        "unknown_fallback": unknown,
        "invalid_k_status": invalid_status,
        "all_rankings_valid_unseen": True,
    }
    (args.output / "http_smoke.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in record.items()
                if key not in {"sample_responses", "unknown_fallback"}
            }
        )
    )


if __name__ == "__main__":
    main()
