"""Нагрузочный тест и генератор трафика для дашбордов.

Два режима:
- `bench` — измеряет латентность и RPS на /recommend;
- `traffic` — имитирует сессию: рекомендации, показы, клики, покупки,
  чтобы на Grafana появились CTR-прокси, покрытие и дрейф.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np

from recsys.config import ARTIFACTS_DIR


async def fetch_users(base_url: str, limit: int) -> list[str]:
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        response = await client.get("/health")
        response.raise_for_status()
        n_users = response.json()["n_users"]
    # ID пользователей детерминированно берем из артефактов, чтобы не
    # угадывать формат.
    import pandas as pd

    category = "Video_Games"
    path = ARTIFACTS_DIR / "serving" / category / "user_ids.parquet"
    users = pd.read_parquet(path)["user_id"].tolist()
    rng = random.Random(0)
    rng.shuffle(users)
    return users[: min(limit, len(users), n_users)]


async def bench(base_url: str, users: list[str], requests: int, concurrency: int, k: int) -> dict:
    latencies: list[float] = []
    errors = 0
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:

        async def one(user_id: str) -> None:
            nonlocal errors
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.get(f"/recommend/{user_id}", params={"k": k})
                    response.raise_for_status()
                except Exception:
                    errors += 1
                finally:
                    latencies.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        await asyncio.gather(*(one(users[i % len(users)]) for i in range(requests)))
        wall = time.perf_counter() - started

    arr = np.array(latencies)
    return {
        "requests": requests,
        "concurrency": concurrency,
        "errors": errors,
        "rps": round(requests / wall, 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(arr.max()), 2),
    }


async def traffic(base_url: str, users: list[str], seconds: int, rps: int, k: int) -> dict:
    clicks = impressions = purchases = 0
    rng = random.Random(42)
    delay = 1.0 / rps
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        started = time.perf_counter()
        while time.perf_counter() - started < seconds:
            user = rng.choice(users)
            response = await client.get(f"/recommend/{user}", params={"k": k})
            if response.status_code != 200:
                await asyncio.sleep(delay)
                continue
            items = response.json()["items"]
            impressions += len(items)
            # Модель клика: позиция и бренд влияют — так CTR-прокси не случаен.
            for position, item in enumerate(items):
                p_click = 0.25 / (1 + position) + (0.05 if item["brand"] else 0.0)
                if rng.random() < p_click:
                    await client.post(
                        "/feedback",
                        json={
                            "user_id": user,
                            "item_id": item["item_id"],
                            "event": "click",
                            "position": position,
                        },
                    )
                    clicks += 1
                    if rng.random() < 0.2:
                        await client.post(
                            "/feedback",
                            json={
                                "user_id": user,
                                "item_id": item["item_id"],
                                "event": "purchase",
                                "position": position,
                            },
                        )
                        purchases += 1
                    break
            await asyncio.sleep(delay)
    return {
        "seconds": seconds,
        "impressions": impressions,
        "clicks": clicks,
        "purchases": purchases,
        "ctr_proxy": round(clicks / max(impressions, 1), 4),
        "conversion_proxy": round(purchases / max(clicks, 1), 4),
    }


async def main_async(args: argparse.Namespace) -> None:
    users = await fetch_users(args.base_url, args.users)
    if args.mode == "bench":
        result = await bench(args.base_url, users, args.requests, args.concurrency, args.k)
    else:
        result = await traffic(args.base_url, users, args.seconds, args.rps, args.k)
    out = Path(args.output) if args.output else ARTIFACTS_DIR / "loadtest" / f"{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=["bench", "traffic"], default="bench")
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--rps", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default=None)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
