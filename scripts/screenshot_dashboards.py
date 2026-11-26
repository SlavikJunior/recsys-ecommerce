"""Скрины дашбордов через headless Chromium.

Ручные скриншоты «на телефон» — плохое доказательство: их нельзя
воспроизвести. Скрипт открывает те же URL, что и проверяющий, и сохраняет
PNG в artifacts/screenshots вместе с метаданными (время, URL).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from recsys.config import ARTIFACTS_DIR

TARGETS = [
    (
        "grafana-service",
        "http://127.0.0.1:3000/d/recsys-service/recsys-service?orgId=1&kiosk&from=now-30m&to=now&refresh=5s",
        16000,
        "domcontentloaded",
    ),
    (
        "grafana-infra",
        "http://127.0.0.1:3000/d/recsys-infra/recsys-infra?orgId=1&kiosk&from=now-30m&to=now&refresh=5s",
        12000,
        "domcontentloaded",
    ),
    ("prometheus-alerts", "http://127.0.0.1:9090/alerts", 3000, "domcontentloaded"),
    ("prometheus-targets", "http://127.0.0.1:9090/targets", 3000, "domcontentloaded"),
    ("alertmanager", "http://127.0.0.1:9093/#/alerts", 4000, "domcontentloaded"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "screenshots")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--only", default=None, help="один target по имени")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "screenshots.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        for name, url, settle_ms, wait_mode in TARGETS:
            if args.only and name != args.only:
                continue
            page.goto(url, wait_until=wait_mode, timeout=90_000)
            page.wait_for_timeout(settle_ms)
            path = args.out / f"{name}.png"
            page.screenshot(path=str(path), full_page=True)
            meta[name] = {
                "url": url,
                "file": str(path),
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            print(f"{name}: {path}")
        browser.close()

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
