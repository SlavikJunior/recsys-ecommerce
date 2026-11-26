"""Скриншоты произвольных страниц (CI, API, Prometheus UI) через Playwright.

Нужен, чтобы в отчёте были воспроизводимые доказательства: скриншот делается
из того же URL, который указан в подписи, а не «сфотографирован» вручную.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from recsys.config import ARTIFACTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="+", help="name=url")
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "screenshots")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument("--settle-ms", type=int, default=5000)
    parser.add_argument("--full-page", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "screenshots.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        for pair in args.pairs:
            name, url = pair.split("=", 1)
            page.goto(url, wait_until="networkidle", timeout=90_000)
            page.wait_for_timeout(args.settle_ms)
            path = args.out / f"{name}.png"
            page.screenshot(path=str(path), full_page=args.full_page)
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
