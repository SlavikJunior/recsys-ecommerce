"""Приёмник вебхуков Alertmanager.

В учебном стенде роль Telegram/Slack заменяет локальный приёмник: он пишет
алерты в artifacts/alerts/alerts.jsonl. Это позволяет показать полный путь
«правило -> Alertmanager -> получатель» без внешних секретов и сети.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from recsys.config import ARTIFACTS_DIR

OUT = ARTIFACTS_DIR / "alerts" / "alerts.jsonl"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — имя задано базовым классом
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        for alert in payload.get("alerts", []):
            record = {
                "received_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "status": alert.get("status"),
                "alertname": alert.get("labels", {}).get("alertname"),
                "severity": alert.get("labels", {}).get("severity"),
                "summary": alert.get("annotations", {}).get("summary"),
                "value": alert.get("annotations", {}).get("description", ""),
            }
            with OUT.open("a") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9099)
    args = parser.parse_args()
    print(f"alert sink: http://127.0.0.1:{args.port}/alerts -> {OUT}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
