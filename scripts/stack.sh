#!/usr/bin/env bash
# Локальный стенд наблюдаемости: Prometheus, Alertmanager, Grafana, node_exporter,
# приёмник алертов. Всё конфигурируется файлами из deploy/, без ручных кликов.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

LOGS="$PROJECT_DIR/artifacts/logs"
DATA="$PROJECT_DIR/deploy/data"
mkdir -p "$LOGS" "$DATA/prometheus" "$DATA/grafana"

start() {
  if curl -fsS http://127.0.0.1:9090/-/ready >/dev/null 2>&1; then
    echo "Prometheus уже запущен"
    return
  fi
  /opt/homebrew/opt/node_exporter/bin/node_exporter --web.listen-address=127.0.0.1:9100 \
    >"$LOGS/node_exporter.log" 2>&1 &
  /opt/homebrew/opt/prometheus/bin/prometheus \
    --config.file=deploy/prometheus.yml \
    --storage.tsdb.path="$DATA/prometheus" \
    --web.listen-address=127.0.0.1:9090 >"$LOGS/prometheus.log" 2>&1 &
  ./deploy/bin/alertmanager --config.file=deploy/alertmanager.yml \
    --storage.path="$DATA/alertmanager" \
    --web.listen-address=127.0.0.1:9093 >"$LOGS/alertmanager.log" 2>&1 &
  uv run python scripts/alert_sink.py >"$LOGS/alert_sink.log" 2>&1 &
  GF_PATHS_PROVISIONING="$PROJECT_DIR/deploy/grafana/provisioning" \
    GF_PATHS_DATA="$DATA/grafana" \
    GF_SERVER_HTTP_ADDR=127.0.0.1 \
    GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    /opt/homebrew/opt/grafana/bin/grafana server \
    --homepath /opt/homebrew/opt/grafana/share/grafana >"$LOGS/grafana.log" 2>&1 &
  sleep 8
  status
}

stop() {
  for port in 3000 9090 9093 9099 9100; do
    lsof -ti "tcp:$port" | xargs -r kill 2>/dev/null || true
  done
  echo "стенд остановлен"
}

status() {
  for name_port in "prometheus:9090/-/ready" "grafana:3000/api/health" "alertmanager:9093/-/ready"; do
    name="${name_port%%:*}"
    path="${name_port#*:}"
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$path" || true)
    echo "$name: $code"
  done
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|status}" >&2; exit 1 ;;
esac
