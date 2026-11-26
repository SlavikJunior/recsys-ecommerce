#!/usr/bin/env bash
# Локальный стенд наблюдаемости.
#
# Запуск через tmux: на этой машине launchd отказывается держать часть
# homebrew-бинарников (EX_CONFIG), а tmux-сессия переживает закрытие терминала
# и не мешает демонстрации. В отчёте это ограничение указано явно.
# API запускается через launchd (deploy/deploy.sh), потому что это и есть
# проверяемый CD-шаг.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

LOGS="$PROJECT_DIR/artifacts/logs"
DATA="$PROJECT_DIR/deploy/data"
SESSION=recsys-stack
mkdir -p "$LOGS" "$DATA/prometheus" "$DATA/grafana"

start() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "сессия $SESSION уже запущена"
    status
    return
  fi
  tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" \
    "node_exporter=1; /opt/homebrew/opt/node_exporter/bin/node_exporter --web.listen-address=127.0.0.1:9100 2>&1 | tee $LOGS/node_exporter.log"
  tmux new-window -t "$SESSION" -c "$PROJECT_DIR" \
    "/opt/homebrew/opt/prometheus/bin/prometheus --config.file=$PROJECT_DIR/deploy/prometheus.yml --storage.tsdb.path=$DATA/prometheus --web.listen-address=127.0.0.1:9090 2>&1 | tee $LOGS/prometheus.log"
  tmux new-window -t "$SESSION" -c "$PROJECT_DIR" \
    "$PROJECT_DIR/deploy/bin/alertmanager --config.file=$PROJECT_DIR/deploy/alertmanager.yml --storage.path=$DATA/alertmanager --web.listen-address=127.0.0.1:9093 2>&1 | tee $LOGS/alertmanager.log"
  tmux new-window -t "$SESSION" -c "$PROJECT_DIR" \
    "PYTHONPATH=$PROJECT_DIR/src $PROJECT_DIR/.venv/bin/python scripts/alert_sink.py 2>&1 | tee $LOGS/alert_sink.log"
  tmux new-window -t "$SESSION" -c "$PROJECT_DIR" \
    "GF_PATHS_PROVISIONING=$PROJECT_DIR/deploy/grafana/provisioning GF_PATHS_DATA=$DATA/grafana GF_SERVER_HTTP_ADDR=127.0.0.1 GF_AUTH_ANONYMOUS_ENABLED=true GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer /opt/homebrew/opt/grafana/bin/grafana server --homepath /opt/homebrew/opt/grafana/share/grafana 2>&1 | tee $LOGS/grafana.log"
  sleep 12
  status
}

stop() {
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  launchctl bootout "gui/$UID/com.recsys.api" 2>/dev/null || true
  echo "стенд остановлен"
}

status() {
  for name_port in "prometheus:9090/-/ready" "grafana:3000/api/health" "alertmanager:9093/-/ready" "api:8000/health"; do
    name="${name_port%%:*}"
    path="${name_port#*:}"
    code=$(curl -s -o /dev/null -m 3 -w "%{http_code}" "http://127.0.0.1:$path" || true)
    echo "$name: $code"
  done
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|status}" >&2; exit 1 ;;
esac
