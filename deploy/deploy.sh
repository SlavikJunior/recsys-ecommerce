#!/usr/bin/env bash
# Деплой на локальный стенд: зависимости, артефакты, launchd, smoke-тест.
# Один и тот же скрипт запускается руками и из CD-workflow на self-hosted runner.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "== зависимости =="
uv sync --frozen --no-dev

echo "== артефакты модели =="
if [ ! -f "artifacts/serving/Video_Games/manifest.json" ]; then
  uv run python -m recsys.serving.build --category Video_Games
else
  echo "артефакты уже есть, пересборка пропущена"
fi

echo "== redis =="
if ! redis-cli ping >/dev/null 2>&1; then
  brew services start redis >/dev/null || true
  for _ in $(seq 1 20); do
    redis-cli ping >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

echo "== launchd =="
mkdir -p artifacts/logs
sed "s#__PROJECT__#$PROJECT_DIR#g" deploy/com.recsys.api.plist > "$HOME/Library/LaunchAgents/com.recsys.api.plist"
launchctl unload "$HOME/Library/LaunchAgents/com.recsys.api.plist" 2>/dev/null || true
launchctl load "$HOME/Library/LaunchAgents/com.recsys.api.plist"

echo "== smoke-тест =="
for _ in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    curl -fsS http://127.0.0.1:8000/health
    echo
    curl -fsS "http://127.0.0.1:8000/recommend/$(uv run python - <<'PY'
import pandas as pd
print(pd.read_parquet("artifacts/serving/Video_Games/user_ids.parquet")["user_id"].iloc[0])
PY
)?k=3" | head -c 400
    echo
    echo "deploy: OK"
    exit 0
  fi
  sleep 1
done

echo "deploy: сервис не поднялся за 40 секунд" >&2
tail -n 50 artifacts/logs/api.err.log >&2 || true
exit 1
