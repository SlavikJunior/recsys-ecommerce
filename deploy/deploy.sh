#!/usr/bin/env bash
# Деплой на локальный стенд: зависимости, артефакты, перезапуск сервиса,
# smoke-тест. Один и тот же скрипт запускается руками и из CD-workflow.
#
# Почему tmux, а не launchd: на macOS 26 launchd отказывается запускать
# сервисы, чьи логи и рабочий каталог лежат в ~/Documents (exit 78 EX_CONFIG),
# а обход через /bin/sh нестабилен. Для учебного стенда выбран tmux — он даёт
# тот же контракт «перезапуск + health-check», но воспроизводится на этой
# машине. Plist для launchd приложен в deploy/launchd/ и используется на
# CI-раннере, где каталог не защищён.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SESSION=recsys-api

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
  brew services start redis >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do
    redis-cli ping >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

echo "== перезапуск сервиса =="
mkdir -p artifacts/logs
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" \
  ".venv/bin/uvicorn recsys.serving.api:app --host 127.0.0.1 --port 8000 --workers 1 2>&1 | tee -a artifacts/logs/api.out.log"

echo "== smoke-тест =="
for _ in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    curl -fsS http://127.0.0.1:8000/health
    echo
    user="$(.venv/bin/python - <<'PY'
import pandas as pd
print(pd.read_parquet("artifacts/serving/Video_Games/user_ids.parquet")["user_id"].iloc[0])
PY
)"
    curl -fsS "http://127.0.0.1:8000/recommend/${user}?k=3" | head -c 400
    echo
    echo "deploy: OK"
    exit 0
  fi
  sleep 1
done

echo "deploy: сервис не поднялся за 40 секунд" >&2
tmux capture-pane -pt "$SESSION" 2>/dev/null | tail -n 40 >&2 || true
exit 1
