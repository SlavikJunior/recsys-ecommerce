#!/usr/bin/env bash
# Деплой на локальный стенд: зависимости, артефакты, перезапуск сервиса,
# smoke-тест. Один и тот же скрипт запускается руками и из CD-workflow.
#
# Каталоги данных и артефактов отделены от кода: на CI-раннере деплоится
# свежий git-чекаут, а модель берётся из постоянного хранилища (RECSYS_*_DIR).
#
# Почему tmux, а не launchd: на macOS 26 launchd не запускает сервисы, чей
# рабочий каталог и логи лежат в ~/Documents (exit 78 EX_CONFIG). Tmux даёт
# тот же контракт «перезапуск + health-check». Plist для launchd приложен в
# deploy/launchd/ и работает на машинах без этого ограничения.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

DATA_DIR="${RECSYS_DATA_DIR:-$PROJECT_DIR/data}"
ARTIFACTS_DIR="${RECSYS_ARTIFACTS_DIR:-$PROJECT_DIR/artifacts}"
SESSION=recsys-api
CATEGORY="${RECSYS_CATEGORY:-Video_Games}"

export RECSYS_DATA_DIR="$DATA_DIR"
export RECSYS_ARTIFACTS_DIR="$ARTIFACTS_DIR"

echo "== зависимости =="
uv sync --frozen --no-dev

echo "== артефакты модели =="
if [ ! -f "$ARTIFACTS_DIR/serving/$CATEGORY/manifest.json" ]; then
  echo "артефактов нет — собираю из данных в $DATA_DIR"
  uv run python -m recsys.serving.build --category "$CATEGORY"
else
  echo "артефакты найдены: $ARTIFACTS_DIR/serving/$CATEGORY"
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
mkdir -p "$ARTIFACTS_DIR/logs"
tmux kill-session -t "$SESSION" 2>/dev/null || true
# Переменные передаём явно: tmux-сервер мог быть запущен раньше с другим окружением.
tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" \
  "RECSYS_DATA_DIR='$DATA_DIR' RECSYS_ARTIFACTS_DIR='$ARTIFACTS_DIR' RECSYS_CATEGORY='$CATEGORY' \
   .venv/bin/uvicorn recsys.serving.api:app --host 127.0.0.1 --port 8000 --workers 1 \
   2>&1 | tee -a '$ARTIFACTS_DIR/logs/api.out.log'"

echo "== smoke-тест =="
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    curl -fsS http://127.0.0.1:8000/health
    echo
    user="$(.venv/bin/python - "$ARTIFACTS_DIR" "$CATEGORY" <<'PY'
import sys

import pandas as pd

artifacts, category = sys.argv[1], sys.argv[2]
print(pd.read_parquet(f"{artifacts}/serving/{category}/user_ids.parquet")["user_id"].iloc[0])
PY
)"
    curl -fsS "http://127.0.0.1:8000/recommend/${user}?k=3" | head -c 400
    echo
    echo "deploy: OK"
    exit 0
  fi
  sleep 1
done

echo "deploy: сервис не поднялся за 60 секунд" >&2
tmux capture-pane -pt "$SESSION" 2>/dev/null | tail -n 40 >&2 || true
exit 1
