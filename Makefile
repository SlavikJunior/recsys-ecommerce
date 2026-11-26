CATEGORY ?= Video_Games

.PHONY: sync data prepare eda eval test lint serve monitor

sync:
	uv sync --python 3.12

data:
	uv run python -m recsys.data.download --category $(CATEGORY)

prepare:
	uv run python -m recsys.data.prepare --category $(CATEGORY)

eda:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

eval:
	uv run python -m recsys.eval.evaluate --category $(CATEGORY)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

serve:
	uv run uvicorn recsys.serving.api:app --host 127.0.0.1 --port 8000
