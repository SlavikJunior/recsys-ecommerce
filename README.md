# recsys-ecommerce

Рекомендательная система для e-commerce на датасете
[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
(категория `Video_Games`): классические подходы, гибрид, two-tower retrieval,
онлайн-сервис, мониторинг и CI/CD.

## Что внутри

| Слой | Состав |
|---|---|
| Данные | 5-core фильтрация, leave-last-out по времени, метаданные товаров |
| База | popularity (в т.ч. time-decayed), content-based (TF-IDF), item-CF, user-CF, hybrid (RRF) |
| Современная модель | Two-Tower retrieval (PyTorch, MPS) + FAISS |
| Оценка | Recall@k, NDCG@k, MRR/MAP@k, coverage, novelty, diversity, Gini, сегменты |
| Сервис | FastAPI, Redis-кэш, fallback на популярное |
| Наблюдаемость | Prometheus + Grafana: latency, QPS, кэш, CTR-прокси, дрейф |
| CI/CD | GitHub Actions: lint → mypy → pytest → build → деплой на self-hosted runner |

## Быстрый старт

```bash
make sync
make data            # ~3 ГБ с Hugging Face
make prepare         # 5-core + сплиты -> data/processed/Video_Games
make eval            # таблица метрик -> artifacts/eval/
make serve           # http://127.0.0.1:8000/docs
```

## Протокол оценки

Один таргет на пользователя (последний товар по времени), ранжирование по
полному каталогу, просмотренное исключается, популярные модели оцениваются на
`valid` только для выбора гиперпараметров. Тест не используется при обучении и
подборе. Это стандартный протокол Amazon Reviews 2023 и он не даёт модели
подсмотреть будущее.

Почему не `All_Beauty`: после 5-core фильтрации категория схлопывается до
357 пользователей и 479 товаров (проверено в `notebooks/01_eda.ipynb`),
поэтому статистически значимые выводы на ней невозможны.

## Структура

```
src/recsys/
  data/        загрузка, подготовка, матрицы взаимодействий
  models/      popularity, heuristics, content, item_cf, user_cf, hybrid, two_tower
  eval/        метрики, сегментный анализ, запуск сравнения
  serving/     FastAPI-приложение
  monitoring/  Prometheus-метрики, дрейф
notebooks/     EDA, обучение и анализ
tests/         юнит- и интеграционные тесты
docs/          диаграммы и отчёт
```

## Результаты

Таблица метрик генерируется в `artifacts/eval/` и вставляется в отчёт.
Полный отчёт — `docs/report.pdf`.
