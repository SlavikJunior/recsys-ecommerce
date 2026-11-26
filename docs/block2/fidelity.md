# OneTrans: фактически реализованные блоки и доказательства

Дата: 2026-11-26. Это evidence после Execute, не план. Метод: OneTrans,
arXiv:2510.26104v3 (первая публикация 2025-10-30), промышленное применение
ByteDance описано в [первичной статье §4.6](https://arxiv.org/html/2510.26104v3#S4.SS6).
Результаты ниже — собственная уменьшенная next-item адаптация на Amazon, а не
повтор production A/B авторов. Подробный протокол: [onetrans.md](onetrans.md).

| ID | Реальный блок / точка кода | Исполнимое доказательство |
|---|---|---|
| F01 | Общий S/NS causal graph: `src/recsys/models/onetrans.py:277`, эквивалентный split: `:243`, `:258` | `tests/test_onetrans.py:36`: combined vs cached, padding/empty |
| F02 | NS Auto-Split из user/candidate/context: `onetrans.py:234` | `tests/test_onetrans.py:36`, `:70`: эквивалентность и gradient |
| F03 | S field projection: `onetrans.py:228`; chronological train ordering: `onetrans_features.py:27`; merge helper: `:74` | `tests/test_onetrans.py:102`, `:119` |
| F04 | RMSNorm pre-norm/residual: `onetrans.py:103`, `:168` | `tests/test_onetrans.py:36`, `:86`: finite values, реальное снижение loss |
| F05 | QKV shared для S, отдельные NS: `onetrans.py:66`; causal mask `:168` | `tests/test_onetrans.py:59`, `:70`: future isolation, независимые параметры |
| F06 | Mixed FFN: `onetrans.py:81` | `tests/test_onetrans.py:36`, `:70`, `:86` |
| F07 | Pyramid Q tail / полный K,V входа слоя: `onetrans.py:129`, `:168` | `tests/test_onetrans.py:48`: exact full-attention затем slice |
| F08 | Настоящий candidate score head: `onetrans.py:258`, `:277` | `tests/test_onetrans.py:127`; `test_onetrans_api.py:18` |
| F09 | Все layer caches общего S prefix: `onetrans.py:48`; LRU: `onetrans_recommender.py:103` | `tests/test_onetrans.py:127`: chunks 1/32 совпадают; HTTP counters |
| F10 | Между запросами: exact hits; append KV первого слоя; глубокие пересчитываются: `onetrans_recommender.py:103`, `onetrans.py:129` | `tests/test_onetrans.py:149`: append равен fresh; window shift invalidates; явные layer/recompute counters |
| F11 | Train-only features/prefix: `onetrans_features.py:27`, `:59`; train/eval CLI: `src/recsys/eval/block2.py` | leakage test `test_onetrans.py:102`; negative/bootstrap tests `tests/test_block2_evaluation.py`; fingerprint rejection `:127`; замороженный `selection.npz` |
| F12 | Bounded catalog batches/streaming top-k: `onetrans_recommender.py:153`; PyTorch SDPA | `benchmark_cpu.json`, `benchmark_mps.json`, `test_onetrans.json`, 100-request `http_smoke.json` |

Пути коротких имён в таблице относятся к `src/recsys/models/`. Артефакты относятся
к `artifacts/block2/run-20261126/`. Функции тестов и смысл проверок также приведены
в runbook; строки даны для текущего Execute snapshot.

Важные границы fidelity: один реальный review stream, один S behavior type;
helper для нескольких потоков проверен отдельно, но multi-behavior production
данные не фабриковались. История 32 события, width 32, heads 4, NS 3,
pyramid `[8,0]`, expansion 2; 3 963 041 параметр. Metadata выключены.
Custom next-item sampled-softmax head заменяет недоступные CTR/CVR labels;
никаких synthetic watch-duration/RL substitutes. FlashAttention CUDA kernels,
private features, distributed ByteDance serving и полный deep-layer incremental
cache не заявляются. Безопасная invalidation/пересчёт сохраняет точное scoring.

## Фактический эксперимент

Train: 628 416 событий, 533 148 prefix examples ×3 epochs, seed 42. Loss:
2.095163 →1.825208 →1.648127. Train CPU: 196.003 s суммарно, validation 36.537 s.
Checkpoint выбран по valid NDCG@10 на 256 заранее выбранных users: epoch 1,
valid NDCG@10=0.017762. SHA256:
`de64463ec829c44e96953379b402494b71a6216fd655446edea43485f8071dc7`.

Test: **1000 замороженных пользователей, полный каталог 26 322**, одинаковые
users/targets/split для всех моделей, train-seen исключены. Это не оценка всех
95 268 test users. Данные/код/config/cohort SHA записаны в manifest и provenance.

| Модель | Recall@10 | NDCG@10 | Coverage@20 | Scoring 1000 users, s |
|---|---:|---:|---:|---:|
| OneTrans | 0.024 | 0.011454 | 0.005281 | 47.402 |
| Popularity | 0.022 | 0.010517 | 0.000988 | 0.173 |
| Content | 0.014 | 0.007442 | 0.315630 | 0.631 |
| Item-CF | 0.051 | 0.028973 | 0.311564 | 0.072 |
| User-CF | 0.055 | 0.027156 | 0.169554 | 0.060 |
| Hybrid | 0.054 | 0.028388 | 0.204164 | 1.060 |

OneTrans проиграл hybrid: paired mean ΔNDCG@10=-0.016934,
95% bootstrap CI [-0.025100,-0.009093], 2000 paired resamples.
Отличие от popularity неубедительно: CI [-0.000962,0.002869] включает ноль.
У 521 tail target (train count<50) у OneTrans нет попаданий@10, у hybrid —12.
Результат показывает popularity concentration; positive production uplift не
заявляется, hyperparameters после просмотра test не менялись.

## HTTP и gates

Реальный HTTP deployment: `recsys.serving.onetrans_api`, localhost:8002.
100 последовательных запросов: 4.772855 s, 20.9518 req/s,
p50=46.857 ms, p95=52.462 ms, max=68.421 ms. Все 100 ранжирований имеют
20 уникальных допустимых unseen IDs и SHA ожидаемого checkpoint. Проверены
unknown-user popularity fallback и invalid k→422. Это serial нагрузка,
не предел concurrent throughput. Сырые ответы и метрики: `http_smoke.json`,
`http_metrics.prom`. API оставлен работающим для демонстрации; restart-команда
в runbook. Старые API/registry/deploy не изменялись.

Проверки Execute worktree: 28 pytest passed; ruff check pass;
ruff format --check:50 files; mypy:29 source files без ошибок;
`uv build` собрал wheel и sdist. Независимая Validate стадия ещё должна оценить
это доказательство относительно полного задания.
