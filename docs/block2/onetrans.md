# OneTrans: воспроизводимый второй блок

Реализован промышленный ranker OneTrans (ByteDance/NTU, первая публикация
30.10.2025), адаптированный к next-item задаче Amazon Video_Games. Это собственная
уменьшенная реализация по [первичной статье, v3](https://arxiv.org/html/2510.26104v3),
а не production веса ByteDance. Факт промышленного применения описан авторами
в §4.6 и Table 2. Чужой GMV uplift не является результатом этого стенда.

## Архитектура и fidelity

История S и non-sequential NS токены проходят единый causal stack. NS получаются
Auto-Split проекцией, параметры QKV/FFN общие у S и отдельные у каждого NS.
Блоки имеют RMSNorm pre-norm и residual; pyramid сокращает queries/outputs,
сохраняя все K/V входа соответствующего слоя. Отдельные S/NS проходы вычисляют
тот же результат, что общий reference graph. [OneTrans §3](https://arxiv.org/html/2510.26104v3#S3).

| Fidelity | Код | Проверка |
|---|---|---|
| F01 unified S/NS | `OneTrans.uncached`, `encode_prefix`, `score_cached` | `test_cache_matches_combined_graph_with_padding_and_empty` |
| F02 Auto-Split | `OneTrans.nonsequential_tokens` | combined/reference equivalence, gradients |
| F03 S projection/time merge | `OneTrans.sequential_tokens`, `OneTransFeatures.from_data`, `chronological_merge` | `test_chronological_multistream_merge_preserves_real_type`, train-prefix test |
| F04 RMSNorm/residual | `OneTransBlock` | reference equivalence, finite loss/gradients |
| F05 mixed causal QKV | `MixedProjection`, `OneTransBlock.forward` | `test_causal_future_and_ns_do_not_change_prefix`, specific params test |
| F06 mixed FFN | `MixedFFN` | specific params/gradients, loss reduction |
| F07 pyramid full K/V | `OneTransBlock.prefix`, `forward` | `test_pyramid_queries_see_full_keys_before_slicing` |
| F08 candidate head | `OneTrans.score_cached`, `uncached` | candidate/chunk equivalence |
| F09 candidate cache | `PrefixCache`, `OneTransRecommender.prefix_for` | `test_checkpoint_mapping_and_chunk_invariance` |
| F10 cross-request reuse | `prefix_for`, first-layer reuse в `prefix` | `test_append_reuses_first_layer_kv_and_recomputes_deep` |
| F11 objective/protocol | `recsys.eval.block2` | negative sampler, frozen selection, leakage and fingerprint tests |
| F12 bounded runtime | `recommend`, candidate chunks | CPU/MPS benchmark, full-catalog final evaluation |

Набор данных содержит один реальный review поток. Общая функция merge сохраняет
типы нескольких потоков, но основной модельный прогон обучает единственный
S projection и один behavior type; cart/click события не фабрикуются. Метаданные
brand/category поддержаны как опция, в основном эксперименте выключены.
Snapshot `average_rating`/`rating_number` не используются из-за риска будущих
агрегатов. Item identity vocabulary обучается только на train; holdout-only item
получает unknown embedding. User embedding относится к известным train users.

Cross-request cache не обещает полный инкрементальный deep stack при движущейся
pyramid. При append до лимита переиспользуются K/V первого слоя; глубокие слои
пересчитываются. При смене окна/порядка история инвалидирует cache. Это явно
отражено счётчиками `append_first_layer_kv_reused`, `deep_layers_recomputed`,
`invalidations`, `exact_hits`, `misses`. Повтор одного префикса переиспользует
все его layer caches. Переобучение/load создаёт новый cache. CUDA FlashAttention
не заявляется: реализация использует математически эквивалентный PyTorch SDPA.

## Протокол и ограничения

Train: 628 416 событий, 533 148 next-item prefix примеров за эпоху. Все train
примеры используются три эпохи с seed 42. Отрицательные альтернативы выбираются
равномерно из train-observed каталога с исключением target и представленной
истории; это sampled-softmax, не observed negative clicks. История обрезается
до 32 событий. Возможны false negatives среди будущих/старых взаимодействий;
отсутствие impression logs не позволяет оценить настоящий CTR.

Valid: заранее выбранные 256 users; checkpoint выбирается по valid NDCG@10.
Test: заранее замороженная uniform выборка 1000 users с selection seed 20261126;
все модели оцениваются на тех же users/targets по полному каталогу 26 322 items.
Полнокаталожная оценка не означает всех 95 268 test users. Отдельные segments и
paired bootstrap показывают неопределённость этой выборки. Valid/test используют
только train историю, исключаются train-seen items. Результат не является
индуктивной оценкой новых items: mapping строится по union split.

`stats.json` содержит 98 906 users и 26 354 items после five-core до окончательного
user фильтра; `make_splits` удаляет пользователей с недостаточной train историей.
`load_dataset` строит mapping по фактически оставшимся train+valid+test, поэтому
получается 95 268 users и 26 322 items. Использованы существующие parquet без
перегенерации; SHA каждой части сохранён в manifest. У всех baseline те же данные.

Width 32, четыре головы, три NS токена, pyramid `[8, 0]`, FFN expansion 2,
3 963 041 параметр. Собственный head/softmax loss, размер и single behavior input
обозначены адаптацией; online CTR/CVR, private features и distributed serving
ByteDance не воспроизводятся. Нельзя выдавать этот эксперимент за повтор их A/B.

## Команды

Из корня `recsys-ecommerce`; существующая среда `uv sync` содержит зависимости.
Для worktree явно задать `PYTHONPATH=src` и `RECSYS_DATA_DIR` на общий каталог data.

```sh
uv run python -m recsys.eval.block2 benchmark --config configs/block2.toml --output artifacts/block2/run-20261126 --device cpu
uv run python -m recsys.eval.block2 train --config configs/block2.toml --output artifacts/block2/run-20261126 --device cpu
uv run python -m recsys.eval.block2 evaluate --config configs/block2.toml --output artifacts/block2/run-20261126 --device cpu
uv run python -m recsys.serving.onetrans_api --checkpoint artifacts/block2/run-20261126/model.pt --port 8002
```

Существующий checkpoint и финальные test результаты не перезаписываются: для
нового эксперимента нужен новый `--output`. Нельзя изменять config/inputs/cohort
внутри существующего run. Source SHA каждого этапа сохраняется в отдельном
`provenance_<stage>_<device>.json`; исходный training snapshot сохранён локально
в `training-source/`. Последующие изменения только typing/документации не заменяют
это происхождение фактического обучения.

HTTP: `GET /health` показывает SHA реального checkpoint и dataset fingerprint;
`GET /recommend?user_id=...&k=10` отдаёт ранжирование; неизвестный user получает
явно помеченный popularity fallback. `GET /metrics` — отдельные Prometheus
requests/latency/cache counters. Старый API и registry не изменены.

Артефакты: `manifest.json`, frozen `selection.npz`, CPU/MPS benchmarks,
`training.json`, `checkpoint.json`, `model.pt`, `test_*.json`, `predictions_*.npz`,
`paired_ndcg10.json`. `model.pt` — локальный бинарный checkpoint, намеренно не Git
asset; отчёт ссылается на его SHA. Rollback — остановить отдельный процесс 8002;
предыдущий сервис и исходные данные не затронуты.
