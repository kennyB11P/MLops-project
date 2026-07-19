# Данные сервиса

## Runtime source of truth

Рабочий сервис запускается через:

```bash
docker compose -f service/docker-compose.dev.yml up
```

Актуальное runtime-состояние хранится в Docker named volumes этого compose-файла:

- `reviews_postgres_data` — PostgreSQL с таблицами `reviews` и `review_labels`;
- `reviews_qdrant_data` — Qdrant collection `wb_reviews_bge_m3`.

Папки `storage/postgres_data` и `storage/qdrant_storage` относились к старому корневому
`docker-compose.yml` и не являются источником правды для текущего сервиса.

## Rebuild/source exports

Актуальные файловые экспорты для восстановления сервисных БД:

- `data/db_exports/postgres_reviews_bge_m3_linearsvc_saved_model/reviews_for_postgres.csv`
- `data/db_exports/qdrant_vectors_baai_bge_m3/qdrant_payload.csv`
- `data/db_exports/qdrant_vectors_baai_bge_m3/qdrant_vectors.npy`
- `data/db_exports/qdrant_vectors_baai_bge_m3/vector_parts/`
- `data/db_exports/review_nm_mapping.csv`
- `data/db_exports/wb_products_slice.parquet`
- `data/db_exports/postgres_reviews_bge_m3_linearsvc_saved_model/postgres_export_manifest.json`
- `data/db_exports/qdrant_vectors_baai_bge_m3/qdrant_manifest.json`

CSV-файлы содержат актуальные `review_date`; Qdrant payload также содержит `text`,
`labels` и `predicted_labels`.

`data/db_exports/review_nm_mapping.csv` восстанавливает связь текущих `review_id`
с Wildberries `nm_id`. Этот mapping покрывает весь текущий service export и
используется для заполнения:

- `product_id` / `nm_id` — артикул товара WB;
- `rating` — оценка из исходного feedbacks-среза.

Текущий `wb_products_slice.parquet` построен streaming-проходом по
`nyuuzyou/wb-products` без загрузки всего датасета в память. Покрытие текущих
ненулевых `nm_id`: `104990 / 113638` (`92.39%`). Для ненайденных товаров
сервис сохраняет `product_id` и `rating`, а `product_name`/`category`/`brand`
остаются пустыми.

Slice можно перестроить так:

```bash
docker compose -f service/docker-compose.dev.yml run --rm \
  --volume /ABS/PATH/service/backend:/app \
  --volume /ABS/PATH/data/db_exports:/workdata \
  --env PYTHONUNBUFFERED=1 \
  db-loader sh -lc 'python -m pip install -q zstandard==0.23.0 && \
    python -m app.offline.build_wb_products_slice \
      --mapping /workdata/review_nm_mapping.csv \
      --output /workdata/wb_products_slice.parquet'
```

Затем товарные поля можно подмешать через:

```bash
docker compose -f service/docker-compose.dev.yml run --rm \
  --volume /ABS/PATH/service/backend:/app \
  --volume /ABS/PATH/data/db_exports:/workdata \
  --env POSTGRES_EXPORT_DIR=/workdata/postgres_reviews_bge_m3_linearsvc_saved_model \
  --env QDRANT_EXPORT_DIR=/workdata/qdrant_vectors_baai_bge_m3 \
  --env REVIEW_NM_MAPPING_PATH=/workdata/review_nm_mapping.csv \
  db-loader python -m app.offline.enrich_review_products \
  --products-path /workdata/path/to/wb_products_slice.parquet
```

## Deprecated data

Устаревшие parquet-файлы без `review_date` и старые storage-папки заархивированы в:

```text
archive/deprecated_data_2026-06-24/
```

В архив перенесены:

- `storage/postgres_data`
- `storage/qdrant_storage`
- `data/db_exports/postgres_reviews_bge_m3_linearsvc_saved_model/reviews_for_postgres.parquet`
- `data/db_exports/postgres_reviews_bge_m3_linearsvc_saved_model/postgres_parts/`
- `data/db_exports/qdrant_vectors_baai_bge_m3/qdrant_payload.parquet`

Не использовать deprecated parquet для загрузки текущего сервиса: они старее CSV и не
содержат актуальные даты отзывов.
