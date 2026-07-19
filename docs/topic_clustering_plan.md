# Topic Clustering Plan

Цель: добавить слой `label -> topic_cluster_id -> topic_cluster_name`, не ломая
текущие `review_labels`.

## Входные данные

- `reviews.review_id`, `reviews.text`, `reviews.product_id`, `reviews.rating`
- `review_primary_labels.review_id`, `review_primary_labels.primary_label`
- BGE-M3 vectors из `data/db_exports/qdrant_vectors_baai_bge_m3/qdrant_vectors.npy`
- payload/order из `data/db_exports/qdrant_vectors_baai_bge_m3/qdrant_payload.csv`

## Первый безопасный pipeline

1. Для каждого problem label взять отзывы из `review_primary_labels`.
2. Внутри label кластеризовать embeddings:
   - старт: `MiniBatchKMeans`, потому что он есть в стандартном sklearn-подходе и
     предсказуем по памяти;
   - следующий шаг: HDBSCAN, если будет установлен отдельно и нужен variable cluster count.
3. Для каждого кластера сохранить:
   - `review_id`
   - `primary_label`
   - `topic_cluster_id`
   - `topic_cluster_name`
   - `cluster_score`
4. Названия кластеров сначала назначать rule-based по частотным словам:
   - качество: `сломано/разбито`, `материал`, `запах`, `швы/дыры`, `не работает`, `грязное`;
   - размер: `маломерит`, `большемерит`, `тесно`, `длина/ширина`;
   - упаковка: `помятая коробка`, `не хватает`, `вскрытая упаковка`.
5. Сохранять результат в отдельный CSV/table, не изменяя `review_labels`.

## Артефакты

Предлагаемый файл:

```text
data/db_exports/review_topic_clusters.csv
```

Предлагаемая таблица:

```sql
CREATE TABLE review_topic_clusters (
    review_id TEXT PRIMARY KEY REFERENCES reviews(review_id) ON DELETE CASCADE,
    primary_label TEXT NOT NULL,
    topic_cluster_id TEXT NOT NULL,
    topic_cluster_name TEXT,
    cluster_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Проверка качества

- Для каждого кластера смотреть 20 ближайших отзывов к центроиду.
- Не показывать кластеры с очень малым размером как устойчивую причину.
- В UI/ответах писать “похоже на тему …”, если cluster name назначен rule-based.
