# service — MVP-приложение для анализа отзывов

Эта папка лежит внутри `med_project/` и отвечает за продуктовый слой:

```text
med_project/
├── dataset/      # ноутбуки и данные
├── algs/         # эксперименты с моделями
├── docs/         # документация
└── service/      # backend/frontend приложения
```

Текущий MVP-срез:

```text
CSV с размеченными отзывами
    ↓
PostgreSQL
    ↓
FastAPI backend
    ↓
шаблонная аналитика с реальными числами
```

Qdrant и online ingestion пока оставлены как следующий этап. Frontend для проверки MVP-сценариев лежит в `frontend/`.

---

## 1. Запуск через Docker Compose

Из папки проекта:

```bash
cd ~/Documents/med_project/service
cp backend/.env.docker.example backend/.env
docker compose -f docker-compose.dev.yml up --build
```

Все Python/Node-зависимости ставятся внутри Docker-образов. Локально нужен только Docker.

Если нужен LLM parser, аналитический вывод и RAG-поиск через OpenAI embeddings, заполни в `backend/.env`:

```text
OPENAI_API_KEY="..."
OPENAI_MODEL="gpt-5"
OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
CHAT_PARSER_MODE="auto"
```

Важно: embedding-модель запроса должна совпадать с embedding-моделью, которой построена коллекция Qdrant. Если коллекция построена BGE-M3, ее нужно либо перестроить под OpenAI embeddings, либо подключить совместимый локальный encoder.

Backend будет здесь:

```text
http://localhost:8000
http://localhost:8000/docs
```

Frontend будет здесь:

```text
http://localhost:5173
```

Проверка backend:

```bash
curl http://localhost:8000/health
```

Проверка frontend production-сборки тоже выполняется через Docker:

```bash
docker compose -f docker-compose.dev.yml build frontend-build
```

PostgreSQL проброшен наружу на порт `5433`, потому что `5432` часто уже занят старым Postgres:

```text
localhost:5433 -> postgres:5432 внутри Docker
```

---

## 2. Создать demo CSV

В другом терминале, пока compose запущен:

```bash
cd ~/Documents/med_project/service
docker compose -f docker-compose.dev.yml exec api \
  python -m app.offline.make_demo_reviews_csv --output /tmp/demo_reviews.csv
```

---

## 3. Загрузить реальные экспорты из `data/db_exports`

Если у тебя уже лежат файлы:

```text
../data/db_exports/postgres_reviews_bge_m3_linearsvc_saved_model/reviews_for_postgres.parquet
../data/db_exports/qdrant_vectors_baai_bge_m3/qdrant_vectors.npy
../data/db_exports/qdrant_vectors_baai_bge_m3/qdrant_payload.parquet
```

то загрузи их в Docker-базы сервиса так:

```bash
docker compose -f docker-compose.dev.yml --profile tools run --rm db-loader
```

Если нужно пересоздать Qdrant collection и очистить PostgreSQL перед загрузкой:

```bash
docker compose -f docker-compose.dev.yml --profile tools run --rm db-loader \
  python -m app.offline.load_db_exports \
  --target all \
  --truncate-postgres \
  --recreate-qdrant
```

После загрузки проверь:

```bash
curl http://localhost:8000/api/v1/debug/db-stats
curl http://localhost:6333/collections
```

По умолчанию сервис использует:

```text
QDRANT_COLLECTION=wb_reviews_bge_m3
EMBEDDING_PROVIDER=bge_m3
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

Это важно: экспорт Qdrant построен BGE-M3-векторами размерности 1024, поэтому online RAG тоже должен строить query embedding через BGE-M3.

### Вынести BGE-M3 embeddings в RunPod

Backend умеет ходить в RunPod Serverless endpoint вместо локальной загрузки BGE-M3. Ключ не хранится в коде и `.env`: передавай его только через переменную окружения хоста.

```bash
export RUNPOD_API_KEY="..."
export RUNPOD_ENDPOINT_ID="..."
export EMBEDDING_PROVIDER="runpod"
docker compose -f docker-compose.dev.yml up -d api
```

RunPod worker должен принимать payload:

```json
{"input": {"text": "текст запроса", "model": "BAAI/bge-m3"}}
```

И возвращать один из форматов:

```json
{"output": {"embedding": [0.1, 0.2]}}
```

Также поддерживаются `output.embeddings[0]`, `output.dense_vecs[0]`, `output.vector` или сразу список float.

---

## 4. Загрузить demo CSV в PostgreSQL

```bash
docker compose -f docker-compose.dev.yml exec api \
  python -m app.offline.import_reviews_to_postgres \
  --input /tmp/demo_reviews.csv \
  --dsn postgresql://reviews_user:reviews_password@postgres:5432/reviews_db \
  --init-schema
```

Проверить, что данные появились:

```bash
curl http://localhost:8000/api/v1/debug/db-stats
```

Ожидаемо: `reviews_count` должен быть больше нуля.

---

## 5. Проверить аналитику

Количество отзывов с проблемой `Проблема с комплектацией / упаковкой` в категории `Книги`:

```bash
curl -X POST "http://localhost:8000/api/v1/templates/count_by_problem/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "labels": ["Проблема с комплектацией / упаковкой"],
      "category": "Книги"
    },
    "add_analytical_summary": false,
    "limit": 20
  }'
```

Топ проблем:

```bash
curl -X POST "http://localhost:8000/api/v1/templates/top_problems/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "category": "Книги"
    },
    "add_analytical_summary": false,
    "limit": 20
  }'
```

---

## 6. Загрузить свой размеченный CSV

Пример для файла с колонками `отзыв` и `labels`:

```bash
cd ~/Documents/med_project/service

docker compose -f docker-compose.dev.yml exec api \
  python -m app.offline.import_reviews_to_postgres \
  --input /app/data/chatgpt_labeled_reviews_mvp_combined.csv \
  --dsn postgresql://reviews_user:reviews_password@postgres:5432/reviews_db \
  --init-schema \
  --default-date 2025-01-01 \
  --default-category "Тестовая категория" \
  --model-name "gpt5_imported_labels"
```

Файл должен быть доступен внутри контейнера `api`. Для разового импорта проще положить CSV в папку `service/backend/data/` или подключить нужную папку отдельным volume в `docker-compose.dev.yml`.

Если колонки называются иначе:

```bash
docker compose -f docker-compose.dev.yml exec api \
  python -m app.offline.import_reviews_to_postgres \
  --input /app/data/reviews.csv \
  --dsn postgresql://reviews_user:reviews_password@postgres:5432/reviews_db \
  --text-column "отзыв" \
  --labels-column "labels" \
  --default-date 2025-01-01
```

---

## Что сейчас работает

```text
GET  /health
GET  /api/v1/templates
GET  /api/v1/facets
GET  /api/v1/debug/db-stats
POST /api/v1/templates/{template_id}/execute
POST /api/v1/query/execute
POST /api/v1/chat/ask
```

Сейчас реально рабочие PostgreSQL-сценарии:

```text
count_by_problem
top_problems
problem_dynamics
top_products_by_problem
review_samples
period_comparison
problem_share
problem_growth
label_cooccurrence
keyword_search
positive_vs_problem
```

RAG-сценарии через Qdrant работают при наличии `OPENAI_API_KEY` и совместимой Qdrant collection. Если ключ не задан или размерность embedding не совпадает с коллекцией, backend вернет warning.
