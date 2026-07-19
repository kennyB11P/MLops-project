"""Загрузка экспортов из data/db_exports в сервисные PostgreSQL и Qdrant.

Экспортные файлы были подготовлены под схему:
- reviews_for_postgres.parquet с review_text и predicted_labels;
- qdrant_vectors.npy + qdrant_payload.parquet с BGE-M3 векторами.

Этот loader адаптирует их к online-схеме сервиса:
- reviews + review_labels в PostgreSQL;
- Qdrant collection с payload, совместимым с QdrantTool.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from tqdm.auto import tqdm

from app.core.config import get_settings


POSTGRES_EXPORT_DIR = Path(
    os.getenv("POSTGRES_EXPORT_DIR", "/app/data/db_exports/postgres_reviews_bge_m3_linearsvc_saved_model")
)
QDRANT_EXPORT_DIR = Path(os.getenv("QDRANT_EXPORT_DIR", "/app/data/db_exports/qdrant_vectors_baai_bge_m3"))
REVIEW_NM_MAPPING_PATH = Path(os.getenv("REVIEW_NM_MAPPING_PATH", "/app/data/db_exports/review_nm_mapping.csv"))

POSTGRES_BATCH_SIZE = int(os.getenv("POSTGRES_LOAD_BATCH_SIZE", "5000"))
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_LOAD_BATCH_SIZE", "1000"))


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [clean_value(item) for item in value.tolist()]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _parse_labels_value(value: Any) -> list[str]:
    value = clean_value(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]

    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null", "[]"}:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass

    for separator in ("|", ";", ","):
        if separator in raw:
            return [item.strip() for item in raw.split(separator) if item.strip()]

    return [raw]


def parse_labels(value: Any, fallback: Any = None) -> list[str]:
    labels = _parse_labels_value(value)
    if labels:
        return labels
    return _parse_labels_value(fallback)


def first_non_empty(*values: Any) -> Any:
    for value in values:
        value = clean_value(value)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def to_int(value: Any) -> int | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def init_service_schema(conn: psycopg.Connection) -> None:
    schema_path = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"
    conn.execute(schema_path.read_text(encoding="utf-8"))
    conn.commit()


def read_export_dataframe(
    *,
    export_dir: Path,
    csv_name: str,
    parquet_name: str,
    preferred_csv_columns: set[str],
) -> pd.DataFrame:
    csv_path = export_dir / csv_name
    parquet_path = export_dir / parquet_name

    if csv_path.exists():
        csv_columns = set(pd.read_csv(csv_path, nrows=0).columns)
        if "review_date" in csv_columns and preferred_csv_columns.issubset(csv_columns):
            print(f"Using CSV export with review_date: {csv_path}")
            return pd.read_csv(csv_path, dtype={"review_id": "string"})

    if not parquet_path.exists():
        raise FileNotFoundError(f"Не найден {parquet_path}")

    print(f"Using parquet export: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if csv_path.exists():
        csv_columns = set(pd.read_csv(csv_path, nrows=0).columns)
        should_merge_dates = "review_date" in csv_columns and (
            "review_date" not in df.columns or df["review_date"].isna().all()
        )
        if should_merge_dates:
            print(f"Merging review_date from CSV: {csv_path}")
            date_df = pd.read_csv(csv_path, usecols=["review_id", "review_date"], dtype={"review_id": "string"})
            df["review_id"] = df["review_id"].astype("string")
            df = df.drop(columns=["review_date"], errors="ignore").merge(date_df, on="review_id", how="left")

    return df


def merge_review_nm_mapping(df: pd.DataFrame) -> pd.DataFrame:
    if not REVIEW_NM_MAPPING_PATH.exists():
        return df
    if "review_id" not in df.columns:
        return df

    mapping_columns = pd.read_csv(REVIEW_NM_MAPPING_PATH, nrows=0).columns
    usecols = ["review_id", "nm_id"]
    if "rating" in mapping_columns:
        usecols.append("rating")

    mapping = pd.read_csv(REVIEW_NM_MAPPING_PATH, usecols=usecols, dtype={"review_id": "string", "nm_id": "string"})
    missing_columns = {"review_id", "nm_id"} - set(mapping.columns)
    if missing_columns:
        raise ValueError(f"В review_nm_mapping не хватает колонок: {missing_columns}")
    if mapping["review_id"].isna().any() or mapping["nm_id"].isna().any():
        raise ValueError("review_nm_mapping содержит пустой review_id или nm_id")
    if mapping["review_id"].duplicated().any():
        duplicate_count = int(mapping["review_id"].duplicated(keep=False).sum())
        raise ValueError(f"review_nm_mapping содержит дубли review_id: rows={duplicate_count}")

    result = df.copy()
    result["review_id"] = result["review_id"].astype("string")
    mapping = mapping.rename(columns={"nm_id": "_mapped_nm_id", "rating": "_mapped_rating"})
    result = result.merge(mapping, on="review_id", how="left", validate="one_to_one")

    missing_ids = int(result["_mapped_nm_id"].isna().sum())
    if missing_ids:
        raise ValueError(f"review_nm_mapping не покрывает все отзывы export: missing={missing_ids}")

    result["nm_id"] = result["_mapped_nm_id"]
    result["product_id"] = result["_mapped_nm_id"]
    if "_mapped_rating" in result.columns:
        if "rating" in result.columns:
            result["rating"] = result["rating"].where(result["rating"].notna(), result["_mapped_rating"])
        else:
            result["rating"] = result["_mapped_rating"]

    return result.drop(columns=[col for col in ("_mapped_nm_id", "_mapped_rating") if col in result.columns])


def load_postgres(*, truncate: bool) -> None:
    settings = get_settings()
    if not settings.postgres_dsn:
        raise RuntimeError("POSTGRES_DSN не задан")

    df = read_export_dataframe(
        export_dir=POSTGRES_EXPORT_DIR,
        csv_name="reviews_for_postgres.csv",
        parquet_name="reviews_for_postgres.parquet",
        preferred_csv_columns={"review_id", "review_text", "review_date"},
    )
    df = merge_review_nm_mapping(df)
    required = {"review_id", "review_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В PostgreSQL export не хватает колонок: {missing}")
    if "predicted_labels" not in df.columns and "predicted_labels_str" not in df.columns:
        raise ValueError("В PostgreSQL export нет predicted_labels или predicted_labels_str")

    with psycopg.connect(settings.postgres_dsn, row_factory=dict_row) as conn:
        init_service_schema(conn)
        with conn.cursor() as cur:
            if truncate:
                cur.execute("TRUNCATE TABLE review_labels;")
                cur.execute("TRUNCATE TABLE reviews CASCADE;")
                conn.commit()

            for start in tqdm(range(0, len(df), POSTGRES_BATCH_SIZE), desc="Loading PostgreSQL"):
                batch = df.iloc[start:start + POSTGRES_BATCH_SIZE]

                review_rows = []
                label_rows = []
                for _, row in batch.iterrows():
                    review_id = str(row["review_id"])
                    labels = parse_labels(row.get("predicted_labels"), row.get("predicted_labels_str"))
                    review_rows.append(
                        (
                            review_id,
                            clean_value(row.get("review_date")),
                            str(first_non_empty(row.get("review_text"), row.get("review_text_preview"), row.get("text")) or ""),
                            to_int(row.get("rating")),
                            str(clean_value(row.get("product_id"))) if clean_value(row.get("product_id")) is not None else None,
                            clean_value(row.get("product_name")),
                            clean_value(row.get("brand")),
                            clean_value(row.get("category")) or clean_value(row.get("subject")),
                            str(clean_value(row.get("seller_id"))) if clean_value(row.get("seller_id")) is not None else None,
                        )
                    )
                    label_rows.extend((review_id, label, None, clean_value(row.get("classifier_model"))) for label in labels)

                cur.executemany(
                    """
                    INSERT INTO reviews (
                        review_id, review_date, text, rating, product_id, product_name, brand, category, seller_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (review_id) DO UPDATE SET
                        review_date = EXCLUDED.review_date,
                        text = EXCLUDED.text,
                        rating = EXCLUDED.rating,
                        product_id = EXCLUDED.product_id,
                        product_name = EXCLUDED.product_name,
                        brand = EXCLUDED.brand,
                        category = EXCLUDED.category,
                        seller_id = EXCLUDED.seller_id;
                    """,
                    review_rows,
                )

                if label_rows:
                    cur.executemany(
                        """
                        INSERT INTO review_labels (review_id, label, confidence, model_name)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (review_id, label) DO UPDATE SET
                            confidence = EXCLUDED.confidence,
                            model_name = EXCLUDED.model_name;
                        """,
                        label_rows,
                    )
                conn.commit()

            cur.execute("SELECT COUNT(*) AS count FROM reviews;")
            reviews_count = cur.fetchone()["count"]
            cur.execute("SELECT COUNT(*) AS count FROM review_labels;")
            labels_count = cur.fetchone()["count"]
            print(f"PostgreSQL loaded: reviews={reviews_count}, review_labels={labels_count}")


def payload_from_row(row: pd.Series) -> dict[str, Any]:
    payload = {}
    for col, value in row.items():
        value = clean_value(value)
        if value is None:
            continue
        payload[col] = value

    labels = parse_labels(payload.get("predicted_labels"), payload.get("predicted_labels_str"))
    payload["predicted_labels"] = labels
    payload["labels"] = labels
    payload["text"] = first_non_empty(
        payload.get("review_text"),
        payload.get("text"),
        payload.get("review_text_preview"),
    ) or ""
    if "review_id" in payload:
        payload["review_id"] = str(payload["review_id"])
    return payload


def load_qdrant(*, recreate: bool) -> None:
    settings = get_settings()
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL не задан")

    vectors_path = QDRANT_EXPORT_DIR / "qdrant_vectors.npy"
    payload_path = QDRANT_EXPORT_DIR / "qdrant_payload.parquet"
    if not vectors_path.exists():
        raise FileNotFoundError(f"Не найден {vectors_path}")

    vectors = np.load(vectors_path, mmap_mode="r")
    payload_df = read_export_dataframe(
        export_dir=QDRANT_EXPORT_DIR,
        csv_name="qdrant_payload.csv",
        parquet_name="qdrant_payload.parquet",
        preferred_csv_columns={"review_id", "review_date", "review_text_preview", "predicted_labels_str"},
    )
    payload_df = merge_review_nm_mapping(payload_df)
    if len(vectors) != len(payload_df):
        raise ValueError(f"vectors rows={len(vectors)} != payload rows={len(payload_df)}")
    if "review_id" not in payload_df.columns:
        raise ValueError("В Qdrant payload export нет колонки review_id")

    collection = settings.qdrant_collection
    vector_dim = int(vectors.shape[1])
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    existing = [item.name for item in client.get_collections().collections]
    if collection in existing and recreate:
        client.delete_collection(collection)
        existing.remove(collection)

    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    for start in tqdm(range(0, len(payload_df), QDRANT_BATCH_SIZE), desc="Loading Qdrant"):
        end = min(start + QDRANT_BATCH_SIZE, len(payload_df))
        batch_payload = payload_df.iloc[start:end]
        batch_vectors = vectors[start:end]
        points = []
        for i, (_, row) in enumerate(batch_payload.iterrows()):
            review_id = int(row["review_id"])
            points.append(
                PointStruct(
                    id=review_id,
                    vector=batch_vectors[i].astype("float32").tolist(),
                    payload=payload_from_row(row),
                )
            )
        client.upsert(collection_name=collection, points=points, wait=True)

    info = client.get_collection(collection)
    print(f"Qdrant loaded: collection={collection}, points={info.points_count}, vector_dim={vector_dim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["postgres", "qdrant", "all"], default="all")
    parser.add_argument("--truncate-postgres", action="store_true")
    parser.add_argument("--recreate-qdrant", action="store_true")
    args = parser.parse_args()

    if args.target in {"postgres", "all"}:
        load_postgres(truncate=args.truncate_postgres)
    if args.target in {"qdrant", "all"}:
        load_qdrant(recreate=args.recreate_qdrant)


if __name__ == "__main__":
    main()
