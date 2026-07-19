"""Repair live PostgreSQL and Qdrant payloads from current CSV exports.

Use this when service databases were loaded from older parquet exports that
miss review_date or normalized Qdrant payload fields.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client import models
from tqdm.auto import tqdm

from app.core.config import get_settings
from app.offline.load_db_exports import first_non_empty, parse_labels


POSTGRES_EXPORT_DIR = Path(
    os.getenv("POSTGRES_EXPORT_DIR", "/app/data/db_exports/postgres_reviews_bge_m3_linearsvc_saved_model")
)
QDRANT_EXPORT_DIR = Path(os.getenv("QDRANT_EXPORT_DIR", "/app/data/db_exports/qdrant_vectors_baai_bge_m3"))

POSTGRES_CSV = POSTGRES_EXPORT_DIR / "reviews_for_postgres.csv"
QDRANT_CSV = QDRANT_EXPORT_DIR / "qdrant_payload.csv"

POSTGRES_COPY_BATCH_SIZE = int(os.getenv("POSTGRES_REPAIR_COPY_BATCH_SIZE", "10000"))
QDRANT_REPAIR_BATCH_SIZE = int(os.getenv("QDRANT_REPAIR_BATCH_SIZE", "500"))


def repair_postgres_dates() -> None:
    settings = get_settings()
    if not settings.postgres_dsn:
        raise RuntimeError("POSTGRES_DSN не задан")
    if not POSTGRES_CSV.exists():
        raise FileNotFoundError(f"Не найден {POSTGRES_CSV}")

    total_rows = count_csv_rows(POSTGRES_CSV)
    copied_rows = 0

    with psycopg.connect(settings.postgres_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TEMP TABLE repair_review_dates (
                    review_id TEXT PRIMARY KEY,
                    review_date DATE NOT NULL
                ) ON COMMIT DROP;
                """
            )
            with cur.copy("COPY repair_review_dates (review_id, review_date) FROM STDIN") as copy:
                for chunk in pd.read_csv(
                    POSTGRES_CSV,
                    usecols=["review_id", "review_date"],
                    dtype={"review_id": "string"},
                    chunksize=POSTGRES_COPY_BATCH_SIZE,
                ):
                    for row in chunk.itertuples(index=False):
                        review_id = str(row.review_id)
                        review_date = clean_str(row.review_date)
                        if review_id and review_date:
                            copy.write_row((review_id, review_date))
                            copied_rows += 1

            cur.execute(
                """
                UPDATE reviews r
                SET review_date = d.review_date
                FROM repair_review_dates d
                WHERE r.review_id = d.review_id;
                """
            )
            updated_rows = cur.rowcount
            conn.commit()

    print(f"PostgreSQL repair: csv_rows={total_rows}, copied_dates={copied_rows}, updated_reviews={updated_rows}")


def repair_qdrant_payload() -> None:
    settings = get_settings()
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL не задан")
    if not QDRANT_CSV.exists():
        raise FileNotFoundError(f"Не найден {QDRANT_CSV}")

    total_rows = count_csv_rows(QDRANT_CSV)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    updated_rows = 0

    for chunk in tqdm(
        pd.read_csv(QDRANT_CSV, dtype={"review_id": "string"}, chunksize=QDRANT_REPAIR_BATCH_SIZE),
        total=(total_rows + QDRANT_REPAIR_BATCH_SIZE - 1) // QDRANT_REPAIR_BATCH_SIZE,
        desc="Repairing Qdrant payload",
    ):
        operations = []
        for row in chunk.to_dict(orient="records"):
            review_id = clean_str(row.get("review_id"))
            if not review_id:
                continue

            payload = build_qdrant_payload(row)
            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload=payload,
                        points=[point_id_from_review_id(review_id)],
                    )
                )
            )

        if operations:
            client.batch_update_points(
                collection_name=settings.qdrant_collection,
                update_operations=operations,
                wait=True,
            )
            updated_rows += len(operations)

    print(f"Qdrant repair: csv_rows={total_rows}, updated_points={updated_rows}")


def build_qdrant_payload(row: dict[str, Any]) -> dict[str, Any]:
    labels = parse_labels(row.get("predicted_labels"), row.get("predicted_labels_str"))
    text = first_non_empty(row.get("text"), row.get("review_text"), row.get("review_text_preview")) or ""

    payload: dict[str, Any] = {
        "review_id": clean_str(row.get("review_id")),
        "review_date": clean_str(row.get("review_date")),
        "text": str(text),
        "labels": labels,
        "predicted_labels": labels,
    }

    for key in ("review_text_preview", "predicted_labels_str"):
        value = clean_str(row.get(key))
        if value:
            payload[key] = value

    return payload


def clean_str(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def point_id_from_review_id(review_id: str) -> int | str:
    try:
        return int(review_id)
    except ValueError:
        return review_id


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return max(sum(1 for _ in csv.reader(file)) - 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["postgres", "qdrant", "all"], default="all")
    args = parser.parse_args()

    if args.target in {"postgres", "all"}:
        repair_postgres_dates()
    if args.target in {"qdrant", "all"}:
        repair_qdrant_payload()


if __name__ == "__main__":
    main()
