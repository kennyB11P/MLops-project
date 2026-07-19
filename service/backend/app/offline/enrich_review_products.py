"""Enrich service exports and live stores with Wildberries product ids.

The current review exports can lose the original Wildberries nmId. This script
uses data/db_exports/review_nm_mapping.csv to restore product_id/nm_id and
rating, then optionally joins a local wb-products export by nm_id to add product
name, category and brand.
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
from app.offline.load_db_exports import first_non_empty


DATA_ROOT = Path(os.getenv("DATA_ROOT", "/app/data"))
POSTGRES_EXPORT_DIR = Path(
    os.getenv("POSTGRES_EXPORT_DIR", DATA_ROOT / "db_exports" / "postgres_reviews_bge_m3_linearsvc_saved_model")
)
QDRANT_EXPORT_DIR = Path(os.getenv("QDRANT_EXPORT_DIR", DATA_ROOT / "db_exports" / "qdrant_vectors_baai_bge_m3"))
MAPPING_CSV = Path(os.getenv("REVIEW_NM_MAPPING_PATH", DATA_ROOT / "db_exports" / "review_nm_mapping.csv"))

POSTGRES_CSV = POSTGRES_EXPORT_DIR / "reviews_for_postgres.csv"
QDRANT_CSV = QDRANT_EXPORT_DIR / "qdrant_payload.csv"

CSV_CHUNK_SIZE = int(os.getenv("PRODUCT_ENRICH_CSV_CHUNK_SIZE", "50000"))
POSTGRES_COPY_BATCH_SIZE = int(os.getenv("PRODUCT_ENRICH_POSTGRES_COPY_BATCH_SIZE", "10000"))
QDRANT_BATCH_SIZE = int(os.getenv("PRODUCT_ENRICH_QDRANT_BATCH_SIZE", "500"))


PRODUCT_COLUMN_ALIASES = {
    "product_name": ("product_name", "imt_name", "name", "title"),
    "category": ("category", "subj_name", "subject", "subject_name"),
    "category_root": ("category_root", "subj_root_name", "subject_root", "root_category"),
    "brand": ("brand", "brand_name"),
    "color": ("color", "nm_colors_names"),
    "vendor_code": ("vendor_code",),
    "description": ("description",),
    "imt_id": ("imt_id",),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["exports", "postgres", "qdrant", "all"], default="all")
    parser.add_argument("--mapping", type=Path, default=MAPPING_CSV)
    parser.add_argument("--products-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mapping = load_and_validate_mapping(args.mapping)
    product_info = load_product_info(args.products_path, set(mapping["nm_id"])) if args.products_path else None
    enriched = build_enrichment_frame(mapping, product_info)

    print_validation_summary(mapping, enriched, product_info)
    if args.dry_run:
        return

    if args.target in {"exports", "all"}:
        enrich_exports(enriched)
    if args.target in {"postgres", "all"}:
        enrich_postgres(enriched)
    if args.target in {"qdrant", "all"}:
        enrich_qdrant(enriched)


def load_and_validate_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден mapping-файл: {path}")

    required = {"review_id", "nm_id"}
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = required - columns
    if missing:
        raise ValueError(f"В mapping-файле не хватает колонок: {missing}")

    usecols = ["review_id", "nm_id"]
    optional = ["match_method", "match_confidence", "source_file", "source_row_idx", "rating"]
    usecols.extend([col for col in optional if col in columns])
    mapping = pd.read_csv(path, usecols=usecols, dtype={"review_id": "string", "nm_id": "string"})

    mapping["review_id"] = mapping["review_id"].map(clean_str)
    mapping["nm_id"] = mapping["nm_id"].map(clean_str)
    null_review_id = int(mapping["review_id"].isna().sum())
    null_nm_id = int(mapping["nm_id"].isna().sum())
    if null_review_id or null_nm_id:
        raise ValueError(f"Mapping содержит пустые ключи: review_id={null_review_id}, nm_id={null_nm_id}")

    duplicated_rows = int(mapping["review_id"].duplicated(keep=False).sum())
    if duplicated_rows:
        conflicts = mapping.groupby("review_id")["nm_id"].nunique(dropna=True)
        conflict_count = int((conflicts > 1).sum())
        raise ValueError(f"Mapping содержит дубли review_id: rows={duplicated_rows}, conflicts={conflict_count}")

    expected_review_ids = load_expected_review_ids()
    mapped_ids = set(mapping["review_id"])
    missing_ids = expected_review_ids - mapped_ids
    extra_ids = mapped_ids - expected_review_ids
    if missing_ids or extra_ids:
        raise ValueError(
            "Mapping не совпадает с текущим reviews export: "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}"
        )

    if "rating" in mapping.columns:
        mapping["rating"] = pd.to_numeric(mapping["rating"], errors="coerce").astype("Int64")

    return mapping


def load_expected_review_ids() -> set[str]:
    if not POSTGRES_CSV.exists():
        raise FileNotFoundError(f"Не найден PostgreSQL export: {POSTGRES_CSV}")
    ids = pd.read_csv(POSTGRES_CSV, usecols=["review_id"], dtype={"review_id": "string"})["review_id"].map(clean_str)
    return set(ids.dropna())


def load_product_info(path: Path, needed_nm_ids: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден products-файл: {path}")

    if path.suffix.lower() == ".parquet":
        products = pd.read_parquet(path)
    else:
        products = pd.read_csv(path)

    nm_column = find_first_column(products.columns, ("nm_id", "nmId", "product_id"))
    if nm_column is None:
        raise ValueError("В products-файле нет nm_id/nmId/product_id")

    products[nm_column] = products[nm_column].map(clean_str)
    products = products[products[nm_column].isin(needed_nm_ids)].copy()
    products = products.drop_duplicates(subset=[nm_column], keep="first")

    result = pd.DataFrame({"nm_id": products[nm_column]})
    for output_col, candidates in PRODUCT_COLUMN_ALIASES.items():
        source_col = find_first_column(products.columns, candidates)
        if source_col:
            result[output_col] = products[source_col].map(clean_str)

    return result


def build_enrichment_frame(mapping: pd.DataFrame, product_info: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["review_id", "nm_id"]
    if "rating" in mapping.columns:
        columns.append("rating")
    result = mapping[columns].copy()
    result["product_id"] = result["nm_id"]

    if product_info is not None:
        result = result.merge(product_info, on="nm_id", how="left", validate="many_to_one")
        if "category" not in result.columns and "category_root" in result.columns:
            result["category"] = result["category_root"]
    return result


def print_validation_summary(mapping: pd.DataFrame, enriched: pd.DataFrame, product_info: pd.DataFrame | None) -> None:
    print(
        "Mapping validation: "
        f"rows={len(mapping)}, review_ids={mapping['review_id'].nunique()}, nm_ids={mapping['nm_id'].nunique()}"
    )
    if "match_method" in mapping.columns:
        print("match_method:")
        print(mapping["match_method"].value_counts(dropna=False).head(20).to_string())
    if "match_confidence" in mapping.columns:
        confidence = pd.to_numeric(mapping["match_confidence"], errors="coerce")
        print(
            "match_confidence: "
            f"min={confidence.min()}, max={confidence.max()}, mean={round(float(confidence.mean()), 4)}"
        )
    if "rating" in enriched.columns:
        print("rating coverage:", int(enriched["rating"].notna().sum()))
    if product_info is None:
        print("Product catalog not provided: will enrich product_id/nm_id/rating only.")
        return
    print(
        "Product catalog match: "
        f"matched_nm_ids={len(product_info)}, needed_nm_ids={mapping['nm_id'].nunique()}"
    )
    for col in ("product_name", "category", "category_root", "brand", "color"):
        if col in enriched.columns:
            print(f"{col} coverage:", int(enriched[col].notna().sum()))


def enrich_exports(enriched: pd.DataFrame) -> None:
    enrich_csv(POSTGRES_CSV, enriched, text="PostgreSQL export")
    enrich_csv(QDRANT_CSV, enriched, text="Qdrant payload export")


def enrich_csv(path: Path, enriched: pd.DataFrame, *, text: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Не найден {text}: {path}")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    first = True
    rows = 0
    for chunk in pd.read_csv(path, dtype={"review_id": "string"}, chunksize=CSV_CHUNK_SIZE):
        chunk["review_id"] = chunk["review_id"].map(clean_str)
        merged = chunk.merge(enriched, on="review_id", how="left", suffixes=("", "_enriched"), validate="one_to_one")
        missing = int(merged["nm_id"].isna().sum())
        if missing:
            raise ValueError(f"{text}: enrichment missing rows={missing}")

        for col in enrichment_payload_columns(enriched):
            enriched_col = f"{col}_enriched"
            if enriched_col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[enriched_col])
                merged = merged.drop(columns=[enriched_col])
            elif col in merged.columns:
                continue
            else:
                merged[col] = merged[col]

        merged.to_csv(tmp_path, index=False, mode="w" if first else "a", header=first)
        rows += len(merged)
        first = False

    tmp_path.replace(path)
    print(f"{text} enriched: rows={rows}, path={path}")


def enrich_postgres(enriched: pd.DataFrame) -> None:
    settings = get_settings()
    if not settings.postgres_dsn:
        raise RuntimeError("POSTGRES_DSN не задан")

    columns = [col for col in postgres_enrichment_columns() if col in enriched.columns]
    with psycopg.connect(settings.postgres_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(build_temp_table_sql(columns))
            with cur.copy(build_copy_sql(columns)) as copy:
                for chunk_start in range(0, len(enriched), POSTGRES_COPY_BATCH_SIZE):
                    chunk = enriched.iloc[chunk_start:chunk_start + POSTGRES_COPY_BATCH_SIZE]
                    for row in chunk.to_dict(orient="records"):
                        copy.write_row(tuple(value_for_copy(row.get(col)) for col in ["review_id", *columns]))

            set_sql = ",\n                    ".join(f"{col} = COALESCE(e.{col}, r.{col})" for col in columns)
            cur.execute(
                f"""
                UPDATE reviews r
                SET {set_sql}
                FROM product_enrichment e
                WHERE r.review_id = e.review_id;
                """
            )
            updated = cur.rowcount
            conn.commit()
    print(f"PostgreSQL product enrichment: updated_reviews={updated}")


def enrich_qdrant(enriched: pd.DataFrame) -> None:
    settings = get_settings()
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL не задан")

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    updated = 0
    total_batches = (len(enriched) + QDRANT_BATCH_SIZE - 1) // QDRANT_BATCH_SIZE
    for start in tqdm(range(0, len(enriched), QDRANT_BATCH_SIZE), total=total_batches, desc="Enriching Qdrant"):
        chunk = enriched.iloc[start:start + QDRANT_BATCH_SIZE]
        operations = []
        for row in chunk.to_dict(orient="records"):
            review_id = clean_str(row.get("review_id"))
            if not review_id:
                continue
            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload=payload_from_enriched_row(row),
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
            updated += len(operations)
    print(f"Qdrant product enrichment: updated_points={updated}")


def payload_from_enriched_row(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for col in enrichment_payload_columns(pd.DataFrame([row])):
        value = clean_value(row.get(col))
        if value is not None:
            payload[col] = value
    return payload


def enrichment_payload_columns(enriched: pd.DataFrame) -> list[str]:
    ordered = [
        "nm_id",
        "product_id",
        "rating",
        "product_name",
        "brand",
        "category",
        "category_root",
        "color",
        "vendor_code",
        "description",
        "imt_id",
    ]
    return [col for col in ordered if col in enriched.columns]


def postgres_enrichment_columns() -> list[str]:
    return ["product_id", "rating", "product_name", "brand", "category"]


def build_temp_table_sql(columns: list[str]) -> str:
    type_map = {
        "product_id": "TEXT",
        "rating": "INTEGER",
        "product_name": "TEXT",
        "brand": "TEXT",
        "category": "TEXT",
    }
    field_sql = ",\n                    ".join(f"{col} {type_map[col]}" for col in columns)
    return f"""
                CREATE TEMP TABLE product_enrichment (
                    review_id TEXT PRIMARY KEY,
                    {field_sql}
                ) ON COMMIT DROP;
                """


def build_copy_sql(columns: list[str]) -> str:
    return f"COPY product_enrichment (review_id, {', '.join(columns)}) FROM STDIN"


def value_for_copy(value: Any) -> Any:
    value = clean_value(value)
    if pd.isna(value):
        return None
    return value


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def clean_str(value: Any) -> str | None:
    value = clean_value(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text


def find_first_column(columns: Any, candidates: tuple[str, ...]) -> str | None:
    normalized = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        found = normalized.get(candidate.lower())
        if found:
            return found
    return None


def point_id_from_review_id(review_id: str) -> int | str:
    try:
        return int(review_id)
    except ValueError:
        return review_id


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return max(sum(1 for _ in csv.reader(file)) - 1, 0)


if __name__ == "__main__":
    main()
