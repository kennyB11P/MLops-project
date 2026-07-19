"""Build a local wb-products slice for nm_id values used by service reviews.

The source dataset is large, so this script streams compressed JSONL basket
files and keeps only product cards whose nm_id is present in review_nm_mapping.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import zstandard as zstd


DEFAULT_MAPPING = Path("/app/data/db_exports/review_nm_mapping.csv")
DEFAULT_OUTPUT = Path("/app/data/db_exports/wb_products_slice.parquet")
DEFAULT_URL_TEMPLATE = "https://huggingface.co/datasets/nyuuzyou/wb-products/resolve/main/basket-{basket:02d}.json.zst"

PRODUCT_COLUMNS = [
    "nm_id",
    "imt_name",
    "subj_name",
    "subj_root_name",
    "brand_name",
    "nm_colors_names",
    "vendor_code",
    "description",
    "imt_id",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url-template", default=DEFAULT_URL_TEMPLATE)
    parser.add_argument("--basket-from", type=int, default=1)
    parser.add_argument("--basket-to", type=int, default=12)
    parser.add_argument("--progress-every", type=int, default=500_000)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--max-baskets", type=int, default=None)
    args = parser.parse_args()

    needed_nm_ids = load_needed_nm_ids(args.mapping)
    found: dict[str, dict[str, Any]] = {}
    baskets = list(range(args.basket_from, args.basket_to + 1))
    if args.max_baskets is not None:
        baskets = baskets[: args.max_baskets]

    print(f"Needed nm_id: {len(needed_nm_ids)}")
    print(f"Output: {args.output}")

    for basket in baskets:
        if len(found) == len(needed_nm_ids):
            break
        url = args.url_template.format(basket=basket)
        before = len(found)
        rows_seen = stream_basket(
            url=url,
            needed_nm_ids=needed_nm_ids,
            found=found,
            request_timeout=args.request_timeout,
            progress_every=args.progress_every,
        )
        print(
            f"basket-{basket:02d}: rows_seen={rows_seen}, "
            f"new_found={len(found) - before}, found_total={len(found)}"
        )

    save_slice(found, args.output)
    print_coverage(needed_nm_ids, found, args.output)


def load_needed_nm_ids(mapping_path: Path) -> set[str]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Не найден mapping: {mapping_path}")
    mapping = pd.read_csv(mapping_path, usecols=["nm_id"], dtype={"nm_id": "string"})
    nm_ids = mapping["nm_id"].dropna().map(clean_nm_id)
    return {nm_id for nm_id in nm_ids if nm_id and nm_id != "0"}


def stream_basket(
    *,
    url: str,
    needed_nm_ids: set[str],
    found: dict[str, dict[str, Any]],
    request_timeout: int,
    progress_every: int,
) -> int:
    rows_seen = 0
    with requests.get(url, stream=True, timeout=(30, request_timeout)) as response:
        response.raise_for_status()
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(response.raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                rows_seen += 1
                row = json.loads(line)
                nm_id = clean_nm_id(row.get("nm_id"))
                if nm_id in needed_nm_ids and nm_id not in found:
                    found[nm_id] = normalize_product_row(row, nm_id)

                if progress_every and rows_seen % progress_every == 0:
                    print(f"  {url.rsplit('/', 1)[-1]} rows={rows_seen}, found_total={len(found)}")
    return rows_seen


def normalize_product_row(row: dict[str, Any], nm_id: str) -> dict[str, Any]:
    normalized = {column: clean_value(row.get(column)) for column in PRODUCT_COLUMNS}
    normalized["nm_id"] = nm_id
    return normalized


def save_slice(found: dict[str, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(found.values(), columns=PRODUCT_COLUMNS)
    df.to_parquet(output_path, index=False)
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved parquet: {output_path} rows={len(df)}")
    print(f"Saved csv: {csv_path} rows={len(df)}")


def print_coverage(needed_nm_ids: set[str], found: dict[str, dict[str, Any]], output_path: Path) -> None:
    found_ids = set(found)
    missing_ids = sorted(needed_nm_ids - found_ids)
    coverage_pct = round(len(found_ids) / max(len(needed_nm_ids), 1) * 100, 2)
    print(
        "Coverage: "
        f"found={len(found_ids)}, needed={len(needed_nm_ids)}, "
        f"missing={len(missing_ids)}, coverage_pct={coverage_pct}"
    )

    if found:
        df = pd.DataFrame(found.values())
        for column in ["imt_name", "subj_name", "subj_root_name", "brand_name"]:
            if column in df:
                print(f"{column}_coverage={int(df[column].notna().sum())}")

    report_path = output_path.with_suffix(".coverage.json")
    report = {
        "needed": len(needed_nm_ids),
        "found": len(found_ids),
        "missing": len(missing_ids),
        "coverage_pct": coverage_pct,
        "missing_sample": missing_ids[:100],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved coverage report: {report_path}")


def clean_nm_id(value: Any) -> str | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        return str(int(float(str(value).strip())))
    except ValueError:
        text = str(value).strip()
        return text or None


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
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


if __name__ == "__main__":
    main()
