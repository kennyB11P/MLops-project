"""Skeleton for offline topic clustering inside primary problem labels.

This module intentionally avoids running heavy clustering by default. It
documents the input contract and produces an empty, schema-compatible CSV that
can be filled by a later MiniBatchKMeans/HDBSCAN implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "review_id",
    "primary_label",
    "topic_cluster_id",
    "topic_cluster_name",
    "cluster_score",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/app/data/db_exports/review_topic_clusters.csv"))
    parser.add_argument("--write-empty-schema", action="store_true")
    args = parser.parse_args()

    if not args.write_empty_schema:
        raise SystemExit(
            "Topic clustering skeleton only writes an empty schema for now. "
            "Run with --write-empty-schema or implement clustering in this module."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(args.output, index=False)
    print(f"Wrote topic cluster schema: {args.output}")


if __name__ == "__main__":
    main()
