#!/usr/bin/env python3
"""Create a training CSV with the unverified 80-row source removed."""

import argparse
import csv
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data/processed_public_datasets/combined_v4_training_data.csv"
DEFAULT_EXCLUDED = PROJECT_ROOT / "data/raw/cas12a_mismatch_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "data/processed_public_datasets/"
    "combined_v4_training_data_cleaned_exclude_unverified80.csv"
)
REQUIRED_COLUMNS = ("spacer_seq", "target_seq", "mismatch_pos", "relative_activity")


def record_key(row):
    try:
        activity = Decimal(row["relative_activity"].strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f"Invalid relative_activity value: {row.get('relative_activity')!r}") from exc

    return (
        row["spacer_seq"].strip().upper(),
        row["target_seq"].strip().upper(),
        row["mismatch_pos"].strip(),
        activity,
    )


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        rows = list(reader)
        return reader.fieldnames, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    source_header, source_rows = read_csv(args.source)
    _, excluded_rows = read_csv(args.exclude)
    removal_counts = Counter(record_key(row) for row in excluded_rows)

    available_counts = Counter(record_key(row) for row in source_rows)
    missing = removal_counts - available_counts
    if missing:
        raise ValueError(
            f"Cannot safely exclude all source rows: {sum(missing.values())} "
            "full-record occurrence(s) are missing from the combined CSV."
        )

    remaining = removal_counts.copy()
    cleaned_rows = []
    removed_count = 0
    for row in source_rows:
        key = record_key(row)
        if remaining[key] > 0:
            remaining[key] -= 1
            removed_count += 1
        else:
            cleaned_rows.append(row)

    if any(remaining.values()):
        raise RuntimeError("Not all requested source records were removed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_header)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"Source rows: {len(source_rows)}")
    print(f"Excluded source rows: {removed_count}")
    print(f"Cleaned rows: {len(cleaned_rows)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
