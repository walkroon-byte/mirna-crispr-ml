#!/usr/bin/env python3
"""Convert Sashital Figshare extracted-count tables to sequence-level tables.

The downloaded files contain one row per observed 20-nt target and columns for
library controls plus Cas12a samples.  This script keeps the original counts,
normalizes each sample by its library depth, and reports a log2 enrichment
against the matched nicked library control.  It intentionally keeps this
in-vitro proxy separate from EasyDesign's activity scale.
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "public_datasets" / "raw" / "sashital_data"
OUT = ROOT / "data" / "processed_public_datasets" / "sashital"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "KMlib001": RAW / "pLibrary_PS4_ExtractedCountsData_mismatches.tsv",
    "KMlib003": RAW / "pLibrary_EMX1_ExtractedCountsData_mismatches.tsv",
    "KMlib004": RAW / "pLibrary_CCR5_ExtractedCountsData_mismatches.tsv",
}


def parse_sample(name: str, library: str) -> tuple[str, int, int] | None:
    """Return ortholog, replicate, and timepoint for a nicked sample column."""
    # The published tables use both '-' and '_' in replicate names.
    m = re.match(r"^(As|Fn|Lb)[-_]" + re.escape(library) + r"[-_]rep(\d+)[-_](\d+)$", name)
    if not m:
        return None
    ortholog, replicate, timepoint = m.groups()
    timepoint = int(timepoint)
    # Use the 30-minute nicked fraction for a focused, reproducible table.
    if timepoint != 7:
        return None
    return ortholog, int(replicate), timepoint


def sequence_columns(columns: list[str]) -> list[str]:
    """Identify the first 20 nucleotide-position columns despite header variants."""
    if all(str(i) in columns for i in range(1, 21)):
        return [str(i) for i in range(1, 21)]
    kmer = [c for c in columns if c == "kmer" or str(c).startswith("kmer.")]
    return kmer[:20]


def process_file(library: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, sep="\t", nrows=0).columns.astype(str).tolist()
    seq_cols = sequence_columns(header)
    if len(seq_cols) != 20:
        raise ValueError(f"{path.name}: expected 20 sequence columns, got {len(seq_cols)}")

    control_candidates = [
        f"N-{library}-control",
        f"control-{library}-rep1",
    ]
    control_col = next((c for c in control_candidates if c in header), None)
    if control_col is None:
        raise ValueError(f"{path.name}: no matched nicked control column")

    samples = []
    for c in header:
        parsed = parse_sample(c, library)
        if parsed:
            samples.append((c, *parsed))
    if not samples:
        raise ValueError(f"{path.name}: no nicked Cas12a sample columns")

    usecols = seq_cols + ["MisMatches", control_col] + [x[0] for x in samples]
    count_cols = [control_col] + [x[0] for x in samples]
    totals = {c: 0.0 for c in count_cols}
    for count_chunk in pd.read_csv(path, sep="\t", usecols=count_cols, chunksize=50000):
        for col in count_cols:
            totals[col] += float(pd.to_numeric(count_chunk[col], errors="coerce").fillna(0).sum())

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=50000):
        seq = chunk[seq_cols].astype(str).agg("".join, axis=1).str.upper()
        valid = seq.str.fullmatch(r"[ACGT]{20}")
        if not valid.any():
            continue
        base = pd.DataFrame(
            {
                "guide_sequence": seq[valid].to_numpy(),
                "mismatches": pd.to_numeric(chunk.loc[valid, "MisMatches"], errors="coerce").to_numpy(),
                "control_count": pd.to_numeric(chunk.loc[valid, control_col], errors="coerce").fillna(0).to_numpy(),
            }
        )
        base = base.dropna(subset=["mismatches"])
        base["mismatches"] = base["mismatches"].astype(int)
        base["library"] = library
        base["library_name"] = {"KMlib001": "PS4", "KMlib003": "EMX1", "KMlib004": "CCR5"}[library]
        for sample_col, ortholog, replicate, timepoint in samples:
            sample_count = pd.to_numeric(chunk.loc[valid, sample_col], errors="coerce").fillna(0).to_numpy()
            # Match rows after dropping malformed mismatch values.
            sample_count = sample_count[pd.to_numeric(chunk.loc[valid, "MisMatches"], errors="coerce").notna().to_numpy()]
            row = base.copy()
            row["enzyme"] = ortholog
            row["replicate"] = replicate
            row["timepoint_code"] = timepoint
            row["timepoint_min"] = 30
            row["sample_name"] = sample_col
            row["sample_count"] = sample_count
            row["sample_total"] = totals[sample_col]
            row["control_total"] = totals[control_col]
            # Pseudocount prevents zero-count ratios from becoming infinite.
            row["activity_log2_ratio"] = np.log2(
                (row["sample_count"] + 0.5) / (row["sample_total"] + 0.5 * len(row))
            ) - np.log2(
                (row["control_count"] + 0.5) / (row["control_total"] + 0.5 * len(row))
            )
            chunks.append(row)
    if not chunks:
        return pd.DataFrame(), pd.DataFrame()
    long = pd.concat(chunks, ignore_index=True)
    long["assay_type"] = "in_vitro_plasmid_HTS_nicked_fraction"
    long["activity_definition"] = "log2 normalized nicked-fraction count ratio vs matched library control"
    long["guide_length"] = 20
    long["pam"] = "TTTV (library context; PAM not present in extracted target column)"
    long["sequence_activity_pair_verified"] = True
    grouped = (
        long[long["timepoint_min"] == 30]
        .groupby(["guide_sequence", "mismatches", "library", "library_name", "enzyme"], as_index=False)
        .agg(
            activity_log2_ratio=("activity_log2_ratio", "median"),
            activity_iqr=("activity_log2_ratio", lambda x: float(x.quantile(0.75) - x.quantile(0.25))),
            n_replicates=("replicate", "nunique"),
            total_sample_count=("sample_count", "sum"),
            total_control_count=("control_count", "sum"),
        )
    )
    grouped["assay_type"] = "in_vitro_plasmid_HTS_nicked_fraction"
    grouped["activity_definition"] = "30 min nicked fraction log2 normalized count ratio vs matched library control"
    grouped["guide_length"] = 20
    grouped["pam"] = "TTTV (library context; PAM not present in extracted target column)"
    grouped["sequence_activity_pair_verified"] = True
    return long, grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-long",
        action="store_true",
        help="also write replicate/timepoint long tables; these can be several GB",
    )
    args = parser.parse_args()
    all_grouped = []
    for library, path in FILES.items():
        long, grouped = process_file(library, path)
        if long.empty:
            continue
        all_grouped.append(grouped)
        if args.write_long:
            long.to_csv(OUT / f"{library}_sequence_activity_long.csv", index=False)
        grouped.to_csv(OUT / f"{library}_sequence_activity_30min.csv", index=False)
        print(f"{library}: {len(long):,} long records; {len(grouped):,} 30-min records")
    if all_grouped:
        pd.concat(all_grouped, ignore_index=True).to_csv(OUT / "sashital_sequence_activity_30min.csv", index=False)
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()
