#!/usr/bin/env python3
"""Export run-level metadata from a saved NCBI SRA ESummary response."""

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "public_datasets" / "raw" / "ncbi_cas12a_human_cells"
SOURCE = DATA_DIR / "sra_esummary.json"
OUTPUT = DATA_DIR / "sra_run_manifest.csv"


def capture(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    result = payload["result"]
    rows = []
    for uid in result["uids"]:
        item = result[uid]
        xml = item.get("expxml", "")
        experiment = re.search(r'<Experiment acc="(SRX\d+)"[^>]*name="([^"]*)', xml)
        run = re.search(r'<Run acc="(SRR\d+)', item.get("runs", ""))
        rows.append(
            {
                "run_accession": run.group(1) if run else "",
                "experiment_accession": experiment.group(1) if experiment else "",
                "experiment_name": experiment.group(2) if experiment else "",
                "library_name": capture(r"<LIBRARY_NAME>(.*?)</LIBRARY_NAME>", xml),
                "library_strategy": capture(r"<LIBRARY_STRATEGY>(.*?)</LIBRARY_STRATEGY>", xml),
                "spots": capture(r'total_spots="(\d+)', xml),
                "bases": capture(r'total_bases="(\d+)', xml),
                "compressed_bytes": capture(r'total_size="(\d+)', xml),
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {OUTPUT}")


if __name__ == "__main__":
    main()
