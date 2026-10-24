#!/usr/bin/env python3
"""Generate transparent RT-RPA-Cas12a design candidates for miR-29a.

This is a forward-design candidate generator, not a final oligo-ordering tool.
It makes the design assumptions explicit so they can be checked experimentally:

    mature miRNA -> RT/RPA amplicon -> PAM -> 25-nt protospacer -> crRNA

The script compares two PAM-source labels (RT-primer and RPA-tail), keeps
candidate spacers that cover the miR-29a differences from both miR-29b and
miR-29c, and performs simple RPA-primer quality checks. It deliberately does
not claim that a heuristic score is measured Cas12a activity.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


# Mature miRNA sequences from the project protocol, converted U -> T for DNA.
MIRNA_DNA: Dict[str, str] = {
    "miR-29a": "TAGCACCATCTGAAATCGGTTA",
    "miR-29b": "TAGCACCATTTGAAATCAGTGTT",
    "miR-29c": "TAGCACCATTTGAAATCGGTTA",
}

# Existing project adapter/RT-primer sequence. These are design inputs, not
# experimentally validated final primers.
RT_HANDLE = "AGTGCGTACTCGAGGCATCGATCAGTACGGATCG"
RPA_HANDLE = "GTCGTATCCAGTGCAGGGTCCGAGGTATTCG"

PAM_PREFIX = "TTT"
PAM_BASES = "ACG"  # LbCas12a TTTV; V means A, C, or G (not T).


@dataclass
class PrimerCheck:
    sequence: str
    length: int
    gc_fraction: float
    tm_approx: float
    max_homopolymer: int
    hairpin_flag: bool
    self_dimer_3p_flag: bool

    @property
    def passes_basic_rules(self) -> bool:
        return (
            18 <= self.length <= 35
            and 0.30 <= self.gc_fraction <= 0.70
            and self.max_homopolymer <= 4
            and not self.hairpin_flag
            and not self.self_dimer_3p_flag
        )


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGT", "TGCA")
    return seq.translate(table)[::-1]


def gc_fraction(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def approximate_tm(seq: str) -> float:
    """Wallace estimate for a quick screen; this is not an RPA thermodynamic model."""
    return 2.0 * (seq.count("A") + seq.count("T")) + 4.0 * (seq.count("G") + seq.count("C"))


def max_homopolymer(seq: str) -> int:
    best = run = 1
    for left, right in zip(seq, seq[1:]):
        run = run + 1 if left == right else 1
        best = max(best, run)
    return best if seq else 0


def has_internal_hairpin(seq: str, stem: int = 4, min_loop: int = 3) -> bool:
    """Conservative exact-complement screen, not a substitute for ViennaRNA."""
    for i in range(len(seq) - stem + 1):
        left = seq[i : i + stem]
        rc = reverse_complement(left)
        for j in range(i + stem + min_loop, len(seq) - stem + 1):
            if seq[j : j + stem] == rc:
                return True
    return False


def has_3prime_self_dimer(seq: str, min_match: int = 4) -> bool:
    """Flag a simple 3' self-complementarity pattern."""
    tail = seq[-min_match:]
    rc_tail = reverse_complement(tail)
    return rc_tail in seq[:-min_match]


def check_primer(seq: str) -> PrimerCheck:
    seq = seq.upper()
    return PrimerCheck(
        sequence=seq,
        length=len(seq),
        gc_fraction=gc_fraction(seq),
        tm_approx=approximate_tm(seq),
        max_homopolymer=max_homopolymer(seq),
        hairpin_flag=has_internal_hairpin(seq),
        self_dimer_3p_flag=has_3prime_self_dimer(seq),
    )


def difference_positions(reference: str, other: str) -> List[int]:
    """Return aligned miRNA difference positions; unequal tails are included."""
    return [
        i
        for i in range(max(len(reference), len(other)))
        if reference[i : i + 1] != other[i : i + 1]
    ]


def insert_pam(handle: str, pam_offset: int, pam: str) -> str:
    return handle[:pam_offset] + pam + handle[pam_offset:]


def build_amplicon(mirna: str, pam_offset: int, pam: str, pam_source: str) -> Tuple[str, int, int]:
    """Return product, PAM start, and miRNA start for one explicit architecture.

    Both source labels currently use the same final sequence abstraction. The
    label records whether the proposed PAM is to be introduced during RT or by
    an RPA 5' overhang; the exact primer chemistry must be verified separately.
    """
    if pam_source not in {"RT-primer", "RPA-tail"}:
        raise ValueError(f"Unsupported PAM source: {pam_source}")
    left = insert_pam(RT_HANDLE, pam_offset, pam)
    product = left + MIRNA_DNA[mirna] + RPA_HANDLE
    return product, pam_offset, len(left)


def choose_primers(product: str, pam_start: int, spacer_start: int, spacer_length: int,
                   primer_length: int, pam_source: str, pam: str) -> Tuple[str, str]:
    """Choose flanking primers; for RPA-tail, PAM is a 5' overhang, not annealing bases."""
    pam_end = pam_start + 4
    if pam_source == "RT-primer":
        # PAM is already present in the RT-derived template; F binds upstream.
        fwd_start, fwd_end = pam_start - primer_length, pam_start
        forward = product[fwd_start:fwd_end]
    else:
        # The RPA forward primer anneals after the insertion point and carries
        # PAM as a non-annealing 5' tail, which becomes part of the amplicon.
        fwd_start, fwd_end = pam_end, pam_end + primer_length
        forward = pam + product[fwd_start:fwd_end]
    rev_start = spacer_start + spacer_length
    rev_end = rev_start + primer_length
    if fwd_start < 0 or fwd_end > len(product) or rev_end > len(product):
        raise ValueError("Insufficient sequence for the requested primer layout")
    reverse = reverse_complement(product[rev_start:rev_end])
    return forward, reverse


def mismatch_count(a: str, b: str) -> int:
    return sum(left != right for left, right in zip(a, b)) + abs(len(a) - len(b))


def make_candidate_rows(
    guide_length: int = 25,
    primer_length: int = 30,
    min_mirna_overlap: int = 8,
) -> Iterable[dict]:
    reference = MIRNA_DNA["miR-29a"]
    diff_b = difference_positions(reference, MIRNA_DNA["miR-29b"])
    diff_c = difference_positions(reference, MIRNA_DNA["miR-29c"])

    for pam_source in ("RT-primer", "RPA-tail"):
        for pam_offset in range(len(RT_HANDLE) + 1):
            for base in PAM_BASES:
                pam = PAM_PREFIX + base
                product, pam_start, mirna_start = build_amplicon(
                    "miR-29a", pam_offset, pam, pam_source
                )
                spacer_start = pam_start + len(pam)
                spacer_end = spacer_start + guide_length
                if spacer_end > len(product):
                    continue

                spacer = product[spacer_start:spacer_end]
                mirna_end = mirna_start + len(reference)
                overlap_start = max(spacer_start, mirna_start)
                overlap_end = min(spacer_end, mirna_end)
                mirna_overlap = max(0, overlap_end - overlap_start)
                if mirna_overlap < min_mirna_overlap:
                    continue

                covered_b = [p for p in diff_b if spacer_start <= mirna_start + p < spacer_end]
                covered_c = [p for p in diff_c if spacer_start <= mirna_start + p < spacer_end]
                if not covered_b or not covered_c:
                    continue

                try:
                    forward, reverse = choose_primers(
                        product, pam_start, spacer_start, guide_length, primer_length,
                        pam_source, pam,
                    )
                except ValueError:
                    continue

                f_check = check_primer(forward)
                r_check = check_primer(reverse)
                non_target_rows = {}
                for name in ("miR-29b", "miR-29c"):
                    other_product, _, _ = build_amplicon(name, pam_offset, pam, pam_source)
                    other_spacer = other_product[spacer_start:spacer_end]
                    non_target_rows[name] = mismatch_count(spacer, other_spacer)

                covered_positions = sorted(set(covered_b + covered_c))
                proximity_score = sum(
                    2.0 if spacer_start <= mirna_start + p - pam_start <= 8 else 1.0
                    for p in covered_positions
                )
                gc_score = 1.0 - min(abs(gc_fraction(spacer) - 0.50) / 0.50, 1.0)
                primer_penalty = int(not f_check.passes_basic_rules) + int(not r_check.passes_basic_rules)
                heuristic_score = proximity_score + gc_score - 1.5 * primer_penalty

                yield {
                    "pam_source": pam_source,
                    "pam": pam,
                    "pam_offset_in_rt_handle": pam_offset,
                    "pam_start_0based": pam_start,
                    "protospacer_start_0based": spacer_start,
                    "guide_length": guide_length,
                    "crrna_spacer_dna_5to3": spacer,
                    "crrna_spacer_rna_5to3": spacer.replace("T", "U"),
                    "mirna_overlap_nt": mirna_overlap,
                    "miR29a_vs_29b_difference_positions": ";".join(map(str, covered_b)),
                    "miR29a_vs_29c_difference_positions": ";".join(map(str, covered_c)),
                    "mismatches_vs_miR29b": non_target_rows["miR-29b"],
                    "mismatches_vs_miR29c": non_target_rows["miR-29c"],
                    "crrna_gc_fraction": round(gc_fraction(spacer), 4),
                    "forward_primer": forward,
                    "forward_primer_annealing_length": primer_length,
                    "forward_primer_5prime_pam_overhang": pam if pam_source == "RPA-tail" else "",
                    "reverse_primer": reverse,
                    "forward_primer_pass": f_check.passes_basic_rules,
                    "reverse_primer_pass": r_check.passes_basic_rules,
                    "forward_primer_gc": round(f_check.gc_fraction, 4),
                    "reverse_primer_gc": round(r_check.gc_fraction, 4),
                    "forward_hairpin_flag": f_check.hairpin_flag,
                    "reverse_hairpin_flag": r_check.hairpin_flag,
                    "forward_3prime_self_dimer_flag": f_check.self_dimer_3p_flag,
                    "reverse_3prime_self_dimer_flag": r_check.self_dimer_3p_flag,
                    "heuristic_score": round(heuristic_score, 4),
                    "status": "candidate; verify PAM orientation, primer chemistry, and experiment",
                }


def write_csv(rows: Sequence[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No candidates passed the current constraints")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-length", type=int, default=25)
    parser.add_argument("--primer-length", type=int, default=30)
    parser.add_argument("--min-mirna-overlap", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/forward_design/mir29a_candidates.csv"),
    )
    args = parser.parse_args()

    rows = list(make_candidate_rows(args.guide_length, args.primer_length, args.min_mirna_overlap))
    rows.sort(key=lambda row: row["heuristic_score"], reverse=True)
    write_csv(rows, args.output)
    print(f"Generated {len(rows)} candidates")
    print(f"Saved: {args.output}")
    print("Scores are heuristic design scores, not measured Cas12a activity.")


if __name__ == "__main__":
    main()
