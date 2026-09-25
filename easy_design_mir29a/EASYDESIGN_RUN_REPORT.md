# EasyDesign Run Report: miR-29a RT-RPA-Cas12a

## Objective

Design a miR-29a-3p RT-RPA-Cas12a assay using EasyDesign's bundled Cas12a activity models, while excluding miR-29b-3p and miR-29c-3p.

## Inputs

- Primary input: `mir29a_primary_rt_cdna.fasta`
- Specificity exclusion input: `mir29bc_exclusion_rt_cdna.fasta`
- Primary mature miRNA: `UAGCACCAUCUGAAAUCGGUUA`
- RT-cDNA construct length: 66 nt
- Cas12a guide length: 25 nt (the guide length stored in both bundled Cas12a models)
- RPA primer length: 30 nt
- Guide-target activity model: bundled EasyDesign Cas12a classifier and regression SavedModels

## Command configuration

The local run used RPA mode, a 25 nt guide, 30 nt primers, zero tolerated guide mismatches, zero tolerated primer mismatches, and `miR-29b/29c` as a `--specific-against-fastas` input. The resulting output header is retained in `easydesign_mir29a_rpa.tsv`.

## Result

EasyDesign returned no complete targets:

```
Zero targets were found. The number of total primer pairs found was 36 and the number of them that were suitable (passing basic criteria, e.g., on length) was 0.
```

This is expected for the current construct. EasyDesign's complete-target search needs space for two 30 nt primers and one 25 nt guide within one non-overlapping design region. The 66 nt RT-cDNA construct cannot accommodate that approximately 85 nt minimum layout.

## Interpretation

This run does **not** invalidate the corrected v4 model. The v4 global top candidate, `CRRNA-CORR-008`, targets miR-29c and remains the corrected top result for that model's cross-family ranking. It is not a miR-29a assay reagent.

It also does **not** establish final miR-29a primers or a crRNA. Any sequence selected from a short construct by relaxing the 25 nt guide or 30 nt primer requirements would no longer be an EasyDesign-model-backed design.

## Required next construct

Before a valid EasyDesign complete-target run, create an experimentally supported cDNA/amplicon architecture of at least 85 nt with these features:

1. A validated miR-29a reverse-transcription strategy, such as stem-loop RT or a ligated/polyadenylated adapter workflow.
2. Two defined, non-target-derived amplification handles flanking the miRNA-derived segment.
3. A deliberately placed `TTTV` LbCas12a PAM in the final double-stranded RPA product, with orientation checked against the selected guide.
4. A 25 nt guide target that crosses the miRNA-derived region and includes the miR-29a/29c difference, rather than a guide confined to a universal handle.
5. Re-run EasyDesign with the same miR-29b/29c exclusion FASTA and then validate the selected guide, primer pair, and PAM orientation using synthetic miR-29a/b/c standards.

## Environment

- Python 3.8.20
- TensorFlow 2.3.0
- NumPy 1.18.5
- SciPy 1.5.2

The upstream EasyDesign file pins SciPy 1.4.1 for Linux. On Windows, TensorFlow 2.3.0's compatible Conda build requires SciPy 1.5.2; this is the only deviation from the upstream numerical stack. The EasyDesign core and its bundled models imported successfully before this run.
