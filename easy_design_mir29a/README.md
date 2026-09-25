# miR-29 RT-RPA-Cas12a design input

This directory contains the reconstructed input architecture for an EasyDesign-assisted miR-29a assay.

## Design basis

- Primary target: `miR-29a-3p` (`UAGCACCAUCUGAAAUCGGUUA`).
- Exclusion controls: `miR-29b-3p` and `miR-29c-3p`.
- The guide ranking in `v4_audited_2.ipynb` remains a separate model output. Its corrected global top candidate, `CRRNA-CORR-008`, targets miR-29c, not the miR-29a assay target.
- Mature miRNA sequences are too short for direct Cas12a/RPA design. The FASTA file therefore represents stem-loop RT cDNA constructs. The shared stem-loop adapter is followed by each target's cDNA sequence.
- The 3-prime six bases of the proposed miR-29a RT primer (`TAACCG`) are complementary to the miR-29a 3-prime terminal sequence (`CGGUUA`). This is a proposed architecture, not a validated wet-lab reagent.

## Required validation before ordering

1. Confirm the stem-loop RT chemistry and primer sequence with the supervising laboratory.
2. Add the LbCas12a-compatible `TTTV` PAM in an RPA primer 5-prime overhang only after guide orientation is verified from the EasyDesign output.
3. Test all selected crRNAs against synthetic miR-29a, miR-29b, and miR-29c standards. Do not infer selectivity from the in-silico rank alone.
4. Screen RPA primer dimers and validate amplicon size experimentally.

## Platform note

EasyDesign's Cas12a model uses a 25 nt guide. Its official environment file is Linux-oriented; on Windows, TensorFlow 2.3 requires SciPy 1.5.2 rather than the upstream SciPy 1.4.1 pin.
