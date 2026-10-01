#!/bin/bash
# Run EasyDesign for miR-29a selective detection
# Design version: v2 (110 nt construct)
# Date: 2026-10-01

echo "=========================================="
echo "EasyDesign Run Script for miR-29a v2"
echo "=========================================="
echo ""

# Check if we're in the correct directory
if [ ! -d "tools/EasyDesign" ]; then
    echo "ERROR: tools/EasyDesign directory not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Navigate to EasyDesign directory
cd tools/EasyDesign

echo "Step 1: Verifying input files..."
INPUT_FILE="../../data/rt_cdna_designs/mir29a_extended_rt_cdna_v2.fasta"
EXCLUDE_FILE="../../data/rt_cdna_designs/mir29bc_extended_rt_cdna_v2.fasta"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

if [ ! -f "$EXCLUDE_FILE" ]; then
    echo "ERROR: Exclusion file not found: $EXCLUDE_FILE"
    exit 1
fi

echo "✓ Input file found: $INPUT_FILE"
echo "✓ Exclusion file found: $EXCLUDE_FILE"
echo ""

# Create output directory if it doesn't exist
mkdir -p ../../results/easydesign_v2

echo "Step 2: Running EasyDesign..."
echo "Parameters:"
echo "  - Mode: RPA"
echo "  - Guide length: 25 nt"
echo "  - Primer length: 30 nt"
echo "  - Max guide mismatches: 0"
echo "  - Max primer mismatches: 0"
echo "  - Construct length: 110 nt"
echo ""

# Run EasyDesign
python bin/design.py \
    --input "$INPUT_FILE" \
    --specific-against-fastas "$EXCLUDE_FILE" \
    --guide-length 25 \
    --primer-length 30 \
    --mode rpa \
    --output ../../results/easydesign_v2/mir29a_110nt_output.tsv \
    --max-primer-mismatches 0 \
    --max-guide-mismatches 0

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "EasyDesign run completed successfully!"
    echo "Output saved to: results/easydesign_v2/mir29a_110nt_output.tsv"
    echo ""
    echo "Next steps:"
    echo "1. Review the output file to see recommended crRNA and RPA primers"
    echo "2. Check the activity scores and select top candidates"
    echo "3. Order the recommended sequences from a synthesis provider"
    echo "4. Proceed with wet-lab validation"
else
    echo "EasyDesign run failed with exit code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "- Python environment not activated (check conda env)"
    echo "- Missing dependencies (TensorFlow, NumPy, etc.)"
    echo "- Still not enough space in 110 nt construct"
    echo ""
    echo "If still getting 0 targets, see troubleshooting in:"
    echo "  data/rt_cdna_designs/easydesign_design_params_v2.json"
fi
echo "=========================================="

# Return to project root
cd ../..
