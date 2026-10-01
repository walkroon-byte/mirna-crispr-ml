# RT-cDNA Design v2 Summary

## Quick Reference

### Construct Specifications
- **Total length:** 110 nt (meets ≥100 nt requirement ✅)
- **Architecture:** 5'-[38nt adapter]-[30nt miRNA]-[42nt adapter]-3'
- **Design space margin:** 25 nt above minimum 85 nt requirement

### Files Created
1. `mir29a_extended_rt_cdna_v2.fasta` - Primary input for EasyDesign
2. `mir29bc_extended_rt_cdna_v2.fasta` - Exclusion targets (29b and 29c)
3. `rt_primers_v2.txt` - RT primer sequences and protocols
4. `easydesign_design_params_v2.json` - Complete design parameters
5. `run_easydesign_v2.sh` - Bash script to run EasyDesign
6. `RT_cDNA_DESIGN_v2.md` - Full design documentation

### Key Sequences

#### miR-29a construct (110 nt):
```
GTCGTATCCAGTGCAGGGTCCGAGGTATTCGACTAGCTAATCGTGGTAACCGATTTCAGATGGTGCTAAGTGCGTACTCGAGGCATCGATCAGTACGGATCG
```

#### RT Primer (Universal):
```
5'-AGTGCGTACTCGAGGCATCGATCAGTACGGATCG-3' (35 nt)
```

## How to Run EasyDesign

### Option 1: Use the provided script (Linux/Mac/WSL)
```bash
cd D:/Evelyn/mirna-crispr-ml
bash data/rt_cdna_designs/run_easydesign_v2.sh
```

### Option 2: Run manually (Windows PowerShell)
```powershell
cd D:\Evelyn\mirna-crispr-ml\tools\EasyDesign

python bin/design.py `
  --input ../../data/rt_cdna_designs/mir29a_extended_rt_cdna_v2.fasta `
  --specific-against-fastas ../../data/rt_cdna_designs/mir29bc_extended_rt_cdna_v2.fasta `
  --guide-length 25 `
  --primer-length 30 `
  --mode rpa `
  --output ../../results/easydesign_v2/mir29a_110nt_output.tsv `
  --max-primer-mismatches 0 `
  --max-guide-mismatches 0
```

### Option 3: Python script (if conda env is active)
```python
import os
os.chdir('tools/EasyDesign')
os.system('python bin/design.py --input ../../data/rt_cdna_designs/mir29a_extended_rt_cdna_v2.fasta --specific-against-fastas ../../data/rt_cdna_designs/mir29bc_extended_rt_cdna_v2.fasta --guide-length 25 --primer-length 30 --mode rpa --output ../../results/easydesign_v2/mir29a_110nt_output.tsv --max-primer-mismatches 0 --max-guide-mismatches 0')
```

## Expected Output

If successful, you should see:
```
Found X complete targets
Output saved to: results/easydesign_v2/mir29a_110nt_output.tsv
```

The output file will contain:
- Recommended crRNA guide sequence (25 nt)
- Forward RPA primer (30 nt)
- Reverse RPA primer (30 nt)
- Activity score (predicted by EasyDesign model)
- PAM location

## What to Do Next

### If you get ≥1 target ✅
1. **Review the output TSV file** - look at activity scores
2. **Select top candidate** - highest activity + specificity
3. **Order reagents:**
   - Synthetic miR-29a/b/c standards (~$900)
   - RT primers (~$30)
   - crRNA guides (~$300)
   - RPA primers (~$100)
4. **Start wet-lab validation** as described in RT_cDNA_DESIGN_v2.md

### If you still get 0 targets ❌
**Don't panic!** Try these options in order:

1. **Relax the mismatch tolerance:**
   ```bash
   --max-guide-mismatches 1  # Allow 1 mismatch
   ```

2. **Extend to 120 nt:**
   - I can help you design a longer construct

3. **Try a different adapter:**
   - Adjust GC content or sequence composition

4. **Manual design:**
   - Skip EasyDesign, manually design guide and primers
   - Use your trained v4 model to predict activity

## Comparison: v1 vs v2

| Feature | v1 (66 nt) | v2 (110 nt) |
|---------|-----------|-------------|
| Total length | 66 nt | 110 nt |
| EasyDesign space | ❌ Insufficient | ✅ Sufficient |
| Design margin | -19 nt | +25 nt |
| Expected targets | 0 | ≥1 |
| Adapter strategy | Minimal | Universal adapters |

## Cost Estimate

| Item | Cost (USD) |
|------|-----------|
| Synthetic miRNA standards (×3) | $900 |
| RT primers | $30 |
| crRNA guides (×3) | $300 |
| RPA primers (×6) | $100 |
| **Total** | **$1,330** |

*Note: This excludes enzyme costs (SuperScript IV, LbCas12a, RPA kit)*

## Timeline Estimate

- **Ordering reagents:** 1-2 weeks
- **RT optimization:** 1 week
- **RPA optimization:** 1 week
- **Cas12a assay setup:** 1 week
- **Validation testing:** 2 weeks
- **Total:** ~6 weeks

## Questions?

See the full documentation in `RT_cDNA_DESIGN_v2.md` for:
- Detailed design rationale
- RT protocol
- Troubleshooting guide
- Quality control methods
- Cross-reactivity testing plan

---

**Design completed:** 2026-10-01  
**All files location:** `D:\Evelyn\mirna-crispr-ml\data\rt_cdna_designs\`
