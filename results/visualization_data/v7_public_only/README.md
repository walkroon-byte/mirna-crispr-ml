# v7_public_only visualization data

This directory is the handoff between model production and publication plotting.

1. `scripts/export_v7_public_only_visualization_data.py` reads the saved model metrics and test predictions, validates residuals, derives ranks/errors/calibration tables, and writes the CSV/JSON data package.
2. `scripts/plot_v7_public_only_from_data.py` reads only this directory and renders the publication figures into `results/figures/v7_public_only_from_data/`.
3. Edit the plotting script for colors, fonts, panel layout, labels, or annotations. Re-run the plotting stage without loading a model or touching training data.

The `manifest.json` records the schema version, source artifact hashes, row counts, and processing order. The current historical v7 artifact contains a training-history PNG but no numeric history; future training runs now save `v7_public_only_training_history.json`, which the export and plotting stages will handle automatically.

Commands from the repository root:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' scripts\export_v7_public_only_visualization_data.py
& 'C:\ProgramData\anaconda3\python.exe' scripts\plot_v7_public_only_from_data.py
```
