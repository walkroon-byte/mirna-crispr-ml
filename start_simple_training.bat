@echo off
cd /d D:\Evelyn\mirna-crispr-ml

cls
echo ============================================================
echo    Training v7 Simple Model (No ViennaRNA)
echo ============================================================
echo.
echo This version uses:
echo   - 10 simplified features (no RNA structure)
echo   - 9,894 samples (public datasets only)
echo   - Deep learning: CNN + Bi-LSTM
echo.
echo Expected time: 15-30 minutes
echo.
echo IMPORTANT: Do NOT close this window during training
echo.
echo ============================================================
echo.
echo Press any key to start training...
pause >nul

echo.
echo Starting training...
echo.

C:\Users\Administrator\anaconda3\python.exe scripts\train_model_v7_simple.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo Training completed successfully!
    echo ============================================================
    echo.
    echo Results saved in: models\v7_public_only_simple\
    echo.
    echo Key files:
    echo   - v7_simple_best_model.h5
    echo   - v7_simple_metrics.json
    echo.
) else (
    echo.
    echo ============================================================
    echo Training failed! Error code: %ERRORLEVEL%
    echo ============================================================
)

echo.
echo Press any key to exit...
pause >nul
