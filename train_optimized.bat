@echo off
cd /d D:\Evelyn\mirna-crispr-ml

cls
echo ============================================================
echo    Training v7_optimized Model
echo ============================================================
echo.
echo Strategy: Enhanced regularization to compensate for missing data
echo Data: 9,894 samples (excluding original 80)
echo Features: 20 (with ViennaRNA)
echo.
echo Improvements:
echo   - Stronger L2 regularization
echo   - Higher dropout rates (0.3-0.4)
echo   - More training epochs (150)
echo   - Lower learning rate (0.0005)
echo.
echo Expected time: 20-35 minutes
echo.
echo ============================================================
echo.
echo Press any key to start training...
pause >nul

echo.
echo Starting optimized training...
echo.

C:\Users\Administrator\anaconda3\python.exe scripts\train_model_v7_optimized.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo Training completed successfully!
    echo ============================================================
    echo.
    echo Check results in: models\v7_public_optimized\
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
