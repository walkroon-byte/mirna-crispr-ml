@echo off
cd /d D:\Evelyn\mirna-crispr-ml

cls
echo ============================================================
echo    Training v7_public_only Model
echo ============================================================
echo.
echo Data: 9,894 samples (public datasets only)
echo Time: 15-30 minutes
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

C:\Users\Administrator\anaconda3\python.exe scripts\train_model_v7_public_only.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo Training completed successfully!
    echo ============================================================
    echo.
    echo Press any key to view results...
    pause >nul

    C:\Users\Administrator\anaconda3\python.exe check_training_status.py
) else (
    echo.
    echo ============================================================
    echo Training failed! Error code: %ERRORLEVEL%
    echo ============================================================
)

echo.
echo Press any key to exit...
pause >nul
