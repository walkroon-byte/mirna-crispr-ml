@echo off
setlocal EnableDelayedExpansion
cd /d D:\Evelyn\mirna-crispr-ml

cls
echo ============================================================
echo    Installing ViennaRNA via pip
echo ============================================================
echo.
echo IMPORTANT: Do NOT type anything in this window!
echo.
echo ============================================================
echo.

C:\Users\Administrator\anaconda3\Scripts\pip.exe install ViennaRNA

echo.
echo ============================================================
echo    Testing ViennaRNA installation
echo ============================================================
echo.

C:\Users\Administrator\anaconda3\python.exe -c "import RNA; print('ViennaRNA version:', RNA.__version__ if hasattr(RNA, '__version__') else 'installed'); print('Test fold result:', RNA.fold('GGGGAAAACCCC'))"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo    SUCCESS! ViennaRNA is now installed
    echo ============================================================
    echo.
    echo Next step: Run start_training.bat to train the model
    echo.
) else (
    echo.
    echo ============================================================
    echo    Installation may have failed
    echo ============================================================
    echo.
    echo Trying alternative method...
    echo.
    C:\Users\Administrator\anaconda3\python.exe -m pip install ViennaRNA
)

echo.
echo ============================================================
echo Press any key to exit (this time it's safe)
pause >nul
