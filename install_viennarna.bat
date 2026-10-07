@echo off
cd /d D:\Evelyn\mirna-crispr-ml

echo ============================================================
echo Installing ViennaRNA via pip
echo ============================================================
echo.

C:\Users\Administrator\anaconda3\Scripts\pip.exe install ViennaRNA

echo.
echo ============================================================
echo Testing ViennaRNA installation
echo ============================================================
echo.

C:\Users\Administrator\anaconda3\python.exe -c "import RNA; print('ViennaRNA version:', RNA.__version__ if hasattr(RNA, '__version__') else 'installed'); print('Test:', RNA.fold('GGGGAAAACCCC'))"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! ViennaRNA is now installed
    echo ============================================================
    echo.
    echo You can now run: start_training.bat
) else (
    echo.
    echo ============================================================
    echo Installation failed. See error above.
    echo ============================================================
)

echo.
pause
