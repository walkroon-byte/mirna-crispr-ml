@echo off
cd /d D:\Evelyn\mirna-crispr-ml

echo ============================================================
echo    Installing Required Python Packages
echo ============================================================
echo.
echo This will install:
echo   - tensorflow
echo   - scikit-learn
echo   - pandas
echo   - numpy
echo   - scipy
echo   - matplotlib
echo   - ViennaRNA (if available)
echo.
echo This may take 5-10 minutes...
echo.
echo ============================================================
echo.
pause

echo Installing packages...
echo.

C:\Users\Administrator\anaconda3\Scripts\pip.exe install tensorflow scikit-learn pandas numpy scipy matplotlib

echo.
echo ============================================================
echo Checking installations...
echo ============================================================
echo.

C:\Users\Administrator\anaconda3\python.exe -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)"
C:\Users\Administrator\anaconda3\python.exe -c "import sklearn; print('scikit-learn installed')"
C:\Users\Administrator\anaconda3\python.exe -c "import pandas; print('pandas installed')"

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Now you can run: start_training.bat
echo.
pause
