@echo off
echo ============================================================
echo  MagicFace -- Angry Expression (Dominant + Submissive)
echo ============================================================
echo.

call C:\Users\faris3\AppData\Local\miniconda3\Scripts\activate.bat magicface

cd /d "%~dp0"

python run_batch.py --expression angry --categories dominant submissive --force

echo.
echo Done. Press any key to close.
pause >nul
