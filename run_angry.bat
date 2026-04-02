@echo off
echo ============================================================
echo  MagicFace -- Angry Expression
echo  Dominant: 100 images  |  Submissive: 150 images
echo ============================================================
echo.

call C:\Users\faris3\AppData\Local\miniconda3\Scripts\activate.bat magicface

cd /d "%~dp0"

echo [1/2] Processing dominant (100 images)...
python run_batch.py --expression angry --categories dominant --n_images 100

echo.
echo [2/2] Processing submissive (150 images)...
python run_batch.py --expression angry --categories submissive --n_images 150

echo.
echo Done. Press any key to close.
pause >nul
