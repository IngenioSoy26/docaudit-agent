@echo off
setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo   DocAudit Agent - MAX Accuracy (scanned_blurry_pdf)
echo ============================================================
echo.
echo Carpeta del proyecto:
echo   %CD%
echo.

set "PY_CMD="

if exist ".venv\Scripts\python.exe" set "PY_CMD=.venv\Scripts\python.exe"
if not defined PY_CMD if exist "venv\Scripts\python.exe" set "PY_CMD=venv\Scripts\python.exe"

if not defined PY_CMD (
    where py >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py"
)

if not defined PY_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo ERROR: No se encontro ningun interprete de Python.
    pause
    exit /b 1
)

echo Interprete detectado:
echo   %PY_CMD%
echo.

echo Ajustando modo MAX accuracy para borrosos...
set "SCANNED_PDF_TEXT_MODE=ocr_only"
set "ENABLE_SCANNED_PDF_VISION_FALLBACK=true"
set "ENABLE_VISION_PAGE_SUBPROCESS=true"
set "VISION_PAGE_TIMEOUT_S=180"
set "EASYOCR_PAGE_MAX_DIM=384"
set "EASYOCR_PAGE_JPEG_QUALITY=70"
set "VISION_PAGE_MAX_DIM=384"
set "VISION_PAGE_JPEG_QUALITY=70"
set "OLLAMA_VISION_NUM_PREDICT=256"

echo.
echo Ejecutando evaluacion AISLADA (30 borrosos)...
echo.

call %PY_CMD% -m tools.evaluate_isolated --backend llm --only-doc-type scanned_blurry_pdf --out reports\evaluation_results_llm_scanned_blurry_pdf_maxacc.csv
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ============================================================
    echo   ERROR EN LA EVALUACION (scanned_blurry_pdf)
    echo ============================================================
    echo Codigo de salida: %EXIT_CODE%
    pause
    exit /b %EXIT_CODE%
)

echo ============================================================
echo   LISTO
echo ============================================================
echo.
echo Resultados:
echo   - reports\evaluation_results_llm_scanned_blurry_pdf_maxacc.csv
echo   - reports\evaluation_results_llm_scanned_blurry_pdf_maxacc.summary.json
echo   - reports\evaluation_results_llm_scanned_blurry_pdf_maxacc_mismatches\
echo.
pause
exit /b 0

