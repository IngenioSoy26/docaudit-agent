@echo off
setlocal enableextensions enabledelayedexpansion

rem Evaluación completa de DocAudit Agent.
rem Procesa 120 documentos, genera logs, gráficos y PDFs.

cd /d "%~dp0"

echo.
echo ============================================================
echo   DocAudit Agent - Evaluación Completa
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
    echo ERROR: No se encontró ningún intérprete de Python.
    echo.
    echo Opciones esperadas:
    echo   - .venv\Scripts\python.exe
    echo   - venv\Scripts\python.exe
    echo   - comando ^`py^`
    echo   - comando ^`python^`
    echo.
    pause
    exit /b 1
)

echo Intérprete detectado:
echo   %PY_CMD%
echo.
echo NOTA:
echo   Este BAT ejecuta la evaluación completa:
echo   1. Procesamiento de 120 documentos del corpus
echo   2. Generación de logs en test_data\execution_logs
echo   3. Generación de gráficos en temp_plots
echo   4. Generación de informes PDF en reports
echo.
echo ADVERTENCIA: Esto tardará varias horas!
echo.

choice /C SN /M "¿Deseas continuar"
if errorlevel 2 exit /b 0

echo.
echo Iniciando evaluación...
echo.

call %PY_CMD% run_full_evaluation.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ============================================================
    echo   ERROR EN LA EVALUACIÓN
    echo ============================================================
    echo Código de salida: %EXIT_CODE%
    echo.
    echo Revisa la salida mostrada arriba.
    echo.
    pause
    exit /b %EXIT_CODE%
)

echo ============================================================
echo   EVALUACIÓN COMPLETA!
echo ============================================================
echo.
echo Revisa los resultados en:
echo   - test_data\execution_logs: logs de ejecución
echo   - temp_plots: gráficos generados
echo   - reports: informes PDF y CSVs
echo.
echo Puedes abrir la aplicación Streamlit para ver los resultados en:
echo   http://localhost:8501
echo.
pause
exit /b 0
