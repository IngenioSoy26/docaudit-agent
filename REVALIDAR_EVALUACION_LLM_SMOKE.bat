@echo off
setlocal enableextensions enabledelayedexpansion

rem Prueba corta del runner oficial de evaluacion con LLM.
rem Ejecuta solo 6 muestras para verificar que el pipeline arranca y genera artefactos.

cd /d "%~dp0"

echo.
echo ============================================================
echo   DocAudit Agent - Smoke Test de Revalidacion LLM
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
    echo.
    pause
    exit /b 1
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"

set "OUT_CSV=reports\evaluation_results_llm_smoke6_%STAMP%.csv"
set "OUT_JSON=reports\evaluation_results_llm_smoke6_%STAMP%.summary.json"
set "OUT_MISMATCHES=reports\evaluation_results_llm_smoke6_%STAMP%_mismatches"

echo Interprete detectado:
echo   %PY_CMD%
echo.
echo Esta prueba ejecuta:
echo   --backend llm
echo   --limit 6
echo   sobre el corpus estructurado real, sin degradacion sintetica adicional
echo.
echo Salidas esperadas:
echo   %OUT_CSV%
echo   %OUT_JSON%
echo   %OUT_MISMATCHES%
echo.
echo Iniciando smoke test...
echo.

call %PY_CMD% -m tools.evaluate --backend llm --limit 6 --out "%OUT_CSV%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ============================================================
    echo   ERROR EN EL SMOKE TEST
    echo ============================================================
    echo Codigo de salida: %EXIT_CODE%
    echo.
    pause
    exit /b %EXIT_CODE%
)

echo ============================================================
echo   SMOKE TEST FINALIZADO
echo ============================================================
echo.

if exist "%OUT_CSV%" (
    echo CSV generado:
    echo   %OUT_CSV%
) else (
    echo AVISO: no se encontro el CSV esperado.
)

if exist "%OUT_JSON%" (
    echo Resumen generado:
    echo   %OUT_JSON%
) else (
    echo AVISO: no se encontro el summary.json esperado.
)

if exist "%OUT_MISMATCHES%" (
    echo Mismatches generados:
    echo   %OUT_MISMATCHES%
) else (
    echo AVISO: no se encontro la carpeta de mismatches esperada.
)

echo.
echo Si este smoke test funciona, ya puedes lanzar:
echo   REVALIDAR_EVALUACION_LLM.bat
echo.
pause
exit /b 0
