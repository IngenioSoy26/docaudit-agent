@echo off
setlocal enableextensions enabledelayedexpansion

rem Smoke final por tipo documental.
rem Objetivo:
rem - detectar el interprete correcto,
rem - ejecutar 1 caso por cada doc_type,
rem - verificar csv + summary + mismatches,
rem - reportar estado claro por doc_type.

cd /d "%~dp0"

echo.
echo ============================================================
echo   DocAudit Agent - Smoke Final por Doc Type
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

echo Interprete detectado:
echo   %PY_CMD%
echo.

set "OVERALL_EXIT=0"

call :run_smoke native_pdf
call :run_smoke scanned_blurry_pdf
call :run_smoke image_photo
call :run_smoke image_handwritten

echo.
echo ============================================================
echo   RESUMEN FINAL DEL SMOKE
echo ============================================================
echo.
call :print_status native_pdf
call :print_status scanned_blurry_pdf
call :print_status image_photo
call :print_status image_handwritten
echo.

if not "%OVERALL_EXIT%"=="0" (
    echo Se detectaron incidencias en uno o mas doc_types.
    echo Revisa el detalle anterior antes de lanzar la corrida completa.
    echo.
    pause
    exit /b %OVERALL_EXIT%
)

echo Todos los doc_types generaron los artefactos esperados.
echo Ya puedes continuar con la evaluacion completa.
echo.
pause
exit /b 0

:run_smoke
set "DOC_TYPE=%~1"
set "OUT_CSV=reports\smoke_%DOC_TYPE%_1.csv"
set "OUT_SUMMARY=reports\smoke_%DOC_TYPE%_1.summary.json"
set "OUT_MISMATCHES=reports\smoke_%DOC_TYPE%_1_mismatches"

set "STATUS_%DOC_TYPE%=PENDIENTE"
set "EXIT_%DOC_TYPE%=0"

if exist "%OUT_CSV%" del /q "%OUT_CSV%" >nul 2>nul
if exist "%OUT_SUMMARY%" del /q "%OUT_SUMMARY%" >nul 2>nul
if exist "%OUT_MISMATCHES%" rmdir /s /q "%OUT_MISMATCHES%" >nul 2>nul

echo ------------------------------------------------------------
echo Ejecutando smoke para: %DOC_TYPE%
echo ------------------------------------------------------------
echo Comando:
echo   %PY_CMD% -m tools.evaluate --backend llm --only-doc-type %DOC_TYPE% --limit 1 --out "%OUT_CSV%"
echo.

call %PY_CMD% -m tools.evaluate --backend llm --only-doc-type %DOC_TYPE% --limit 1 --out "%OUT_CSV%"
set "CMD_EXIT=%ERRORLEVEL%"
set "EXIT_%DOC_TYPE%=%CMD_EXIT%"

set "STATUS_MSG="
if exist "%OUT_SUMMARY%" (
    set "HAS_SUMMARY=1"
) else (
    set "HAS_SUMMARY=0"
)

if exist "%OUT_MISMATCHES%" (
    set "HAS_MISMATCHES=1"
) else (
    set "HAS_MISMATCHES=0"
)

if not "%CMD_EXIT%"=="0" (
    set "STATUS_MSG=ERROR_EJECUCION"
    set "OVERALL_EXIT=1"
)

if "!HAS_SUMMARY!"=="0" (
    if defined STATUS_MSG (
        set "STATUS_MSG=!STATUS_MSG! + FALTA summary"
    ) else (
        set "STATUS_MSG=FALTA summary"
    )
    set "OVERALL_EXIT=1"
)

if "!HAS_MISMATCHES!"=="0" (
    if defined STATUS_MSG (
        set "STATUS_MSG=!STATUS_MSG! + FALTA mismatches"
    ) else (
        set "STATUS_MSG=FALTA mismatches"
    )
    set "OVERALL_EXIT=1"
)

if not defined STATUS_MSG set "STATUS_MSG=OK"

set "STATUS_%DOC_TYPE%=!STATUS_MSG!"

echo Estado %DOC_TYPE%:
echo   !STATUS_MSG!
echo.
goto :eof

:print_status
set "DOC_TYPE=%~1"
call echo %DOC_TYPE%: %%STATUS_%DOC_TYPE%%%
call echo   Exit code: %%EXIT_%DOC_TYPE%%%
call echo   CSV: reports\smoke_%DOC_TYPE%_1.csv
call echo   Summary: reports\smoke_%DOC_TYPE%_1.summary.json
call echo   Mismatches: reports\smoke_%DOC_TYPE%_1_mismatches
echo.
goto :eof
