@echo off
setlocal
cd /d "%~dp0"

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" goto run_api

echo No se encontro un entorno virtual valido.
echo.
echo Se esperaba esta ruta:
echo   %PROJECT_DIR%.venv\Scripts\python.exe
echo.
echo Crea o reconstruye la .venv dentro de esta misma carpeta del proyecto.
pause
exit /b 1

:run_api
echo Iniciando API de DocAudit Agent...
"%PYTHON_EXE%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
pause
