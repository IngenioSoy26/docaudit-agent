@echo off
setlocal
cd /d "%~dp0"

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" goto run_ui

echo No se encontro un entorno virtual valido.
echo.
echo Se esperaba esta ruta:
echo   %PROJECT_DIR%.venv\Scripts\python.exe
echo.
echo Crea o reconstruye la .venv dentro de esta misma carpeta del proyecto.
pause
exit /b 1

:run_ui
echo Iniciando UI de DocAudit Agent...
"%PYTHON_EXE%" -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501
pause
