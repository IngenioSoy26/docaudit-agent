# DocAudit Agent

Sistema multi-agente para la auditoría y extracción inteligente de documentos hipotecarios, fiscales y KYC.

## Estado actual (MVP)

Este repositorio ya incluye un MVP funcional que:

- Carga un esquema YAML desde `schemas/`.
- Ejecuta un pipeline mínimo: clasifica → extrae campos vía LLM local (Ollama) → devuelve JSON.
- Normaliza el resultado extraído (ej. fechas e importes a formatos consistentes) antes de validar.
- Valida el resultado extraído contra reglas del esquema (requeridos, min/max, etc.).
- Expone un endpoint FastAPI para probar la extracción.
- Incluye una prueba unitaria básica.

## Estructura del Repositorio

- `schemas/`: Esquemas YAML por caso de uso.
- `agents/`: Un módulo Python por agente.
- `core/`: Orquestador LangGraph + schema loader.
- `api/`: FastAPI endpoints.
- `ui/`: Streamlit app + componentes Stitch.
- `tests/`: Pruebas unitarias e integración.
- `data/sample_docs/`: Documentos anonimizados de prueba.
- `notebooks/`: Jupyter Notebooks de exploración.
- `docs/`: Documentación técnica.

## Instalación (Windows + PowerShell)

1) Clona el repo dentro de tu carpeta de trabajo:

```powershell
cd "C:\Users\gusta\Desktop\Maestria\0. TFM\DocAudit Agent"
git clone https://github.com/IngenioSoy26/docaudit-agent.git
cd ".\docaudit-agent"
```

2) Crea el entorno virtual dentro del repo:

```powershell
py -m venv .venv
```

3) Activa el entorno virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

4) Instala dependencias:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verificación de Ollama

1) Asegúrate de tener Ollama abierto (icono en la bandeja del sistema).
2) Verifica que los modelos existan:

```powershell
ollama list
```

Modelos esperados:

- `qwen2.5vl:7b`
- `mistral:7b-instruct`

## Ejecutar pruebas

```powershell
python -m pytest -q
```

## Ejecutar la API (FastAPI)

En una terminal con `(.venv)` activo:

```powershell
uvicorn api.main:app --reload
```

- Healthcheck: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs

Ejemplo de petición:

```powershell
curl -X POST "http://127.0.0.1:8000/extract" -H "Content-Type: application/json" -d "{ \"text\": \"Escritura de hipoteca. Titular: Juan Pérez García. Entidad financiera: Banco X. Importe del préstamo: 120000. Fecha de firma: 2024-03-10.\" }"
```

## Componentes implementados (MVP)

- `schemas/hipotecario.yaml`: esquema inicial de ejemplo.
- `core/schema_loader.py`: carga y valida YAML con Pydantic.
- `agents/classifier.py`: clasificador heurístico mínimo.
- `agents/extractor.py`: extractor con LLM local (Ollama) que devuelve JSON.
- `core/normalizer.py`: normalización de tipos/formato (fechas, números, strings).
- `core/orchestrator.py`: orquesta clasificador + esquema + extractor.
- `core/validator.py`: validación de campos extraídos contra el esquema.
- `api/main.py`: endpoints `/health` y `/extract`.

## Ejecutar la UI (Streamlit)

En una terminal con `(.venv)` activo:

```powershell
streamlit run ui/app.py
```
