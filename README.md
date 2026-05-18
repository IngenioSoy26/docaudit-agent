# DocAudit Agent

Sistema multi-agente para la auditoría y extracción inteligente de documentos hipotecarios, fiscales y KYC.

## Estado actual (plataforma)

Este repositorio ya incluye una plataforma funcional que:

- Orquesta 5 agentes en un grafo LangGraph: clasificador → extractor → normalizador → validador → auditor.
- Extrae campos con LLM local (Ollama) devolviendo estructura por campo (valor, confianza, evidencia, página).
- Soporta PDFs nativos (texto embebido) y PDFs escaneados (visión con Qwen2.5-VL).
- Normaliza y valida contra reglas declarativas (required, min/max, regex, enum).
- Genera un informe de auditoría (JSON + Markdown) con score y evaluación de reglas de decisión.
- Robustez frente a salidas “sucias” del LLM (comentarios, comas finales, fences, números malformados como `5.489.11`).
- Evidencias con RAG (ChromaDB) con caché y persistencia por documento (hash).
- Expone API (FastAPI) y UI (Streamlit).
- Incluye pruebas unitarias (schema/normalización/validación/auditoría/grafo).

## Arquitectura (resumen)

Entrada: texto (o PDF → extracción de texto) + esquema YAML.

Grafo (LangGraph) con estado compartido:

1) Agente Clasificador: selecciona el caso de uso (schema).
2) Agente Extractor: extrae campos (LLM local) en JSON.
3) Agente Normalizador: homogeniza tipos/formato (fechas/importes/strings).
4) Agente Validador: aplica reglas declarativas del esquema.
5) Agente Auditor: produce informe (JSON + Markdown) + score y reglas de decisión.

## Estructura del Repositorio

- `schemas/`: Esquemas YAML por caso de uso.
- `agents/`: Un módulo Python por agente.
- `core/`: Orquestador LangGraph + schema loader.
- `api/`: FastAPI endpoints.
- `ui/`: Streamlit app + componentes Stitch.
- `tests/`: Pruebas unitarias e integración.
- `data/sample_docs/`: Documentos anonimizados de prueba (por caso de uso).
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
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Verificación de Ollama

1) Asegúrate de tener Ollama abierto (icono en la bandeja del sistema) y que responda en `http://localhost:11434`.
2) Verifica que los modelos existan:

```powershell
ollama list
```

Modelos esperados:

- `qwen2.5vl:7b`
- `llama3.2:3b` (texto por defecto + clasificación)
- `mistral:7b-instruct` (opcional; suele ser más lento en CPU)
- `nomic-embed-text` (embeddings para ChromaDB)

Si falta alguno:

```powershell
ollama pull qwen2.5vl:7b
ollama pull llama3.2:3b
ollama pull mistral:7b-instruct
ollama pull nomic-embed-text
```

## Configuración (variables de entorno)

Puedes ajustar el comportamiento sin tocar código:

- `LLM_BACKEND` (default: `local`) valores: `local` (Ollama) o `gpt4mini` (OpenAI)
- `OPENAI_API_KEY` (requerida si `LLM_BACKEND=gpt4mini`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_TEXT_MODEL` (default: `llama3.2:3b`)
- `OLLAMA_CLASSIFIER_MODEL` (default: `llama3.2:3b`)
- `OLLAMA_EMBEDDING_MODEL` (default: `nomic-embed-text`)
- `OLLAMA_TIMEOUT_S` (default: `600`)
- `OLLAMA_TEXT_NUM_PREDICT` (default: `256`)
- `OLLAMA_CLASSIFIER_NUM_PREDICT` (default: `64`)
- `ENABLE_PII_REDACTION` (default: `false`) aplica seudonimización básica de PII en el texto antes de extraer

## Ejecutar pruebas

```powershell
.\.venv\Scripts\python -m pytest -q
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

Para PDF:

- En Swagger (o cliente) usa `POST /extract_pdf` subiendo un archivo PDF.
- Parámetro `mode`: `auto` (intenta texto; si no hay, intenta visión) o `vision` (fuerza visión).

Endpoints compatibles con la Propuesta Técnica:

- `POST /upload` (PDF → doc_id)
- `POST /process` (doc_id o text → resultado del pipeline)
- `GET /report/{doc_id}` (último resultado cacheado por doc_id)

## Ejecutar la UI (Streamlit)

En una terminal con `(.venv)` activo:

```powershell
streamlit run ui/app.py
```

Uso recomendado en la UI:

1) Pega texto en “Texto de entrada” y pulsa “Ejecutar”, o sube un PDF.
2) Si el PDF no tiene texto embebido, deja activada la opción de visión (Qwen2.5-VL).
3) Revisa pestañas:
   - Extracción: raw + normalizado
   - Normalización: cambios aplicados
   - Validación: incidencias de reglas
   - Auditoría: informe en Markdown + JSON (score + reglas de decisión)

## Datos de prueba (Capítulo 6 — Código Fuente y Datos)

El repositorio incluye un corpus de prueba bajo `data/sample_docs/` siguiendo la estructura descrita en la propuesta técnica:

- `data/sample_docs/caso_uso_1_auditoria_hipotecaria/`: PDFs `contrato_hipoteca_esp_*.pdf` + `ground_truth_esp_*.json`.
- `data/sample_docs/caso_uso_2_auditoria_fiscal/`: PDFs `factura_fiscal_*.pdf` + `factura_fiscal_*.json`.
- `data/sample_docs/caso_uso_3_kyc_onboarding/`: PDFs `expediente_kyc_*.pdf` + `expediente_kyc_*.json`.

Los esquemas “canónicos” que usa el sistema están en `schemas/`:
- `schemas/credito_hipotecario.yaml`
- `schemas/auditoria_fiscal.yaml`
- `schemas/kyc_onboarding.yaml`

## Esquemas YAML

El sistema usa el formato extendido (empresa) con `caso_uso`, `documentos`, `campos`, `reglas_decision` e `informe`.

Esquemas del proyecto:

- [credito_hipotecario.yaml](schemas/credito_hipotecario.yaml)
- [auditoria_fiscal.yaml](schemas/auditoria_fiscal.yaml)
- [kyc_onboarding.yaml](schemas/kyc_onboarding.yaml)

Estructura:

- `caso_uso`, `version`, `descripcion`
- `documentos[]` con `tipo` y `campos[]` (`nombre`, `etiqueta`, `tipo_dato`, `requerido`, `patron`, `validacion`)
- `reglas_decision[]` (expresiones booleanas evaluadas en auditoría)
- `informe` (opciones de salida)

## Soporte PDF (MVP)

La UI permite subir un PDF “nativo” (con texto seleccionable). Al subirlo, se extrae el texto y se ejecuta el pipeline.

Si el PDF es escaneado y no tiene texto embebido, se intenta la ruta de visión con `qwen2.5vl:7b` (Ollama) para transcribir imágenes incrustadas.

Para PDFs nativos, si tienes instalada la librería `docling`, el sistema intenta usarla primero (layout → markdown). Si no está instalada, hace fallback a PyPDF.

## Componentes (mapa rápido)

- Clasificador: [classifier.py](agents/classifier.py)
- Extractor (LLM → JSON): [extractor.py](agents/extractor.py)
- Normalizador: [normalizer.py](core/normalizer.py)
- Validador: [validator.py](core/validator.py)
- Auditor (informe + score + reglas): [auditor.py](agents/auditor.py)
- Grafo LangGraph: [orchestrator.py](core/orchestrator.py)
- Loader de schemas: [schema_loader.py](core/schema_loader.py)
- PDF (texto/visión): [document_loader.py](core/document_loader.py)
- API: [main.py](api/main.py)
- UI: [app.py](ui/app.py)

## Checklist de ejecución (paso a paso)

1) Crear y activar `.venv`.
2) Instalar dependencias (`requirements.txt`).
3) Instalar y abrir Ollama.
4) Descargar modelos (`qwen2.5vl:7b`, `llama3.2:3b`, `nomic-embed-text`; `mistral:7b-instruct` opcional).
5) Ejecutar pruebas (`pytest`).
6) Levantar UI (`streamlit run ui/app.py`) o API (`uvicorn api.main:app --reload`).
7) Probar:
   - Texto pegado (cualquier caso) + “Ejecutar”.
   - PDF nativo (debería rellenar el texto automáticamente).
   - PDF escaneado (activar visión).

## Troubleshooting (rápido)

- `python` no reconocido en PowerShell: usa `.\.venv\Scripts\python ...` o `py -m ...`.
- `&&` no funciona en tu PowerShell: usa `;` para encadenar comandos.
- Error “could not connect to running instance” en Ollama: abre la app de Ollama y reintenta `ollama list`.
- Modelo de visión: el tag correcto es `qwen2.5vl:7b` (sin guion entre 5 y vl).
- `ReadTimeout` en extracción: sube `OLLAMA_TIMEOUT_S` (ej: 600) y/o usa `llama3.2:3b` como modelo de texto (más rápido en CPU).
- La terminal no para un proceso `python.exe`: mata el PID con `Stop-Process -Id <PID> -Force`.
- Reglas “NO EVALUABLE”: faltan campos en el contexto (ej: reglas hipotecarias requieren IRPF + extracto; si solo envías extracto, se marca como `REVISAR`).

## Stack recomendado (opcional)

Estas librerías no son obligatorias para correr el MVP, pero alinean el proyecto con la propuesta técnica:

```powershell
.\.venv\Scripts\python -m pip install docling marker-pdf chromadb ollama
```
