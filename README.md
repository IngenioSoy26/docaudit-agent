# DocAudit Agent — Manual de instalación y uso

DocAudit Agent es una plataforma **multi‑agente** para **extraer**, **normalizar**, **validar** y **auditar** documentos (hipotecarios, fiscales y KYC). Está pensada para convertir documentos en **datos estructurados y trazables** que apoyen el **análisis** y la **toma de decisiones**, mostrando además evidencias y reglas evaluadas.

El sistema **no sustituye a un analista** ni debe usarse como “decisión crediticia automática”. Genera resultados, evidencias y alertas para revisión humana.

---

## 1) Qué hace (en 60 segundos)

- Procesa texto o PDFs (nativos o escaneados) y extrae campos definidos en **YAML**.
- Normaliza formatos (fechas, importes, booleanos) y valida reglas declarativas.
- Evalúa reglas de decisión (auditoría) de forma segura y genera informe (JSON/Markdown).
- Permite evaluar calidad con métricas (exact match, precisión/recall/F1, latencia, RAM) sobre un corpus con ground truth.

---

## 2) Arquitectura (visión general)

Pipeline (LangGraph) con estado compartido:

1) Clasificador → selecciona el caso de uso (schema).
2) Extractor → produce JSON de campos.
3) Normalizador → homogeniza tipos/formato.
4) Validador → aplica reglas declarativas del schema.
5) Auditor → evalúa reglas de decisión y genera informe.

---

## 3) Roles de modelos/tecnologías (qué hace cada uno)

- **LLaMA 3.2 3B (Ollama)**: extracción estructurada (texto → JSON) y tareas textuales (clasificación).
- **Qwen2.5‑VL (Ollama)**: ruta visual para documentos escaneados o con componente visual.
- **Docling (opcional)**: extracción más rica en PDFs nativos (layout → texto/markdown) cuando está disponible.
- **nomic-embed-text (Ollama)**: embeddings para recuperación de evidencias.
- **ChromaDB**: almacén vectorial local persistente para evidencias/RAG.
- **Streamlit**: interfaz web para el usuario.
- **FastAPI**: API REST para integración.

---

## 4) Requisitos (para cualquier equipo)

Mínimos recomendados:
- Python **3.10+**
- Git (solo si vas a clonar desde GitHub)
- Ollama instalado y ejecutándose en `http://localhost:11434` (requerido para extracción con LLM y visión)

Recomendación de recursos:
- CPU moderna y al menos **8 GB de RAM** (mejor 16 GB si usarás visión con Qwen2.5‑VL).

---

## 5) Instalación (paso a paso)

### 5.1) Clonar el repositorio

```bash
git clone https://github.com/IngenioSoy26/docaudit-agent.git
cd docaudit-agent
```

### 5.2) Crear entorno virtual e instalar dependencias

#### Windows (PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux (Terminal)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 6) Ollama (instalación y modelos)

1) Instala Ollama desde su instalador oficial y confirma que el servicio está activo.
2) Verifica conexión:

```bash
ollama list
```

3) Descarga los modelos necesarios:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5vl:7b
ollama pull nomic-embed-text
```

---

## 7) Ejecutar el sistema

### 7.1) Ejecutar pruebas

```bash
python -m pytest -q
```

### 7.2) Ejecutar UI (Streamlit)

```bash
python -m streamlit run ui/app.py
```

Abre en el navegador la URL que Streamlit imprime en consola (normalmente `http://127.0.0.1:8501`).

### 7.3) Ejecutar API (FastAPI)

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

---

## 8) Uso de la UI (manual)

### 8.1) Procesar un documento (1 PDF)

1) Sube un PDF.
2) El sistema mostrará un resumen con:
   - Documento analizado: 1
   - Folios: X
   - Método de extracción (pypdf/docling/vision)
3) Pulsa “Ejecutar análisis”.
4) Revisa:
   - **Resumen**: validez, incidencias, confianza y reglas.
   - **Extracción / Normalización / Validación / Auditoría**: vista explicada y JSON desplegable.

### 8.2) Procesar un expediente (varios PDFs)

1) Sube varios PDFs.
2) Se mostrará una tabla “Documentos cargados” con:
   - Nombre del documento, folios, método, caracteres extraídos.
3) Pulsa “Ejecutar análisis”.
4) En la pestaña **Expediente** verás:
   - Tabla por documento (válido, incidencias, confianza, reglas OK/total).
   - Selector para ver el detalle por documento (extracción/normalización/validación/auditoría).
5) El sistema también genera resultados agregados del expediente (campos fusionados + auditoría global).

### 8.3) Procesar PDFs desde una carpeta

1) En la barra lateral, pega la ruta de una carpeta que contenga PDFs (ejemplo):
   - `C:\Users\TU_USUARIO\Documents\MiCarpetaPDFs`
2) Pulsa “Cargar PDFs desde carpeta”.
3) Se cargarán automáticamente todos los `*.pdf` de esa carpeta y se tratarán como expediente.

---

## 9) Configuración (variables de entorno)

Puedes ajustar el comportamiento sin modificar código usando variables de entorno (o un archivo `.env`):

- `LLM_BACKEND` (default: `local`) valores: `local` (Ollama) o `gpt4mini` (OpenAI).
- `OPENAI_API_KEY` (solo si `LLM_BACKEND=gpt4mini`).
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`).
- `OLLAMA_TEXT_MODEL` (default: `llama3.2:3b`).
- `OLLAMA_CLASSIFIER_MODEL` (default: `llama3.2:3b`).
- `OLLAMA_VISION_MODEL` (default: `qwen2.5vl:7b`).
- `OLLAMA_EMBEDDING_MODEL` (default: `nomic-embed-text`).
- `OLLAMA_TIMEOUT_S` (default: `600`).
- `ENABLE_PII_REDACTION` (default: `false`).

Ejemplo Windows (PowerShell):

```powershell
$env:OLLAMA_TIMEOUT_S = "600"
$env:ENABLE_PII_REDACTION = "false"
```

Ejemplo macOS/Linux:

```bash
export OLLAMA_TIMEOUT_S=600
export ENABLE_PII_REDACTION=false
```

---

## 10) Evaluación cuantitativa reproducible (métricas)

El runner `tools.evaluate` genera:
- CSV por documento con métricas y tiempos.
- Resumen agregado `.summary.json`.
- Mismatches por documento en carpeta `*_mismatches/`.

### 10.1) Baseline determinista (sin LLM)

```bash
python -m tools.evaluate --include-degraded --backend heuristic --out reports/evaluation_results_heuristic_full.csv
```

### 10.2) Evaluación con LLM (Ollama)

```bash
python -m tools.evaluate --include-degraded --backend llm --out reports/evaluation_results_llm_full.csv
```

---

## 11) Esquemas YAML (configurable sin programar)

Esquemas canónicos:
- [credito_hipotecario.yaml](schemas/credito_hipotecario.yaml)
- [auditoria_fiscal.yaml](schemas/auditoria_fiscal.yaml)
- [kyc_onboarding.yaml](schemas/kyc_onboarding.yaml)

Estructura (alto nivel):
- `caso_uso`, `version`, `descripcion`
- `documentos[]` con `tipo` y `campos[]` (nombre, tipo_dato, requerido, validación, patrón, etc.)
- `reglas_decision[]` (expresiones booleanas evaluadas de forma segura)
- `informe` (opciones de salida)

---

## 12) Git y GitHub (manual rápido)

### 12.1) Crear rama, commit y push

```bash
git checkout -b feature/mi-cambio
git status
git add .
git commit -m "Describe el cambio en español"
git push -u origin feature/mi-cambio
```

### 12.2) Mantener tu rama sincronizada con main

```bash
git fetch origin
git rebase origin/main
```

---

## 13) Solución de problemas (FAQ)

- **No conecta con Ollama**: abre Ollama y ejecuta `ollama list`.
- **Timeout**: sube `OLLAMA_TIMEOUT_S` o usa un modelo de texto más rápido.
- **Activación de venv en Windows falla**: usa `Set-ExecutionPolicy ... -Scope Process`.
- **Procesos colgados**: en Windows puedes detener por PID con `Stop-Process -Id <PID> -Force`.
