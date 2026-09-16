# ========================================================================
# EJECUTOR DE 21 CORRIDAS EXPERIMENTALES - DocAudit Agent (TFM)
# ========================================================================
#  7 configuraciones de ablacion x 3 semillas independientes = 21 corridas
#  Frente-D (lectura): auto | pypdf_only | ocr_only | vlm_only
#  Frente-E (arquitectura): auto | llm_flat_pipeline | heuristic | llm
#  Frente-F (reproducibilidad): 3 semillas fijas distintas
# ========================================================================

$ErrorActionPreference = "Stop"

# ---------- RUTAS ----------
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) {
    $PYTHON = "python"
}
$EVALUATE = Join-Path $PROJECT_ROOT "tools\evaluate.py"
$DATASET = Join-Path $PROJECT_ROOT "data\sample_docs"
$REPORTS_DIR = Join-Path $PROJECT_ROOT "reports"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$RUNS_DIR = Join-Path $REPORTS_DIR ("ablacion_21_runs_" + $TIMESTAMP)

# ---------- SEMILLAS (3) ----------
$SEEDS = @(20260501, 20260514, 20260528)

# ---------- 7 CONFIGURACIONES ----------
# Cada config = @(backend, reading_mode, etiqueta_corta)
$CONFIGURATIONS = @(
    @("auto",              "auto",       "C1_LANGGRAPH_CASCADA_OPTIMA"),
    @("auto",              "pypdf_only", "C2_SOLO_PYPDF"),
    @("auto",              "ocr_only",   "C3_SOLO_OCR"),
    @("auto",              "vlm_only",   "C4_SOLO_VLM"),
    @("llm_flat_pipeline", "auto",       "C5_PIPELINE_PLANO_2PASOS"),
    @("heuristic",         "auto",       "C6_BASELINE_HEURISTICO"),
    @("llm",               "auto",       "C7_SOLO_LLM_SIN_FALLBACK")
)

# ========================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DocAudit Agent - 21 CORRIDAS EXPERIMENTALES (Ablacion)"   -ForegroundColor Cyan
Write-Host "  7 configuraciones x 3 semillas = 21 evaluaciones"         -ForegroundColor Cyan
Write-Host ("  Proyecto: TFM Visual Analytics " + [char]38 + " Big Data")    -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host ("  Salida       : " + $RUNS_DIR)
Write-Host ("  Python       : " + $PYTHON)
Write-Host ("  Dataset      : " + $DATASET)
Write-Host ("  Semillas     : " + ($SEEDS -join ", "))
Write-Host ""

if (-not (Test-Path $RUNS_DIR)) {
    New-Item -ItemType Directory -Path $RUNS_DIR -Force | Out-Null
}

$TOTAL_RUNS = $CONFIGURATIONS.Count * $SEEDS.Count
$CURRENT_RUN = 0
$FAILED_RUNS = @()
$SUCCESS_RUNS = @()

# ---------- LOOP PRINCIPAL ----------
foreach ($config in $CONFIGURATIONS) {
    $backend = $config[0]
    $reading = $config[1]
    $label = $config[2]

    foreach ($seed in $SEEDS) {
        $CURRENT_RUN++
        $csv_name = ("{0:D2}_seed{1}_{2}.csv" -f $CURRENT_RUN, $seed, $label)
        $csv_path = Join-Path $RUNS_DIR $csv_name
        $log_name = ("{0:D2}_seed{1}_{2}.log" -f $CURRENT_RUN, $seed, $label)
        $log_path = Join-Path $RUNS_DIR $log_name

        Write-Host ""
        Write-Host ("----------------------------------------------------------------") -ForegroundColor DarkGray
        Write-Host ("  [{0,2}/{1}]  {2,-35}  seed={3}" -f $CURRENT_RUN, $TOTAL_RUNS, $label, $seed) -ForegroundColor Yellow
        Write-Host ("         backend={0,-18}  reading_mode={1}" -f $backend, $reading)
        Write-Host ("         -> " + $csv_name)
        Write-Host ("----------------------------------------------------------------") -ForegroundColor DarkGray

        $ARGUMENTS = @(
            $EVALUATE,
            "--dataset", $DATASET,
            "--out", $csv_path,
            "--backend", $backend,
            "--reading-mode", $reading,
            "--run-seed", $seed
        )

        $start = Get-Date
        try {
            & $PYTHON @ARGUMENTS 2>&1 | Tee-Object -FilePath $log_path | Out-Host
            $rc = $LASTEXITCODE
        }
        catch {
            Write-Host ("  EXCEPCION: " + $_.Exception.Message) -ForegroundColor Red
            $rc = 999
        }
        $elapsed = (Get-Date) - $start

        if ($rc -eq 0 -and (Test-Path $csv_path)) {
            $rows = Import-Csv $csv_path | Measure-Object | Select-Object -ExpandProperty Count
            Write-Host ""
            Write-Host ("  OK  - {0} filas en {1:mm}m {1:ss}s" -f ($rows - 1), $elapsed) -ForegroundColor Green
            $SUCCESS_RUNS += @{ run = $CURRENT_RUN; label = $label; seed = $seed; elapsed = $elapsed; csv = $csv_path }
        }
        else {
            Write-Host ("  FALLO - exit code = {0}, duracion {1:mm}m {1:ss}s" -f $rc, $elapsed) -ForegroundColor Red
            $FAILED_RUNS += @{ run = $CURRENT_RUN; label = $label; seed = $seed; rc = $rc; log = $log_path }
        }
    }
}

# ========================================================================
# RESUMEN FINAL
# ========================================================================
Write-Host ""
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  FIN DE LAS 21 CORRIDAS - RESUMEN"                          -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
$colorOK = if ($SUCCESS_RUNS.Count -eq $TOTAL_RUNS) { "Green" } else { "Yellow" }
Write-Host ("  EXITOSAS  : {0}/{1}" -f $SUCCESS_RUNS.Count, $TOTAL_RUNS) -ForegroundColor $colorOK
$colorFAIL = if ($FAILED_RUNS.Count -eq 0) { "Gray" } else { "Red" }
Write-Host ("  FALLIDAS  : {0}" -f $FAILED_RUNS.Count) -ForegroundColor $colorFAIL
Write-Host ("  DIRECTORIO: " + $RUNS_DIR)

if ($FAILED_RUNS.Count -gt 0) {
    Write-Host ""
    Write-Host "  Corridas fallidas:" -ForegroundColor Red
    foreach ($f in $FAILED_RUNS) {
        Write-Host ("    [{0,2}] {1} seed={2} -> rc={3}  log={4}" -f $f.run, $f.label, $f.seed, $f.rc, $f.log)
    }
}

Write-Host ""
Write-Host "  PASO SIGUIENTE: Consolidar resultados con:" -ForegroundColor Yellow
$consolidatePath = Join-Path $PROJECT_ROOT "consolidate_21_runs.py"
Write-Host ("    " + [char]38 + " '" + $consolidatePath + "' '" + $RUNS_DIR + "'")
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
