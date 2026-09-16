# ========================================================================
# RELANZADOR DE 3 CORRIDAS FALTANTES 19/20/21  -  DocAudit Agent (TFM)
# ========================================================================
#  PASO 0 (OPCIONAL PERO RECOMENDADO):
#    - Cierra procesos python/ollama residuales que hayan quedado colgados
#    - Reinicia ollama serve para limpiar memoria del modelo
#
#  SOLO re-lanza C7 (Solo LLM sin fallback) x las 3 semillas,
#  escribiendo los CSVs directamente en el DIRECTORIO EXISTENTE
#  de la campana que se interrumpio por corte de energia.
#
#  Valida AL FINAL de cada corrida que el CSV tenga EXACTAMENTE 120 docs.
# ========================================================================

$ErrorActionPreference = "Stop"

# ---------- RUTAS ----------
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) { $PYTHON = "python" }
$EVALUATE = Join-Path $PROJECT_ROOT "tools\evaluate.py"
$DATASET = Join-Path $PROJECT_ROOT "data\sample_docs"
$EXPECTED_DOCS = 120

# ====== MODIFICA ESTA LINEA SI EL DIRECTORIO NO ES EL CORRECTO ======
$RUNS_DIR = Join-Path $PROJECT_ROOT "reports\ablacion_21_runs_20260912_214540"
# ====================================================================

# ---------- SEMILLAS Y CORRIDAS FALTANTES ----------
$RUNS = @(
    @{ run = 19; seed = 20260501; label = "C7_SOLO_LLM_SIN_FALLBACK"; backend = "llm"; reading = "auto" },
    @{ run = 20; seed = 20260514; label = "C7_SOLO_LLM_SIN_FALLBACK"; backend = "llm"; reading = "auto" },
    @{ run = 21; seed = 20260528; label = "C7_SOLO_LLM_SIN_FALLBACK"; backend = "llm"; reading = "auto" }
)

# ========================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  RELANZADOR DE 3 CORRIDAS FALTANTES (19 / 20 / 21)"        -ForegroundColor Magenta
Write-Host "  Configuracion: C7 - Solo LLM sin fallback (backend llm)"  -ForegroundColor Magenta
Write-Host "  Umbral exito: EXACTAMENTE $EXPECTED_DOCS filas de datos / CSV"  -ForegroundColor Magenta
Write-Host "  Directorio destino: $RUNS_DIR"                            -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

if (-not (Test-Path $RUNS_DIR)) {
    Write-Host "ERROR: No existe el directorio $RUNS_DIR" -ForegroundColor Red
    Write-Host "Edita la variable RUNS_DIR dentro de este script." -ForegroundColor Yellow
    exit 1
}

# ====== PASO 0: LIMPIAR PROCESOS RESIDUALES ======
Write-Host "[PASO 0] Limpiando procesos python/ollama residuales..." -ForegroundColor Yellow
$pythonKilled = 0
$ollamaKilled = 0
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*docaudit-agent*" -or $_.MainWindowTitle -like "*evaluate*"
} | ForEach-Object {
    try { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue; $pythonKilled++ } catch {}
}
# No matamos ollama serve global para no romper si el usuario lo tiene abierto fuera.
# Si el usuario quiere reiniciar Ollama, lo hace manualmente.
Write-Host ("         Procesos python residuales detenidos: {0}" -f $pythonKilled) -ForegroundColor Gray
Write-Host ""

# ====== PASO 1: BORRAR CSVs 19/20/21 EXISTENTES (incompletos) ======
Write-Host "[PASO 1] Borrando CSV/LOG 19, 20, 21 existentes (si existen y estan incompletos)..." -ForegroundColor Yellow
foreach ($R in $RUNS) {
    $csv_name = ("{0:D2}_seed{1}_{2}.csv" -f $R.run, $R.seed, $R.label)
    $log_name = ("{0:D2}_seed{1}_{2}.log" -f $R.run, $R.seed, $R.label)
    $sum_name = ("{0:D2}_seed{1}_{2}.summary.json" -f $R.run, $R.seed, $R.label)
    $mis_name = ("{0:D2}_seed{1}_{2}_mismatches" -f $R.run, $R.seed, $R.label)

    $csv_path = Join-Path $RUNS_DIR $csv_name
    $log_path = Join-Path $RUNS_DIR $log_name
    $sum_path = Join-Path $RUNS_DIR $sum_name
    $mis_path = Join-Path $RUNS_DIR $mis_name

    $need_delete = $false
    if (Test-Path $csv_path) {
        try {
            $n_docs = (Import-Csv $csv_path | Measure-Object).Count
            if ($n_docs -lt $EXPECTED_DOCS) {
                Write-Host ("         [X] {0}: solo {1}/{2} docs -> BORRAR" -f $csv_name, $n_docs, $EXPECTED_DOCS) -ForegroundColor Red
                $need_delete = $true
            } else {
                Write-Host ("         [OK] {0}: {1} docs -> conservar" -f $csv_name, $n_docs) -ForegroundColor Green
                continue  # no borres ni re-lances
            }
        } catch {
            Write-Host ("         [!] {0}: ilegible -> BORRAR" -f $csv_name) -ForegroundColor Red
            $need_delete = $true
        }
    }

    if ($need_delete) {
        foreach ($f in @($csv_path, $log_path, $sum_path)) {
            if (Test-Path $f) {
                try { Remove-Item $f -Force -ErrorAction SilentlyContinue } catch {
                    Write-Host ("             * Aviso: no se pudo borrar {0} (sigue en uso? se sobrescribira al escribir)" -f (Split-Path $f -Leaf)) -ForegroundColor DarkYellow
                }
            }
        }
        if (Test-Path $mis_path) {
            try { Remove-Item $mis_path -Recurse -Force -ErrorAction SilentlyContinue } catch {}
        }
    }
}
Write-Host ""

# ========================================================================
# LOOP PRINCIPAL: EJECUTAR LAS 3 CORRIDAS
# ========================================================================
$FAILED = @()
$OK = @()
$SKIPPED = @()

foreach ($R in $RUNS) {
    $csv_name = ("{0:D2}_seed{1}_{2}.csv" -f $R.run, $R.seed, $R.label)
    $csv_path = Join-Path $RUNS_DIR $csv_name
    $log_name = ("{0:D2}_seed{1}_{2}.log" -f $R.run, $R.seed, $R.label)
    $log_path = Join-Path $RUNS_DIR $log_name

    # Si ya existe y tiene 120 docs, skipea
    if (Test-Path $csv_path) {
        try {
            $n_docs = (Import-Csv $csv_path | Measure-Object).Count
            if ($n_docs -eq $EXPECTED_DOCS) {
                Write-Host ("[SKIP #{0:D2}] {1} ya tiene 120 docs -> se salta" -f $R.run, $csv_name) -ForegroundColor Gray
                $SKIPPED += $R
                continue
            }
        } catch {}
    }

    Write-Host ("----------------------------------------------------------------") -ForegroundColor DarkGray
    Write-Host ("  [{0,2}/21]  {1,-35}  seed={2}" -f $R.run, $R.label, $R.seed) -ForegroundColor Yellow
    Write-Host ("         backend={0,-18}  reading_mode={1}" -f $R.backend, $R.reading)
    Write-Host ("         -> " + $csv_name)
    Write-Host ("         -> " + "Duracion estimada: ~12-18 minutos (120 docs x ~7-9 s/doc)")
    Write-Host ("----------------------------------------------------------------") -ForegroundColor DarkGray

    $ARGUMENTS = @(
        $EVALUATE,
        "--dataset", $DATASET,
        "--out", $csv_path,
        "--backend", $R.backend,
        "--reading-mode", $R.reading,
        "--run-seed", $R.seed,
        "--limit", 0
    )

    $start = Get-Date
    $rc = 9999
    try {
        & $PYTHON @ARGUMENTS 2>&1 | Tee-Object -FilePath $log_path | Out-Host
        $rc = $LASTEXITCODE
    }
    catch {
        Write-Host ("  EXCEPCION: " + $_.Exception.Message) -ForegroundColor Red
        $rc = 998
    }
    $elapsed = (Get-Date) - $start

    # Validacion final
    $n_docs = -1
    if (Test-Path $csv_path) {
        try { $n_docs = (Import-Csv $csv_path | Measure-Object).Count } catch { $n_docs = -2 }
    }

    if ($rc -eq 0 -and $n_docs -eq $EXPECTED_DOCS) {
        Write-Host ""
        Write-Host ("  OK  - $n_docs docs en {0:hh}h {0:mm}m {0:ss}s (exit=$rc)" -f $elapsed) -ForegroundColor Green
        $OK += @{ run = $R.run; label = $R.label; seed = $R.seed; elapsed = $elapsed; csv = $csv_path }
    }
    else {
        Write-Host ""
        Write-Host ("  FALLO - docs=$n_docs/$EXPECTED_DOCS, exit=$rc, duracion {0:mm}m {0:ss}s" -f $elapsed) -ForegroundColor Red
        if ($rc -eq 0 -and $n_docs -gt 0 -and $n_docs -lt $EXPECTED_DOCS) {
            Write-Host ("         El CSV se escribio pero se corto. Puede ser:") -ForegroundColor Yellow
            Write-Host ("          - Se cerro la ventana PowerShell") -ForegroundColor Yellow
            Write-Host ("          - El PC se suspendio / hiberno") -ForegroundColor Yellow
            Write-Host ("          - Ollama murio a mitad (mira el log en las ultimas lineas)") -ForegroundColor Yellow
        }
        $FAILED += @{ run = $R.run; label = $R.label; seed = $R.seed; rc = $rc; docs = $n_docs; log = $log_path }
    }
}

# ========================================================================
Write-Host ""
Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  FIN RELANZADOR 19/20/21"                                   -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ("  EXITOSAS  : {0}/3" -f $OK.Count)
Write-Host ("  SALTADAS  : {0}/3" -f $SKIPPED.Count)
Write-Host ("  FALLIDAS  : {0}"   -f $FAILED.Count)
Write-Host ("  DIRECTORIO: $RUNS_DIR")

if ($FAILED.Count -gt 0) {
    Write-Host ""
    Write-Host "  Corridas fallidas:" -ForegroundColor Red
    foreach ($f in $FAILED) {
        Write-Host ("    [{0,2}] seed={1} -> docs={2}/120, rc={3}  log={4}" -f $f.run, $f.seed, $f.docs, $f.rc, $f.log)
    }
    Write-Host ""
    Write-Host "  Re-lanza SOLO las fallidas con el BLOQUE DE COMANDOS INDIVIDUALES del manual (run_3_missing_runs_NOTAS.md)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  PASO SIGUIENTE: Consolidar las 21 corridas con:" -ForegroundColor Yellow
$consolidatePath = Join-Path $PROJECT_ROOT "consolidate_21_runs.py"
Write-Host ("    cd '" + $PROJECT_ROOT + "'")
Write-Host ("    " + [char]38 + " '" + $consolidatePath + "' '" + $RUNS_DIR + "'")
Write-Host ""
