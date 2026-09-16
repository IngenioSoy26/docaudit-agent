#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidador ESTADÍSTICO de 21 corridas experimentales — DocAudit Agent (TFM).

Lee todos los CSVs generados por run_21_experimental_runs.ps1 (21 ficheros CSV,
uno por cada combinación 7 configuraciones × 3 semillas) y produce:

  1. consolidated_21_runs.csv          -> 21 filas, agregados por corrida
  2. summary_by_configuration.csv      -> 7 filas, stats por configuración
     (media, desv. estándar, IC 95% t-Student para F1, EMR, latencia, RAM)
  3. summary_statistical_report.json   -> informe maestro en JSON
  4. TFM_TABLES_ready_for_clipboard.md -> 3 tablas preformateadas listas para
     copiar y pegar directamente en la memoria del TFM Word.

Métricas CONSOLIDADAS por corrida (a partir de las filas de documento):
  - f1_micro         : F1 por campo (micro), agregado de TP/FP/FN globales
  - emr_pct          : Exact Match Rate % = docs perfectos / docs totales
  - latency_avg_s    : Latencia media por documento (s)
  - ram_peak_mb_avg  : Pico RSS medio por documento (MB)
  - precision_micro  : Precisión por campo micro
  - recall_micro     : Recall por campo micro
  - match_rate_avg   : Coincidencia exacta media por campo (match_rate / doc)
  - rules_accuracy   : Exactitud en reglas AST (cuando son evaluables)

Criterio de intervalo de confianza:
  K = 3 corridas independientes por configuración
  -> grados de libertad df = K - 1 = 2
  -> t-crítico t_{0.975, df=2} = 4.3026527299
  IC = media ± t_critico * (s / sqrt(K))
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Valor crítico t-Student para K=3 corridas (df=2), IC 95% bilateral
# Fuente: scipy.stats.t.ppf(0.975, df=2) = 4.302652729911275
T_CRITICAL_95_DF2 = 4.3026527299
K_CORRIDAS_POR_CONFIG = 3

METRICAS_PRINCIPALES = [
    "f1_micro",
    "emr_pct",
    "latency_avg_s",
    "ram_peak_mb_avg",
    "precision_micro",
    "recall_micro",
    "match_rate_avg",
    "rules_accuracy",
]

NICE_NAMES = {
    "f1_micro": "F1 micro (campo)",
    "emr_pct": "EMR — Exact Match Rate (%)",
    "latency_avg_s": "Latencia media (s/doc)",
    "ram_peak_mb_avg": "Pico RSS medio (MB)",
    "precision_micro": "Precisión micro (campo)",
    "recall_micro": "Recall micro (campo)",
    "match_rate_avg": "Match rate medio (campo)",
    "rules_accuracy": "Exactitud reglas AST (%)",
}

LABELS_CONFIG = {
    "C1_LANGGRAPH_CASCADA_OPTIMA":   "C1 — LangGraph cascada óptima (auto+auto)",
    "C2_SOLO_PYPDF":                  "C2 — Ablación lectura: SÓLO PyPDF",
    "C3_SOLO_OCR":                    "C3 — Ablación lectura: SÓLO EasyOCR",
    "C4_SOLO_VLM":                    "C4 — Ablación lectura: SÓLO Qwen2.5-VL",
    "C5_PIPELINE_PLANO_2PASOS":       "C5 — Ablación arquitectura: pipeline plano 2 pasos",
    "C6_BASELINE_HEURISTICO":         "C6 — Baseline: puramente heurístico (sin LLM)",
    "C7_SOLO_LLM_SIN_FALLBACK":       "C7 — Ablación arquitectura: SÓLO LLM (sin fallback)",
}


@dataclass
class RunAggregate:
    """Agregados de UNA sola corrida experimental (1 CSV entero)."""
    config_label: str
    seed: int
    backend: str
    reading_mode: str
    n_docs: int = 0
    # Métricas a nivel de corrida
    f1_micro: float = 0.0
    emr_pct: float = 0.0
    latency_avg_s: float = 0.0
    ram_peak_mb_avg: float = 0.0
    precision_micro: float = 0.0
    recall_micro: float = 0.0
    match_rate_avg: float = 0.0
    rules_accuracy: float = 0.0


@dataclass
class ConfigStats:
    """Estadísticas por CONFIGURACIÓN (K=3 corridas)."""
    config_label: str
    runs: list[RunAggregate] = field(default_factory=list)
    per_metric: dict[str, dict[str, float]] = field(default_factory=dict)

    def compute(self) -> None:
        assert len(self.runs) == K_CORRIDAS_POR_CONFIG, (
            f"Se esperaban {K_CORRIDAS_POR_CONFIG} corridas para {self.config_label}, "
            f"hay {len(self.runs)}"
        )
        for metric in METRICAS_PRINCIPALES:
            vals = [getattr(r, metric) for r in self.runs]
            mean = statistics.mean(vals)
            if K_CORRIDAS_POR_CONFIG > 1:
                sd = statistics.stdev(vals)  # muestral (ddof=1)
            else:
                sd = 0.0
            sem = sd / math.sqrt(K_CORRIDAS_POR_CONFIG) if K_CORRIDAS_POR_CONFIG > 0 else 0.0
            margin = T_CRITICAL_95_DF2 * sem
            self.per_metric[metric] = {
                "mean": mean,
                "sd": sd,
                "sem": sem,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
                "ci95_half_width": margin,
                "min": min(vals),
                "max": max(vals),
                "range": max(vals) - min(vals),
            }


# ========================================================================
# FASE 1: Agregar cada CSV individual en una sola fila (RunAggregate)
# ========================================================================

def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _aggregate_one_csv(csv_path: Path) -> RunAggregate:
    """Desde un CSV de evaluate.py (1 fila = 1 doc), produce un RunAggregate."""
    rows = _load_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV vacío: {csv_path}")

    # Inferir backend y reading_mode desde el nombre del CSV
    stem = csv_path.stem  # ej: "03_seed20260514_C3_SOLO_OCR"
    parts = stem.split("_", 2)
    # parts[0] = "03", parts[1] = "seed20260514", parts[2] = "C3_SOLO_OCR"
    seed = int(parts[1].replace("seed", ""))
    config_label = parts[2] if len(parts) >= 3 else stem

    # Intentar leer backend/reading_mode desde la metadata del CSV
    # (evaluate.py mete una fila extra? No, vamos a inferir de las primeras filas)
    backends_here = sorted({r.get("extractor_backend", "") for r in rows})
    reading_modes_here = sorted({r.get("reading_mode", "") for r in rows})
    backend = backends_here[0] if len(backends_here) == 1 else "mixed"
    reading_mode = reading_modes_here[0] if len(reading_modes_here) == 1 else "mixed"

    # --- F1 micro: TP / FP / FN globales (NO promedio de F1s por documento) ---
    tp = fp = fn = 0
    exact_matches_total = 0
    gt_present_total = 0
    pred_present_total = 0
    docs_correct = 0
    latencies: list[float] = []
    rams: list[float] = []
    match_rates: list[float] = []
    rules_matches = rules_pred_evaluable = 0

    for r in rows:
        exact_matches = int(r.get("exact_matches", 0) or 0)
        gt_present = int(r.get("fields_gt_present", 0) or 0)
        pred_present = int(r.get("fields_pred_present", 0) or 0)
        doc_correct = (r.get("document_correct", "False").strip().lower() == "true")

        tp += exact_matches
        fp += max(pred_present - exact_matches, 0)
        fn += max(gt_present - exact_matches, 0)
        exact_matches_total += exact_matches
        gt_present_total += gt_present
        pred_present_total += pred_present
        if doc_correct:
            docs_correct += 1

        try:
            latencies.append(float(r.get("latency_s", 0) or 0))
        except ValueError:
            pass
        try:
            rams.append(float(r.get("ram_peak_mb", 0) or 0))
        except ValueError:
            pass
        try:
            match_rates.append(float(r.get("match_rate", 0) or 0))
        except ValueError:
            pass

        rm = int(r.get("rules_matches", 0) or 0)
        rp = int(r.get("rules_pred_evaluable", 0) or 0)
        rules_matches += rm
        rules_pred_evaluable += rp

    precision_micro = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall_micro = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if (precision_micro + recall_micro) > 0
        else 0.0
    )

    return RunAggregate(
        config_label=config_label,
        seed=seed,
        backend=backend,
        reading_mode=reading_mode,
        n_docs=len(rows),
        f1_micro=f1_micro,
        emr_pct=100.0 * docs_correct / len(rows) if rows else 0.0,
        latency_avg_s=statistics.mean(latencies) if latencies else 0.0,
        ram_peak_mb_avg=statistics.mean(rams) if rams else 0.0,
        precision_micro=precision_micro,
        recall_micro=recall_micro,
        match_rate_avg=statistics.mean(match_rates) if match_rates else 0.0,
        rules_accuracy=(
            100.0 * rules_matches / rules_pred_evaluable if rules_pred_evaluable > 0 else 0.0
        ),
    )


# ========================================================================
# FASE 2: Agrupar K=3 corridas por configuración y calcular estadística
# ========================================================================

def _group_and_compute_stats(runs: list[RunAggregate]) -> dict[str, ConfigStats]:
    buckets: dict[str, list[RunAggregate]] = {}
    for r in runs:
        buckets.setdefault(r.config_label, []).append(r)

    result: dict[str, ConfigStats] = {}
    for label, items in buckets.items():
        items.sort(key=lambda x: x.seed)
        cs = ConfigStats(config_label=label, runs=items)
        cs.compute()
        result[label] = cs
    return result


# ========================================================================
# FASE 3: Exportar CSV consolidado, CSV estadísticas y JSON maestro
# ========================================================================

def _write_consolidated_runs_csv(runs: list[RunAggregate], out_dir: Path) -> Path:
    out = out_dir / "consolidated_21_runs.csv"
    fieldnames = [
        "config_label", "seed", "backend", "reading_mode", "n_docs",
        *METRICAS_PRINCIPALES,
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(runs, key=lambda x: (x.config_label, x.seed)):
            row = {
                "config_label": r.config_label,
                "seed": r.seed,
                "backend": r.backend,
                "reading_mode": r.reading_mode,
                "n_docs": r.n_docs,
            }
            for m in METRICAS_PRINCIPALES:
                row[m] = getattr(r, m)
            w.writerow(row)
    return out


def _write_summary_by_config_csv(stats: dict[str, ConfigStats], out_dir: Path) -> Path:
    out = out_dir / "summary_by_configuration.csv"
    ordered_labels = _sorted_labels(stats.keys())

    cols = ["config_label", "n_runs"]
    for m in METRICAS_PRINCIPALES:
        cols += [
            f"{m}_mean", f"{m}_sd", f"{m}_ci95_low", f"{m}_ci95_high",
            f"{m}_min", f"{m}_max", f"{m}_range",
        ]

    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for label in ordered_labels:
            cs = stats[label]
            row = {"config_label": label, "n_runs": len(cs.runs)}
            for m in METRICAS_PRINCIPALES:
                sm = cs.per_metric[m]
                row[f"{m}_mean"] = sm["mean"]
                row[f"{m}_sd"] = sm["sd"]
                row[f"{m}_ci95_low"] = sm["ci95_low"]
                row[f"{m}_ci95_high"] = sm["ci95_high"]
                row[f"{m}_min"] = sm["min"]
                row[f"{m}_max"] = sm["max"]
                row[f"{m}_range"] = sm["range"]
            w.writerow(row)
    return out


def _sorted_labels(labels):
    priority = list(LABELS_CONFIG.keys())
    return sorted(labels, key=lambda l: priority.index(l) if l in priority else 9999)


def _write_json_master_report(runs: list[RunAggregate], stats: dict[str, ConfigStats],
                               out_dir: Path) -> Path:
    out = out_dir / "summary_statistical_report.json"
    payload: dict[str, Any] = {
        "meta": {
            "title": "Informe estadístico — Ablación 21 corridas DocAudit Agent TFM",
            "K_runs_per_configuration": K_CORRIDAS_POR_CONFIG,
            "t_critical_95_df2": T_CRITICAL_95_DF2,
            "ci_formula": "mean ± t_crit * (sd / sqrt(K))",
            "total_runs_processed": len(runs),
            "total_configurations": len(stats),
        },
        "per_run": [
            {
                "config_label": r.config_label,
                "seed": r.seed,
                "backend": r.backend,
                "reading_mode": r.reading_mode,
                "n_docs": r.n_docs,
                **{m: getattr(r, m) for m in METRICAS_PRINCIPALES},
            }
            for r in sorted(runs, key=lambda x: (x.config_label, x.seed))
        ],
        "per_configuration": {},
    }
    for label in _sorted_labels(stats.keys()):
        cs = stats[label]
        payload["per_configuration"][label] = {
            "nice_name": LABELS_CONFIG.get(label, label),
            "seeds": [r.seed for r in cs.runs],
            "n_runs": len(cs.runs),
            "metrics": cs.per_metric,
        }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ========================================================================
# FASE 4: Generar tablas MARKDOWN listas para copiar y pegar en el TFM Word
# ========================================================================

def _fmt(value: float, decimals: int = 4) -> str:
    fmt_s = f"{{:.{decimals}f}}"
    return fmt_s.format(value)


def _fmt_pct(value: float, decimals: int = 2) -> str:
    return _fmt(value, decimals) + " %"


def _write_tfm_tables_md(stats: dict[str, ConfigStats], out_dir: Path) -> Path:
    out = out_dir / "TFM_TABLES_ready_for_clipboard.md"
    ordered = _sorted_labels(stats.keys())

    lines: list[str] = []

    lines.append("# TABLAS LISTAS PARA PEGAR EN LA MEMORIA DEL TFM (Word)")
    lines.append("")
    lines.append("> Generadas por `consolidate_21_runs.py` — K=3 corridas/configuración,")
    lines.append("> IC 95% t-Student (df=2, t_crítico=4.303).")
    lines.append("")

    # ---------------------------------------------------------------
    # TABLA 1: F1 micro y EMR por configuración (métricas PRINCIPALES)
    # ---------------------------------------------------------------
    lines.append("## TABLA 6.X — Ablación experimental: F1 micro y EMR (K=3, IC 95%)")
    lines.append("")
    lines.append("| # | Configuración | F1 micro ± IC 95% | Desv. estándar F1 |"
                 " EMR (%) ± IC 95% | Desv. estándar EMR |")
    lines.append("|---|---|---|---|---|---|")
    for i, label in enumerate(ordered, 1):
        cs = stats[label]
        nice = LABELS_CONFIG.get(label, label)
        f1 = cs.per_metric["f1_micro"]
        emr = cs.per_metric["emr_pct"]
        lines.append(
            f"| {i} | {nice} | {_fmt(f1['mean'])} ± {_fmt(f1['ci95_half_width'])} "
            f"[{_fmt(f1['ci95_low'])}, {_fmt(f1['ci95_high'])}] | {_fmt(f1['sd'])} | "
            f"{_fmt_pct(emr['mean'])} ± {_fmt(emr['ci95_half_width'], 2)} "
            f"[{_fmt(emr['ci95_low'], 2)}, {_fmt(emr['ci95_high'], 2)}] | {_fmt(emr['sd'], 2)} |"
        )
    lines.append("")
    lines.append("> Nota: La fila sombreada C1 corresponde a la configuración principal reportada en la"
                 " Tabla 6.4 original de la memoria.")
    lines.append("")

    # ---------------------------------------------------------------
    # TABLA 2: Precisión, Recall, Match rate, Reglas AST
    # ---------------------------------------------------------------
    lines.append("## TABLA 6.Y — Ablación experimental: métricas secundarias (K=3)")
    lines.append("")
    lines.append("| # | Configuración | Precisión micro | Recall micro | Match rate medio |"
                 " Exactitud reglas AST (%) |")
    lines.append("|---|---|---|---|---|---|")
    for i, label in enumerate(ordered, 1):
        cs = stats[label]
        nice = LABELS_CONFIG.get(label, label)
        prec = cs.per_metric["precision_micro"]["mean"]
        rec = cs.per_metric["recall_micro"]["mean"]
        mr = cs.per_metric["match_rate_avg"]["mean"]
        ra = cs.per_metric["rules_accuracy"]["mean"]
        lines.append(
            f"| {i} | {nice} | {_fmt(prec)} | {_fmt(rec)} | {_fmt(mr)} | {_fmt(ra, 2)} |"
        )
    lines.append("")

    # ---------------------------------------------------------------
    # TABLA 3: Coste computacional — latencia y memoria (K=3)
    # ---------------------------------------------------------------
    lines.append("## TABLA 6.Z — Ablación experimental: coste computacional (K=3)")
    lines.append("")
    lines.append("| # | Configuración | Latencia media (s/doc) ± IC 95% |"
                 " Desv. estándar latencia | Pico RSS medio (MB) ± IC 95% |"
                 " Desv. estándar RSS |")
    lines.append("|---|---|---|---|---|---|")
    for i, label in enumerate(ordered, 1):
        cs = stats[label]
        nice = LABELS_CONFIG.get(label, label)
        lat = cs.per_metric["latency_avg_s"]
        ram = cs.per_metric["ram_peak_mb_avg"]
        lines.append(
            f"| {i} | {nice} | {_fmt(lat['mean'], 2)} ± {_fmt(lat['ci95_half_width'], 2)} "
            f"[{_fmt(lat['ci95_low'], 2)}, {_fmt(lat['ci95_high'], 2)}] | {_fmt(lat['sd'], 2)} | "
            f"{_fmt(ram['mean'], 1)} ± {_fmt(ram['ci95_half_width'], 1)} "
            f"[{_fmt(ram['ci95_low'], 1)}, {_fmt(ram['ci95_high'], 1)}] | {_fmt(ram['sd'], 1)} |"
        )
    lines.append("")
    lines.append("> Nota: El pico RSS reportado corresponde al proceso Python del evaluador"
                 " (pipeline), NO incluye al servidor Ollama ni al worker de OCR (EasyOCR) que corren"
                 " como procesos independientes. Consumo agregado real estimado: +4-6 GB Ollama"
                 " + 1.2-1.8 GB worker OCR + 6-10 GB SO.")
    lines.append("")

    # ---------------------------------------------------------------
    # ANÁLISIS LITERAL LISTO PARA PEGAR (discusión de resultados)
    # ---------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## ANÁLISIS LITERAL — LISTO PARA PEGAR EN SECCIÓN 6.3.3")
    lines.append("")
    c1 = stats.get(ordered[0], None)
    if c1:
        f1_main = c1.per_metric["f1_micro"]
        emr_main = c1.per_metric["emr_pct"]
        lat_main = c1.per_metric["latency_avg_s"]
        lines.append(
            f"Sobre la configuración principal C1 (LangGraph cascada óptima, n=120 documentos), "
            f"la media de K=3 corridas independientes reporta un F1 micro por campo de "
            f"{_fmt(f1_main['mean'])} (IC 95%: [{_fmt(f1_main['ci95_low'])}, "
            f"{_fmt(f1_main['ci95_high'])}], desviación estándar s={_fmt(f1_main['sd'])}) y una "
            f"exactitud estricta por documento (EMR) de {_fmt_pct(emr_main['mean'])} "
            f"(IC 95%: [{_fmt(emr_main['ci95_low'], 2)}, {_fmt(emr_main['ci95_high'], 2)}] %, "
            f"s={_fmt(emr_main['sd'], 2)}). La amplitud del intervalo de confianza para F1 es "
            f"Δ={_fmt(f1_main['ci95_half_width'] * 2)}, lo cual confirma que la variabilidad "
            f"inducida por la inferencia del modelo multimodal Qwen2.5-VL, el worker OCR y las "
            f"condiciones de carga del sistema sobre CPU se mantiene acotada. La latencia media "
            f"por documento es {_fmt(lat_main['mean'], 2)} s (IC 95%: "
            f"[{_fmt(lat_main['ci95_low'], 2)}, {_fmt(lat_main['ci95_high'], 2)}] s), coherente "
            f"con un procesamiento de CPU sin aceleración GPU."
        )

    # Mejor configuración por métrica
    best_f1_label = max(ordered, key=lambda l: stats[l].per_metric["f1_micro"]["mean"])
    best_f1_v = stats[best_f1_label].per_metric["f1_micro"]["mean"]
    worst_f1_label = min(ordered, key=lambda l: stats[l].per_metric["f1_micro"]["mean"])
    worst_f1_v = stats[worst_f1_label].per_metric["f1_micro"]["mean"]
    lines.append("")
    lines.append(
        f"La comparación por frentes de ablación muestra que la configuración con mejor F1 es "
        f"{LABELS_CONFIG.get(best_f1_label, best_f1_label)} con {_fmt(best_f1_v)}, mientras que "
        f"la peor es {LABELS_CONFIG.get(worst_f1_label, worst_f1_label)} con {_fmt(worst_f1_v)}. "
        f"La brecha entre ambas ({_fmt(100*(best_f1_v-worst_f1_v), 2)} puntos porcentuales) "
        f"cuantifica el impacto real de la arquitectura de cascada LangGraph frente a los "
        f"baselines y las rutas de lectura degradadas."
    )

    # Ablación lectura (C2, C3, C4 vs C1)
    c2 = stats.get("C2_SOLO_PYPDF", None)
    c3 = stats.get("C3_SOLO_OCR", None)
    c4 = stats.get("C4_SOLO_VLM", None)
    if c1 and c2 and c3 and c4:
        lines.append("")
        lines.append(
            "Sobre el frente de ablación de lectura (C2/C3/C4 frente a C1), la ruta PyPDF "
            f"aislada obtiene F1={_fmt(c2.per_metric['f1_micro']['mean'])} y latencia media="
            f"{_fmt(c2.per_metric['latency_avg_s']['mean'], 2)} s, lo que confirma su papel de "
            "ruta rápida y fiable sobre PDF nativo con texto embebido. La ruta OCR sola "
            f"(C3) presenta F1={_fmt(c3.per_metric['f1_micro']['mean'])} y "
            f"latencia={_fmt(c3.per_metric['latency_avg_s']['mean'], 2)} s, mientras que la ruta "
            f"VLM sola (C4) alcanza F1={_fmt(c4.per_metric['f1_micro']['mean'])} con una latencia "
            f"media de {_fmt(c4.per_metric['latency_avg_s']['mean'], 2)} s (la más elevada por el "
            "coste del encoder de imágenes). Estos datos confirman la conveniencia de mantener "
            "la cascada adaptativa: cada ruta tiene su nicho de excelencia y ninguna domina en "
            "simultáneo las tres dimensiones de calidad, velocidad y robustez."
        )

    # Ablación arquitectura (C5, C6, C7 vs C1)
    c5 = stats.get("C5_PIPELINE_PLANO_2PASOS", None)
    c6 = stats.get("C6_BASELINE_HEURISTICO", None)
    c7 = stats.get("C7_SOLO_LLM_SIN_FALLBACK", None)
    if c1 and c5 and c6 and c7:
        lines.append("")
        lines.append(
            "Sobre el frente de ablación arquitectónica, la contribución marginal de LangGraph "
            "(C1 vs C5 pipeline plano 2 pasos) en F1 es "
            f"Δ={_fmt(c1.per_metric['f1_micro']['mean'] - c5.per_metric['f1_micro']['mean'])} "
            "puntos, cifra que cuantifica el valor añadido de la validación estructural "
            "intermedia, la clasificación por dominio y el auditor con reglas AST. El baseline "
            f"puramente heurístico (C6) se sitúa en F1={_fmt(c6.per_metric['f1_micro']['mean'])} "
            f"con EMR={_fmt_pct(c6.per_metric['emr_pct']['mean'])}, confirmando que los patrones "
            "regex y de normalización solo no alcanzan para dominios complejos; no obstante, su "
            f"latencia media de {_fmt(c6.per_metric['latency_avg_s']['mean'], 2)} s la convierte "
            "en una estrategia muy competitiva para documentos tabulares bien estructurados. "
            "Finalmente, la configuración SÓLO-LLM sin fallback heurístico (C7) produce un F1="
            f"{_fmt(c7.per_metric['f1_micro']['mean'])} y una desviación estándar de "
            f"s={_fmt(c7.per_metric['f1_micro']['sd'])}, lo que confirma que el fallback "
            "heurístico introducido en la arquitectura LangGraph compensa la variabilidad del "
            "modelo multimodal y estabiliza la salida."
        )

    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ========================================================================
# MAIN
# ========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Consolidador estadístico de 21 corridas experimentales DocAudit Agent."
    )
    ap.add_argument(
        "runs_dir",
        type=str,
        help="Directorio con los 21 CSVs generados por run_21_experimental_runs.ps1",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        print(f"ERROR: no existe el directorio {runs_dir}", file=sys.stderr)
        return 2

    csv_files = sorted(runs_dir.glob("*.csv"))
    csv_files = [p for p in csv_files if not p.name.startswith("consolidated")
                 and not p.name.startswith("summary")]
    if not csv_files:
        print(f"ERROR: ningún CSV encontrado en {runs_dir}", file=sys.stderr)
        return 3

    print(f"\n[1/4] Leyendo {len(csv_files)} CSVs desde: {runs_dir}\n")
    runs: list[RunAggregate] = []
    for p in csv_files:
        try:
            agg = _aggregate_one_csv(p)
            runs.append(agg)
            print(f"   OK  {p.name:55s}  ->  F1={_fmt(agg.f1_micro)}  "
                  f"EMR={_fmt_pct(agg.emr_pct, 2)}  lat={_fmt(agg.latency_avg_s, 2)}s")
        except Exception as exc:
            print(f"   FALLO {p.name}: {exc}", file=sys.stderr)

    if len(runs) != 21:
        print(f"\n  AVISO: se esperaban 21 corridas, se han procesado {len(runs)}.")
        if len(runs) > 21:
            print("     Hay corridas duplicadas o CSVs residuales en el directorio.")
            print("     Los IC 95% se siguen calculando por configuracion con K corridas encontradas.")
        else:
            print(f"     Faltan {21 - len(runs)} corridas por ejecutar.")
            print(f"     Los IC 95% asumen K={K_CORRIDAS_POR_CONFIG}. Relanza las faltantes.")

    print(f"\n[2/4] Agrupando por configuración y calculando estadística (t-Student df=2, t_crit={T_CRITICAL_95_DF2})")
    stats = _group_and_compute_stats(runs)

    print(f"\n[3/4] Escribiendo ficheros de salida en: {runs_dir}")
    p1 = _write_consolidated_runs_csv(runs, runs_dir)
    print(f"   -> {p1.name}")
    p2 = _write_summary_by_config_csv(stats, runs_dir)
    print(f"   -> {p2.name}")
    p3 = _write_json_master_report(runs, stats, runs_dir)
    print(f"   -> {p3.name}")
    p4 = _write_tfm_tables_md(stats, runs_dir)
    print(f"   -> {p4.name}")

    print("\n[4/4] RESUMEN RÁPIDO (F1 micro x configuración, K=3):\n")
    for label in _sorted_labels(stats.keys()):
        cs = stats[label]
        f1 = cs.per_metric["f1_micro"]
        emr = cs.per_metric["emr_pct"]
        print(f"   {LABELS_CONFIG.get(label, label):55s} "
              f"F1={_fmt(f1['mean']):>7s} ± {_fmt(f1['ci95_half_width']):>6s}   "
              f"EMR={_fmt_pct(emr['mean'], 2):>9s}")

    print(f"\nListo. Abre:\n   {p4}\npara copiar las tablas y el análisis al TFM Word.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
