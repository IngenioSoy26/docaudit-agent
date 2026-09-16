# -*- coding: utf-8 -*-
"""
Extrae los datos reales que necesita la guía, desde los CSV brutos de ablación:
1) C1: TP, FP, FN POR cada una de las 3 semillas (para demostrar media vs conteo)
2) Resumen: precisión / recall / F1 por C1 semilla (ver desviación)
3) Matriz ruta (C1, C2, C3, C4) × variante visual × F1
4) AST C5 = N/A comprobación
5) C2 vs C1 dominancia Pareto comprobación
"""
import csv
import os
from collections import defaultdict

BASE = r"c:\Users\gusta\Desktop\Maestria\0. TFM\DocAudit Agent\docaudit-agent\reports\ablacion_21_runs_20260912_214540"
CONS = os.path.join(BASE, "consolidated_21_runs.csv")
SUM = os.path.join(BASE, "summary_by_configuration.csv")

VARIANT_ORDER = ["native_pdf", "scanned_blurry_pdf", "image_photo", "image_handwritten"]
ROUTE_CONFIGS = {"C1": "C1_LANGGRAPH_CASCADA_OPTIMA",
                 "C2": "C2_SOLO_PYPDF",
                 "C3": "C3_SOLO_OCR",
                 "C4": "C4_SOLO_VLM"}

print("=" * 90)
print("(1) C1 - TP / FP / FN POR SEMILLA")
print("=" * 90)
# Leer los 3 CSV individuales C1 (ya traen campo tp/fp/fn O global? Leer summary.json individual)
import json
for seed in ["20260501", "20260514", "20260528"]:
    prefix = {"20260501": "01", "20260514": "02", "20260528": "03"}[seed]
    s_path = os.path.join(BASE, f"{prefix}_seed{seed}_C1_LANGGRAPH_CASCADA_OPTIMA.summary.json")
    if os.path.exists(s_path):
        with open(s_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        ov = d.get("overall", {})
        # Buscar claves TP/FP/FN
        keys_of_interest = [k for k in ov.keys() if "tp" in k.lower() or "fp" in k.lower() or "fn" in k.lower() or "match" in k.lower() or "f1" in k.lower() or "precision" in k.lower() or "recall" in k.lower() or "exact" in k.lower()]
        print(f"\n--- seed={seed} ---")
        for k in sorted(ov.keys()):
            if any(x in k.lower() for x in ["tp", "fp", "fn", "match", "f1", "precision", "recall", "exact", "fields"]):
                print(f"  {k:45s} = {ov[k]}")

print("\n" + "=" * 90)
print("(2) RESUMEN POR CONFIGURACIÓN - summary_by_configuration.csv")
print("=" * 90)
if os.path.exists(SUM):
    with open(SUM, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"\nColumnas disponibles: {list(rows[0].keys())}")
    for r in rows:
        cfg = r.get("configuration_id", r.get("config", ""))
        print(f"\n--- {cfg} ---")
        for k, v in r.items():
            if any(x in k.lower() for x in ["f1", "precision", "recall", "emr", "exact", "match", "tp", "fp", "fn", "rules", "latency", "ram"]):
                print(f"  {k:40s} = {v}")

print("\n" + "=" * 90)
print("(3) LEER consolidated_21_runs.csv COLUMNAS")
print("=" * 90)
if os.path.exists(CONS):
    with open(CONS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row0 = next(reader)
    cols = list(row0.keys())
    print(f"\nColumnas: {cols}")
    # Agrupar por ruta × variante: calcular F1 medio
    agg = defaultdict(list)
    with open(CONS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cfg = r.get("configuration_id", r.get("config", ""))
            variant = r.get("variant", r.get("doc_variant", r.get("visual_variant", "")))
            # Buscar F1
            f1_val = None
            for key in r:
                if "f1" in key.lower():
                    try:
                        f1_val = float(r[key])
                        break
                    except:
                        pass
            if f1_val is not None:
                agg[(cfg, variant)].append(f1_val)
    print("\nMATRIZ RUTA × VARIANTE (F1 medio):")
    # Imprimir
    print(f"{'Ruta':<6s}", end="")
    for v in VARIANT_ORDER:
        print(f" {v:<22s}", end="")
    print()
    for route_label, cfg_contains in ROUTE_CONFIGS.items():
        print(f"{route_label:<6s}", end="")
        for v in VARIANT_ORDER:
            vals = []
            for (cfg, variant), fs in agg.items():
                if cfg_contains in cfg and variant == v:
                    vals.extend(fs)
            if vals:
                avg = sum(vals) / len(vals)
                print(f" {avg:<22.4f}", end="")
            else:
                # Buscar por coincidencia aproximada
                found = [(k, v) for k, v in agg.items() if (cfg_contains in k[0]) and (v in k[1])]
                if found:
                    vs = [x for sub in [f for k, f in found] for x in sub]
                    avg = sum(vs) / len(vs)
                    print(f" {avg:<22.4f}", end="")
                else:
                    print(f" {'N/A':<22s}", end="")
        print()
