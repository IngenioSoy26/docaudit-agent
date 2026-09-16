"""
TFM DOCUMENTAUDIT AGENT - RECÁLCULO BLOQUEANTE v2
Usa columnas REALES detectadas: schema / pdf / precision / recall / f1 / fields_gt_present / fields_pred_present / document_correct / rules_accuracy / run_seed
"""
import csv
from collections import defaultdict
from pathlib import Path

REPORTS_DIR = Path(r"c:\Users\gusta\Desktop\Maestria\0. TFM\DocAudit Agent\docaudit-agent\reports\ablacion_21_runs_20260912_214540")
C1_FILES = [
    (20260501, "01_seed20260501_C1_LANGGRAPH_CASCADA_OPTIMA.csv"),
    (20260514, "02_seed20260514_C1_LANGGRAPH_CASCADA_OPTIMA.csv"),
    (20260528, "03_seed20260528_C1_LANGGRAPH_CASCADA_OPTIMA.csv"),
]

rows_all = []
cols_found = None
for seed, fname in C1_FILES:
    fpath = REPORTS_DIR / fname
    assert fpath.exists(), f"NO EXISTE {fpath}"
    with open(fpath, "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if cols_found is None:
                cols_found = sorted(row.keys())
                print("=" * 110)
                print(f"COLUMNAS:")
                for c in cols_found: print(f"   {c!r}")
                print("=" * 110)
            row["__seed"] = seed
            row["__fname"] = fname
            # Parse numérico
            def f(v):
                try: return float(v)
                except: return None
            P = f(row["precision"])
            R = f(row["recall"])
            F1_csv = f(row["f1"])
            GT = f(row["fields_gt_present"])   # = TP + FN
            PRED = f(row["fields_pred_present"])  # = TP + FP
            # Algebra: TP_predicciones = P * PRED ; TP_GT = R * GT
            # Deben ser casi iguales si el CSV es consistente. Usamos media redondeada si difieren poco.
            TP_from_P = P * PRED if (P is not None and PRED is not None) else None
            TP_from_R = R * GT if (R is not None and GT is not None) else None
            TP_candidates = [x for x in [TP_from_P, TP_from_R] if x is not None]
            if not TP_candidates:
                TP = None
            else:
                # Usamos el promedio y luego redondeamos a entero (TP son aciertos de campos discretos)
                TP = round(sum(TP_candidates) / len(TP_candidates))
            if TP is not None:
                FP = round(PRED) - TP if PRED is not None else None
                FN = round(GT) - TP if GT is not None else None
            else:
                FP = FN = None
            row["__TP"] = TP; row["__FP"] = FP; row["__FN"] = FN
            row["__P"] = P; row["__R"] = R; row["__F1_csv"] = F1_csv
            rows_all.append(row)

print(f"\nTotal filas = {len(rows_all)} (esperadas 360 = 3 × 120)")
assert len(rows_all) == 360

def bool_true(v):
    return str(v).strip().lower() in {"true", "1", "yes", "si", "sí"}

def agg(group, label):
    n_rows = len(group)
    assert n_rows % 3 == 0, f"filas={n_rows} no divisible por 3"
    n_unique_docs = n_rows // 3
    TP = sum(r["__TP"] for r in group if r["__TP"] is not None)
    FP = sum(r["__FP"] for r in group if r["__FP"] is not None)
    FN = sum(r["__FN"] for r in group if r["__FN"] is not None)
    # F1 micro a partir de sum TP / sum FP / sum FN (la forma canónica para agregación micro)
    P_micro = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R_micro = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1_micro = (2 * P_micro * R_micro / (P_micro + R_micro)) if (P_micro + R_micro) > 0 else 0.0
    # docs perfectos por semilla
    perfect_by_seed = defaultdict(int)
    rules_acc_sum = 0.0; rules_count = 0
    for r in group:
        if bool_true(r["document_correct"]):
            perfect_by_seed[r["__seed"]] += 1
        ra = r.get("rules_accuracy")
        if ra not in (None, ""):
            try:
                rules_acc_sum += float(ra); rules_count += 1
            except: pass
    perfect_mean = sum(perfect_by_seed.values()) / 3
    EMR_pct = perfect_mean / n_unique_docs * 100.0
    rules_acc_mean = rules_acc_sum / rules_count if rules_count else None
    return dict(label=label, n_unique=n_unique_docs, rows=n_rows,
                TP=TP, FP=FP, FN=FN, P=P_micro, R=R_micro, F1=F1_micro,
                perfect_mean=perfect_mean, perfect_by_seed=dict(perfect_by_seed), EMR_pct=EMR_pct,
                rules_acc=rules_acc_mean)

def f4(x): return f"{x:.4f}"
def f2(x): return f"{x:.2f}"
HR = "-" * 115

def print_header():
    print(f"{'GRUPO':<28} {'N':>3} {'TP':>6} {'FP':>6} {'FN':>6} {'P micro':>8} {'R micro':>8} {'F1 micro':>9} {'Perfectos':>11} {'EMR %':>8} {'rules_acc':>9}")
    print(HR)

def print_res(res):
    ra = f4(res['rules_acc']) if res['rules_acc'] is not None else "N/A"
    print(f"{res['label']:<28} {res['n_unique']:>3} {res['TP']:>6} {res['FP']:>6} {res['FN']:>6} {f4(res['P']):>8} {f4(res['R']):>8} {f4(res['F1']):>9} {res['perfect_mean']:>7.2f}/{res['n_unique']:<3} {res['EMR_pct']:>7.2f}% {ra:>9}")

print("\n" + "=" * 115); print("1. GLOBAL C1 3 semillas"); print("=" * 115)
g = agg(rows_all, "GLOBAL C1"); print_header(); print_res(g)
print(f"   perfect_by_seed: {g['perfect_by_seed']}")

# --- Verificar si el valor F1_csv promedio coincide con F1 micro agregado ---
avg_f1_csv = sum(r["__F1_csv"] for r in rows_all if r["__F1_csv"] is not None) / len(rows_all)
print(f"   Promedio F1 (csv promedio fila) = {avg_f1_csv:.6f}  vs  F1 micro agregado = {g['F1']:.6f}")
print(f"   (Nota: F1 micro agregado CANÓNICO es el correcto para agregación de grupos, NO la media de F1s individuales)")

print("\n" + "=" * 115); print("2. POR DOMINIO (columna 'schema')"); print("=" * 115)
doms = sorted(set(r["schema"] for r in rows_all))
print_header()
for d in doms:
    group = [r for r in rows_all if r["schema"] == d]
    res = agg(group, d); print_res(res)
    if res["perfect_mean"] > 0:
        sample = next(r for r in group if bool_true(r["document_correct"]))
        print(f"   [muestra doc perfecto en {d}] case_id={sample.get('case_id')!r} pdf={sample.get('pdf')!r}")

print("\n" + "=" * 115); print("3. POR VARIANTE VISUAL (columna 'pdf' / extraído prefijo del filename)"); print("=" * 115)
# Extraer variante desde pdf: el PDF contiene nombre del tipo en el filename: p. ej. "native_pdf_conforme_041.pdf"
# o la columna reading_mode / pdf
def variant_of(r):
    p = r.get("pdf", "")
    for v in ["native_pdf", "scanned_blurry_pdf", "image_photo", "image_handwritten"]:
        if v in p: return v
    return r.get("reading_mode", "UNKNOWN") + "_??"
# Añadir variante
for r in rows_all:
    r["__var"] = variant_of(r)
vars_u = sorted(set(r["__var"] for r in rows_all))
print(f"Variantes detectadas: {vars_u}")
print_header()
for v in vars_u:
    group = [r for r in rows_all if r["__var"] == v]
    res = agg(group, v); print_res(res)
    if res["perfect_mean"] > 0:
        sample = next(r for r in group if bool_true(r["document_correct"]))
        print(f"   [muestra doc perfecto en {v}] schema={sample['schema']!r} case_id={sample.get('case_id')!r}")

print("\n" + "=" * 115); print("4. DISEÑO EXPERIMENTAL (dominio × variante) por SEMILLA"); print("=" * 115)
for seed in [20260501, 20260514, 20260528]:
    rs = [r for r in rows_all if r["__seed"] == seed]
    counts = defaultdict(int)
    for r in rs: counts[(r["schema"], r["__var"])] += 1
    dvars = sorted(set(d for d,_ in counts.keys()))
    vvars = sorted(set(v for _,v in counts.keys()))
    print(f"\n--- seed={seed}: N unique docs por (dominio × variante) ---")
    print(f"{'dominio':<25}" + "".join(f"{v:>22}" for v in vvars) + f"{'| TOTAL dom.':>15}")
    for d in dvars:
        tot = 0
        line = f"{d:<25}"
        for v in vvars:
            c = counts[(d,v)]; tot += c
            line += f"{c:>22}"
        line += f"{tot:>15}"; print(line)
    total_line = f"{'| TOTAL var.':<25}"
    gt = 0
    for v in vvars:
        tv = sum(counts[(d,v)] for d in dvars); gt += tv
        total_line += f"{tv:>22}"
    total_line += f"{gt:>15}"; print(total_line)

print("\n" + "=" * 115); print("5. RELACIÓN: document_correct vs rules_accuracy vs rules_coverage"); print("=" * 115)
perfect_rows = [r for r in rows_all if bool_true(r["document_correct"])]
print(f"Total filas document_correct = {len(perfect_rows)} (se esperan 3 docs × 3 semillas = 9 o 0 docs × 3 = 0)")
for i, r in enumerate(perfect_rows[:15]):
    ra = r.get("rules_accuracy"); rc = r.get("rules_coverage"); f1 = r.get("f1")
    print(f"   i={i} seed={r['__seed']} schema={r['schema']!r} var={r['__var']!r} F1_csv={f1} rules_accuracy={ra!r} rules_coverage={rc!r}")
    print(f"        fields_gt={r.get('fields_gt_present')} fields_pred={r.get('fields_pred_present')} mismatch_count={r.get('mismatch_count')} exact_matches={r.get('exact_matches')}")
