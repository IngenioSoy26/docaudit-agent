from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.evaluate import (
    _compute_field_metrics,
    _iter_samples,
    _load_json,
    _measure_peak_rss_mb,
    _resolve_under_project,
    _run_single,
    _summarize,
)


FIELDNAMES = [
    "case_id",
    "pdf",
    "schema",
    "degraded_text",
    "text_method",
    "extractor_backend",
    "fields_total",
    "fields_gt_present",
    "fields_pred_present",
    "exact_matches",
    "match_rate",
    "precision",
    "recall",
    "f1",
    "document_correct",
    "llm_returned_empty",
    "latency_s",
    "ram_peak_mb",
    "mismatch_count",
]


def _build_row(sample: Any, result: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": sample["case_id"],
        "pdf": sample["pdf_name"],
        "schema": sample["schema_name"],
        "degraded_text": False,
        "text_method": result.get("text_method"),
        "extractor_backend": result.get("extractor_backend"),
        "fields_total": metrics["fields_total"],
        "fields_gt_present": metrics["fields_gt_present"],
        "fields_pred_present": metrics["fields_pred_present"],
        "exact_matches": metrics["exact_matches"],
        "match_rate": metrics["match_rate"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "document_correct": metrics["document_correct"],
        "llm_returned_empty": result.get("llm_returned_empty", False),
        "latency_s": result.get("latency_s", 0.0),
        "ram_peak_mb": result.get("ram_peak_mb", 0.0),
        "mismatch_count": len(metrics["mismatches"]),
    }


def _run_worker(args: argparse.Namespace) -> int:
    pdf_path = Path(args.sample_pdf)
    gt_path = Path(args.sample_gt)
    started_payload = {
        "sample": {
            "case_id": args.case_id,
            "pdf_name": pdf_path.name,
            "schema_name": args.schema_name,
        }
    }

    pdf_bytes = pdf_path.read_bytes()
    gt = _load_json(gt_path)
    started = time.perf_counter()

    def _run():
        return _run_single(
            pdf_bytes=pdf_bytes,
            schema_name=args.schema_name,
            degraded=False,
            float_tol=args.float_tol,
            pdf_name=pdf_path.name,
            backend=args.backend,
        )

    result, peak_rss_mb = _measure_peak_rss_mb(_run)
    elapsed_s = time.perf_counter() - started
    if isinstance(result, dict):
        result["latency_s"] = elapsed_s
        result["ram_peak_mb"] = peak_rss_mb
    metrics = _compute_field_metrics(result["extracted"], gt, float_tol=args.float_tol)
    payload = dict(started_payload)
    payload["result"] = result
    payload["metrics"] = metrics
    Path(args.worker_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _run_parent(args: argparse.Namespace) -> int:
    dataset_root = _resolve_under_project(args.dataset)
    samples = _iter_samples(dataset_root)
    if args.only_doc_type:
        needle = args.only_doc_type.lower()
        samples = [s for s in samples if needle in s.pdf_path.name.lower()]
    if args.offset and args.offset > 0:
        samples = samples[args.offset :]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    out_path = _resolve_under_project(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mismatches_out = out_path.with_suffix("")
    mismatches_dir = mismatches_out.parent / (mismatches_out.name + "_mismatches")
    mismatches_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    print(f"\n{'=' * 80}")
    print(f"INICIANDO EVALUACION AISLADA DE {len(samples)} DOCUMENTOS")
    print(f"{'=' * 80}\n")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        for idx, sample in enumerate(samples, 1):
            print(f"[{idx}/{len(samples)}] Procesando: {sample.pdf_path.name}")
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                worker_out = Path(tmp.name)
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "tools.evaluate_isolated",
                    "--worker",
                    "--sample-pdf",
                    str(sample.pdf_path),
                    "--sample-gt",
                    str(sample.ground_truth_path),
                    "--schema-name",
                    sample.schema_name,
                    "--case-id",
                    sample.case_id,
                    "--backend",
                    args.backend,
                    "--float-tol",
                    str(args.float_tol),
                    "--worker-out",
                    str(worker_out),
                ]
                completed = subprocess.run(
                    cmd,
                    cwd=str(_PROJECT_ROOT),
                    env=os.environ.copy(),
                    timeout=args.per_doc_timeout_s,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if completed.returncode != 0 or not worker_out.exists():
                    raise RuntimeError(
                        f"Worker fallo para {sample.pdf_path.name}. rc={completed.returncode}."
                    )

                payload = json.loads(worker_out.read_text(encoding="utf-8"))
                result = payload["result"]
                metrics = payload["metrics"]
                row = _build_row(
                    {
                        "case_id": sample.case_id,
                        "pdf_name": sample.pdf_path.name,
                        "schema_name": sample.schema_name,
                    },
                    result,
                    metrics,
                )
                rows.append(row)
                writer.writerow(row)
                f.flush()
                mm_path = mismatches_dir / f"{sample.pdf_path.stem}.json"
                mm_path.write_text(json.dumps(metrics["mismatches"], ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"  [OK] Hecho | F1: {metrics['f1']:.3f} | Backend: {result.get('extractor_backend')} | Metodo: {result.get('text_method')}"
                )
            except subprocess.TimeoutExpired:
                print(f"  X Timeout en {sample.pdf_path.name}")
            except Exception as exc:
                print(f"  X Error en {sample.pdf_path.name}: {exc}")
            finally:
                try:
                    worker_out.unlink(missing_ok=True)
                except Exception:
                    pass

    if not rows:
        print(f"ERROR: no se pudo evaluar ningun documento de {args.only_doc_type or 'subset'}")
        return 2

    summary = _summarize(rows)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCSV: {out_path}")
    print(f"Resumen JSON: {summary_path}")
    print(f"Mismatches: {mismatches_dir}")
    print(f"{'=' * 80}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluacion aislada por documento.")
    parser.add_argument("--worker", action="store_true", help="Ejecuta un unico documento y vuelca JSON intermedio.")
    parser.add_argument("--dataset", type=str, default=str(Path("data") / "sample_docs"))
    parser.add_argument("--out", type=str, default=str(Path("reports") / "evaluation_results_isolated.csv"))
    parser.add_argument("--backend", type=str, default="llm", choices=["auto", "llm", "heuristic"])
    parser.add_argument("--float-tol", type=float, default=0.01)
    parser.add_argument("--only-doc-type", type=str, default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-doc-timeout-s", type=int, default=900)
    parser.add_argument("--sample-pdf", type=str, default="")
    parser.add_argument("--sample-gt", type=str, default="")
    parser.add_argument("--schema-name", type=str, default="")
    parser.add_argument("--case-id", type=str, default="")
    parser.add_argument("--worker-out", type=str, default="")
    args = parser.parse_args()
    if args.worker:
        return _run_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
