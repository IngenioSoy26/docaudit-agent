from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_eval(
    *,
    ocr_backend: str,
    doc_type: str,
    limit: int,
    out_csv: Path,
    extract_backend: str,
) -> dict:
    env = dict(os.environ)
    env["OCR_BACKEND"] = ocr_backend
    env[f"OCR_BACKEND_{doc_type.upper()}"] = ocr_backend
    cmd = [
        sys.executable,
        "-m",
        "tools.evaluate",
        "--backend",
        extract_backend,
        "--only-doc-type",
        doc_type,
        "--limit",
        str(limit),
        "--out",
        str(out_csv),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
    summary_path = out_csv.with_suffix(".summary.json")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark rapido de backends OCR por tipo documental.")
    parser.add_argument("--limit", type=int, default=6, help="Muestras por tipo documental.")
    parser.add_argument("--out-dir", type=str, default=str(Path("reports") / "ocr_benchmark"))
    parser.add_argument(
        "--extract-backend",
        type=str,
        default="llm",
        choices=["auto", "llm", "heuristic"],
        help="Backend de extracción para la evaluación (auto/llm/heuristic).",
    )
    parser.add_argument(
        "--doc-types",
        type=str,
        default="scanned_blurry_pdf,image_photo,image_handwritten",
        help="Lista separada por comas de tipos documentales a evaluar.",
    )
    parser.add_argument(
        "--ocr-backends",
        type=str,
        default="easyocr,rapidocr",
        help="Lista separada por comas de backends OCR a comparar.",
    )
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_types = [x.strip() for x in (args.doc_types or "").split(",") if x.strip()]
    backends = [x.strip() for x in (args.ocr_backends or "").split(",") if x.strip()]
    rows: list[dict] = []

    for doc_type in doc_types:
        for backend in backends:
            out_csv = out_dir / f"{doc_type}_{backend}.csv"
            summary = _run_eval(
                ocr_backend=backend,
                doc_type=doc_type,
                limit=args.limit,
                out_csv=out_csv,
                extract_backend=args.extract_backend,
            )
            metrics = summary["by_doc_type"].get(doc_type, summary["overall"])
            rows.append(
                {
                    "doc_type": doc_type,
                    "ocr_backend": backend,
                    "runs": metrics["runs"],
                    "micro_f1": metrics["micro_f1"],
                    "micro_precision": metrics["micro_precision"],
                    "micro_recall": metrics["micro_recall"],
                    "micro_rules_accuracy": metrics.get("micro_rules_accuracy", 0.0),
                    "micro_rules_coverage": metrics.get("micro_rules_coverage", 0.0),
                    "avg_latency_s": metrics["avg_latency_s"],
                    "document_exact_match": metrics["document_exact_match"],
                }
            )

    result_path = out_dir / "benchmark_summary.json"
    result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
