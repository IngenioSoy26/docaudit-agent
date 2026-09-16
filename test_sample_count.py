
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.evaluate import _iter_samples

dataset_root = _PROJECT_ROOT / "data" / "sample_docs"
samples = _iter_samples(dataset_root)

print(f"Total de documentos encontrados: {len(samples)}")
print("\nMuestras encontradas:")
for i, sample in enumerate(samples, 1):
    print(f"{i}. {sample.pdf_path.name} -> {sample.ground_truth_path.name}")
