from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
REPORT_DIR = BASE_DIR / os.getenv("REPORT_DIR", "reports")
ACCURACY_DROP_THRESHOLD = float(os.getenv("ACCURACY_DROP_THRESHOLD", "0.015"))
MAX_REFINEMENT_ITERATIONS = int(os.getenv("MAX_REFINEMENT_ITERATIONS", "5"))
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
