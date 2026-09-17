from __future__ import annotations

import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Thresholds and Iterations
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "1.5"))  # in percent, e.g. 1.5%
MAX_REFINEMENT_ITERATIONS = int(os.getenv("MAX_REFINEMENT_ITERATIONS", "5"))
DEFAULT_PRUNING_RATIO = float(os.getenv("DEFAULT_PRUNING_RATIO", "0.30"))
DEFAULT_QUANTIZATION = os.getenv("DEFAULT_QUANTIZATION", "dynamic_int8")

# Upload and Security Configuration
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))

# Directories
MODEL_DIRECTORY = BASE_DIR / os.getenv("MODEL_DIR", "models")
OUTPUT_DIRECTORY = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
REPORT_DIRECTORY = BASE_DIR / os.getenv("REPORT_DIR", "reports")
RUNS_DIRECTORY = BASE_DIR / os.getenv("RUNS_DIR", "runs")

TENSORBOARD_DIRECTORY = RUNS_DIRECTORY / "neuromorphic_ops"

# Ensure directories exist
for directory in [
    MODEL_DIRECTORY,
    OUTPUT_DIRECTORY,
    REPORT_DIRECTORY,
    RUNS_DIRECTORY,
    TENSORBOARD_DIRECTORY,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging and API
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_URL = os.getenv("API_URL", "http://localhost:8000")
OUTPUT_DIR = OUTPUT_DIRECTORY
REPORT_DIR = REPORT_DIRECTORY
TENSORBOARD_DIR = TENSORBOARD_DIRECTORY