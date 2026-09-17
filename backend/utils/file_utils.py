from __future__ import annotations

import os
import re
from pathlib import Path

from backend.config import (
    MODEL_DIRECTORY,
    OUTPUT_DIRECTORY,
    REPORT_DIRECTORY,
)


# ============================================================
# FILE SIZE
# ============================================================

def get_file_size_mb(path: str | Path) -> float:
    """Returns file size in Megabytes."""

    p = Path(path)

    if not p.exists():
        return 0.0

    if not p.is_file():
        return 0.0

    return p.stat().st_size / (1024 * 1024)


# ============================================================
# FILENAME SANITIZATION
# ============================================================

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename to prevent directory traversal and
    remove unsafe characters.
    """

    if not filename:
        return "uploaded_model.pt"

    # Normalize Windows-style separators as well as Unix-style.
    normalized = filename.replace("\\", "/")

    # Keep only the final filename component.
    base = os.path.basename(normalized)

    # Remove unsafe characters.
    clean = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "",
        base,
    )

    # Prevent special path names.
    if clean in {"", ".", ".."}:
        clean = "uploaded_model.pt"

    return clean


# ============================================================
# SAFE PATH VALIDATION
# ============================================================

def validate_safe_path(
    base_dir: Path,
    target_path: str | Path,
) -> Path:
    """
    Ensures target_path is safely contained inside base_dir.

    Protects against:
        - ../ traversal
        - ..\\ traversal
        - absolute paths outside the base directory
        - symlink-based escape from the base directory

    Returns:
        Resolved target path.

    Raises:
        ValueError:
            When the target escapes the allowed base directory.
    """

    resolved_base = Path(base_dir).resolve()
    resolved_target = Path(target_path).resolve()

    try:
        resolved_target.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(
            f"Unsafe path detected: "
            f"{target_path} is outside {base_dir}"
        ) from exc

    return resolved_target


# ============================================================
# MODEL PATH VALIDATION
# ============================================================

def validate_model_path(
    target_path: str | Path,
) -> Path:
    """
    Validates that a model path is inside either the configured
    model directory or output directory.
    """

    path = Path(target_path)

    for base_dir in (
        MODEL_DIRECTORY,
        OUTPUT_DIRECTORY,
    ):
        try:
            return validate_safe_path(
                Path(base_dir),
                path,
            )
        except ValueError:
            continue

    raise ValueError(
        f"Unsafe model path detected: {target_path}"
    )


# ============================================================
# REPORT PATH VALIDATION
# ============================================================

def validate_report_path(
    target_path: str | Path,
) -> Path:
    """
    Validates that a report path is safely contained inside
    the configured report directory.
    """

    return validate_safe_path(
        Path(REPORT_DIRECTORY),
        Path(target_path),
    )