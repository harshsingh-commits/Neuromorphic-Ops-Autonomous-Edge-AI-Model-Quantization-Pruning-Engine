from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import OUTPUT_DIRECTORY
from backend.main import app
from backend.services.onnx_service import (
    validate_onnx_artifact,
    validate_onnx_input,
)
from backend.utils.file_utils import (
    sanitize_filename,
    validate_safe_path,
)


client = TestClient(app)


# ============================================================
# 21.4 PATH TRAVERSAL TESTS
# ============================================================

def test_safe_path_allows_file_inside_base_directory(tmp_path: Path) -> None:
    base_dir = tmp_path / "outputs"
    base_dir.mkdir()

    target = base_dir / "model.pth"

    result = validate_safe_path(
        base_dir,
        target,
    )

    assert result == target.resolve()


def test_safe_path_rejects_parent_traversal(tmp_path: Path) -> None:
    base_dir = tmp_path / "outputs"
    base_dir.mkdir()

    target = base_dir / ".." / "evil.pth"

    with pytest.raises(ValueError, match="Unsafe path detected"):
        validate_safe_path(
            base_dir,
            target,
        )


def test_safe_path_rejects_absolute_path_outside_base(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "outputs"
    base_dir.mkdir()

    outside = tmp_path / "evil.pth"

    with pytest.raises(ValueError, match="Unsafe path detected"):
        validate_safe_path(
            base_dir,
            outside,
        )


# ============================================================
# 21.4 FILENAME SANITIZATION TESTS
# ============================================================

def test_sanitize_filename_removes_path_components() -> None:
    result = sanitize_filename(
        r"..\..\evil_model.pth"
    )

    assert result == "evil_model.pth"


def test_sanitize_filename_removes_unsafe_characters() -> None:
    result = sanitize_filename(
        "my model@#$%^&.pth"
    )

    assert result == "mymodel.pth"


def test_sanitize_filename_handles_empty_filename() -> None:
    result = sanitize_filename("")

    assert result == "uploaded_model.pt"


# ============================================================
# 21.3 ONNX ARTIFACT VALIDATION TESTS
# ============================================================

def test_invalid_onnx_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    bad_model = tmp_path / "bad.onnx"

    bad_model.write_bytes(
        b"this is not a valid onnx model"
    )

    result = validate_onnx_artifact(
        bad_model
    )

    assert result["valid"] is False
    assert result["status"] == "invalid"
    assert result["error"] is not None


def test_empty_onnx_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    empty_model = tmp_path / "empty.onnx"
    empty_model.touch()

    result = validate_onnx_artifact(
        empty_model
    )

    assert result["valid"] is False
    assert result["status"] == "invalid"
    assert "empty" in result["error"].lower()


def test_wrong_onnx_extension_is_rejected(
    tmp_path: Path,
) -> None:
    wrong_file = tmp_path / "model.txt"

    wrong_file.write_text(
        "invalid model",
        encoding="utf-8",
    )

    result = validate_onnx_artifact(
        wrong_file
    )

    assert result["valid"] is False
    assert result["status"] == "invalid"
    assert "onnx" in result["error"].lower()


def test_missing_onnx_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    missing_model = tmp_path / "missing.onnx"

    result = validate_onnx_artifact(
        missing_model
    )

    assert result["valid"] is False
    assert result["status"] == "invalid"


# ============================================================
# 21.5 ONNX INPUT VALIDATION TESTS
# ============================================================

class FakeInput:
    name = "input"
    shape = [1, 3, 32, 32]
    type = "tensor(float)"


class FakeSession:
    @staticmethod
    def get_inputs():
        return [FakeInput()]


def test_onnx_input_rejects_wrong_rank() -> None:
    session = FakeSession()

    tensor = np.zeros(
        (1, 3, 32),
        dtype=np.float32,
    )

    result = validate_onnx_input(
        session,
        tensor,
    )

    assert result["valid"] is False
    assert "rank mismatch" in result["error"].lower()


def test_onnx_input_rejects_wrong_dimensions() -> None:
    session = FakeSession()

    tensor = np.zeros(
        (1, 3, 64, 64),
        dtype=np.float32,
    )

    result = validate_onnx_input(
        session,
        tensor,
    )

    assert result["valid"] is False
    assert "dimension mismatch" in result["error"].lower()


def test_onnx_input_rejects_wrong_dtype() -> None:
    session = FakeSession()

    tensor = np.zeros(
        (1, 3, 32, 32),
        dtype=np.float64,
    )

    result = validate_onnx_input(
        session,
        tensor,
    )

    assert result["valid"] is False
    assert "float32" in result["error"].lower()


def test_onnx_input_rejects_nan() -> None:
    session = FakeSession()

    tensor = np.zeros(
        (1, 3, 32, 32),
        dtype=np.float32,
    )

    tensor[0, 0, 0, 0] = np.nan

    result = validate_onnx_input(
        session,
        tensor,
    )

    assert result["valid"] is False
    assert "nan" in result["error"].lower()


def test_onnx_input_rejects_infinity() -> None:
    session = FakeSession()

    tensor = np.zeros(
        (1, 3, 32, 32),
        dtype=np.float32,
    )

    tensor[0, 0, 0, 0] = np.inf

    result = validate_onnx_input(
        session,
        tensor,
    )

    assert result["valid"] is False
    assert (
        "nan" in result["error"].lower()
        or "infinite" in result["error"].lower()
    )


def test_onnx_input_accepts_valid_tensor() -> None:
    session = FakeSession()

    tensor = np.zeros(
        (1, 3, 32, 32),
        dtype=np.float32,
    )

    result = validate_onnx_input(
        session,
        tensor,
    )

    assert result["valid"] is True
    assert result["error"] is None


# ============================================================
# 21.1 UPLOAD API SECURITY TESTS
# ============================================================

def test_upload_rejects_unsupported_extension() -> None:
    response = client.post(
        "/upload",
        files={
            "file": (
                "malicious.exe",
                b"fake content",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert "pt" in response.json()["detail"]
    assert "pth" in response.json()["detail"]


def test_upload_rejects_empty_model() -> None:
    response = client.post(
        "/upload",
        files={
            "file": (
                "empty.pth",
                b"",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_sanitizes_traversal_filename() -> None:
    response = client.post(
        "/upload",
        files={
            "file": (
                "../../evil.pth",
                b"test model",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code in {
        200,
        201,
    }

    body = response.json()

    assert body["status"] == "uploaded"

    uploaded_path = Path(
        body["model_path"]
    )

    assert uploaded_path.name == "evil.pth"

    # Uploaded file must remain inside the configured
    # output directory.
    assert uploaded_path.parent.resolve() == (
        Path(OUTPUT_DIRECTORY).resolve()
    )


# ============================================================
# SECURITY TEST MARKER
# ============================================================

def test_security_suite_loaded() -> None:
    """
    Simple sanity test ensuring pytest discovers this module.
    """
    assert True