
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.models.schemas import (
    ApprovalRequest,
    HealthResponse,
    OptimizationRequest,
    WorkflowResponse,
)
from backend.services.monitoring import monitor
from backend.services.onnx_service import (
    benchmark_onnx_runtime,
    run_onnx_inference,
)
from backend.services.orchestrator import orchestrator
from backend.utils.config import OUTPUT_DIR, REPORT_DIR
from backend.utils.logging import configure_logging


# ============================================================
# APPLICATION SETUP
# ============================================================

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Neuromorphic-Ops",
    version="1.0.0",
    description=(
        "Autonomous Edge-AI Model Quantization "
        "and Pruning Engine"
    ),
)


# ============================================================
# PHASE 22.3 - REQUEST / ERROR MONITORING
# ============================================================

@app.middleware("http")
async def monitoring_middleware(request, call_next):
    """
    Automatically monitor every HTTP request.

    Tracks:
        - HTTP method
        - request path
        - status code
        - request latency
        - unexpected errors
    """

    start_time = time.perf_counter()

    try:

        response = await call_next(request)

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        monitor.record_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        return response

    except Exception as exc:

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        monitor.record_request(
            method=request.method,
            path=request.url.path,
            status_code=500,
            latency_ms=latency_ms,
        )

        monitor.record_error(
            str(exc)
        )

        raise


# ============================================================
# CONFIGURATION
# ============================================================

MAX_UPLOAD_SIZE = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pt",
    ".pth",
}

DEFAULT_ONNX_PATH = (
    OUTPUT_DIR / "optimized_model.onnx"
)

MAX_INFERENCE_ELEMENTS = 10_000_000


# ============================================================
# PATH SECURITY HELPERS
# ============================================================

def get_safe_output_path(
    filename: str,
) -> Path:
    """
    Return a resolved path guaranteed to remain inside
    OUTPUT_DIR.

    Prevents path traversal such as:
        ../../model.pt
        ..\\..\\model.pt
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_dir = OUTPUT_DIR.resolve()

    safe_name = Path(
        filename
    ).name

    if not safe_name:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    target = (
        base_dir / safe_name
    ).resolve()

    try:

        target.relative_to(
            base_dir
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail="Invalid model path.",
        ) from exc

    return target


def validate_model_path(
    model_path: str,
) -> Path:
    """
    Validate an existing model path.

    The model must:
        - exist
        - be a regular file
        - use .pt or .pth
        - remain inside OUTPUT_DIR
    """

    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="Model path is required.",
        )

    base_dir = OUTPUT_DIR.resolve()

    candidate = Path(
        model_path
    ).resolve()

    try:

        candidate.relative_to(
            base_dir
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=403,
            detail=(
                "Model path must be located inside "
                "the configured model output directory."
            ),
        ) from exc

    if not candidate.exists():

        raise HTTPException(
            status_code=404,
            detail="Model path does not exist.",
        )

    if not candidate.is_file():

        raise HTTPException(
            status_code=400,
            detail="Model path must point to a file.",
        )

    if (
        candidate.suffix.lower()
        not in ALLOWED_EXTENSIONS
    ):

        raise HTTPException(
            status_code=400,
            detail="Unsupported model file type.",
        )

    if candidate.stat().st_size <= 0:

        raise HTTPException(
            status_code=400,
            detail="Model file is empty.",
        )

    return candidate


# ============================================================
# ONNX INFERENCE REQUEST
# ============================================================

class ONNXInferenceRequest(BaseModel):
    """
    Request body for CPU-based ONNX Runtime inference.

    The server generates a float32 test tensor from input_shape.

    DemoCNN:
        [1, 3, 32, 32]
    """

    input_shape: list[int] = Field(
        default=[1, 3, 32, 32],
        min_length=1,
        max_length=6,
        description=(
            "Input tensor shape. "
            "For DemoCNN use [1, 3, 32, 32]."
        ),
    )


# ============================================================
# ONNX BENCHMARK REQUEST
# ============================================================

class ONNXBenchmarkRequest(BaseModel):
    """
    Request body for ONNX Runtime CPU benchmarking.
    """

    input_shape: list[int] = Field(
        default=[1, 3, 32, 32],
        min_length=1,
        max_length=6,
        description=(
            "Input tensor shape. "
            "For DemoCNN use [1, 3, 32, 32]."
        ),
    )

    warmup_runs: int = Field(
        default=10,
        ge=0,
        le=1000,
        description="Number of warmup inference runs.",
    )

    runs: int = Field(
        default=50,
        ge=1,
        le=5000,
        description="Number of timed inference runs.",
    )


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """
    Service health check.
    """

    return HealthResponse(
        status="ok",
        version="1.0.0",
    )


# ============================================================
# PHASE 22.2 - LIVENESS
# ============================================================

@app.get("/health/live")
def liveness() -> dict[str, str]:
    """
    Liveness check.

    Confirms that the API process is alive and responding.
    """

    return {
        "status": "alive",
    }


# ============================================================
# PHASE 22.2 - READINESS
# ============================================================

@app.get("/health/ready")
def readiness() -> dict[str, Any]:
    """
    Readiness check.

    Verifies that the runtime environment and the latest
    optimized ONNX artifact are available.
    """

    checks: dict[str, Any] = {}

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    try:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        checks["output_directory"] = (
            OUTPUT_DIR.exists()
            and OUTPUT_DIR.is_dir()
        )

    except Exception:

        checks["output_directory"] = False

    # --------------------------------------------------------
    # Optimized ONNX artifact
    # --------------------------------------------------------

    try:

        checks["onnx_artifact"] = (
            DEFAULT_ONNX_PATH.exists()
            and DEFAULT_ONNX_PATH.is_file()
            and DEFAULT_ONNX_PATH.stat().st_size > 0
        )

    except OSError:

        checks["onnx_artifact"] = False

    # --------------------------------------------------------
    # Overall readiness
    # --------------------------------------------------------

    ready = all(
        checks.values()
    )

    return {
        "status": (
            "ready"
            if ready
            else "not_ready"
        ),
        "checks": checks,
        "monitoring": monitor.snapshot(),
    }


# ============================================================
# PHASE 22.3 - LIVE MONITORING
# ============================================================

@app.get("/monitoring")
def monitoring_status() -> dict[str, Any]:
    """
    Return live runtime monitoring metrics from the
    currently running FastAPI process.
    """

    return monitor.snapshot()


# ============================================================
# SECURE MODEL UPLOAD
# ============================================================

@app.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
) -> dict[str, str]:
    """
    Securely upload a PyTorch model artifact.

    Security checks:
        - filename required
        - .pt / .pth only
        - path traversal protection
        - 100 MB size limit
        - empty file rejection

    Upload behavior:
        - streams the file in 1 MB chunks
        - writes to a temporary file first
        - atomically replaces the destination
        - prevents partial model artifacts
        - logs the real exception
    """

    # --------------------------------------------------------
    # Filename validation
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    original_name = Path(
        file.filename
    ).name

    if not original_name:

        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    suffix = Path(
        original_name
    ).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PyTorch model files are accepted: "
                ".pt or .pth"
            ),
        )

    # --------------------------------------------------------
    # Safe destination
    # --------------------------------------------------------

    target = get_safe_output_path(
        original_name
    )

    temporary_target = target.with_name(
        f".{target.name}.uploading"
    )

    total_size = 0

    try:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Write upload to temporary file
        # ----------------------------------------------------

        with temporary_target.open(
            "wb"
        ) as destination:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(
                    chunk
                )

                # --------------------------------------------
                # Upload size protection
                # --------------------------------------------

                if total_size > MAX_UPLOAD_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Model file exceeds 100 MB."
                        ),
                    )

                destination.write(
                    chunk
                )

            destination.flush()

        # ----------------------------------------------------
        # Empty-file validation
        # ----------------------------------------------------

        if total_size <= 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded model file is empty."
                ),
            )

        # ----------------------------------------------------
        # Verify the temporary artifact exists
        # ----------------------------------------------------

        if not temporary_target.exists():

            raise RuntimeError(
                "Temporary upload file was not created."
            )

        temporary_size = (
            temporary_target.stat().st_size
        )

        if temporary_size != total_size:

            raise RuntimeError(
                "Uploaded file size verification failed."
            )

        # ----------------------------------------------------
        # Atomic replacement
        # ----------------------------------------------------

        temporary_target.replace(
            target
        )

        logger.info(
            "Model uploaded successfully: "
            "%s (%d bytes)",
            target,
            total_size,
        )

        return {
            "model_path": str(
                target
            ),
            "status": "uploaded",
        }

    except HTTPException:

        temporary_target.unlink(
            missing_ok=True
        )

        raise

    except Exception as exc:

        temporary_target.unlink(
            missing_ok=True
        )

        logger.exception(
            "Model upload failed for '%s': %s",
            original_name,
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Model upload failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    finally:

        try:
            await file.close()
        except Exception as exc:
            logger.warning(
                "Could not close uploaded file '%s': %s",
                original_name,
                exc,
            )


# ============================================================
# PHASE 22.4 - OPTIMIZATION MONITORING
# ============================================================

@app.post(
    "/optimize",
    response_model=WorkflowResponse,
)
def optimize(
    request: OptimizationRequest,
) -> WorkflowResponse:
    """
    Start the LangGraph optimization workflow.

    The model must be located inside OUTPUT_DIR.

    Monitoring:
        - started count
        - completed count
        - failed count
    """

    model_path = validate_model_path(
        request.model_path
    )

    # --------------------------------------------------------
    # Optimization started
    # --------------------------------------------------------

    monitor.record_optimization_started()

    try:

        thread_id, state = orchestrator.start(
            model_path=str(
                model_path
            ),
            strategy=request.strategy,
            pruning_ratio=request.pruning_ratio,
            accuracy_threshold=request.accuracy_threshold,
            quantization_type=request.quantization_type,
        )

        workflow_status = str(
            state.get(
                "workflow_status",
                "",
            )
        ).lower()

        # ----------------------------------------------------
        # Only count genuine terminal completion.
        # ----------------------------------------------------

        terminal_statuses = {
            "workflow_completed",
            "deployment_ready",
            "completed",
        }

        if workflow_status in terminal_statuses:

            monitor.record_optimization_completed()

    except Exception as exc:

        monitor.record_optimization_failed(
            str(exc)
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return WorkflowResponse(
        thread_id=thread_id,
        status=state.get(
            "workflow_status",
            state.get(
                "deployment_status",
                "running",
            ),
        ),
        state=state,
    )


# ============================================================
# ONNX RUNTIME INFERENCE
# ============================================================

@app.post(
    "/inference/onnx",
)
def onnx_inference(
    request: ONNXInferenceRequest,
) -> dict[str, Any]:
    """
    Run CPU inference using the generated optimized ONNX model.

    The endpoint always uses:
        outputs/optimized_model.onnx
    """

    onnx_path = DEFAULT_ONNX_PATH

    if not onnx_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "No optimized ONNX model has been generated. "
                "Run /optimize first."
            ),
        )

    if not onnx_path.is_file():

        raise HTTPException(
            status_code=400,
            detail="Optimized ONNX artifact is invalid.",
        )

    input_shape = [
        int(dimension)
        for dimension in request.input_shape
    ]

    if not input_shape:

        raise HTTPException(
            status_code=400,
            detail="input_shape must not be empty.",
        )

    if any(
        dimension <= 0
        for dimension in input_shape
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "All input_shape dimensions "
                "must be greater than zero."
            ),
        )

    element_count = 1

    for dimension in input_shape:

        element_count *= dimension

        if element_count > MAX_INFERENCE_ELEMENTS:

            raise HTTPException(
                status_code=413,
                detail=(
                    "Requested input tensor is too large."
                ),
            )

    try:

        input_tensor = torch.randn(
            tuple(input_shape),
            dtype=torch.float32,
        )

        result = run_onnx_inference(
            onnx_path=onnx_path,
            input_tensor=input_tensor,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except TypeError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "ONNX Runtime inference failed: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# ONNX RUNTIME BENCHMARK
# ============================================================

@app.post(
    "/inference/onnx/benchmark",
)
def benchmark_onnx(
    request: ONNXBenchmarkRequest,
) -> dict[str, Any]:
    """
    Benchmark the optimized ONNX model using
    ONNX Runtime CPUExecutionProvider.
    """

    onnx_path = DEFAULT_ONNX_PATH

    if not onnx_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "No optimized ONNX model has been generated. "
                "Run /optimize first."
            ),
        )

    if not onnx_path.is_file():

        raise HTTPException(
            status_code=400,
            detail="Optimized ONNX artifact is invalid.",
        )

    input_shape = [
        int(dimension)
        for dimension in request.input_shape
    ]

    if not input_shape:

        raise HTTPException(
            status_code=400,
            detail="input_shape must not be empty.",
        )

    if any(
        dimension <= 0
        for dimension in input_shape
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "All input_shape dimensions "
                "must be greater than zero."
            ),
        )

    element_count = 1

    for dimension in input_shape:

        element_count *= dimension

        if element_count > MAX_INFERENCE_ELEMENTS:

            raise HTTPException(
                status_code=413,
                detail=(
                    "Requested input tensor is too large."
                ),
            )

    try:

        result = benchmark_onnx_runtime(
            onnx_path=onnx_path,
            input_shape=tuple(
                input_shape
            ),
            warmup_runs=request.warmup_runs,
            runs=request.runs,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except TypeError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "ONNX Runtime benchmark failed: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# PHASE 22.4 - APPROVAL / COMPLETION MONITORING
# ============================================================

@app.post(
    "/approve/{thread_id}",
    response_model=WorkflowResponse,
)
def approve(
    thread_id: str,
    request: ApprovalRequest,
) -> WorkflowResponse:
    """
    Approve or reject an optimization workflow.

    Monitoring:
        - completed count is incremented when approval resumes
          the workflow into a terminal completed state
        - failed count is incremented when approval rejects or
          blocks the workflow
        - invalid thread IDs are API errors, not optimization
          failures
    """

    try:

        state = orchestrator.approve(
            thread_id,
            request.approved,
        )

    except KeyError:

        raise HTTPException(
            status_code=404,
            detail="Workflow thread not found.",
        )

    except Exception as exc:

        monitor.record_optimization_failed(
            str(exc),
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # Determine resulting workflow state
    # --------------------------------------------------------

    workflow_status = str(
        state.get(
            "workflow_status",
            "",
        )
    ).lower()

    deployment_status = str(
        state.get(
            "deployment_status",
            "",
        )
    ).lower()

    approval_status = str(
        state.get(
            "approval_status",
            "",
        )
    ).lower()

    # --------------------------------------------------------
    # Completed optimization
    # --------------------------------------------------------

    terminal_success_states = {
        "completed",
        "workflow_completed",
        "deployment_ready",
    }

    if (
        workflow_status in terminal_success_states
        or deployment_status == "deployment_ready"
    ):

        monitor.record_optimization_completed()

    # --------------------------------------------------------
    # Rejected / blocked / failed optimization
    # --------------------------------------------------------

    elif (
        workflow_status == "failed"
        or deployment_status == "blocked"
        or approval_status in {
            "rejected",
            "blocked",
        }
    ):

        monitor.record_optimization_failed(
            state.get(
                "error",
                "Optimization workflow was rejected or blocked.",
            )
        )

    return WorkflowResponse(
        thread_id=thread_id,
        status=state.get(
            "workflow_status",
            state.get(
                "deployment_status",
                "running",
            ),
        ),
        state=state,
    )


# ============================================================
# STATUS
# ============================================================

@app.get(
    "/status/{thread_id}"
)
def status(
    thread_id: str,
) -> dict:
    """
    Return current optimization workflow status.
    """

    try:

        return orchestrator.status(
            thread_id
        )

    except KeyError:

        raise HTTPException(
            status_code=404,
            detail="Workflow thread not found.",
        )

    except Exception as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


# ============================================================
# REPORT
# ============================================================

@app.get("/report")
def report() -> FileResponse:
    """
    Download the latest optimization report.
    """

    path = (
        REPORT_DIR
        / "optimization_report.json"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="No report has been generated.",
        )

    return FileResponse(
        path=path,
        media_type="application/json",
        filename=path.name,
    )


# ============================================================
# DOWNLOAD ONNX
# ============================================================

@app.get("/download/onnx")
def download_onnx() -> FileResponse:
    """
    Download the optimized ONNX artifact.
    """

    path = (
        OUTPUT_DIR
        / "optimized_model.onnx"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "No optimized ONNX artifact "
                "has been generated."
            ),
        )

    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename=path.name,
    )

