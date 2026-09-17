
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


logger = logging.getLogger(__name__)


# ============================================================
# SECURITY / VALIDATION CONFIGURATION
# ============================================================

ALLOWED_ONNX_EXTENSION = ".onnx"
MAX_ONNX_FILE_SIZE = 500 * 1024 * 1024
MAX_INPUT_ELEMENTS = 10_000_000


# ============================================================
# ONNX RUNTIME SESSION CACHE
# ============================================================

_ONNX_SESSION_CACHE: dict[
    str,
    tuple[
        tuple[int, int, int],
        ort.InferenceSession,
    ],
] = {}

_ONNX_SESSION_CACHE_LOCK = threading.RLock()


def _file_signature(
    onnx_path: str | Path,
) -> tuple[str, tuple[int, int, int]]:
    """
    Return the resolved model path and file signature.

    Signature:
        - file size
        - modification time
        - creation/change time
    """

    path = Path(
        onnx_path
    ).resolve()

    stat = path.stat()

    signature = (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )

    return (
        str(path),
        signature,
    )


def _get_cached_session(
    onnx_path: str | Path,
) -> ort.InferenceSession:
    """
    Return a reusable ONNX Runtime session.

    A new session is created only when:
        - model is not cached
        - model file has changed
    """

    cache_key, signature = _file_signature(
        onnx_path
    )

    with _ONNX_SESSION_CACHE_LOCK:

        cached = _ONNX_SESSION_CACHE.get(
            cache_key
        )

        if cached is not None:

            cached_signature, cached_session = (
                cached
            )

            if cached_signature == signature:

                logger.debug(
                    "Reusing cached ONNX Runtime session: %s",
                    cache_key,
                )

                return cached_session

            logger.info(
                "ONNX artifact changed; rebuilding session: %s",
                cache_key,
            )

            _ONNX_SESSION_CACHE.pop(
                cache_key,
                None,
            )

        logger.info(
            "Creating ONNX Runtime session: %s",
            cache_key,
        )

        session = ort.InferenceSession(
            cache_key,
            providers=[
                "CPUExecutionProvider"
            ],
        )

        _ONNX_SESSION_CACHE[
            cache_key
        ] = (
            signature,
            session,
        )

        return session


def clear_onnx_session_cache() -> None:
    """
    Clear all cached ONNX Runtime sessions.
    """

    with _ONNX_SESSION_CACHE_LOCK:

        cache_size = len(
            _ONNX_SESSION_CACHE
        )

        _ONNX_SESSION_CACHE.clear()

        logger.info(
            "Cleared %d cached ONNX Runtime session(s).",
            cache_size,
        )


# ============================================================
# PATH + FILE VALIDATION
# ============================================================

def validate_onnx_artifact(
    onnx_path: str | Path,
) -> dict[str, Any]:
    """
    Validate an ONNX artifact before inference
    or benchmarking.
    """

    path = Path(
        onnx_path
    )

    # --------------------------------------------------------
    # Existence
    # --------------------------------------------------------

    if not path.exists():

        return {
            "valid": False,
            "status": "invalid",
            "error": (
                "ONNX model file does not exist."
            ),
        }

    # --------------------------------------------------------
    # Regular file
    # --------------------------------------------------------

    if not path.is_file():

        return {
            "valid": False,
            "status": "invalid",
            "error": (
                "ONNX model path must point to a regular file."
            ),
        }

    # --------------------------------------------------------
    # Extension
    # --------------------------------------------------------

    if (
        path.suffix.lower()
        != ALLOWED_ONNX_EXTENSION
    ):

        return {
            "valid": False,
            "status": "invalid",
            "error": (
                "Only .onnx model files are supported."
            ),
        }

    # --------------------------------------------------------
    # Size
    # --------------------------------------------------------

    try:

        file_size = int(
            path.stat().st_size
        )

    except OSError as exc:

        return {
            "valid": False,
            "status": "invalid",
            "error": (
                f"Unable to inspect ONNX model file: {exc}"
            ),
        }

    if file_size <= 0:

        return {
            "valid": False,
            "status": "invalid",
            "error": (
                "ONNX model file is empty."
            ),
        }

    if file_size > MAX_ONNX_FILE_SIZE:

        return {
            "valid": False,
            "status": "invalid",
            "error": (
                "ONNX model file exceeds the maximum "
                "allowed file size."
            ),
        }

    # --------------------------------------------------------
    # ONNX structural validation
    # --------------------------------------------------------

    try:

        model = onnx.load(
            str(path)
        )

        onnx.checker.check_model(
            model
        )

    except Exception as exc:

        logger.exception(
            "ONNX structural validation failed."
        )

        return {
            "valid": False,
            "status": "invalid",
            "error": (
                f"Invalid ONNX model: {exc}"
            ),
        }

    # --------------------------------------------------------
    # Graph validation
    # --------------------------------------------------------

    if not model.graph.input:

        return {
            "valid": False,
            "status": "invalid",
            "error": (
                "ONNX model contains no graph inputs."
            ),
        }

    if not model.graph.output:

        return {
            "valid": False,
            "status": "invalid",
            "error": (
                "ONNX model contains no graph outputs."
            ),
        }

    # --------------------------------------------------------
    # Runtime compatibility
    # --------------------------------------------------------

    try:

        session = _get_cached_session(
            path
        )

        runtime_inputs = (
            session.get_inputs()
        )

        runtime_outputs = (
            session.get_outputs()
        )

    except Exception as exc:

        logger.exception(
            "ONNX Runtime compatibility validation failed."
        )

        return {
            "valid": False,
            "status": "invalid",
            "error": (
                f"ONNX Runtime could not load model: {exc}"
            ),
        }

    return {
        "valid": True,
        "status": "valid",
        "path": str(
            path.resolve()
        ),
        "file_size_bytes": file_size,
        "input_count": len(
            runtime_inputs
        ),
        "output_count": len(
            runtime_outputs
        ),
        "input_names": [
            item.name
            for item
            in runtime_inputs
        ],
        "output_names": [
            item.name
            for item
            in runtime_outputs
        ],
        "providers": list(
            session.get_providers()
        ),
        "error": None,
    }


# ============================================================
# CREATE ONNX SESSION
# ============================================================

def create_onnx_session(
    onnx_path: str | Path,
) -> ort.InferenceSession:
    """
    Validate an ONNX artifact and return a cached session.
    """

    validation = validate_onnx_artifact(
        onnx_path
    )

    if not validation["valid"]:

        raise ValueError(
            validation["error"]
        )

    return _get_cached_session(
        onnx_path
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_onnx_input(
    session: ort.InferenceSession,
    input_data: Any,
) -> dict[str, Any]:
    """
    Validate a NumPy input tensor or an input shape.

    Checks:
        - dtype
        - NaN / infinity
        - rank
        - dimensions
        - tensor size
    """

    try:

        inputs = session.get_inputs()

        if not inputs:

            return {
                "valid": False,
                "error": (
                    "ONNX model has no inputs."
                ),
            }

        model_input = inputs[0]

        expected_shape = model_input.shape

        expected_type = str(
            getattr(
                model_input,
                "type",
                "",
            )
        ).lower()

        # ----------------------------------------------------
        # NumPy tensor
        # ----------------------------------------------------

        if isinstance(
            input_data,
            np.ndarray,
        ):

            tensor = input_data

            actual_shape = [
                int(value)
                for value
                in tensor.shape
            ]

            # ------------------------------------------------
            # Dtype validation
            # ------------------------------------------------

            if (
                "tensor(float)" in expected_type
                or expected_type in {
                    "float32",
                    "float",
                }
            ):

                if tensor.dtype != np.float32:

                    return {
                        "valid": False,
                        "error": (
                            "Input dtype must be float32; "
                            f"got {tensor.dtype}."
                        ),
                    }

            # ------------------------------------------------
            # NaN / infinity
            # ------------------------------------------------

            if np.issubdtype(
                tensor.dtype,
                np.number,
            ):

                if not np.isfinite(
                    tensor
                ).all():

                    return {
                        "valid": False,
                        "error": (
                            "Input tensor contains NaN "
                            "or infinite values."
                        ),
                    }

            total_elements = int(
                tensor.size
            )

        # ----------------------------------------------------
        # Shape list / tuple
        # ----------------------------------------------------

        else:

            try:

                actual_shape = [
                    int(value)
                    for value
                    in input_data
                ]

            except (
                TypeError,
                ValueError,
            ):

                return {
                    "valid": False,
                    "error": (
                        "Input must be a NumPy tensor "
                        "or a valid input shape."
                    ),
                }

            if not actual_shape:

                return {
                    "valid": False,
                    "error": (
                        "Input shape cannot be empty."
                    ),
                }

            total_elements = int(
                np.prod(
                    actual_shape
                )
            )

        # ----------------------------------------------------
        # Shape sanity
        # ----------------------------------------------------

        if not actual_shape:

            return {
                "valid": False,
                "error": (
                    "Input shape cannot be empty."
                ),
            }

        if any(
            value <= 0
            for value
            in actual_shape
        ):

            return {
                "valid": False,
                "error": (
                    "All input dimensions must be positive."
                ),
            }

        # ----------------------------------------------------
        # Maximum elements
        # ----------------------------------------------------

        if (
            total_elements <= 0
            or total_elements
            > MAX_INPUT_ELEMENTS
        ):

            return {
                "valid": False,
                "error": (
                    "Input tensor exceeds the maximum "
                    "allowed element count."
                ),
            }

        # ----------------------------------------------------
        # Rank
        # ----------------------------------------------------

        if (
            len(actual_shape)
            != len(expected_shape)
        ):

            return {
                "valid": False,
                "error": (
                    "Rank mismatch: expected "
                    f"{len(expected_shape)} dimensions, "
                    f"got {len(actual_shape)}."
                ),
            }

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        for index, (
            actual_dimension,
            expected_dimension,
        ) in enumerate(
            zip(
                actual_shape,
                expected_shape,
            )
        ):

            if isinstance(
                expected_dimension,
                int,
            ):

                if (
                    actual_dimension
                    != expected_dimension
                ):

                    return {
                        "valid": False,
                        "error": (
                            "Dimension mismatch at "
                            f"index {index}: expected "
                            f"{expected_dimension}, got "
                            f"{actual_dimension}."
                        ),
                    }

        return {
            "valid": True,
            "shape": actual_shape,
            "total_elements": total_elements,
            "error": None,
        }

    except Exception as exc:

        logger.exception(
            "ONNX input validation failed."
        )

        return {
            "valid": False,
            "error": str(exc),
        }


# ============================================================
# ONNX INFERENCE
# ============================================================

def run_onnx_inference(
    onnx_path: str | Path,
    input_tensor: Any = None,
    input_shape: tuple[int, ...] | list[int] | None = None,
    batch_size: int = 1,
    input_data: Any = None,
) -> dict[str, Any]:
    """
    Execute ONNX Runtime inference using a cached session.

    Supports both:
        input_tensor=
        input_data=

    User-provided input is validated before dtype conversion.
    """

    try:

        # ----------------------------------------------------
        # Normalize input aliases
        # ----------------------------------------------------

        if (
            input_tensor is not None
            and input_data is not None
        ):

            return {
                "status": "failed",
                "latency_ms": None,
                "input_shape": None,
                "output_count": 0,
                "output_shapes": [],
                "providers": [],
                "error": (
                    "Provide either input_tensor or "
                    "input_data, not both."
                ),
            }

        if input_tensor is not None:

            input_data = input_tensor

        path = Path(
            onnx_path
        )

        # ----------------------------------------------------
        # Validate artifact
        # ----------------------------------------------------

        validation = validate_onnx_artifact(
            path
        )

        if not validation["valid"]:

            return {
                "status": "failed",
                "latency_ms": None,
                "input_shape": None,
                "output_count": 0,
                "output_shapes": [],
                "providers": [],
                "error": validation.get(
                    "error",
                    "Invalid ONNX artifact.",
                ),
            }

        # ----------------------------------------------------
        # Cached session
        # ----------------------------------------------------

        session = create_onnx_session(
            path
        )

        inputs = session.get_inputs()

        if not inputs:

            return {
                "status": "failed",
                "latency_ms": None,
                "input_shape": None,
                "output_count": 0,
                "output_shapes": [],
                "providers": list(
                    session.get_providers()
                ),
                "error": (
                    "ONNX model has no inputs."
                ),
            }

        input_metadata = inputs[0]

        input_name = (
            input_metadata.name
        )

        # ----------------------------------------------------
        # Default/generated input
        # ----------------------------------------------------

        if input_data is None:

            if input_shape is None:

                model_shape = (
                    input_metadata.shape
                )

                resolved_shape: list[int] = []

                for index, dimension in enumerate(
                    model_shape
                ):

                    if isinstance(
                        dimension,
                        int,
                    ):

                        if dimension <= 0:

                            raise ValueError(
                                "ONNX model contains an invalid input dimension."
                            )

                        resolved_shape.append(
                            int(dimension)
                        )

                    elif index == 0:

                        resolved_shape.append(
                            max(
                                1,
                                int(batch_size),
                            )
                        )

                    else:

                        resolved_shape.append(
                            1
                        )

                input_shape = resolved_shape

            else:

                input_shape = [
                    int(value)
                    for value
                    in input_shape
                ]

                if input_shape:

                    input_shape[0] = int(
                        batch_size
                    )

            shape_validation = (
                validate_onnx_input(
                    session,
                    input_shape,
                )
            )

            if not shape_validation["valid"]:

                return {
                    "status": "failed",
                    "latency_ms": None,
                    "input_shape": input_shape,
                    "output_count": 0,
                    "output_shapes": [],
                    "providers": list(
                        session.get_providers()
                    ),
                    "error": shape_validation.get(
                        "error",
                        "Invalid ONNX input.",
                    ),
                }

            numpy_input = np.zeros(
                input_shape,
                dtype=np.float32,
            )

        # ----------------------------------------------------
        # User supplied input
        # ----------------------------------------------------

        else:

            # IMPORTANT:
            # Do not cast before validation.

            numpy_input = np.asarray(
                input_data
            )

            validation_result = (
                validate_onnx_input(
                    session,
                    numpy_input,
                )
            )

            if not validation_result["valid"]:

                return {
                    "status": "failed",
                    "latency_ms": None,
                    "input_shape": list(
                        numpy_input.shape
                    ),
                    "output_count": 0,
                    "output_shapes": [],
                    "providers": list(
                        session.get_providers()
                    ),
                    "error": validation_result.get(
                        "error",
                        "Invalid ONNX input.",
                    ),
                }

            # Convert only after validation.

            numpy_input = numpy_input.astype(
                np.float32,
                copy=False,
            )

        # ----------------------------------------------------
        # Final numeric safety check
        # ----------------------------------------------------

        if not np.isfinite(
            numpy_input
        ).all():

            return {
                "status": "failed",
                "latency_ms": None,
                "input_shape": list(
                    numpy_input.shape
                ),
                "output_count": 0,
                "output_shapes": [],
                "providers": list(
                    session.get_providers()
                ),
                "error": (
                    "Input tensor contains NaN "
                    "or infinite values."
                ),
            }

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        outputs = session.run(
            None,
            {
                input_name: numpy_input
            },
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        # ----------------------------------------------------
        # Output shapes
        # ----------------------------------------------------

        output_shapes: list[list[int]] = []

        for output in outputs:

            if hasattr(
                output,
                "shape",
            ):

                output_shapes.append(
                    [
                        int(value)
                        for value
                        in output.shape
                    ]
                )

            else:

                output_shapes.append([])

        return {
            "status": "success",
            "latency_ms": float(
                latency_ms
            ),
            "input_shape": [
                int(value)
                for value
                in numpy_input.shape
            ],
            "output_count": len(
                outputs
            ),
            "output_shapes": output_shapes,
            "providers": list(
                session.get_providers()
            ),
            "error": None,
        }

    except Exception as exc:

        logger.exception(
            "ONNX inference failed."
        )

        return {
            "status": "failed",
            "latency_ms": None,
            "input_shape": None,
            "output_count": 0,
            "output_shapes": [],
            "providers": [],
            "error": str(exc),
        }


# ============================================================
# ONNX RUNTIME BENCHMARK
# ============================================================

def benchmark_onnx_runtime(
    onnx_path: str | Path,
    input_shape: tuple[int, ...] | list[int] | None = None,
    batch_size: int = 1,
    warmup_runs: int = 10,
    runs: int = 50,
) -> dict[str, Any]:
    """
    Benchmark ONNX Runtime inference latency.

    Uses the cached InferenceSession.
    """

    try:

        path = Path(
            onnx_path
        )

        validation = validate_onnx_artifact(
            path
        )

        if not validation["valid"]:

            return {
                "status": "failed",
                "onnx_path": str(path),
                "input_shape": None,
                "warmup_runs": int(
                    warmup_runs
                ),
                "runs": int(
                    runs
                ),
                "median_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "providers": [],
                "error": validation.get(
                    "error",
                    "Invalid ONNX artifact.",
                ),
            }

        session = create_onnx_session(
            path
        )

        inputs = session.get_inputs()

        if not inputs:

            return {
                "status": "failed",
                "onnx_path": str(path),
                "input_shape": None,
                "warmup_runs": int(
                    warmup_runs
                ),
                "runs": int(
                    runs
                ),
                "median_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "providers": list(
                    session.get_providers()
                ),
                "error": (
                    "ONNX model has no inputs."
                ),
            }

        input_info = inputs[0]

        input_name = (
            input_info.name
        )

        # ----------------------------------------------------
        # Resolve input shape
        # ----------------------------------------------------

        if input_shape is None:

            model_shape = (
                input_info.shape
            )

            resolved_shape: list[int] = []

            for index, dimension in enumerate(
                model_shape
            ):

                if isinstance(
                    dimension,
                    int,
                ):

                    if dimension <= 0:

                        raise ValueError(
                            "ONNX model contains an invalid input dimension."
                        )

                    resolved_shape.append(
                        int(dimension)
                    )

                elif index == 0:

                    resolved_shape.append(
                        max(
                            1,
                            int(batch_size),
                        )
                    )

                else:

                    resolved_shape.append(
                        1
                    )

            input_shape = resolved_shape

        else:

            input_shape = [
                int(value)
                for value
                in input_shape
            ]

            if input_shape:

                input_shape[0] = int(
                    batch_size
                )

        # ----------------------------------------------------
        # Validate benchmark shape
        # ----------------------------------------------------

        input_validation = (
            validate_onnx_input(
                session,
                input_shape,
            )
        )

        if not input_validation["valid"]:

            return {
                "status": "failed",
                "onnx_path": str(path),
                "input_shape": input_shape,
                "warmup_runs": int(
                    warmup_runs
                ),
                "runs": int(
                    runs
                ),
                "median_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "providers": list(
                    session.get_providers()
                ),
                "error": input_validation.get(
                    "error",
                    "Invalid ONNX input.",
                ),
            }

        numpy_input = np.zeros(
            input_shape,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------

        for _ in range(
            max(
                0,
                int(warmup_runs),
            )
        ):

            session.run(
                None,
                {
                    input_name: numpy_input
                },
            )

        # ----------------------------------------------------
        # Timed runs
        # ----------------------------------------------------

        latencies_ms: list[float] = []

        for _ in range(
            max(
                1,
                int(runs),
            )
        ):

            start_time = (
                time.perf_counter()
            )

            session.run(
                None,
                {
                    input_name: numpy_input
                },
            )

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            latencies_ms.append(
                float(
                    elapsed_ms
                )
            )

        return {
            "status": "success",
            "onnx_path": str(path),
            "input_shape": [
                int(value)
                for value
                in input_shape
            ],
            "warmup_runs": int(
                warmup_runs
            ),
            "runs": int(
                runs
            ),
            "median_latency_ms": float(
                np.median(
                    latencies_ms
                )
            ),
            "min_latency_ms": float(
                np.min(
                    latencies_ms
                )
            ),
            "max_latency_ms": float(
                np.max(
                    latencies_ms
                )
            ),
            "providers": list(
                session.get_providers()
            ),
            "error": None,
        }

    except Exception as exc:

        logger.exception(
            "ONNX Runtime benchmark failed."
        )

        return {
            "status": "failed",
            "onnx_path": str(
                onnx_path
            ),
            "input_shape": None,
            "warmup_runs": int(
                warmup_runs
            ),
            "runs": int(
                runs
            ),
            "median_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
            "providers": [],
            "error": str(exc),
        }


# ============================================================
# ONNX MODEL INSPECTION
# ============================================================

def inspect_onnx_model(
    onnx_path: str | Path,
) -> dict[str, Any]:
    """
    Inspect an ONNX model and return metadata.
    """

    try:

        path = Path(
            onnx_path
        )

        validation = validate_onnx_artifact(
            path
        )

        if not validation["valid"]:

            return {
                "status": "failed",
                "valid": False,
                "error": validation.get(
                    "error",
                    "Invalid ONNX artifact.",
                ),
            }

        model = onnx.load(
            str(path)
        )

        session = create_onnx_session(
            path
        )

        runtime_inputs = (
            session.get_inputs()
        )

        runtime_outputs = (
            session.get_outputs()
        )

        inputs = [
            {
                "name": item.name,
                "shape": [
                    str(value)
                    for value
                    in item.shape
                ],
                "type": item.type,
            }
            for item
            in runtime_inputs
        ]

        outputs = [
            {
                "name": item.name,
                "shape": [
                    str(value)
                    for value
                    in item.shape
                ],
                "type": item.type,
            }
            for item
            in runtime_outputs
        ]

        return {
            "status": "success",
            "valid": True,
            "path": str(
                path.resolve()
            ),
            "file_size_bytes": int(
                path.stat().st_size
            ),
            "ir_version": int(
                model.ir_version
            ),
            "opset_import": [
                {
                    "domain": item.domain,
                    "version": int(
                        item.version
                    ),
                }
                for item
                in model.opset_import
            ],
            "node_count": len(
                model.graph.node
            ),
            "initializer_count": len(
                model.graph.initializer
            ),
            "inputs": inputs,
            "outputs": outputs,
            "providers": list(
                session.get_providers()
            ),
            "error": None,
        }

    except Exception as exc:

        logger.exception(
            "ONNX model inspection failed."
        )

        return {
            "status": "failed",
            "valid": False,
            "error": str(exc),
        }

