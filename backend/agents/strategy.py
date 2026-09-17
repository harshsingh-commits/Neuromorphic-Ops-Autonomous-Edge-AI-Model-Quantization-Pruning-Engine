from __future__ import annotations

import logging
from backend.models.schemas import WorkflowState
from backend.config import (
    DEFAULT_PRUNING_RATIO,
    DEFAULT_QUANTIZATION,
)


logger = logging.getLogger(__name__)

VALID_STRATEGIES = {
    "pruning",
    "quantization",
    "hybrid",
}


def select_strategy(state: WorkflowState) -> WorkflowState:
    """
    Deterministically selects an edge optimization strategy.

    Supported strategies:
    - pruning
    - quantization
    - hybrid

    Behavior:
    1. Respect an explicitly supplied valid strategy.
    2. Automatically select a strategy from analysis_summary.
    3. Fail safely when model analysis is unavailable.
    4. Never leave an ambiguous/invalid strategy in state.
    """

    logger.info("Starting strategy selector agent...")

    state["workflow_status"] = "selecting_strategy"

    # --------------------------------------------------------
    # Initialize collections safely
    # --------------------------------------------------------

    if not state.get("logs"):
        state["logs"] = []

    if not state.get("errors"):
        state["errors"] = []

    state["logs"].append(
        "Strategy Selector Agent: Selecting optimization strategy."
    )

    # --------------------------------------------------------
    # 1. Check explicit user strategy
    # --------------------------------------------------------

    user_strategy = state.get("optimization_strategy")

    if user_strategy is not None:
        user_strategy = str(user_strategy).strip().lower()

        if user_strategy in VALID_STRATEGIES:
            logger.info(
                "User specified valid strategy: %s",
                user_strategy,
            )

            state["optimization_strategy"] = user_strategy

            pruning_ratio = state.get("pruning_ratio")

            if pruning_ratio is None:
                if user_strategy == "quantization":
                    pruning_ratio = 0.0
                else:
                    pruning_ratio = DEFAULT_PRUNING_RATIO

            state["pruning_ratio"] = float(
                pruning_ratio
            )

            quantization_type = state.get(
                "quantization_type"
            )

            if not quantization_type:
                if user_strategy == "pruning":
                    quantization_type = "none"
                else:
                    quantization_type = DEFAULT_QUANTIZATION

            state["quantization_type"] = (
                quantization_type
            )

            state["optimization_reason"] = (
                "Strategy specified explicitly by "
                f"user optimization request: {user_strategy}."
            )

            state["logs"].append(
                "Strategy Selector Agent: "
                f"User selected {user_strategy}."
            )

            state["workflow_status"] = (
                "strategy_selected"
            )

            return state

        # ----------------------------------------------------
        # Invalid explicit strategy
        # ----------------------------------------------------

        error_message = (
            f"Invalid optimization strategy: "
            f"{user_strategy}. "
            f"Expected one of: "
            f"{', '.join(sorted(VALID_STRATEGIES))}."
        )

        logger.error(error_message)

        state["errors"].append(
            error_message
        )

        state["workflow_status"] = (
            "strategy_failed"
        )

        state["error"] = error_message

        return state

    # --------------------------------------------------------
    # 2. Automatic strategy selection
    # --------------------------------------------------------

    summary = (
    state.get("analysis_summary")
    or state.get("analysis")
)

    if not isinstance(summary, dict) or not summary:
        error_message = (
            "No model analysis summary found. "
            "Cannot determine strategy automatically."
        )

        logger.error(error_message)

        state["errors"].append(
            error_message
        )

        state["workflow_status"] = (
            "strategy_failed"
        )

        state["error"] = error_message

        state["logs"].append(
            "Strategy Selector Agent: "
            "Automatic strategy selection failed."
        )

        return state

    # --------------------------------------------------------
    # 3. Extract model profile
    # --------------------------------------------------------

    file_size_mb = float(
        summary.get(
            "file_size_mb",
            0.0,
        )
        or 0.0
    )

    parameter_count = int(
        summary.get(
            "parameter_count",
            0,
        )
        or 0
    )

    supported_pruning_layers = int(
        summary.get(
            "supported_pruning_layers",
            0,
        )
        or 0
    )

    supported_quantization_layers = int(
        summary.get(
            "supported_quantization_layers",
            0,
        )
        or 0
    )

    # --------------------------------------------------------
    # 4. Validate analysis profile
    # --------------------------------------------------------

    if (
        parameter_count < 0
        or file_size_mb < 0
        or supported_pruning_layers < 0
        or supported_quantization_layers < 0
    ):
        error_message = (
            "Invalid model analysis summary: "
            "model profile contains negative values."
        )

        logger.error(error_message)

        state["errors"].append(
            error_message
        )

        state["workflow_status"] = (
            "strategy_failed"
        )

        state["error"] = error_message

        return state

    # --------------------------------------------------------
    # 5. Deterministic heuristic strategy selection
    # --------------------------------------------------------

    # Large models with both optimization capabilities
    if (
        (
            file_size_mb >= 15.0
            or parameter_count >= 1_500_000
        )
        and supported_pruning_layers > 0
        and supported_quantization_layers > 0
    ):
        recommended_strategy = "hybrid"

        reason = (
            f"Model is large "
            f"({file_size_mb:.2f} MB, "
            f"{parameter_count:,} parameters) "
            f"and supports both pruning and quantization. "
            "Recommending hybrid optimization."
        )

        prune_ratio = DEFAULT_PRUNING_RATIO
        quant_type = DEFAULT_QUANTIZATION

    # Models with meaningful pruning support
    elif (
        supported_pruning_layers > 3
        and parameter_count >= 500_000
    ):
        recommended_strategy = "pruning"

        reason = (
            f"Model has "
            f"{supported_pruning_layers} prunable layers "
            f"and {parameter_count:,} parameters. "
            "Recommending pruning to reduce model redundancy."
        )

        prune_ratio = DEFAULT_PRUNING_RATIO
        quant_type = "none"

    # Small/medium models or models without
    # sufficient pruning support
    else:
        recommended_strategy = "quantization"

        reason = (
            f"Model is small/medium "
            f"({file_size_mb:.2f} MB, "
            f"{parameter_count:,} parameters) "
            "or has limited pruning support. "
            "Recommending dynamic INT8 quantization."
        )

        prune_ratio = 0.0
        quant_type = DEFAULT_QUANTIZATION

    # --------------------------------------------------------
    # 6. Persist selected strategy
    # --------------------------------------------------------

    state["optimization_strategy"] = (
        recommended_strategy
    )

    state["pruning_ratio"] = float(
        prune_ratio
    )

    state["quantization_type"] = (
        quant_type
    )

    state["optimization_reason"] = reason

    state["workflow_status"] = (
        "strategy_selected"
    )

    logger.info(
        "Strategy Recommended: %s. Reason: %s",
        recommended_strategy,
        reason,
    )

    state["logs"].append(
        "Strategy Selector Agent: "
        f"Recommended {recommended_strategy}. "
        f"Reason: {reason}"
    )

    return state