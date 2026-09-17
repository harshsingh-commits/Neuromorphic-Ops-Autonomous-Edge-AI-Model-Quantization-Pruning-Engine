
from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.types import Command

from backend.graph.workflow import workflow
from backend.models.schemas import WorkflowState


class Orchestrator:
    """
    Workflow orchestration service.

    LangGraph SQLite checkpoints are the persistent source of truth.
    The in-memory `states` dictionary is only a compatibility/cache
    layer.

    Unexpected workflow exceptions are converted into a controlled,
    checkpointed failure state so that:
      - the API remains responsive
      - the workflow state remains queryable
      - deployment is blocked safely
      - the failure survives container restart
    """

    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {}

    # ========================================================
    # CONFIGURATION
    # ========================================================

    @staticmethod
    def _config(
        thread_id: str,
    ) -> dict[str, dict[str, str]]:
        """
        Build the LangGraph thread configuration.
        """

        return {
            "configurable": {
                "thread_id": thread_id,
            }
        }

    # ========================================================
    # PERSISTENT STATE
    # ========================================================

    @staticmethod
    def _load_state(
        thread_id: str,
    ) -> dict[str, Any]:
        """
        Load state from the persistent LangGraph checkpointer.
        """

        config = Orchestrator._config(
            thread_id
        )

        snapshot = workflow.get_state(
            config
        )

        values = dict(
            snapshot.values
        )

        if not values:
            raise KeyError(
                f"Unknown workflow thread: {thread_id}"
            )

        return values

    # ========================================================
    # GRACEFUL FAILURE RECOVERY
    # ========================================================

    def _persist_failure(
        self,
        thread_id: str,
        error_message: str,
        fallback_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Convert an unexpected exception into a persistent,
        safe workflow failure state.

        The current checkpoint is loaded when available.
        The failure state is then written back through
        LangGraph's persistent checkpointer.
        """

        config = self._config(
            thread_id
        )

        # ----------------------------------------------------
        # Recover the latest persisted state.
        # ----------------------------------------------------

        try:
            current_state = self._load_state(
                thread_id
            )
        except Exception:
            current_state = dict(
                fallback_state or {}
            )

        # ----------------------------------------------------
        # Preserve existing errors and append the new one.
        # ----------------------------------------------------

        errors = list(
            current_state.get(
                "errors",
                [],
            )
        )

        if error_message not in errors:
            errors.append(
                error_message
            )

        failure_patch: dict[str, Any] = {
            "workflow_status": "failed",
            "deployment_status": "blocked",
            "approval_status": "blocked",
            "errors": errors,
            "error": error_message,
        }

        # ----------------------------------------------------
        # Persist failure through LangGraph.
        # ----------------------------------------------------

        try:
            workflow.update_state(
                config,
                failure_patch,
            )

            recovered_state = self._load_state(
                thread_id
            )

        except Exception as persistence_exc:
            # ------------------------------------------------
            # Last-resort local state.
            #
            # This protects the API from crashing even when
            # checkpoint persistence itself becomes unavailable.
            # ------------------------------------------------

            recovered_state = {
                **current_state,
                **failure_patch,
            }

            persistence_error = (
                "Failure state persistence failed: "
                f"{persistence_exc}"
            )

            recovered_errors = list(
                recovered_state.get(
                    "errors",
                    [],
                )
            )

            if persistence_error not in recovered_errors:
                recovered_errors.append(
                    persistence_error
                )

            recovered_state["errors"] = (
                recovered_errors
            )

        # ----------------------------------------------------
        # Refresh compatibility/cache layer.
        # ----------------------------------------------------

        self.states[thread_id] = recovered_state

        return recovered_state

    # ========================================================
    # START
    # ========================================================

    def start(
        self,
        model_path: str,
        strategy: str | None = None,
        pruning_ratio: float | None = None,
        accuracy_threshold: float | None = None,
        quantization_type: str | None = None,
    ) -> tuple[str, dict]:

        thread_id = str(
            uuid4()
        )

        initial: WorkflowState = {
            "model_path": model_path,
            "original_model_path": model_path,
            "iteration": 0,
            "approval_status": "pending",
            "deployment_status": "running",
            "workflow_status": "starting",
            "errors": [],

            # ------------------------------------------------
            # Evaluation configuration
            # ------------------------------------------------

            "evaluation_dataset": "cifar10",
            "evaluation_data_root": "data",
            "evaluation_batch_size": 64,
            "evaluation_max_samples": 1000,
        }

        # ----------------------------------------------------
        # Optimization strategy
        # ----------------------------------------------------

        if strategy:
            initial["optimization_strategy"] = strategy

        # ----------------------------------------------------
        # Pruning configuration
        # ----------------------------------------------------

        if pruning_ratio is not None:
            initial["pruning_ratio"] = (
                pruning_ratio
            )

        # ----------------------------------------------------
        # Accuracy threshold
        # ----------------------------------------------------

        if accuracy_threshold is not None:
            initial["accuracy_threshold"] = (
                accuracy_threshold
            )

        # ----------------------------------------------------
        # Quantization configuration
        # ----------------------------------------------------

        if quantization_type:
            initial["quantization_type"] = (
                quantization_type
            )

        config = self._config(
            thread_id
        )

        # ----------------------------------------------------
        # Workflow execution
        # ----------------------------------------------------

        try:
            result = workflow.invoke(
                initial,
                config,
            )

            state = dict(
                result
            )

            self.states[thread_id] = state

            return thread_id, state

        except Exception as exc:
            error_message = (
                f"Workflow execution failed: {exc}"
            )

            state = self._persist_failure(
                thread_id=thread_id,
                error_message=error_message,
                fallback_state=dict(
                    initial
                ),
            )

            return thread_id, state

    # ========================================================
    # APPROVE
    # ========================================================

    def approve(
        self,
        thread_id: str,
        approved: bool,
    ) -> dict:

        # ----------------------------------------------------
        # Persistent lookup.
        #
        # This supports approval after container restart.
        # ----------------------------------------------------

        current_state = self._load_state(
            thread_id
        )

        # ----------------------------------------------------
        # Prevent approval of an already failed/blocked run.
        # ----------------------------------------------------

        if current_state.get(
            "workflow_status"
        ) == "failed":

            raise ValueError(
                "Workflow has failed and deployment is blocked."
            )

        if current_state.get(
            "deployment_status"
        ) == "blocked":

            raise ValueError(
                "Workflow deployment is blocked."
            )

        config = self._config(
            thread_id
        )

        decision = (
            "approved"
            if approved
            else "rejected"
        )

        # ----------------------------------------------------
        # Resume approval interrupt.
        # ----------------------------------------------------

        try:
            result = workflow.invoke(
                Command(
                    resume=decision
                ),
                config,
            )

            state = dict(
                result
            )

            self.states[thread_id] = state

            return state

        except Exception as exc:

            error_message = (
                f"Approval workflow failed: {exc}"
            )

            state = self._persist_failure(
                thread_id=thread_id,
                error_message=error_message,
                fallback_state=current_state,
            )

            return state

    # ========================================================
    # STATUS
    # ========================================================

    def status(
        self,
        thread_id: str,
    ) -> dict:

        # ----------------------------------------------------
        # ALWAYS read from persistent checkpoint.
        # ----------------------------------------------------

        state = self._load_state(
            thread_id
        )

        self.states[thread_id] = state

        return {
            "thread_id": thread_id,
            "status": state.get(
                "workflow_status",
                "unknown",
            ),
            "state": state,
        }


orchestrator = Orchestrator()

