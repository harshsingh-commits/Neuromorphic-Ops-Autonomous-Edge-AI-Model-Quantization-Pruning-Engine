from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import psutil


@dataclass
class MonitoringState:
    start_time: float = field(
        default_factory=time.perf_counter
    )

    request_count: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    optimization_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0

    total_request_time_ms: float = 0.0
    max_request_time_ms: float = 0.0

    last_request_at: str | None = None
    last_error: str | None = None

    recent_requests: deque[
        dict[str, Any]
    ] = field(
        default_factory=lambda: deque(
            maxlen=100
        )
    )

    lock: threading.Lock = field(
        default_factory=threading.Lock
    )


class RuntimeMonitor:
    def __init__(self) -> None:
        self.state = MonitoringState()

        # ----------------------------------------------------
        # Process resource monitoring
        # ----------------------------------------------------

        self.process = psutil.Process(
            os.getpid()
        )

        # Prime CPU measurement so the next
        # cpu_percent(interval=None) returns
        # a meaningful value.
        self.process.cpu_percent(
            interval=None
        )

    # ========================================================
    # REQUEST MONITORING
    # ========================================================

    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
    ) -> None:

        success = (
            200
            <= status_code
            < 400
        )

        timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        with self.state.lock:

            self.state.request_count += 1

            if success:
                self.state.successful_requests += 1

            else:
                self.state.failed_requests += 1

            self.state.total_request_time_ms += (
                latency_ms
            )

            self.state.max_request_time_ms = max(
                self.state.max_request_time_ms,
                latency_ms,
            )

            self.state.last_request_at = (
                timestamp
            )

            self.state.recent_requests.append(
                {
                    "timestamp": timestamp,
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "latency_ms": round(
                        latency_ms,
                        3,
                    ),
                }
            )

    # ========================================================
    # ERROR MONITORING
    # ========================================================

    def record_error(
        self,
        error: str,
    ) -> None:

        timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        with self.state.lock:

            self.state.last_error = error

            self.state.recent_requests.append(
                {
                    "timestamp": timestamp,
                    "type": "error",
                    "error": error,
                }
            )

    # ========================================================
    # OPTIMIZATION MONITORING
    # ========================================================

    def record_optimization_started(
        self,
    ) -> None:

        with self.state.lock:
            self.state.optimization_runs += 1

    def record_optimization_completed(
        self,
    ) -> None:

        with self.state.lock:
            self.state.completed_runs += 1

    def record_optimization_failed(
        self,
        error: str | None = None,
    ) -> None:

        with self.state.lock:

            self.state.failed_runs += 1

            if error:
                self.state.last_error = error

    # ========================================================
    # PHASE 22.5 - RESOURCE MONITORING
    # ========================================================

    def resource_snapshot(
        self,
    ) -> dict[str, Any]:
        """
        Return live process resource metrics.

        Metrics:
            - CPU utilization
            - resident memory (RSS)
            - virtual memory
            - process thread count
            - process ID
            - process uptime
        """

        try:

            cpu_percent = (
                self.process.cpu_percent(
                    interval=None
                )
            )

            memory_info = (
                self.process.memory_info()
            )

            rss_mb = (
                memory_info.rss
                / (
                    1024
                    * 1024
                )
            )

            vms_mb = (
                memory_info.vms
                / (
                    1024
                    * 1024
                )
            )

            try:
                thread_count = (
                    self.process.num_threads()
                )
            except (
                psutil.Error,
                OSError,
            ):
                thread_count = 0

            uptime_seconds = (
                time.time()
                - self.process.create_time()
            )

            return {
                "status": "available",
                "process_id": (
                    self.process.pid
                ),
                "cpu_percent": round(
                    cpu_percent,
                    2,
                ),
                "memory_rss_mb": round(
                    rss_mb,
                    3,
                ),
                "memory_vms_mb": round(
                    vms_mb,
                    3,
                ),
                "thread_count": int(
                    thread_count
                ),
                "process_uptime_seconds": round(
                    max(
                        uptime_seconds,
                        0.0,
                    ),
                    3,
                ),
            }

        except (
            psutil.Error,
            OSError,
        ) as exc:

            return {
                "status": "unavailable",
                "error": str(exc),
            }

    # ========================================================
    # COMPLETE MONITORING SNAPSHOT
    # ========================================================

    def snapshot(
        self,
    ) -> dict[str, Any]:

        with self.state.lock:

            uptime_seconds = (
                time.perf_counter()
                - self.state.start_time
            )

            average_latency_ms = (
                self.state.total_request_time_ms
                / self.state.request_count
                if self.state.request_count > 0
                else 0.0
            )

            success_rate = (
                self.state.successful_requests
                / self.state.request_count
                * 100.0
                if self.state.request_count > 0
                else 0.0
            )

            snapshot = {
                "status": "running",

                "uptime_seconds": round(
                    uptime_seconds,
                    3,
                ),

                "requests": {
                    "total": self.state.request_count,
                    "successful": (
                        self.state.successful_requests
                    ),
                    "failed": (
                        self.state.failed_requests
                    ),
                    "success_rate_percent": round(
                        success_rate,
                        2,
                    ),
                    "average_latency_ms": round(
                        average_latency_ms,
                        3,
                    ),
                    "max_latency_ms": round(
                        self.state.max_request_time_ms,
                        3,
                    ),
                    "last_request_at": (
                        self.state.last_request_at
                    ),
                },

                "optimization": {
                    "started": (
                        self.state.optimization_runs
                    ),
                    "completed": (
                        self.state.completed_runs
                    ),
                    "failed": (
                        self.state.failed_runs
                    ),
                },

                "last_error": (
                    self.state.last_error
                ),

                "recent_events": list(
                    self.state.recent_requests
                ),
            }

        # Resource metrics are collected
        # outside the lock so a psutil call
        # does not hold the monitoring lock.
        snapshot["resources"] = (
            self.resource_snapshot()
        )

        return snapshot


monitor = RuntimeMonitor()