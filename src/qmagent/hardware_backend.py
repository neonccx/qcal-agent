"""Guarded adapter for user-supplied real-hardware acquisition functions.

This module intentionally contains no vendor driver and never discovers callable
names dynamically.  A deployment must register every permitted experiment and
approve every acquisition at action time.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from .analysis_tools import analyze
from .contracts import TOOLS, validate_scan
from .storage import digest


class HardwareApprovalError(RuntimeError):
    """Raised when a physical acquisition was not explicitly approved."""


class RegisteredHardwareBackend:
    """Adapt an allow-list of acquisition callables to the Agent backend contract.

    Each handler receives one deep-copied request object with ``tool``, ``state``,
    ``scan`` and a monotonically increasing ``sequence``.  It must return the raw
    observation schema accepted by :func:`qmagent.analysis_tools.analyze`.
    """

    backend_name = "registered-hardware-1.0"

    def __init__(
        self,
        handlers: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]],
        approve: Callable[[dict[str, Any]], bool],
        safe_shutdown: Callable[[], None] | None = None,
    ) -> None:
        unknown = set(handlers) - set(TOOLS)
        if unknown:
            raise ValueError(f"Unregistered experiment names: {sorted(unknown)}")
        if not handlers or not callable(approve):
            raise ValueError("At least one handler and an approval callback are required")
        if safe_shutdown is not None and not callable(safe_shutdown):
            raise TypeError("safe_shutdown must be callable")
        self._handlers = dict(handlers)
        self._approve = approve
        self._safe_shutdown = safe_shutdown
        self._sequence = 0

    def checkpoint(self) -> dict[str, int]:
        """Return logical bookkeeping only; physical hardware is never rewound."""
        return {"sequence": self._sequence}

    def restore(self, checkpoint: dict[str, int]) -> None:
        if (set(checkpoint) != {"sequence"} or type(checkpoint["sequence"]) is not int
                or checkpoint["sequence"] < 0):
            raise ValueError("Invalid hardware-backend checkpoint")
        if self._safe_shutdown is not None:
            self._safe_shutdown()
        # A physical acquisition cannot be rolled back. Never reuse its audit ID.
        self._sequence = max(self._sequence, checkpoint["sequence"])

    def acquire(self, tool: str, state: dict, scan: dict) -> dict:
        if tool not in self._handlers:
            raise ValueError(f"No hardware handler registered for {tool}")
        validate_scan(tool, state, scan)
        request = {"tool": tool, "state": copy.deepcopy(state), "scan": copy.deepcopy(scan),
                   "sequence": self._sequence + 1}
        if self._approve(copy.deepcopy(request)) is not True:
            raise HardwareApprovalError(f"Physical acquisition was not approved: {tool}")
        self._sequence += 1
        try:
            raw = self._handlers[tool](copy.deepcopy(request))
            if not isinstance(raw, dict):
                raise TypeError("Hardware handler must return a raw observation object")
            raw = copy.deepcopy(raw)
            raw.setdefault("tool", tool)
            raw.setdefault("current_parameters", copy.deepcopy(state))
            raw.setdefault("scan", copy.deepcopy(scan))
            raw["synthetic"] = False
            raw["backend"] = self.backend_name
            return raw
        except BaseException:
            if self._safe_shutdown is not None:
                self._safe_shutdown()
            raise

    def measure(self, tool: str, state: dict, scan: dict) -> dict:
        raw = self.acquire(tool, state, scan)
        try:
            result = analyze(raw)
            raw_hash = digest(raw)
            return raw | result | {
                "raw_artifact": {"id": "sha256:" + raw_hash, "sha256": raw_hash},
                "tool_trace": [
                    {"tool": tool, "operation": "acquire", "output_sha256": raw_hash},
                    {"tool": result["analysis_tool"], "operation": "analyze", "input_sha256": raw_hash},
                ],
            }
        except BaseException:
            if self._safe_shutdown is not None:
                self._safe_shutdown()
            raise
