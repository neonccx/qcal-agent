"""Hardware-independent acquisition interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MeasurementBackend(ABC):
    """A backend acquires raw data and never performs fitting or decisions."""

    @abstractmethod
    def acquire(self, experiment: str, target: str, **parameters: Any) -> dict:
        raise NotImplementedError

    def connect(self) -> None:
        """Connect hardware resources. The simulator needs no connection."""

    def disconnect(self) -> None:
        """Release hardware resources."""
