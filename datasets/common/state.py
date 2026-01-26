"""Dataset-specific state dataclasses shared across loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class BaseState:
    """Base class for all dataset state representations."""

    id: int

    def __post_init__(self) -> None:
        # Subclasses can hook into dataclass initialisation.
        return None


@dataclass
class CitySimState(BaseState):
    """CitySim vehicle state."""

    center: np.ndarray
    lon: Iterable[float]
    lat: Iterable[float]
    length: float
    width: float
    course_rad: float


@dataclass
class XYState(BaseState):
    """State with longitudinal and lateral time series."""

    x: Iterable[float]
    y: Iterable[float]

    def __post_init__(self) -> None:
        super().__post_init__()
        self._x0 = self._first_value(self.x)
        self._y0 = self._first_value(self.y)

    @staticmethod
    def _first_value(values: Iterable[float]) -> float:
        return float(np.asarray(values, dtype=float)[0])

    @property
    def xy0(self) -> tuple[float, float]:
        """Return the first position sample in (x, y) form."""
        return self._x0, self._y0


@dataclass
class CenteredBoundsState(XYState):
    """State whose centre is derived from axis-aligned bounds."""

    def __post_init__(self) -> None:
        super().__post_init__()
        dx, dy = self._extents()
        x0, y0 = self.xy0
        self.center = (x0 + (dx / 2), y0 + (dy / 2))

    def _extents(self) -> tuple[float, float]:
        raise NotImplementedError


@dataclass
class InteractionState(CenteredBoundsState):
    """Shared state representation for DJI/HighD interaction datasets."""

    width: float
    height: float
    vehicle_type: str
    lane_id: int
    driving_direction: float
    frame_id: Optional[int] = None

    def _extents(self) -> tuple[float, float]:
        return self.width, self.height


# Aliases for clarity in dataset modules.
DJIState = InteractionState
HighDState = InteractionState


@dataclass
class NGSIMState(CenteredBoundsState):
    """NGSIM vehicle state."""

    width: float
    length: float
    vehicle_type: str
    lane_id: int
    frame_id: Optional[int] = None

    def _extents(self) -> tuple[float, float]:
        return self.length, self.width


@dataclass
class FrenetState(CenteredBoundsState):
    """Base state for datasets providing Frenet components."""

    lon: Iterable[float]
    lat: Iterable[float]
    width: float
    height: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.position = np.array(self.xy0, dtype=float)

    def _extents(self) -> tuple[float, float]:
        return self.width, self.height


@dataclass
class InDState(FrenetState):
    """inD vehicle state with Frenet longitudinal/lateral components."""

    heading: float
    vehicle_type: str

    def __post_init__(self) -> None:
        super().__post_init__()
        self.heading_vis = (-self.heading) % 360


@dataclass
class SinDState(FrenetState):
    """sinD vehicle state with Frenet components and heading in radians."""

    heading_rad: float
    vehicle_type: str


State = BaseState


__all__ = [
    "BaseState",
    "XYState",
    "CenteredBoundsState",
    "FrenetState",
    "CitySimState",
    "InteractionState",
    "DJIState",
    "HighDState",
    "NGSIMState",
    "InDState",
    "SinDState",
    "State",
]
