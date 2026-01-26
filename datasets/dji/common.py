"""Common utilities and constants for the DJI trajectory project."""

from __future__ import annotations

from datasets.common.constants import DJI as _DJI
from datasets.common.state import InteractionState as State

globals().update(vars(_DJI))
del _DJI
