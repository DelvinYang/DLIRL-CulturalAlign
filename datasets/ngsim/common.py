"""Common utilities and constants for the NGSIM trajectory project."""

from __future__ import annotations

from datasets.common.constants import NGSIM as _NGSIM
from datasets.common.state import NGSIMState as State

globals().update(vars(_NGSIM))
del _NGSIM
