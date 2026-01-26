from typing import Final, Tuple

from frenet_system_creator import FrenetSystem

from datasets.common.eight_directions import eight_dirs as eight_dirs
from datasets.common.state import InDState as State

BBOX: Final[str]
FRAME: Final[str]
TRACK_ID: Final[str]
X: Final[str]
Y: Final[str]
HEADING: Final[str]
LENGTH: Final[str]
WIDTH: Final[str]
X_VELOCITY: Final[str]
Y_VELOCITY: Final[str]
X_ACCELERATION: Final[str]
Y_ACCELERATION: Final[str]
LON_VELOCITY: Final[str]
LAT_VELOCITY: Final[str]
LON_ACCELERATION: Final[str]
LAT_ACCELERATION: Final[str]
INITIAL_FRAME: Final[str]
FINAL_FRAME: Final[str]
NUM_FRAMES: Final[str]
CLASS: Final[str]
FRAME_RATE: Final[str]
SPEED_LIMIT: Final[str]
ORTHO_PX_TO_METER: Final[str]
AREA_ID: Final[str]
SCALE_DOWN_FACTOR: Final[int]
RELEVANT_AREAS: Final[dict]
dir_names: Final[Tuple[str, ...]]
