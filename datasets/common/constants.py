"""Centralised constant definitions shared across dataset loaders."""

from __future__ import annotations

from types import SimpleNamespace


class _FrozenNamespace(SimpleNamespace):
    """SimpleNamespace variant that prohibits mutation once created."""

    __slots__ = ()

    def __setattr__(self, name, value):  # noqa: D401 - behaviour override
        raise AttributeError("dataset constants are read-only")

    def __delattr__(self, name):  # noqa: D401 - behaviour override
        raise AttributeError("dataset constants are read-only")


def _frozen_namespace(**kwargs) -> _FrozenNamespace:
    """Helper to create a frozen namespace from keyword arguments."""

    return _FrozenNamespace(**kwargs)


UNITS = _frozen_namespace(
    FT_TO_M=0.3048,
    MPH_TO_MPS=0.44704,
    PIXEL_TO_M=0.0421,
)


CITYSIM = _frozen_namespace(
    CENTER_PIC="center_pic",
    CENTER="center",
    BBOX_CORNERS="bboxCorners",
    FRAME="frameNum",
    TRACK_ID="carId",
    LENGTH="length",
    WIDTH="width",
    COURSE="course",
    COURSE_RAD="course_rad",
    SPEED="speed",
    LANE_ID="laneId",
    CAR_CENTER_X_FT="carCenterXft",
    CAR_CENTER_Y_FT="carCenterYft",
    CAR_CENTER_X_PX="carCenterX",
    CAR_CENTER_Y_PX="carCenterY",
    BOUNDING_BOX1_X="boundingBox1X",
    BOUNDING_BOX1_Y="boundingBox1Y",
    BOUNDING_BOX2_X="boundingBox2X",
    BOUNDING_BOX2_Y="boundingBox2Y",
    BOUNDING_BOX3_X="boundingBox3X",
    BOUNDING_BOX3_Y="boundingBox3Y",
    BOUNDING_BOX4_X="boundingBox4X",
    BOUNDING_BOX4_Y="boundingBox4Y",
    HEAD_X="headX",
    HEAD_Y="headY",
    TAIL_X="tailX",
    TAIL_Y="tailY",
    FT_TO_M=UNITS.FT_TO_M,
    MPH_TO_MPS=UNITS.MPH_TO_MPS,
    PIXEL_TO_M=UNITS.PIXEL_TO_M,
)

INTERACTION = _frozen_namespace(
    BBOX="bbox",
    FRAMES="frames",
    FRAME="frame",
    TRACK_ID="id",
    X="x",
    Y="y",
    WIDTH="width",
    HEIGHT="height",
    X_VELOCITY="xVelocity",
    Y_VELOCITY="yVelocity",
    X_ACCELERATION="xAcceleration",
    Y_ACCELERATION="yAcceleration",
    TTC="ttc",
    PRECEDING_ID="precedingId",
    FOLLOWING_ID="followingId",
    LEFT_PRECEDING_ID="leftPrecedingId",
    LEFT_ALONGSIDE_ID="leftAlongsideId",
    LEFT_FOLLOWING_ID="leftFollowingId",
    RIGHT_PRECEDING_ID="rightPrecedingId",
    RIGHT_ALONGSIDE_ID="rightAlongsideId",
    RIGHT_FOLLOWING_ID="rightFollowingId",
    LANE_ID="laneId",
    INITIAL_FRAME="initialFrame",
    FINAL_FRAME="finalFrame",
    NUM_FRAMES="numFrames",
    CLASS="class",
    DRIVING_DIRECTION="drivingDirection",
    ID="id",
    FRAME_RATE="frameRate",
    SPEED_LIMIT="speedLimit",
)

DJI = INTERACTION
HIGHD = INTERACTION

NGSIM = _frozen_namespace(
    VEHICLE_ID="Vehicle_ID",
    FRAME_ID="Frame_ID",
    LOCAL_POSITION="Local_Position",
    VELOCITY="v_Vel",
    ACCELERATION="v_Acc",
    LOCAL_Y="Local_Y",
    LOCAL_X="Local_X",
    LANE_ID="Lane_ID",
    TOTAL_FRAMES="Total_Frames",
    VEHICLE_LENGTH="v_Length",
    VEHICLE_WIDTH="v_Width",
    INITIAL_FRAME="initial_frame",
    FINAL_FRAME="final_frame",
    VEHICLE_CLASS="v_Class",
    PRECEDING_DISTANCE="precedingD",
    FOLLOWING_DISTANCE="followingD",
    LEFT_PRECEDING_DISTANCE="leftPrecedingD",
    LEFT_ALONGSIDE_DISTANCE="leftAlongsideD",
    LEFT_FOLLOWING_DISTANCE="leftFollowingD",
    RIGHT_PRECEDING_DISTANCE="rightPrecedingD",
    RIGHT_ALONGSIDE_DISTANCE="rightAlongsideD",
    RIGHT_FOLLOWING_DISTANCE="rightFollowingD",
    FT_TO_M=UNITS.FT_TO_M,
)

IND = _frozen_namespace(
    BBOX="bbox",
    FRAME="frame",
    TRACK_ID="trackId",
    X="xCenter",
    Y="yCenter",
    HEADING="heading",
    LENGTH="length",
    WIDTH="width",
    X_VELOCITY="xVelocity",
    Y_VELOCITY="yVelocity",
    X_ACCELERATION="xAcceleration",
    Y_ACCELERATION="yAcceleration",
    LON_VELOCITY="lonVelocity",
    LAT_VELOCITY="latVelocity",
    LON_ACCELERATION="lonAcceleration",
    LAT_ACCELERATION="latAcceleration",
    INITIAL_FRAME="initialFrame",
    FINAL_FRAME="finalFrame",
    NUM_FRAMES="numFrames",
    CLASS="class",
    FRAME_RATE="frameRate",
    SPEED_LIMIT="speedLimit",
    ORTHO_PX_TO_METER="orthoPxToMeter",
    AREA_ID="locationId",
    SCALE_DOWN_FACTOR=12,
    RELEVANT_AREAS={
        "1": {"x_lim": [2000, 11500], "y_lim": [9450, 0]},
        "2": {"x_lim": [0, 12500], "y_lim": [7400, 0]},
        "3": {"x_lim": [0, 11500], "y_lim": [9365, 0]},
        "4": {"x_lim": [2700, 15448], "y_lim": [9365, 0]},
    },
    dir_names=(
        "preceding",
        "following",
        "leftPreceding",
        "leftAlongside",
        "leftFollowing",
        "rightPreceding",
        "rightAlongside",
        "rightFollowing",
    ),
)

SIND = _frozen_namespace(
    BBOX="bbox",
    FRAME="frame_id",
    TRACK_ID="track_id",
    X="x",
    Y="y",
    LENGTH="length",
    WIDTH="width",
    X_VELOCITY="vx",
    Y_VELOCITY="vy",
    X_ACCELERATION="ax",
    Y_ACCELERATION="ay",
    LON_VELOCITY="v_lon",
    LAT_VELOCITY="v_lat",
    LON_ACCELERATION="a_lon",
    LAT_ACCELERATION="a_lat",
    HEADING_RAD="heading_rad",
    AGENT_TYPE="agent_type",
)

__all__ = ["UNITS", "INTERACTION", "CITYSIM", "DJI", "HIGHD", "NGSIM", "IND", "SIND"]
