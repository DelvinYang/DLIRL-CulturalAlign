"""Scenario helpers built on top of :mod:`datasets.ngsim.read_data_ngsim`."""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy.signal import savgol_filter

from .common import *
from .read_data_ngsim import DataReaderNGSIM

if TYPE_CHECKING:  # pragma: no cover - aid IDE/static typing only
    from .common import State


class ScenarioNGSIM:
    """Wrapper around :class:`DataReaderNGSIM` providing vehicle utilities."""

    def __init__(self, data_reader: DataReaderNGSIM):
        self.data_reader = data_reader
        self.dt = 0.1  # 10 Hz sampling frequency
        self.vehicles, self.id_list = self.set_vehicles()
        logger.info("[ScenarioNGSIM] Imported {} vehicles", len(self.id_list))

    def set_vehicles(self) -> tuple[Dict[int, "Vehicle"], set[int]]:
        """Filter valid vehicles and build a lookup table."""

        vehicles_dict: Dict[int, Vehicle] = {}
        id_list: set[int] = set()
        for track in self.data_reader.tracks:
            track_id = int(track[VEHICLE_ID])
            if int(track[VEHICLE_CLASS]) == 1:
                continue
            vehicle = Vehicle(track, dt=self.dt)
            vehicles_dict[track_id] = vehicle
            id_list.add(track_id)

        return vehicles_dict, id_list

    def find_vehicle_by_id(self, vehicle_id: int) -> "Vehicle":
        return self.vehicles[vehicle_id]

    def set_reference_path(self, ego_id: int) -> List[tuple[float, float]]:
        """Return the ego vehicle centre trajectory."""

        ego_vehicle = self.find_vehicle_by_id(ego_id)
        return [tuple(center) for center in ego_vehicle.reference_path]

    def find_vehicle_state(self, frame_num: int, vehicle_id: int) -> Optional[State]:
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None:
            return None
        return vehicle.find_vehicle_state(frame_num)

    def find_svs_state(self, frame_num: int, vehicle_id: int) -> List[State]:
        """Return the state of all surrounding vehicles for a frame."""

        svs_state: List[State] = []
        for vehicle in self.vehicles.values():
            if vehicle.vehicle_id == vehicle_id:
                continue
            try:
                state = vehicle.find_vehicle_state(frame_num=frame_num)
            except Exception:  # pragma: no cover - defensive
                continue
            if state is not None:
                svs_state.append(state)
        return svs_state

    def generate_state_list(self, frame_num: int, vehicle_id: int) -> List[float]:
        svs_state = self.find_svs_state(frame_num=frame_num, vehicle_id=vehicle_id)
        ego_state = self.find_vehicle_state(frame_num=frame_num, vehicle_id=vehicle_id)
        if ego_state is None:
            return []
        ev_lane_id = ego_state.lane_id

        distance_values = {
            PRECEDING_DISTANCE: -1.0,
            FOLLOWING_DISTANCE: -1.0,
            LEFT_PRECEDING_DISTANCE: -1.0,
            LEFT_ALONGSIDE_DISTANCE: -1.0,
            LEFT_FOLLOWING_DISTANCE: -1.0,
            RIGHT_PRECEDING_DISTANCE: -1.0,
            RIGHT_ALONGSIDE_DISTANCE: -1.0,
            RIGHT_FOLLOWING_DISTANCE: -1.0,
        }

        def update_distance(distance_key: str, candidate_distance: float) -> None:
            current_distance = distance_values[distance_key]
            if current_distance == -1.0 or candidate_distance < current_distance:
                distance_values[distance_key] = candidate_distance

        for sv_state in svs_state:
            current_distance = calc_distance(sv_state, ego_state)
            sv_lane_id = sv_state.lane_id
            if sv_lane_id == ev_lane_id:  # same lane
                if ego_state.x[0] < sv_state.x[0]:  # front vehicle
                    update_distance(PRECEDING_DISTANCE, current_distance)
                else:  # rear vehicle
                    update_distance(FOLLOWING_DISTANCE, current_distance)
            elif sv_lane_id - ev_lane_id == 1:  # right lane
                if abs(ego_state.x[0] - sv_state.x[0]) < ego_state.length:  # right alongside
                    update_distance(RIGHT_ALONGSIDE_DISTANCE, current_distance)
                elif ego_state.x[0] < sv_state.x[0]:  # front vehicle
                    update_distance(RIGHT_PRECEDING_DISTANCE, current_distance)
                else:  # rear vehicle
                    update_distance(RIGHT_FOLLOWING_DISTANCE, current_distance)
            elif sv_lane_id - ev_lane_id == -1:  # left lane
                if abs(ego_state.x[0] - sv_state.x[0]) < ego_state.length:  # left alongside
                    update_distance(LEFT_ALONGSIDE_DISTANCE, current_distance)
                elif ego_state.x[0] < sv_state.x[0]:  # front vehicle
                    update_distance(LEFT_PRECEDING_DISTANCE, current_distance)
                else:  # rear vehicle
                    update_distance(LEFT_FOLLOWING_DISTANCE, current_distance)

        return [
            distance_values[PRECEDING_DISTANCE],
            distance_values[FOLLOWING_DISTANCE],
            distance_values[LEFT_PRECEDING_DISTANCE],
            distance_values[LEFT_ALONGSIDE_DISTANCE],
            distance_values[LEFT_FOLLOWING_DISTANCE],
            distance_values[RIGHT_PRECEDING_DISTANCE],
            distance_values[RIGHT_ALONGSIDE_DISTANCE],
            distance_values[RIGHT_FOLLOWING_DISTANCE],
        ]


def calc_distance(sv_state: State, ev_state: State) -> float:
    """Return Euclidean distance between two vehicle states."""

    sv_pos = np.asarray(sv_state.center, dtype=float)
    ego_pos = np.asarray(ev_state.center, dtype=float)
    return float(np.linalg.norm(sv_pos - ego_pos))


class Vehicle:
    def __init__(self, track: Dict[str, np.ndarray], dt: float = 0.1):
        self.track = track
        self.dt = float(dt)
        self.vehicle_id = int(track[VEHICLE_ID])
        self.length = float(track[VEHICLE_LENGTH])
        self.width = float(track[VEHICLE_WIDTH])
        self.vehicle_type = int(track[VEHICLE_CLASS])

        (
            self.reference_path,
            self.velocities,
            self.accelerations,
            self.lane_ids,
            self.frames,
        ) = self._set_values()

        if len(self.frames) > 0:
            self.initial_frame = int(self.frames[0])
            self.final_frame = int(self.frames[-1])
        else:
            self.initial_frame = int(track[INITIAL_FRAME])
            self.final_frame = int(track[FINAL_FRAME])

        self.states_by_index: List[State] | None = None
        self.frame_to_index: dict[int, int] | None = None

        self.vx: np.ndarray | None = None
        self.vy: np.ndarray | None = None
        self.ax: np.ndarray | None = None
        self.ay: np.ndarray | None = None

        self.precompute_states()

    def _set_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        centers = np.asarray(self.track[LOCAL_POSITION], dtype=np.float64)
        frames = np.asarray(self.track[FRAME_ID], dtype=np.int64)
        velocities = np.asarray(self.track[VELOCITY], dtype=np.float64)
        accelerations = np.asarray(self.track[ACCELERATION], dtype=np.float64)
        lane_ids = np.asarray(self.track[LANE_ID], dtype=np.int64)

        eps = 1e-6
        start_idx = next((i for i, v in enumerate(velocities) if abs(v) > eps), 0)

        centers = centers[start_idx:]
        frames = frames[start_idx:]
        velocities = velocities[start_idx:]
        accelerations = accelerations[start_idx:]
        lane_ids = lane_ids[start_idx:]

        if centers.size == 0:
            centers = np.asarray(self.track[LOCAL_POSITION], dtype=np.float64)
            frames = np.asarray(self.track[FRAME_ID], dtype=np.int64)
            velocities = np.asarray(self.track[VELOCITY], dtype=np.float64)
            accelerations = np.asarray(self.track[ACCELERATION], dtype=np.float64)
            lane_ids = np.asarray(self.track[LANE_ID], dtype=np.int64)

        smoothed_centers = self._smooth_path(centers)
        return smoothed_centers, velocities, accelerations, lane_ids, frames

    def _smooth_path(self, path: np.ndarray) -> np.ndarray:
        path = np.asarray(path, dtype=np.float64)
        n = len(path)
        if n < 3:
            return path

        max_window = n if (n % 2 == 1) else (n - 1)
        window = min(51, max_window)
        poly = 3
        if window < (poly + 2):
            return path

        x_smooth = savgol_filter(path[:, 0], window_length=window, polyorder=poly)
        y_smooth = savgol_filter(path[:, 1], window_length=window, polyorder=poly)
        return np.vstack([x_smooth, y_smooth]).T

    def precompute_states(self) -> None:
        self.compute_kinematics()

        centers = self.reference_path
        frames = self.frames.astype(int, copy=False)
        lane_ids = self.lane_ids.astype(int, copy=False) if len(self.lane_ids) else np.array([], dtype=int)

        n = len(centers)
        self.frame_to_index = {int(frames[i]): i for i in range(n)}

        states: List[State] = []
        for i in range(n):
            center = centers[i]
            corner_x = center[0] - 0.5 * self.length
            corner_y = center[1] - 0.5 * self.width
            lane_id = int(lane_ids[i]) if lane_ids.size > i else (int(lane_ids[-1]) if lane_ids.size else 0)
            state = State(
                self.vehicle_id,
                [float(corner_x), float(self.vx[i]), float(self.ax[i])],
                [float(corner_y), float(self.vy[i]), float(self.ay[i])],
                float(self.width),
                float(self.length),
                int(self.vehicle_type),
                lane_id,
                int(frames[i]),
            )
            states.append(state)

        self.states_by_index = states

    def compute_kinematics(self) -> None:
        centers = self.reference_path
        n = len(centers)
        if n == 0:
            self.vx = np.zeros(0, dtype=float)
            self.vy = np.zeros(0, dtype=float)
            self.ax = np.zeros(0, dtype=float)
            self.ay = np.zeros(0, dtype=float)
            return

        if n == 1:
            self.vx = np.zeros(1, dtype=float)
            self.vy = np.zeros(1, dtype=float)
            self.ax = np.zeros(1, dtype=float)
            self.ay = np.zeros(1, dtype=float)
            return

        frames = self.frames.astype(float, copy=False)
        t = (frames - frames[0]) * self.dt
        dt_nominal = float(np.median(np.diff(t))) if n > 1 else self.dt

        x = centers[:, 0]
        y = centers[:, 1]

        if n < 7:
            vx = np.gradient(x, t, edge_order=1)
            vy = np.gradient(y, t, edge_order=1)
            ax = np.gradient(vx, t, edge_order=1)
            ay = np.gradient(vy, t, edge_order=1)
        else:
            win = min(31, n if n % 2 == 1 else n - 1)
            poly = 3
            if win < (poly + 2):
                vx = np.gradient(x, t, edge_order=1)
                vy = np.gradient(y, t, edge_order=1)
                ax = np.gradient(vx, t, edge_order=1)
                ay = np.gradient(vy, t, edge_order=1)
            else:
                vx = savgol_filter(
                    x,
                    window_length=win,
                    polyorder=poly,
                    deriv=1,
                    delta=dt_nominal,
                    mode="interp",
                )
                vy = savgol_filter(
                    y,
                    window_length=win,
                    polyorder=poly,
                    deriv=1,
                    delta=dt_nominal,
                    mode="interp",
                )
                ax = savgol_filter(
                    x,
                    window_length=win,
                    polyorder=poly,
                    deriv=2,
                    delta=dt_nominal,
                    mode="interp",
                )
                ay = savgol_filter(
                    y,
                    window_length=win,
                    polyorder=poly,
                    deriv=2,
                    delta=dt_nominal,
                    mode="interp",
                )

        self.vx = np.asarray(vx, dtype=float)
        self.vy = np.asarray(vy, dtype=float)
        self.ax = np.asarray(ax, dtype=float)
        self.ay = np.asarray(ay, dtype=float)

    def find_vehicle_state(self, frame_num: int) -> Optional[State]:
        if not (self.initial_frame <= frame_num <= self.final_frame):
            return None
        if not self.frame_to_index or not self.states_by_index:
            return None
        idx = self.frame_to_index.get(int(frame_num))
        if idx is None:
            return None
        return self.states_by_index[idx]
