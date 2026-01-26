import numpy as np
from scipy.signal import savgol_filter
from typing import TYPE_CHECKING

from .read_data_CitySim import DataReaderCitySim
from .common import *

if TYPE_CHECKING:  # pragma: no cover - aid IDE/static typing only
    from .common import State



class ScenarioCitySim(object):
    def __init__(self, data_reader: DataReaderCitySim):
        self.data_reader = data_reader
        self.dt = 1 / 30.0  # 30 Hz sampling frequency
        self.vehicles, self.id_list = self.set_vehicles()

    def set_vehicles(self):
        vehicles_dict = {}
        id_list = set()
        for track in self.data_reader.tracks:
            track_id = track[TRACK_ID][0]
            vehicles_dict[track_id] = Vehicle(track, dt=self.dt)
            id_list.add(track_id)

        return vehicles_dict, id_list

    def find_vehicle_by_id(self, vehicle_id):
        return self.vehicles[vehicle_id]

    def find_vehicle_bbox(self, frame_num, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        if vehicle.initial_frame <= frame_num <= vehicle.final_frame:
            index = frame_num - vehicle.initial_frame
            return vehicle.track[BBOX_CORNERS][index]
        return None

    def find_svs_state(self, frame_num, vehicle_id):
        svs_state = []
        for vehicle in self.vehicles.values():
            if vehicle.vehicle_id != vehicle_id:
                state = vehicle.find_vehicle_state(frame_num=frame_num)
                if state is not None:
                    svs_state.append(state)
        return svs_state


class Vehicle:
    def __init__(self, track, dt=1 / 30.0):
        self.track = track
        self.dt = dt
        self.vehicle_id = track[TRACK_ID][0]
        self.lane_id = self.set_lane_id()

        self.frames = None
        self.initial_frame = None
        self.final_frame = None

        self.reference_path, self.velocities, self.headings = self.set_values()

        self.theta_ref_list = None
        self.vx = None
        self.vy = None
        self.ax = None
        self.ay = None

        self.states_by_index = None
        self.frame_to_index = None

        self.precompute_states()

    def precompute_states(self):
        """
        Precompute State for all frames and build the frame->index mapping.
        Requirement: len(reference_path) == number of frames.
        """
        self.compute_kinematics()

        centers = self.reference_path
        thetas = self.theta_ref_list
        vx, vy = self.vx, self.vy
        ax, ay = self.ax, self.ay

        width = float(self.track[WIDTH][0])
        length = float(self.track[LENGTH][0])

        n = len(centers)
        frames = self.frames.astype(int)  # Actual frame numbers after trimming

        # Correct mapping: frame_num -> idx
        self.frame_to_index = {int(frames[i]): i for i in range(n)}

        # Build all State objects (Python objects are not vectorized).
        self.states_by_index = [
            State(
                self.vehicle_id,
                centers[i],
                [float(vx[i]), float(ax[i])],
                [float(vy[i]), float(ay[i])],
                length,
                width,
                float(thetas[i]),
            )
            for i in range(n)
        ]

    def set_values(self):
        # Convert raw data to ndarray
        reference_path = np.asarray(self.track[CENTER_PIC], dtype=np.float64)  # (N,2)
        speeds = np.asarray(self.track[SPEED], dtype=np.float64)  # (N,)
        headings = np.asarray(self.track[COURSE_RAD], dtype=np.float64)  # (N,)
        frames = np.asarray(self.track[FRAME], dtype=np.int64)  # (N,)

        # 1) Remove zero-speed segments (middle/ends), keep non-zero slices.
        eps = 1e-6
        mask = np.abs(speeds) > eps
        if not np.any(mask):
            path_cut = reference_path[:0]
            speeds_cut = speeds[:0]
            headings_cut = headings[:0]
            frames_cut = frames[:0]
        else:
            path_cut = reference_path[mask]
            speeds_cut = speeds[mask]
            headings_cut = headings[mask]
            frames_cut = frames[mask]
        self.frames = frames_cut

        # 3) Record initial/final using trimmed frames
        self.initial_frame = (
            int(frames_cut[0]) if len(frames_cut) > 0 else int(frames[0])
        )
        self.final_frame = (
            int(frames_cut[-1]) if len(frames_cut) > 0 else int(frames[-1])
        )

        n = len(path_cut)
        if n == 0:
            return path_cut, speeds_cut, headings_cut

        # Integrate trajectory from speed + heading (use dataset start point).
        t_raw = (frames_cut - frames_cut[0]) * self.dt
        headings_unwrap = np.unwrap(headings_cut)
        dt_steps = np.diff(t_raw, prepend=t_raw[0])

        vx = speeds_cut * np.cos(headings_unwrap)
        vy = speeds_cut * np.sin(headings_unwrap)

        x0 = float(path_cut[0, 0])
        y0 = float(path_cut[0, 1])
        x = x0 + np.cumsum(vx * dt_steps)
        y = y0 + np.cumsum(vy * dt_steps)

        if n > 1:
            t_uniform = np.linspace(t_raw[0], t_raw[-1], n)
            x = np.interp(t_uniform, t_raw, x)
            y = np.interp(t_uniform, t_raw, y)
            headings_unwrap = np.interp(t_uniform, t_raw, headings_unwrap)

        path = np.vstack([x, y]).T

        max_window = n if (n % 2 == 1) else (n - 1)
        window = min(51, max_window)
        poly = 3
        if window >= (poly + 2):
            x_smooth = savgol_filter(path[:, 0], window_length=window, polyorder=poly)
            y_smooth = savgol_filter(path[:, 1], window_length=window, polyorder=poly)
            path = np.vstack([x_smooth, y_smooth]).T

        return path, speeds_cut, headings_unwrap

    def set_lane_id(self):
        lane_id = self.track[LANE_ID]
        return np.array(lane_id)


    def compute_kinematics(self):
        centers = self.reference_path
        frames = self.frames

        n = len(centers)
        if n < 2:
            zeros = np.zeros(n, dtype=float)
            self.vx = zeros
            self.vy = zeros
            self.ax = zeros
            self.ay = zeros
            self.theta_ref_list = self.headings
            return

        t_raw = (frames - frames[0]) * self.dt
        t_uniform = np.linspace(t_raw[0], t_raw[-1], n)
        dt_uniform = t_uniform[1] - t_uniform[0]

        x = centers[:, 0]
        y = centers[:, 1]
        theta = np.asarray(self.headings, dtype=float)

        vx = np.gradient(x, dt_uniform, edge_order=1)
        vy = np.gradient(y, dt_uniform, edge_order=1)
        ax = np.gradient(vx, dt_uniform, edge_order=1)
        ay = np.gradient(vy, dt_uniform, edge_order=1)

        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.theta_ref_list = theta

    def find_vehicle_bbox(self, frame_num):
        if self.initial_frame <= frame_num <= self.final_frame:
            index = frame_num - self.initial_frame
            bbox = self.track[BBOX_CORNERS][index]
            return bbox
        else:
            return None

    def find_vehicle_state(self, frame_num):
        if self.initial_frame <= frame_num <= self.final_frame:
            idx = self.frame_to_index.get(int(frame_num), None)
            if idx is not None:
                return self.states_by_index[idx]
        return None
