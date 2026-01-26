"""Scenario helpers built on top of :mod:`src.read_data_highd`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from .common import *
from .read_data_highd import DataReaderHighD

if TYPE_CHECKING:  # pragma: no cover - aid IDE/static typing only
    from .common import State


class ScenarioHighD:
    """Wrapper around :class:`DataReaderHighD` providing vehicle utilities."""

    def __init__(self, data_reader: DataReaderHighD):
        self.data_reader = data_reader
        self.vehicles, self.id_list = self.set_vehicles()
        self.frame_rate = self.data_reader.recordingMeta[FRAME_RATE]
        logger.info("[ScenarioHighD] Imported {} vehicles", len(self.id_list))

    def set_vehicles(self) -> tuple[Dict[int, "Vehicle"], set[int]]:
        """Filter valid vehicles and build a lookup table."""

        vehicles_dict: Dict[int, Vehicle] = {}
        id_list: set[int] = set()
        for track in self.data_reader.tracks:
            track_id = int(track[TRACK_ID][0])
            trackMeta = self.data_reader.tracksMeta[track_id]
            if trackMeta[CLASS] not in {"Truck", "Car", "Bus"}:
                continue
            vehicles_dict[track_id] = Vehicle(track, trackMeta)
            id_list.add(track_id)

        return vehicles_dict, id_list

    def find_vehicle_by_id(self, vehicle_id: int) -> "Vehicle":
        return self.vehicles[vehicle_id]

    def set_reference_path(self, ego_id: int) -> List[tuple[float, float]]:
        """Return the ego vehicle centre trajectory."""

        reference_path = []
        ego_vehicle = self.find_vehicle_by_id(ego_id)
        for x, y, w, h in ego_vehicle.track[BBOX]:
            center = (x, y)
            reference_path.append(center)

        return reference_path

    def find_vehicle_state(self, frame_num: int, vehicle_id: int) -> Optional[State]:
        if vehicle_id not in self.vehicles:  # Skip non-standard surrounding vehicles
            return None
        vehicle = self.vehicles[vehicle_id]
        if vehicle.initial_frame <= frame_num <= vehicle.final_frame:
            index = frame_num - vehicle.initial_frame
            x = vehicle.track[BBOX][index][0]
            y = vehicle.track[BBOX][index][1]
            x_velocity = vehicle.track[X_VELOCITY][index]
            y_velocity = vehicle.track[Y_VELOCITY][index]
            x_acceleration = vehicle.track[X_ACCELERATION][index]
            y_acceleration = vehicle.track[Y_ACCELERATION][index]

            x_value = [x, x_velocity, x_acceleration]
            y_value = [y, y_velocity, y_acceleration]
            width = vehicle.track[BBOX][index][2]
            height = vehicle.track[BBOX][index][3]
            vehicle_type = vehicle.trackMeta[CLASS]

            lane_id = vehicle.track[LANE_ID][index]
            driving_direction = vehicle.direction

            state_f = State(
                vehicle_id,
                x_value,
                y_value,
                width,
                height,
                vehicle_type,
                lane_id,
                driving_direction,
                frame_id=frame_num,
            )
            return state_f
        else:
            return None

    def find_svs_state(self, frame_num: int, vehicle_id: int) -> List[State]:
        """Return the state of all surrounding vehicles for a frame."""

        svs_state = []
        for vehicle in self.vehicles.values():
            if vehicle.vehicle_id != vehicle_id:
                state = self.find_vehicle_state(frame_num=frame_num, vehicle_id=vehicle.vehicle_id)
                if state is not None:
                    svs_state.append(state)
        return svs_state

    def find_eight_vehicles(self, frame_num: int, vehicle_id: int) -> List[Optional[State]]:
        """Return the eight-neighbour vehicle states around the ego vehicle."""

        ego_vehicle = self.find_vehicle_by_id(vehicle_id)
        veh_time = frame_num - ego_vehicle.initial_frame
        ego_track = ego_vehicle.track
        sv_list = [
            ego_track["precedingId"][veh_time],
            ego_track["followingId"][veh_time],
            ego_track["leftPrecedingId"][veh_time],
            ego_track["leftAlongsideId"][veh_time],
            ego_track["leftFollowingId"][veh_time],
            ego_track["rightPrecedingId"][veh_time],
            ego_track["rightAlongsideId"][veh_time],
            ego_track["rightFollowingId"][veh_time],
        ]
        eight_vehicles: List[Optional[State]] = []
        for sv_id in sv_list:
            if sv_id != 0:
                sv_state = self.find_vehicle_state(frame_num, sv_id)
                eight_vehicles.append(sv_state)
            else:
                eight_vehicles.append(None)
        if len(eight_vehicles) != 8:
            logger.error(
                "[ScenarioDJI] find_eight_vehicles: {} at frame {} has {} vehicles",
                vehicle_id,
                frame_num,
                len(eight_vehicles),
            )
        return eight_vehicles


@dataclass
class Vehicle:
    track: Dict[str, list]
    trackMeta: Dict[str, object]

    def __post_init__(self) -> None:
        self.initial_frame = self.trackMeta[INITIAL_FRAME]
        self.final_frame = self.trackMeta[FINAL_FRAME]
        self.vehicle_id = self.track[TRACK_ID][0]
        self.direction = self.trackMeta[DRIVING_DIRECTION]
