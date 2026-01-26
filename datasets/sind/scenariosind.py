from __future__ import annotations

from typing import TYPE_CHECKING

"""Scenario and vehicle utilities for the sinD dataset."""

from .common import *
from .read_data_sinD import DataReaderSinD

if TYPE_CHECKING:  # pragma: no cover - aid IDE/static typing only
    from .common import State


class ScenarioSinD(object):
    """High level access to vehicles within a sinD recording."""

    def __init__(self, data_reader: DataReaderSinD):
        """Create a scenario from a :class:`DataReaderSinD` instance."""
        self.data_reader = data_reader
        self.vehicles, self.id_list = self.set_vehicles()

    def set_vehicles(self):
        """Create :class:`Vehicle` instances for all tracks."""
        vehicles_dict = {}
        id_list = set()
        for track in self.data_reader.tracks:
            track_id = track[TRACK_ID][0]
            if track[AGENT_TYPE] == "bicycle":
                continue
            vehicles_dict[track_id] = Vehicle(track)
            id_list.add(track_id)

        return vehicles_dict, id_list

    def find_vehicle_by_id(self, vehicle_id):
        """Return a :class:`Vehicle` for ``vehicle_id``."""
        vehicle = self.vehicles[vehicle_id]
        return vehicle

    def set_reference_path(self, ego_id):
        """Return the reference path for ``ego_id`` as list of (x, y)."""
        reference_path = []
        ego_vehicle = self.find_vehicle_by_id(ego_id)
        for sublist in ego_vehicle.track[BBOX]:
            x, y, w, h = sublist
            center = (x + 0.5 * w, y + 0.5 * h)
            reference_path.append(center)

        return reference_path

    def find_vehicle_state(self, frame_num, vehicle_id):
        """Return the :class:`State` of ``vehicle_id`` at ``frame_num``."""
        vehicle = self.vehicles[vehicle_id]
        if vehicle.initial_frame <= frame_num <= vehicle.final_frame:
            index = frame_num - vehicle.initial_frame
            x = vehicle.track[BBOX][index][0]
            y = vehicle.track[BBOX][index][1]
            x_velocity = vehicle.track[X_VELOCITY][index]
            y_velocity = vehicle.track[Y_VELOCITY][index]
            x_acceleration = vehicle.track[X_ACCELERATION][index]
            y_acceleration = vehicle.track[Y_ACCELERATION][index]

            lon_velocity = vehicle.track[LON_VELOCITY][index]
            lat_velocity = vehicle.track[LAT_VELOCITY][index]
            lon_acceleration = vehicle.track[LON_ACCELERATION][index]
            lat_acceleration = vehicle.track[LAT_ACCELERATION][index]

            heading_rad = vehicle.track[HEADING_RAD][index]

            x_value = [x, x_velocity, x_acceleration]
            y_value = [y, y_velocity, y_acceleration]
            lon_value = [lon_velocity, lon_acceleration]
            lat_value = [lat_velocity, lat_acceleration]

            width = vehicle.track[BBOX][index][2]
            height = vehicle.track[BBOX][index][3]
            vehicle_type = vehicle.track[AGENT_TYPE]

            state_f = State(
                vehicle_id,
                x_value,
                y_value,
                lon_value,
                lat_value,
                width,
                height,
                heading_rad,
                vehicle_type,
            )
            return state_f
        else:
            return None

    def find_svs_state(self, frame_num, vehicle_id):
        """Return states of all surrounding vehicles at ``frame_num``."""
        svs_state = []
        for vehicle in self.vehicles.values():
            if vehicle.vehicle_id != vehicle_id:
                state = self.find_vehicle_state(
                    frame_num=frame_num, vehicle_id=vehicle.vehicle_id
                )
                if state is not None:
                    svs_state.append(state)
        return svs_state


class Vehicle:
    """Representation of a single track."""

    def __init__(self, track):
        self.track = track
        self.initial_frame = int(track[FRAME].min())
        self.final_frame = int(track[FRAME].max())
        self.vehicle_id = track[TRACK_ID][0]
