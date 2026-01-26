"""Utilities for reading sinD dataset CSV files."""

import os

import numpy as np
from loguru import logger
from .common import *
import pandas as pd


class DataReaderSinD(object):
    """Load sinD recording data from the CSV files produced by the dataset."""

    def __init__(self, location_name, prefix_number, data_path):
        """Create a reader for a single recording."""
        self.location_name = location_name
        self.prefix_number = prefix_number
        self.data_path = data_path
        self.csv_tracks_path = self.generate_path()
        self.tracks, self.id_list = self.read_tracks_csv()
        logger.info(f"read data done, total tracks: {len(self.tracks)}")

    def generate_path(self):
        """Return the path to `Veh_smoothed_tracks.csv` for this recording."""
        subfolder_name = f"{self.location_name}_{self.prefix_number}"
        csv_tracks_path = os.path.join(
            self.data_path,
            self.location_name,
            subfolder_name,
            "Veh_smoothed_tracks.csv",
        )
        return str(csv_tracks_path)

    def read_tracks_csv(self):
        """Read the CSV file and convert it into a list of track dictionaries."""
        df = pd.read_csv(self.csv_tracks_path)
        grouped = df.groupby([TRACK_ID], sort=False)

        tracks = []
        id_list = set()
        for group_id, rows in grouped:
            # Concatenate all velocity and acceleration fields
            v_a_values = np.concatenate(
                [
                    rows[X_VELOCITY].values,
                    rows[Y_VELOCITY].values,
                    rows[X_ACCELERATION].values,
                    rows[Y_ACCELERATION].values,
                    rows[LON_VELOCITY].values,
                    rows[LAT_VELOCITY].values,
                    rows[LON_ACCELERATION].values,
                    rows[LAT_ACCELERATION].values,
                ]
            )

            # Skip tracks that contain only zeros
            if np.all(v_a_values == 0):
                logger.info(f"All Zeros, skipping track {group_id}")
                continue

            bounding_boxes = np.transpose(
                np.array(
                    [
                        rows[X].values,
                        rows[Y].values,
                        rows[LENGTH].values,
                        rows[WIDTH].values,
                    ]
                )
            )
            track_data = {
                TRACK_ID: np.int64(group_id),
                # for compatibility, int would be more space efficient
                FRAME: rows[FRAME].values,
                BBOX: bounding_boxes,
                X_VELOCITY: rows[X_VELOCITY].values,
                Y_VELOCITY: rows[Y_VELOCITY].values,
                X_ACCELERATION: rows[X_ACCELERATION].values,
                Y_ACCELERATION: rows[Y_ACCELERATION].values,
                LON_VELOCITY: rows[LON_VELOCITY].values,
                LAT_VELOCITY: rows[LAT_VELOCITY].values,
                LON_ACCELERATION: rows[LON_ACCELERATION].values,
                LAT_ACCELERATION: rows[LAT_ACCELERATION].values,
                HEADING_RAD: rows[HEADING_RAD].values,
                AGENT_TYPE: str(rows[AGENT_TYPE].values[0]),
            }
            tracks.append(track_data)
            id_ = np.int64(group_id)[0]
            id_list.add(id_)
        return tracks, id_list
