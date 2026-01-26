"""Utility for reading NGSIM interaction dataset annotations."""

from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .common import *


class DataReaderNGSIM:
    """Read the NGSIM dataset for a given recording prefix."""

    def __init__(self, prefix_number: str, data_path: str) -> None:
        self.prefix_number = prefix_number
        self.data_path = data_path
        self.csv_tracks_path = self.generate_path()

        self.tracks, self.id_list = self.read_tracks_csv()
        logger.info("read data done, total tracks: {}", len(self.tracks))

    def generate_path(self) -> str:
        """Compute absolute paths to the dataset files for ``prefix_number``."""

        tracks_path = os.path.join(self.data_path, f"trajectories-{self.prefix_number}.csv")
        return str(tracks_path)

    def read_tracks_csv(self) -> Tuple[List[Dict[str, np.ndarray]], Set[int]]:
        """Read per-frame tracks from the CSV file."""

        df = pd.read_csv(self.csv_tracks_path)
        grouped = df.groupby([VEHICLE_ID], sort=False)

        tracks: List[Dict[str, np.ndarray]] = []
        id_list: Set[int] = set()
        for group_id, rows in grouped:
            # Collect all speed/acceleration related values.
            v_a_values = np.concatenate(
                [
                    rows[VELOCITY].values,
                    rows[ACCELERATION].values,
                ]
            )

            # Skip track if all values are zero.
            if np.all(v_a_values == 0):
                logger.info("All Zeros, skipping track {}", group_id)
                continue
            track_data: Dict[str, np.ndarray] = {
                VEHICLE_ID: np.int64(group_id),
                FRAME_ID: np.int64(rows[FRAME_ID].values),
                LOCAL_POSITION: np.transpose(
                    np.array([rows[LOCAL_Y].values * FT_TO_M, rows[LOCAL_X].values * FT_TO_M])
                ),
                VELOCITY: rows[VELOCITY].values * FT_TO_M,
                ACCELERATION: rows[ACCELERATION].values * FT_TO_M,
                LANE_ID: np.int64(rows[LANE_ID].values),
                TOTAL_FRAMES: len(np.int64(rows[TOTAL_FRAMES].values)),
                VEHICLE_LENGTH: rows[VEHICLE_LENGTH].values[0] * FT_TO_M,
                VEHICLE_WIDTH: rows[VEHICLE_WIDTH].values[0] * FT_TO_M,
                INITIAL_FRAME: np.int64(rows[FRAME_ID].values[0]),
                FINAL_FRAME: np.int64(rows[FRAME_ID].values[-1] + 1),
                VEHICLE_CLASS: np.int64(rows[VEHICLE_CLASS].values[0]),
            }
            tracks.append(track_data)
            id_value = int(np.asarray(group_id).item())
            id_list.add(id_value)
        return tracks, id_list
