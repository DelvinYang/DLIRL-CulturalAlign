"""Utility for reading DJI interaction dataset annotations."""

from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .common import *


class DataReaderDJI:
    """Read the DJI dataset for a given recording prefix."""

    def __init__(self, prefix_number: str, data_path: str) -> None:
        self.prefix_number = prefix_number
        self.data_path = data_path
        (
            self.csv_tracks_path,
            self.csv_tracksMeta_path,
            self.csv_recordingMeta_path,
            self.background_path,
        ) = self.generate_path()

        self.tracks, self.id_list = self.read_tracks_csv()
        self.tracksMeta = self.read_static_info()
        self.recordingMeta = self.read_recording_info()
        logger.info("read data done, total tracks: {}", len(self.tracks))

    def generate_path(self) -> Tuple[str, str, str, str]:
        """Compute absolute paths to the dataset files for ``prefix_number``."""

        data_dir = os.path.join(self.data_path, f"DJI_00{self.prefix_number}")

        tracks_path = os.path.join(data_dir, f"{self.prefix_number}_tracks.csv")
        tracks_meta_path = os.path.join(data_dir, f"{self.prefix_number}_tracksMeta.csv")
        recording_meta_path = os.path.join(data_dir, f"{self.prefix_number}_recordingMeta.csv")
        background_path = os.path.join(data_dir, f"{self.prefix_number}_backgroundpics.jpg")

        return (
            str(tracks_path),
            str(tracks_meta_path),
            str(recording_meta_path),
            str(background_path),
        )

    def read_tracks_csv(self) -> Tuple[List[Dict[str, np.ndarray]], Set[int]]:
        """Read per-frame tracks from the CSV file."""

        df = pd.read_csv(self.csv_tracks_path)
        grouped = df.groupby([TRACK_ID], sort=False)

        tracks: List[Dict[str, np.ndarray]] = []
        id_list: Set[int] = set()
        for group_id, rows in grouped:
            # Collect all speed/acceleration related values.
            v_a_values = np.concatenate(
                [
                    rows[X_VELOCITY].values,
                    rows[Y_VELOCITY].values,
                    rows[X_ACCELERATION].values,
                    rows[Y_ACCELERATION].values,
                ]
            )

            # Skip track if all values are zero.
            if np.all(v_a_values == 0):
                logger.info("All Zeros, skipping track {}", group_id)
                continue

            bounding_boxes = np.transpose(
                np.array(
                    [
                        rows[X].values,
                        rows[Y].values,
                        rows[WIDTH].values,
                        rows[HEIGHT].values,
                    ]
                )
            )
            track_data: Dict[str, np.ndarray] = {
                TRACK_ID: np.asarray(group_id),  # numpy array for compatibility
                FRAME: rows[FRAME].values,
                BBOX: bounding_boxes,
                X_VELOCITY: rows[X_VELOCITY].values,
                Y_VELOCITY: rows[Y_VELOCITY].values,
                X_ACCELERATION: rows[X_ACCELERATION].values,
                Y_ACCELERATION: rows[Y_ACCELERATION].values,
                PRECEDING_ID: rows[PRECEDING_ID].values,
                FOLLOWING_ID: rows[FOLLOWING_ID].values,
                LEFT_FOLLOWING_ID: rows[LEFT_FOLLOWING_ID].values,
                LEFT_ALONGSIDE_ID: rows[LEFT_ALONGSIDE_ID].values,
                LEFT_PRECEDING_ID: rows[LEFT_PRECEDING_ID].values,
                RIGHT_FOLLOWING_ID: rows[RIGHT_FOLLOWING_ID].values,
                RIGHT_ALONGSIDE_ID: rows[RIGHT_ALONGSIDE_ID].values,
                RIGHT_PRECEDING_ID: rows[RIGHT_PRECEDING_ID].values,
                LANE_ID: rows[LANE_ID].values,
            }
            tracks.append(track_data)
            id_value = int(np.asarray(group_id).item())
            id_list.add(id_value)
        return tracks, id_list

    def read_static_info(self) -> Dict[int, Dict[str, object]]:
        """Read per-track static metadata."""

        df = pd.read_csv(self.csv_tracksMeta_path)

        static_dictionary: Dict[int, Dict[str, object]] = {}

        # Iterate over all rows of the csv because we need to create the bounding boxes for each row
        for i_row in range(df.shape[0]):
            track_id = int(df[TRACK_ID][i_row])
            static_dictionary[track_id] = {
                TRACK_ID: track_id,
                WIDTH: int(df[WIDTH][i_row]),
                HEIGHT: int(df[HEIGHT][i_row]),
                INITIAL_FRAME: int(df[INITIAL_FRAME][i_row]),
                FINAL_FRAME: int(df[FINAL_FRAME][i_row]),
                NUM_FRAMES: int(df[NUM_FRAMES][i_row]),
                CLASS: str(df[CLASS][i_row]),
                DRIVING_DIRECTION: float(df[DRIVING_DIRECTION][i_row]),
            }
        return static_dictionary

    def read_recording_info(self) -> Dict[str, object]:
        """Read recording-level metadata such as FPS and speed limit."""

        df = pd.read_csv(self.csv_recordingMeta_path)

        extracted_meta_dictionary = {
            ID: int(df[ID][0]),
            FRAME_RATE: int(df[FRAME_RATE][0]),
            SPEED_LIMIT: float(df[SPEED_LIMIT][0]),
        }
        return extracted_meta_dictionary
