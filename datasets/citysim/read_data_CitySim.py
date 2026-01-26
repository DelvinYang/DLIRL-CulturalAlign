import os

import numpy as np
import pandas as pd
from loguru import logger

from .common import *


class DataReaderCitySim(object):
    def __init__(self, location_name, prefix_number, data_path):
        self.location_name = location_name
        self.prefix_number = prefix_number
        self.data_path = data_path
        self.csv_tracks_path, self.background_path = self.generate_path()
        self.tracks, self.id_list = self.read_tracks_csv()
        logger.info(f"read data done, total tracks: {len(self.tracks)}")

    def generate_path(self):
        # Build filename: {location_name}-{prefix_number}.csv
        filename = f"{self.location_name}-{self.prefix_number}.csv"

        # Build full path: data_path/location_name/Trajectories/...
        full_path = os.path.join(
            self.data_path, self.location_name, "Trajectories", filename
        )

        background_path = os.path.join(
            self.data_path, self.location_name, "background.png"
        )

        return str(full_path), str(background_path)

    def read_tracks_csv(self):
        # Read the csv file, convert it into a useful data structure
        df = pd.read_csv(self.csv_tracks_path)
        grouped = df.groupby([TRACK_ID], sort=False)

        tracks = []
        id_list = set()
        for group_id, rows in grouped:
            speed_values = rows[SPEED].values

            # Skip if speed is all zeros or NaN
            if np.all(np.nan_to_num(speed_values) == 0):
                logger.info(
                    f"Track {group_id} skipped: all speed values are 0 (len={len(speed_values)})"
                )
                continue

            x_pos = rows[CAR_CENTER_X_FT].values
            y_pos = rows[CAR_CENTER_Y_FT].values
            x_pos_pic = rows[CAR_CENTER_X_PX].values
            y_pos_pic = rows[CAR_CENTER_Y_PX].values
            bbox1_x_pic = rows[BOUNDING_BOX1_X].values
            bbox1_y_pic = rows[BOUNDING_BOX1_Y].values
            bbox2_x_pic = rows[BOUNDING_BOX2_X].values
            bbox2_y_pic = rows[BOUNDING_BOX2_Y].values
            bbox3_x_pic = rows[BOUNDING_BOX3_X].values
            bbox3_y_pic = rows[BOUNDING_BOX3_Y].values
            bbox4_x_pic = rows[BOUNDING_BOX4_X].values
            bbox4_y_pic = rows[BOUNDING_BOX4_Y].values
            head_x_pic = rows[HEAD_X].values
            head_y_pic = rows[HEAD_Y].values
            tail_x_pic = rows[TAIL_X].values
            tail_y_pic = rows[TAIL_Y].values
            lane_id = rows[LANE_ID].values

            # Format: [x_center, y_center]
            center = np.stack([x_pos, y_pos], axis=1)
            center_pic = np.stack([x_pos_pic, y_pos_pic], axis=1)

            # =================== Vehicle size (image pixels) =====================
            dx14_pic = bbox1_x_pic - bbox4_x_pic
            dy14_pic = bbox1_y_pic - bbox4_y_pic
            width = np.sqrt(dx14_pic**2 + dy14_pic**2)

            dx_head_tail = head_x_pic - tail_x_pic
            dy_head_tail = head_y_pic - tail_y_pic
            length = np.sqrt(dx_head_tail**2 + dy_head_tail**2)

            # Format: (N, 4, 2)
            bbox_corners = np.stack(
                [
                    np.stack([bbox1_x_pic, bbox1_y_pic], axis=1),
                    np.stack([bbox2_x_pic, bbox2_y_pic], axis=1),
                    np.stack([bbox3_x_pic, bbox3_y_pic], axis=1),
                    np.stack([bbox4_x_pic, bbox4_y_pic], axis=1),
                ],
                axis=1,
            )

            speed_values = rows[SPEED].values * MPH_TO_MPS  # mph → m/s

            course_rad = np.deg2rad(rows[COURSE].values)

            track_data = {
                TRACK_ID: np.int64(group_id),
                FRAME: rows[FRAME].values,
                CENTER: center,
                CENTER_PIC: center_pic,
                BBOX_CORNERS: bbox_corners,
                WIDTH: width,
                LENGTH: length,
                SPEED: speed_values,
                COURSE_RAD: course_rad,
                LANE_ID: lane_id,
            }
            tracks.append(track_data)
            id_ = np.int64(group_id)[0]
            id_list.add(id_)
        return tracks, id_list
