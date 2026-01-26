"""
Build trajectory-level dataset for CitySim.
Each .pt file stores all trajectories in one recording as
per-frame states: [vx, ax, vy, ay, d1..d8].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datasets.citysim import (
    PIXEL_TO_M,
    DataReaderCitySim,
    ScenarioCitySim,
    State,
    Vehicle,
    eight_dirs_by_heading,
)

DEFAULT_DATA_PATH = "/path/to/datasets/CitySim"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"

LOCATION_PREFIXES: dict[str, List[str]] = {
    # "IntersectionA": ["01"],
    "IntersectionA": ["01", "02", "03", "05", "07", "08", "09", "10", "11", "12"],
    "IntersectionB": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
    "IntersectionD": ["01", "02", "03"],
    "IntersectionE": ["01", "02", "03", "04", "05", "06", "07", "08"],
}

DIR_NAMES = (
    "preceding",
    "following",
    "leftPreceding",
    "leftAlongside",
    "leftFollowing",
    "rightPreceding",
    "rightAlongside",
    "rightFollowing",
)

MIN_TRAJ_LEN = 15
MAX_RADIUS_METERS = 60.0
MAX_ABS_ACC = 10.0


def extract_frame_state(
    scenario: ScenarioCitySim,
    ego_vehicle: Vehicle,
    frame_num: int,
    max_radius: float,
) -> torch.Tensor | None:
    """Extract one frame state: [vx, ax, vy, ay, d1..d8]."""

    ego_state: State | None = ego_vehicle.find_vehicle_state(frame_num)
    if ego_state is None:
        return None

    lon_acc = float(ego_state.lon[1])
    lat_acc = float(ego_state.lat[1])
    if abs(lon_acc) > MAX_ABS_ACC or abs(lat_acc) > MAX_ABS_ACC:
        return None

    ego_feature = torch.tensor(
        ego_state.lon + ego_state.lat, dtype=torch.float
    )  # [vx, ax, vy, ay]

    dir_dist = {name: -1.0 for name in DIR_NAMES}
    ego_pos = np.array(ego_state.center, dtype=float)
    ego_pos_m = ego_pos * PIXEL_TO_M
    ego_theta = float(ego_state.course_rad)

    svs_state = scenario.find_svs_state(frame_num, ego_vehicle.vehicle_id)
    for sv in svs_state:
        sv_pos = np.array(sv.center, dtype=float)
        dist_m = np.linalg.norm(sv_pos - ego_pos) * PIXEL_TO_M
        if dist_m > max_radius:
            continue
        sv_pos_m = sv_pos * PIXEL_TO_M
        direction = eight_dirs_by_heading(ego_pos_m, ego_theta, sv_pos_m)
        if direction in dir_dist:
            if dir_dist[direction] < 0 or dist_m < dir_dist[direction]:
                dir_dist[direction] = dist_m

    dist_tensor = torch.tensor(
        [dir_dist[name] for name in DIR_NAMES], dtype=torch.float
    )

    return torch.cat([ego_feature, dist_tensor], dim=0)  # [12]


def collect_trajectories(
    scenario: ScenarioCitySim,
    max_radius: float,
) -> List[torch.Tensor]:
    """Collect all valid trajectories in one scenario."""

    trajectories: List[torch.Tensor] = []

    for ego_id in tqdm(scenario.id_list, desc="Collecting trajectories"):
        ego_vehicle = scenario.find_vehicle_by_id(ego_id)
        start_f = ego_vehicle.initial_frame
        end_f = ego_vehicle.final_frame

        frame_states: List[torch.Tensor] = []

        for frame_num in range(start_f, end_f + 1):
            try:
                state = extract_frame_state(
                    scenario, ego_vehicle, frame_num, max_radius
                )
                if state is None:
                    break
                frame_states.append(state)
            except Exception as exc:  # noqa
                logger.debug(
                    "skip frame {} of ego {}: {}",
                    frame_num,
                    ego_vehicle.vehicle_id,
                    exc,
                )
                break

        if len(frame_states) >= MIN_TRAJ_LEN:
            traj = torch.stack(frame_states, dim=0)  # [T, 12]
            trajectories.append(traj)

    return trajectories


def build_dataset(
    location: str,
    prefixes: List[str],
    data_path: str,
    output_dir: Path,
    max_radius: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_prefixes = set(LOCATION_PREFIXES.get(location, []))

    for prefix in prefixes:
        if valid_prefixes and prefix not in valid_prefixes:
            logger.warning("skip unsupported prefix {}", prefix)
            continue

        logger.info("Processing {}_{}", location, prefix)

        try:
            reader = DataReaderCitySim(location, prefix, data_path)
            scenario = ScenarioCitySim(reader)
        except Exception as exc:  # noqa
            logger.warning("skip prefix {} due to {}", prefix, exc)
            continue

        trajectories = collect_trajectories(
            scenario=scenario,
            max_radius=max_radius,
        )

        save_path = output_dir / f"traj_states_{location}_{prefix}.pt"
        torch.save(
            {
                "location": location,
                "prefix": prefix,
                "state_dim": 12,
                "min_traj_len": MIN_TRAJ_LEN,
                "trajectories": trajectories,
            },
            save_path,
        )

        logger.info(
            "Saved {} trajectories to {}",
            len(trajectories),
            save_path.as_posix(),
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--location", nargs="*", default=None)
    parser.add_argument("--prefix", nargs="*", default=None)
    parser.add_argument("--max-radius", type=float, default=MAX_RADIUS_METERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    locations = args.location or list(LOCATION_PREFIXES.keys())

    for location in locations:
        prefixes = args.prefix or LOCATION_PREFIXES[location]
        build_dataset(
            location=location,
            prefixes=prefixes,
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_radius=args.max_radius,
        )


if __name__ == "__main__":
    logger.info("start building trajectory dataset")
    main()
