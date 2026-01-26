"""
Build trajectory-level dataset for the sinD dataset.
Each .pt file stores all trajectories in one recording as
per-frame states: [vx, ax, vy, ay, d1..d8].
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datasets.sind.common import State, eight_dirs_by_heading
from datasets.sind.read_data_sinD import DataReaderSinD
from datasets.sind.scenariosind import ScenarioSinD, Vehicle

DEFAULT_DATA_PATH = "/path/to/datasets/sinD"
DEFAULT_LOCATION = "tianjin"
DEFAULT_PREFIXES = [str(idx) for idx in range(1, 26)]

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"

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
MAX_RADIUS_METERS = 60.0
MIN_TRAJ_LEN = 15


def extract_frame_state(
    scenario: ScenarioSinD,
    ego_vehicle: Vehicle,
    frame_num: int,
    max_radius: float,
) -> torch.Tensor | None:
    """Extract one frame state: [vx, ax, vy, ay, d1..d8]."""

    ego_state: State | None = scenario.find_vehicle_state(frame_num, ego_vehicle.vehicle_id)
    if ego_state is None:
        return None

    ego_feature = torch.tensor(ego_state.lon + ego_state.lat, dtype=torch.float)

    dir_dist = {name: -1.0 for name in DIR_NAMES}
    ego_pos = np.array([ego_state.x[0], ego_state.y[0]], dtype=float)
    ego_theta = float(ego_state.heading_rad)

    svs_state = scenario.find_svs_state(frame_num, ego_vehicle.vehicle_id)
    for sv in svs_state:
        sv_pos = np.array([sv.x[0], sv.y[0]], dtype=float)
        dist = float(np.linalg.norm(sv_pos - ego_pos))
        if dist > max_radius:
            continue

        direction = eight_dirs_by_heading(ego_pos, ego_theta, sv_pos)
        if direction in dir_dist:
            if dir_dist[direction] < 0 or dist < dir_dist[direction]:
                dir_dist[direction] = dist

    dist_tensor = torch.tensor(
        [dir_dist[name] for name in DIR_NAMES], dtype=torch.float
    )

    return torch.cat([ego_feature, dist_tensor], dim=0)


def collect_trajectories(
    scenario: ScenarioSinD,
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
                    scenario=scenario,
                    ego_vehicle=ego_vehicle,
                    frame_num=frame_num,
                    max_radius=max_radius,
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
            trajectories.append(torch.stack(frame_states, dim=0))

    return trajectories


def build_dataset(
    prefixes: List[str],
    location: str,
    data_path: str,
    output_dir: Path,
    max_radius: float,
) -> None:
    """Iterate over prefixes and persist the resulting tensors."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for prefix_number in prefixes:
        logger.info("Processing prefix {}_{}", location, prefix_number)
        try:
            data_reader = DataReaderSinD(location, prefix_number, data_path)
            scenario = ScenarioSinD(data_reader)
        except Exception as exc:  # noqa
            logger.warning("skip prefix {} due to {}", prefix_number, exc)
            continue

        trajectories = collect_trajectories(scenario, max_radius=max_radius)
        save_path = output_dir / f"traj_states_{location}_{prefix_number}.pt"
        torch.save(
            {
                "location": location,
                "prefix": prefix_number,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Root directory that contains the sinD data folders.",
    )
    parser.add_argument(
        "--location",
        nargs="*",
        default=None,
        help="Location name(s) within the sinD dataset.",
    )
    parser.add_argument(
        "--prefix",
        nargs="*",
        default=None,
        help="Recording prefixes to process. Defaults to known prefixes.",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=MAX_RADIUS_METERS,
        help="Maximum distance (in meters) to consider surrounding vehicles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the generated .pt files.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    locations = args.location or [DEFAULT_LOCATION]

    for location in locations:
        prefixes = args.prefix or DEFAULT_PREFIXES
        build_dataset(
            prefixes=prefixes,
            location=location,
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_radius=args.max_radius,
        )


if __name__ == "__main__":
    logger.info("start building trajectory dataset")
    main()
