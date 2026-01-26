"""
Build trajectory-level dataset for the DJI interaction dataset.
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

from datasets.dji.common import State
from datasets.dji.read_data_dji import DataReaderDJI
from datasets.dji.scenariodji import ScenarioDJI, Vehicle

DEFAULT_DATA_PATH = "/path/to/datasets/DJI"
DEFAULT_PREFIXES = [f"{idx:02d}" for idx in range(1, 69)]
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

MIN_TRAJ_LEN = 15


def extract_frame_state(
    scenario: ScenarioDJI,
    ego_vehicle: Vehicle,
    frame_num: int,
) -> torch.Tensor | None:
    """Extract one frame state: [vx, ax, vy, ay, d1..d8]."""

    ego_state: State | None = scenario.find_vehicle_state(
        frame_num, ego_vehicle.vehicle_id
    )
    if ego_state is None:
        return None

    ego_feature = torch.tensor(
        list(ego_state.x[1:]) + list(ego_state.y[1:]), dtype=torch.float
    )
    ego_pos = np.array([ego_state.x[0], ego_state.y[0]], dtype=float)

    eight_vehicles = scenario.find_eight_vehicles(frame_num, ego_vehicle.vehicle_id)
    distances: List[float] = []
    for sv in eight_vehicles:
        if sv is None:
            distances.append(-1.0)
            continue
        sv_pos = np.array([sv.x[0], sv.y[0]], dtype=float)
        distances.append(float(np.linalg.norm(sv_pos - ego_pos)))

    if len(distances) < len(DIR_NAMES):
        distances.extend([-1.0] * (len(DIR_NAMES) - len(distances)))

    dist_tensor = torch.tensor(distances[: len(DIR_NAMES)], dtype=torch.float)
    return torch.cat([ego_feature, dist_tensor], dim=0)


def collect_trajectories(
    scenario: ScenarioDJI,
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
    data_path: str,
    output_dir: Path,
) -> None:
    """Iterate over prefixes and persist the resulting tensors."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for prefix_number in prefixes:
        logger.info("Processing prefix {}", prefix_number)
        try:
            data_reader = DataReaderDJI(prefix_number, data_path)
            scenario = ScenarioDJI(data_reader)
        except Exception as exc:  # noqa
            logger.warning("skip prefix {} due to {}", prefix_number, exc)
            continue

        trajectories = collect_trajectories(scenario)
        save_path = output_dir / f"traj_states_{prefix_number}.pt"
        torch.save(
            {
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
        help="Root directory that contains the DJI_00XX folders.",
    )
    parser.add_argument(
        "--prefix",
        nargs="*",
        default=None,
        help="Recording prefixes to process. Defaults to known prefixes.",
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
    prefixes = args.prefix or DEFAULT_PREFIXES
    build_dataset(
        prefixes=prefixes,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    logger.info("start building trajectory dataset")
    main()
