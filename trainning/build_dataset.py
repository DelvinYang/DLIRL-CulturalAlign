from pathlib import Path
from datetime import datetime
import random
import torch
from tqdm import tqdm
from loguru import logger


# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "generated"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "psiphi_datasets"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

WINDOW = 12
STRIDE = 13
MIN_TRAJ_LEN = 15  # enforce >= 15
STD_LOW_QUANTILE = 0.3
STD_LOW_KEEP_RATIO = 0.1
STD_FILTER_SEED = 42


# =========================
# Core builder
# =========================
def build_split_dataset(country: str, split: str):
    country_dir = DATA_ROOT / country / split
    assert country_dir.exists(), f"{country_dir} not found"

    pt_files = sorted(country_dir.glob("*.pt"))
    assert len(pt_files) > 0, f"No pt files in {country_dir}"

    s_list = []
    a_list = []
    s_next_list = []
    a_next_list = []
    traj_id_list = []
    t_idx_list = []
    skipped_traj = 0
    std_scores = []

    logger.info(f"Building dataset for {country}/{split}, files={len(pt_files)}")

    traj_id_offset = 0
    for pt_path in tqdm(pt_files, desc=f"{country} files"):
        data = torch.load(pt_path, map_location="cpu")

        trajs = data.get("trajectories", [])
        for traj_id, traj in enumerate(trajs):
            if not torch.is_tensor(traj):
                traj = torch.tensor(traj, dtype=torch.float32)

            T, D = traj.shape
            if T < MIN_TRAJ_LEN:
                skipped_traj += 1
                continue
            assert D == 12, f"Unexpected state dim {D}"

            # stride sampling
            for t in range(0, T - (WINDOW + 1), STRIDE):
                if t + 13 >= T:
                    break

                s = traj[t : t + WINDOW]              # [12,12]
                a = traj[t + WINDOW][[1, 3]]          # ax, ay
                s_next = traj[t + 1 : t + 1 + WINDOW] # [12,12]
                a_next = traj[t + WINDOW + 1][[1, 3]] # ax, ay

                # safety check
                if (
                    s.shape != (12, 12)
                    or s_next.shape != (12, 12)
                    or a.shape != (2,)
                    or a_next.shape != (2,)
                ):
                    continue

                # Window variability score for ax/ay to filter overly smooth samples.
                ax_std = float(s[:, 1].std().item())
                ay_std = float(s[:, 3].std().item())
                std_scores.append(ax_std + ay_std)

                s_list.append(s)
                a_list.append(a)
                s_next_list.append(s_next)
                a_next_list.append(a_next)
                traj_id_list.append(traj_id_offset + traj_id)
                t_idx_list.append(t)
        traj_id_offset += len(trajs)

    if s_list:
        if 0.0 < STD_LOW_QUANTILE < 1.0 and STD_LOW_KEEP_RATIO < 1.0:
            rng = random.Random(STD_FILTER_SEED)
            scores_tensor = torch.tensor(std_scores, dtype=torch.float32)
            thr = torch.quantile(scores_tensor, STD_LOW_QUANTILE).item()
            keep_idx = []
            for i, score in enumerate(std_scores):
                if score >= thr or rng.random() < STD_LOW_KEEP_RATIO:
                    keep_idx.append(i)
            if keep_idx:
                s_list = [s_list[i] for i in keep_idx]
                a_list = [a_list[i] for i in keep_idx]
                s_next_list = [s_next_list[i] for i in keep_idx]
                a_next_list = [a_next_list[i] for i in keep_idx]
                traj_id_list = [traj_id_list[i] for i in keep_idx]
                t_idx_list = [t_idx_list[i] for i in keep_idx]
            logger.info(
                f"[{country}/{split}] std filter: thr={thr:.6f}, kept {len(s_list)}/{len(std_scores)} "
                f"(low_quantile={STD_LOW_QUANTILE}, low_keep_ratio={STD_LOW_KEEP_RATIO})"
            )

        s_tensor = torch.stack(s_list, dim=0)
        a_tensor = torch.stack(a_list, dim=0)
        s_next_tensor = torch.stack(s_next_list, dim=0)
        a_next_tensor = torch.stack(a_next_list, dim=0)
        traj_id_tensor = torch.tensor(traj_id_list, dtype=torch.long)
        t_idx_tensor = torch.tensor(t_idx_list, dtype=torch.long)
    else:
        s_tensor = torch.empty((0, 12, 12), dtype=torch.float32)
        a_tensor = torch.empty((0, 2), dtype=torch.float32)
        s_next_tensor = torch.empty((0, 12, 12), dtype=torch.float32)
        a_next_tensor = torch.empty((0, 2), dtype=torch.float32)
        traj_id_tensor = torch.empty((0,), dtype=torch.long)
        t_idx_tensor = torch.empty((0,), dtype=torch.long)

    out = {
        # =========================
        # Core tensors (SoA)
        # =========================
        "s": s_tensor,
        "a": a_tensor,
        "s_next": s_next_tensor,
        "a_next": a_next_tensor,

        # =========================
        # Optional but useful
        # =========================
        "traj_id": traj_id_tensor,
        "t_idx": t_idx_tensor,

        # =========================
        # Meta (small, python)
        # =========================
        "meta": {
            "country": country,
            "split": split,
            "state_dim": 12,
            "window": WINDOW,
            "stride": STRIDE,
            "min_traj_len": MIN_TRAJ_LEN,
            "feature_desc": [
                "vx", "ax", "vy", "ay",
                "dist_front", "dist_back",
                "dist_left", "dist_right",
                "dist_front_left", "dist_front_right",
                "dist_back_left", "dist_back_right"
            ],
            "created": datetime.now().isoformat(timespec="seconds"),
            "num_samples": int(s_tensor.shape[0]),
        },
    }

    out_path = OUTPUT_ROOT / f"psiphi_{country}_{split}.pt"
    torch.save(out, out_path)

    logger.info(
        f"[{country}/{split}] saved {int(s_tensor.shape[0])} samples "
        f"(skipped trajs={skipped_traj}) -> {out_path}"
    )


def build_country_dataset(country: str):
    for split in ("train", "test"):
        build_split_dataset(country, split)


# =========================
# Main
# =========================
if __name__ == "__main__":
    for country in ["Germany"]:
        build_country_dataset(country)
