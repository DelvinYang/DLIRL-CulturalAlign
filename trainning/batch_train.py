# batch_train_psiphi.py
import os
import subprocess
import random
import time
from pathlib import Path
from datetime import datetime

# ===== config =====
train_script = Path(__file__).resolve().parent / "train_psiphi.py"
base_output_dir = Path("../experiments_psiphi_1227")

# datasets: (name, train pt, test pt)
repo_root = Path(__file__).resolve().parents[1]
datasets = [
    ("US",
     str(repo_root / "data" / "psiphi_datasets" / "psiphi_US_train.pt"),
     str(repo_root / "data" / "psiphi_datasets" / "psiphi_US_test.pt")),
    # ("China",
    #  str(repo_root / "data" / "psiphi_datasets" / "psiphi_China_train.pt"),
    #  str(repo_root / "data" / "psiphi_datasets" / "psiphi_China_test.pt")),
    # ("Germany",
    #  str(repo_root / "data" / "psiphi_datasets" / "psiphi_Germany_train.pt"),
    #  str(repo_root / "data" / "psiphi_datasets" / "psiphi_Germany_test.pt")),
]

batch_size = 8192
epochs = 1000
lr = 2e-3
feature_dim = 64
gamma = 0.95
kneg = 12
neg_mode = "gauss"
neg_uni_range = 3.0
neg_gauss_std = 0.35
acc_coef = 0.1
bc_coef = 0.05
itd_coef = 0.01
num_workers = 4
pin_memory = 1
cache_gpu = 1
accum = 1
warmup = 0
cos_final_lr_ratio = 0.05
early_delta = 5e-6

# train_coeff list (percent -> coeff)
percents = ["p100", "p90", "p80", "p70", "p60", "p50", "p40", "p30", "p20", "p10", "p05", "p0025"]
train_coeff_list = [1.0,   0.9,   0.8,   0.7,   0.6,   0.5,   0.4,   0.3,   0.2,   0.1,   0.05,  0.025]

# percents = ["p100"]
# train_coeff_list = [1.0]
# runs for each percent (set to 1 if you only need one run)
runs_per_case = 1

# training mode config
# Localized
# mode = "train_all"
# sf_path = None
# pv_path = None

# Cross Cultural
mode = "train_all"  #   "train_all" or "ft_sf_fix_pv"

cross_configs = [
    # China-U.S
    {
        "name": "China-US",
        "dataset": "US",
        "sf_path": "/path/to/experiments_psiphi/China/Localized/p100/run_xxxxx/sf_state_xxx.pth",
        "pv_path": "/path/to/experiments_psiphi/US/p100/run_xxxxx/pv_xxx.npy",
    },
    # China-Germany
    {
        "name": "China-Germany",
        "dataset": "Germany",
        "sf_path": "/path/to/experiments_psiphi/China/Localized/p100/run_xxxxx/sf_state_xxx.pth",
        "pv_path": "/path/to/experiments_psiphi/Germany/p100/run_xxxxx/pv_xxx.npy",
    },
    # U.S.-China
    {
        "name": "US-China",
        "dataset": "China",
        "sf_path": "/path/to/experiments_psiphi/US/p100/run_xxxxx/sf_state_xxx.pth",
        "pv_path": "/path/to/experiments_psiphi/China/Localized/p100/run_xxxxx/pv_xxx.npy",
    },
    # U.S.-Germany
    {
        "name": "US-Germany",
        "dataset": "Germany",
        "sf_path": "/path/to/experiments_psiphi/US/p100/run_xxxxx/sf_state_xxx.pth",
        "pv_path": "/path/to/experiments_psiphi/Germany/p100/run_xxxxx/pv_xxx.npy",
    },
]


freeze_sf = 0
lr_mult_sf = 0.1
lr_mult_pv = 1.0
lr_mult_act = 1.0
train_act_head = 1

if mode == "ft_sf_fix_pv":
    lr_mult_pv = 0.0
    train_act_head = 1

# eval / early stop
eval_every = 5
early_patience = 3
stage2_extra_epochs = 0
stage2_patience = 3

# run one process per run; optionally pin to a single GPU
use_torchrun = False
gpu_ids = ["0", "1", "2", "3"]

# ===== batch run =====
def build_cmd(train_path, test_path, train_coeff, run_dir, seed, sf_path=None, pv_path=None):
    cmd = []
    if use_torchrun:
        cmd += ["torchrun", f"--nproc_per_node={len(gpu_ids)}"]
    else:
        cmd += ["python"]
    cmd += [
        str(train_script),
        "--train_data", str(train_path),
        "--test_data", str(test_path),
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--lr", str(lr),
        "--feature_dim", str(feature_dim),
        "--gamma", str(gamma),
        "--kneg", str(kneg),
        "--neg_mode", str(neg_mode),
        "--neg_uni_range", str(neg_uni_range),
        "--neg_gauss_std", str(neg_gauss_std),
        "--acc_coef", str(acc_coef),
        "--bc_coef", str(bc_coef),
        "--itd_coef", str(itd_coef),
        "--mode", str(mode),
        "--freeze_sf", str(freeze_sf),
        "--lr_mult_sf", str(lr_mult_sf),
        "--lr_mult_pv", str(lr_mult_pv),
        "--lr_mult_act", str(lr_mult_act),
        "--train_act_head", str(train_act_head),
        "--train_coeff", str(train_coeff),
        "--eval_every", str(eval_every),
        "--early_patience", str(early_patience),
        "--stage2_extra_epochs", str(stage2_extra_epochs),
        "--stage2_patience", str(stage2_patience),
        "--accum", str(accum),
        "--warmup", str(warmup),
        "--cos_final_lr_ratio", str(cos_final_lr_ratio),
        "--early_delta", str(early_delta),
        "--num_workers", str(num_workers),
        "--pin_memory", str(pin_memory),
        "--save_dir", str(run_dir),
        "--seed", str(seed),
    ]

    if int(cache_gpu) == 1:
        cmd += ["--cache_gpu", "1"]
    if mode == "ft_sf_fix_pv":
        if not sf_path or not pv_path:
            raise ValueError("ft_sf_fix_pv requires sf_path and pv_path")
        cmd += ["--sf_path", str(sf_path), "--pv_path", str(pv_path)]
    return cmd


def wait_for_slot(running, max_slots):
    while len(running) >= max_slots:
        for i, (proc, _) in enumerate(list(running)):
            ret = proc.poll()
            if ret is not None:
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, proc.args)
                running.pop(i)
                break
        else:
            time.sleep(1.0)


def drain_running(running):
    while running:
        proc, _ = running[0]
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, proc.args)
        running.pop(0)


dataset_map = {name: (train_path, test_path) for name, train_path, test_path in datasets}
if mode == "ft_sf_fix_pv":
    run_configs = cross_configs
else:
    run_configs = [
        {"name": dataset_name, "dataset": dataset_name, "sf_path": None, "pv_path": None}
        for dataset_name, _, _ in datasets
    ]

running = []
max_slots = len(gpu_ids) if (not use_torchrun and gpu_ids) else 1
run_idx = 0
for config in run_configs:
    dataset_name = config["dataset"]
    dataset_paths = dataset_map.get(dataset_name)
    if dataset_paths is None:
        raise ValueError(f"Unknown dataset '{dataset_name}' in config '{config['name']}'")
    train_path, test_path = dataset_paths
    sf_path = config.get("sf_path")
    pv_path = config.get("pv_path")
    for _ in range(runs_per_case):
        seed = random.randint(0, 2**32 - 1)
        for percent, train_coeff in zip(percents, train_coeff_list):
            run_dir = base_output_dir / config["name"] / percent / f"run_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = build_cmd(train_path, test_path, train_coeff, run_dir, seed, sf_path=sf_path, pv_path=pv_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Running seed={seed} | ds={dataset_name} | {config['name']} | {percent} (train_coeff={train_coeff})")

            if use_torchrun:
                subprocess.run(cmd, check=True)
                continue

            wait_for_slot(running, max_slots)
            env = None
            if gpu_ids:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[run_idx % len(gpu_ids)]
                run_idx += 1
            proc = subprocess.Popen(cmd, env=env)
            running.append((proc, env))

drain_running(running)
