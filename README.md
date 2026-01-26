# Human-inspired Data-light Cultural Alignment for Cross-regional Deployment of Autonomous Vehicles

This repository is the official codebase for the paper "Human-inspired Data-light Cultural Alignment for Cross-regional Deployment of Autonomous Vehicles".

## Overview
This project provides scripts to build trajectory datasets from multiple public driving datasets, construct PsiPhi training data, and train SF+PV models for cross-regional cultural alignment.

## Requirements
- Python 3.10+
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
The raw datasets are not included. Please obtain access to each dataset and follow its license terms.

### Build per-recording trajectory files
Each script converts a raw dataset into `.pt` files containing per-frame states.

Examples (replace paths with your local dataset paths):
```bash
python scripts/dji/build_dataset_dji_new.py --data-path /path/to/datasets/DJI --output-dir data/generated
python scripts/highd/build_dataset_highd_new.py --data-path /path/to/datasets/highD --output-dir data/generated
python scripts/ind/build_dataset_ind_new.py --data-path /path/to/datasets/inD --output-dir data/generated
python scripts/ngsim/build_dataset_ngsim_new.py --data-path /path/to/datasets/NGSIM --output-dir data/generated
python scripts/sind/build_dataset_sind_new.py --data-path /path/to/datasets/sinD --output-dir data/generated
python scripts/citysim/build_dataset_citysim_new.py --data-path /path/to/datasets/CitySim --output-dir data/generated
```

### Organize train/test splits
`trainning/build_dataset.py` expects the following structure:
```
data/generated/<country>/<split>/*.pt
```
Create your train/test splits by moving or copying the generated `.pt` files into:
```
data/generated/US/train
data/generated/US/test
```
Repeat for other regions (e.g., `China`, `Germany`) as needed.

## Build PsiPhi Dataset
After arranging the train/test splits, build the PsiPhi dataset:
```bash
python trainning/build_dataset.py
```
This script writes outputs to:
```
data/psiphi_datasets/psiphi_<country>_<split>.pt
```
If you need different countries, edit the country list in `trainning/build_dataset.py`.

## Training
Single run:
```bash
python trainning/train_psiphi.py \
  --train_data data/psiphi_datasets/psiphi_US_train.pt \
  --test_data data/psiphi_datasets/psiphi_US_test.pt \
  --save_dir outputs/run_us
```

Fine-tuning with fixed PV:
```bash
python trainning/train_psiphi.py \
  --mode ft_sf_fix_pv \
  --sf_path /path/to/sf_state_xxx.pth \
  --pv_path /path/to/pv_xxx.npy \
  --train_data data/psiphi_datasets/psiphi_US_train.pt \
  --test_data data/psiphi_datasets/psiphi_US_test.pt \
  --save_dir outputs/run_us_ft
```

Batch runs:
- Configure `trainning/batch_train.py` with your datasets and model paths, then run:
```bash
python trainning/batch_train.py
```

## Outputs
Training outputs are written to the chosen `--save_dir`, including:
- Model checkpoints
- Exported SF state dicts and PV vectors
- Metrics JSON logs

## Usage and Data Policy
- This repository does not distribute any datasets.
- You must have proper rights to use each dataset and follow its license terms.
- Replace all dataset/model paths with your own environment-specific paths.

## License and Open-Source Compliance
- Copyright (c) 2025 Authors. All rights reserved unless a license file is added.
- Third-party dependencies are governed by their respective licenses; review `requirements.txt` and comply accordingly.
- If you plan to redistribute or publish derived artifacts, ensure compliance with all dataset and dependency licenses.
