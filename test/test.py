"""Metric utilities used throughout the project."""

from __future__ import annotations

from typing import Sequence
import time

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SF_PV_Model(nn.Module):
    def __init__(
        self,
        input_dim=12,
        hidden_dim=64,
        feature_dim=64,
        action_dim=2,
        pv_fixed: torch.Tensor | None = None,
        tau: float = 1.0,
    ):
        super().__init__()
        self.tau = tau
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.rnn_0 = nn.RNN(input_size=input_dim, hidden_size=32, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.gru_3 = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        fused_dim = hidden_dim + 32

        self.phi_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
        )
        self.psi_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
        )

        self.act_head = nn.Sequential(
            nn.Linear(hidden_dim + 4 + action_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

        if pv_fixed is None:
            self.w = nn.Parameter(torch.zeros(feature_dim))
            nn.init.normal_(self.w, mean=0.0, std=0.2)
            self._pv_fixed = False
        else:
            pv_fixed = pv_fixed.flatten()
            assert pv_fixed.numel() == feature_dim
            self.register_buffer("w", pv_fixed.detach().clone())
            self._pv_fixed = True

        self.l1_coef = 1e-4

    def encode_state(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn_0(x)
        h, _ = self.lstm_1(h)
        h, _ = self.lstm_2(h)
        h, _ = self.gru_3(h)
        return h[:, -1, :]

    def forward_sa(self, s: torch.Tensor, a: torch.Tensor):
        hs = self.encode_state(s)
        ha = self.action_encoder(a)
        z = torch.cat([hs, ha], dim=-1)
        phi = self.phi_head(z)
        psi = self.psi_head(z)
        q = (psi * self.w).sum(dim=-1)
        return phi, psi, q

    def q_on_action_set(self, s: torch.Tensor, actions: torch.Tensor):
        B, K, _ = actions.shape
        hs = self.encode_state(s)
        hs = hs.unsqueeze(1).expand(B, K, hs.shape[-1]).reshape(B * K, -1)
        ha = self.action_encoder(actions.reshape(B * K, self.action_dim))
        z = torch.cat([hs, ha], dim=-1)
        psi = self.psi_head(z)
        q = (psi * self.w).sum(dim=-1)
        return q.view(B, K)

    @torch.no_grad()
    def predict_action(self, s: torch.Tensor, a_curr: torch.Tensor | None = None) -> torch.Tensor:
        hs = self.encode_state(s)
        curr_state = s[:, -1, :4]
        if a_curr is None:
            a_curr = torch.zeros(s.size(0), self.action_dim, device=s.device, dtype=s.dtype)
        return self.act_head(torch.cat([hs, curr_state, a_curr], dim=-1))


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the per-dimension mean absolute error."""

    return torch.mean(torch.abs(pred - target), dim=0)


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the per-dimension root mean squared error."""

    return torch.sqrt(torch.mean((pred - target) ** 2, dim=0))


def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the per-dimension coefficient of determination (R²)."""

    target_mean = torch.mean(target, dim=0)
    ss_tot = torch.sum((target - target_mean) ** 2, dim=0)
    ss_res = torch.sum((target - pred) ** 2, dim=0)

    safe_ss_tot = torch.where(ss_tot > 0, ss_tot, torch.ones_like(ss_tot))
    r2 = 1.0 - ss_res / safe_ss_tot

    zero_var_mask = ss_tot <= 0
    perfect_mask = zero_var_mask & (ss_res <= 0)
    r2 = torch.where(zero_var_mask, torch.zeros_like(r2), r2)
    r2 = torch.where(perfect_mask, torch.ones_like(r2), r2)
    return r2


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean distances between ``x`` and ``y``."""

    x_exp = x.unsqueeze(1)
    y_exp = y.unsqueeze(0)
    return torch.sum((x_exp - y_exp) ** 2, dim=-1)


def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    bandwidths: Sequence[float],
) -> torch.Tensor:
    """Multi-scale Gaussian kernel used for the MMD computation."""

    x = x.to(torch.float64)
    y = y.to(torch.float64)
    dists = pairwise_sq_dists(x, y)
    kernel = torch.zeros_like(dists, dtype=torch.float64)
    for bw in bandwidths:
        gamma = 1.0 / (2.0 * (bw ** 2))
        kernel += torch.exp(-gamma * dists)
    return kernel / float(len(bandwidths))


def _rbf_kernel_values(
    x: torch.Tensor, y: torch.Tensor, bandwidths: Sequence[float]
) -> torch.Tensor:
    """Return the mean RBF kernel value for each pair in ``(x, y)``."""

    x = x.to(torch.float64)
    y = y.to(torch.float64)
    dists = torch.sum((x - y) ** 2, dim=-1)
    kernel_vals = torch.zeros_like(dists, dtype=torch.float64)
    for bw in bandwidths:
        gamma = 1.0 / (2.0 * (bw ** 2))
        kernel_vals += torch.exp(-gamma * dists)
    return kernel_vals / float(len(bandwidths))


def compute_rbf_mmd(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    bandwidths: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    exact_threshold: int = 4096,
    fallback_sample: int = 2048,
    eps: float = 1e-12,
) -> float:

    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")

    num_samples = pred.shape[0]
    if num_samples == 0:
        raise ValueError("pred and target must contain at least one sample")

    pred = pred.to(torch.float64)
    target = target.to(torch.float64)

    if num_samples <= exact_threshold:
        if num_samples < 2:
            raise ValueError("Need at least two samples to estimate MMD")
        k_xx = gaussian_kernel(pred, pred, bandwidths=bandwidths)
        k_yy = gaussian_kernel(target, target, bandwidths=bandwidths)
        k_xy = gaussian_kernel(pred, target, bandwidths=bandwidths)

        # Unbiased U-statistic estimator (exclude diagonals)
        n = num_samples
        mmd_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
        mmd_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1))
        mmd_xy = k_xy.mean()
        mmd = mmd_xx + mmd_yy - 2.0 * mmd_xy
        if mmd < -eps:
            # Fallback to biased estimator which is non-negative in theory
            mmd_biased = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
            mmd = mmd_biased
        return float(max(mmd.item(), 0.0))

    if num_samples < 2:
        raise ValueError("Need at least two samples to estimate MMD")

    if num_samples % 2 == 1:
        pred = pred[:-1]
        target = target[:-1]
        num_samples -= 1

    x_a = pred[0::2]
    x_b = pred[1::2]
    y_a = target[0::2]
    y_b = target[1::2]

    k_xx = _rbf_kernel_values(x_a, x_b, bandwidths)
    k_yy = _rbf_kernel_values(y_a, y_b, bandwidths)
    k_xy = _rbf_kernel_values(x_a, y_b, bandwidths)
    k_yx = _rbf_kernel_values(x_b, y_a, bandwidths)

    mmd = 2.0 * (k_xx + k_yy - k_xy - k_yx).mean()

    if mmd < -eps:
        # Fallback: compute biased estimator on a capped subset to stabilise
        subset = min(num_samples, fallback_sample)
        idx = torch.randperm(num_samples, device=pred.device)[:subset]
        pred_sub = pred[idx]
        target_sub = target[idx]
        k_xx_b = gaussian_kernel(pred_sub, pred_sub, bandwidths=bandwidths)
        k_yy_b = gaussian_kernel(target_sub, target_sub, bandwidths=bandwidths)
        k_xy_b = gaussian_kernel(pred_sub, target_sub, bandwidths=bandwidths)
        mmd_biased = k_xx_b.mean() + k_yy_b.mean() - 2.0 * k_xy_b.mean()
        mmd = mmd_biased

    return float(max(mmd.item(), 0.0))


def _extract_state_dict(ckpt: object) -> dict:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise ValueError("Unsupported checkpoint format; expected a state dict.")


def _infer_model_kwargs(state_dict: dict) -> dict:
    rnn_weight = state_dict["rnn_0.weight_ih_l0"]
    input_dim = rnn_weight.shape[1]
    lstm_weight = state_dict["lstm_1.weight_ih_l0"]
    hidden_dim = lstm_weight.shape[0] // 4
    feature_dim = state_dict["phi_head.2.weight"].shape[0]
    action_dim = state_dict["action_encoder.0.weight"].shape[1]
    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "feature_dim": feature_dim,
        "action_dim": action_dim,
    }

def _summary_stats(x: torch.Tensor) -> dict:
    x = x.flatten()
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "median": float(x.median().item()),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "model_sample/sf_pv_model_best_ep060_20251226_144627.pt"
    data_path = "data/psiphi_datasets/psiphi_Germany_test.pt"

    print("== Runtime Info ==")
    print(f"device: {device}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    print(f"checkpoint: {ckpt_path}")
    print(f"data: {data_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    model_kwargs = _infer_model_kwargs(state_dict)
    print(f"model kwargs: {model_kwargs}")

    model = SF_PV_Model(**model_kwargs).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    data = torch.load(data_path, map_location="cpu")
    s = data["s"].to(device)
    a = data["a"].to(device)
    a_next = data["a_next"].to(device)
    print(f"s shape: {tuple(s.shape)} dtype={s.dtype}")
    print(f"a shape: {tuple(a.shape)} dtype={a.dtype}")
    print(f"a_next shape: {tuple(a_next.shape)} dtype={a_next.dtype}")
    print(f"num samples: {s.shape[0]}")

    t4 = time.perf_counter()
    with torch.no_grad():
        pred = model.predict_action(s, a)
    t5 = time.perf_counter()
    print(f"inference time: {t5 - t4:.3f}s")

    pred_cpu = pred.detach().cpu()
    target_cpu = a_next.detach().cpu()
    error = pred_cpu - target_cpu

    r2 = compute_r2(pred_cpu, target_cpu)
    r2_mean = r2.mean().item()
    rbf_mmd = compute_rbf_mmd(pred_cpu, target_cpu)

    print(f"R2 mean: {r2_mean:.6f}")
    print(f"RBF-MMD: {rbf_mmd:.6f}")

    # Distribution + stats for ax (assume action dim 0)
    ax_true = target_cpu[:, 0]
    ax_pred = pred_cpu[:, 0]

    ax_true_stats = _summary_stats(ax_true)
    ax_pred_stats = _summary_stats(ax_pred)
    median_err = abs(ax_pred_stats["median"] - ax_true_stats["median"])

    print("ax stats (true): min={min:.6f} max={max:.6f} median={median:.6f}".format(**ax_true_stats))
    print("ax stats (pred): min={min:.6f} max={max:.6f} median={median:.6f}".format(**ax_pred_stats))
    print(f"ax median error: {median_err:.6f}")

    # Distribution + stats for ay (assume action dim 1)
    ay_true = target_cpu[:, 1]
    ay_pred = pred_cpu[:, 1]

    ay_true_stats = _summary_stats(ay_true)
    ay_pred_stats = _summary_stats(ay_pred)
    ay_median_err = abs(ay_pred_stats["median"] - ay_true_stats["median"])

    print("ay stats (true): min={min:.6f} max={max:.6f} median={median:.6f}".format(**ay_true_stats))
    print("ay stats (pred): min={min:.6f} max={max:.6f} median={median:.6f}".format(**ay_pred_stats))
    print(f"ay median error: {ay_median_err:.6f}")

    # RMSE for acceleration error (per-dim + mean)
    rmse_per_dim = compute_rmse(pred_cpu, target_cpu)
    rmse_mean = float(rmse_per_dim.mean().item())
    print(f"acc RMSE mean: {rmse_mean:.6f}")



if __name__ == "__main__":
    main()
