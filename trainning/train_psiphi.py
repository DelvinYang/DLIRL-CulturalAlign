# psiphi_train.py
import os
import shutil
import json
import random
import math
from pathlib import Path
from datetime import datetime

import numpy as np
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "psiphi_datasets"
DEFAULT_TRAIN_FILE = DEFAULT_DATA_DIR / "psiphi_US_train.pt"
DEFAULT_TEST_FILE = DEFAULT_DATA_DIR / "psiphi_US_test.pt"

# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Dataset
# =========================
class PsiPhiDataset(Dataset):
    def __init__(self, data_dict: dict):
        self.s = data_dict["s"]
        self.a = data_dict["a"]
        self.s_next = data_dict["s_next"]
        self.a_next = data_dict["a_next"]

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return {
            "s": self.s[idx],
            "a": self.a[idx],
            "s_next": self.s_next[idx],
            "a_next": self.a_next[idx],
        }

# =========================
# Model: SF + PV  (+ action recon head)
# =========================
class SF_PV_Model(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, feature_dim=64, action_dim=2, pv_fixed: torch.Tensor | None = None, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.rnn_0  = nn.RNN(input_size=input_dim, hidden_size=32, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.gru_3  = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 32), nn.ReLU(inplace=True),
        )
        fused_dim = hidden_dim + 32

        self.phi_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim)
        )
        self.psi_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim)
        )

        # NOTE: non-ASCII comment removed.
        self.act_head = nn.Sequential(
            nn.Linear(hidden_dim + 4 + action_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, action_dim)
        )

        # NOTE: This uses a single global w as the average preference over mixed strategies,
        # and does not correspond to the per-demonstrator w_k in the paper.
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
        hs = self.encode_state(s)  # [B,H]
        hs = hs.unsqueeze(1).expand(B, K, hs.shape[-1]).reshape(B*K, -1)
        ha = self.action_encoder(actions.reshape(B*K, self.action_dim))
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

    def pv_mode(self) -> str:
        return "fixed" if self._pv_fixed else "trainable"

    def pv_value(self) -> torch.Tensor:
        return self.w

# =========================
# Losses
# =========================
def forward_sa_with_hs(model: SF_PV_Model, hs: torch.Tensor, a: torch.Tensor):
    ha = model.action_encoder(a)
    z = torch.cat([hs, ha], dim=-1)
    phi = model.phi_head(z)
    psi = model.psi_head(z)
    q = (psi * model.w).sum(dim=-1)
    return phi, psi, q

def q_on_action_set_with_hs(model: SF_PV_Model, hs: torch.Tensor, actions: torch.Tensor):
    B, K, _ = actions.shape
    hs = hs.unsqueeze(1).expand(B, K, hs.shape[-1]).reshape(B*K, -1)
    ha = model.action_encoder(actions.reshape(B*K, model.action_dim))
    z = torch.cat([hs, ha], dim=-1)
    psi = model.psi_head(z)
    q = (psi * model.w).sum(dim=-1)
    return q.view(B, K)

def bc_q_loss_sampled_softmax_cached(model: SF_PV_Model, hs, a_pos, a_negs, temperature=1.0):
    if a_negs is None:
        actions = a_pos.unsqueeze(1)
    else:
        actions = torch.cat([a_pos.unsqueeze(1), a_negs], dim=1)  # [B,K,2]
    q_bk = q_on_action_set_with_hs(model, hs, actions) / max(1e-6, temperature)
    logp = q_bk - torch.logsumexp(q_bk, dim=1, keepdim=True)
    nll = -logp[:, 0].mean()
    l1 = model.l1_coef * model.pv_value().abs().sum()
    return nll + l1

def itd_loss_cached(model: SF_PV_Model, hs, a, hs_next, a_next, gamma=0.97, stop_grad_next=True):
    phi, psi, _ = forward_sa_with_hs(model, hs, a)
    if stop_grad_next:
        with torch.no_grad():
            psi_next = forward_sa_with_hs(model, hs_next, a_next)[1]
    else:
        psi_next = forward_sa_with_hs(model, hs_next, a_next)[1]
    td = psi - (phi + gamma * psi_next)
    return (td.pow(2).sum(dim=-1)).mean()

def acc_mse_loss_cached(model: SF_PV_Model, hs, curr_state, a_curr, a):
    hs_acc = hs.detach() + 0.3 * (hs - hs.detach())
    a_hat = model.act_head(torch.cat([hs_acc, curr_state, a_curr], dim=-1))
    return F.smooth_l1_loss(a_hat, a, beta=0.5)

@torch.no_grad()
def acc_mse_metric_cached(model: SF_PV_Model, hs, curr_state, a_curr, a):
    a_hat = model.act_head(torch.cat([hs.detach(), curr_state, a_curr], dim=-1))
    return F.smooth_l1_loss(a_hat, a, beta=0.5)

@torch.no_grad()
def sample_negative_actions_like(a_pos: torch.Tensor, K_minus_1=16, mode="uniform", uni_range=3.0, gauss_std=0.5):
    B, A = a_pos.shape
    if K_minus_1 <= 0: return None
    if mode == "uniform":
        negs = torch.empty(B, K_minus_1, A, device=a_pos.device).uniform_(-uni_range, uni_range)
    elif mode == "gauss":
        noise = torch.randn(B, K_minus_1, A, device=a_pos.device) * gauss_std
        negs = a_pos.unsqueeze(1) + noise
    else:
        raise ValueError("neg_mode must be 'uniform' or 'gauss'")
    return negs

# =========================
# NOTE: non-ASCII comment removed.
# =========================
def extract_sf_state_dict(model: SF_PV_Model) -> dict:
    """Export all parameters except PV(w) (i.e., SF + auxiliary heads)."""
    sd = model.state_dict()
    return {k: v for k, v in sd.items() if k != "w"}

def load_sf_state_dict(model: SF_PV_Model, sf_path: str):
    sf_sd = torch.load(sf_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sf_sd, strict=False)
    logger.info(f"Loaded SF from {sf_path} | missing={missing} | unexpected={unexpected}")

def load_pv_npy(pv_path: str, feature_dim: int, device="cpu") -> torch.Tensor:
    w = np.load(pv_path)
    w = np.asarray(w).reshape(-1)
    if w.shape[0] != feature_dim:
        raise ValueError(f"PV dim mismatch: file has {w.shape[0]}, expected {feature_dim}")
    return torch.tensor(w, dtype=torch.float32, device=device)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =========================
# Optimizer builder with param groups
# =========================
def build_optimizer(model: SF_PV_Model, base_lr: float,
                    lr_mult_sf: float = 1.0, lr_mult_pv: float = 1.0, lr_mult_act: float = 1.0,
                    freeze_sf: bool = False, freeze_pv: bool = False, train_act_head: bool = True):
    params_sf, params_pv, params_act = [], [], []
    for n, p in model.named_parameters():
        if n == "w":
            params_pv.append(p)
        elif n.startswith(("rnn_0", "lstm_1", "lstm_2", "gru_3", "action_encoder", "phi_head", "psi_head")):
            params_sf.append(p)
        elif n.startswith("act_head"):
            params_act.append(p)

    # NOTE: non-ASCII comment removed.
    for p in params_sf:
        p.requires_grad = not freeze_sf
    # NOTE: non-ASCII comment removed.
    for p in params_pv:
        p.requires_grad = not freeze_pv
    # NOTE: non-ASCII comment removed.
    for p in params_act:
        p.requires_grad = train_act_head

    param_groups = []
    if any(p.requires_grad for p in params_sf):
        param_groups.append({"params": [p for p in params_sf if p.requires_grad], "lr": base_lr * lr_mult_sf})
    if any(p.requires_grad for p in params_act):
        param_groups.append({"params": [p for p in params_act if p.requires_grad], "lr": base_lr * lr_mult_act})
    # NOTE: non-ASCII comment removed.
    if any(p.requires_grad for p in params_pv):
        param_groups.append({"params": [p for p in params_pv if p.requires_grad], "lr": base_lr * lr_mult_pv})

    opt = optim.AdamW(param_groups, lr=base_lr)
    logger.info(f"Optimizer groups -> SF:{len(params_sf)} / ACT:{len(params_act)} / PV:{len(params_pv)} "
                f"| lr(sf)={base_lr*lr_mult_sf:g} lr(act)={base_lr*lr_mult_act:g} lr(pv)={base_lr*lr_mult_pv:g} "
                f"| trainable_params={count_trainable_params(model)}")
    return opt

# =========================
# Train / Eval
# =========================
# NOTE: Default uses Gaussian negative sampling and a smaller acc_coef to avoid
# the regression head dominating policy learning.
def train_sf_pv(
    train_data: dict,
    epochs=200, batch_size=512, lr=2e-3, feature_dim=64,
    gamma=0.97, K_neg=12, neg_mode="gauss", neg_uni_range=3.0, neg_gauss_std=0.35, acc_coef=0.1,
    bc_coef=0.05, itd_coef=0.01,
    mode="train_all", sf_path=None, pv_path=None,
    freeze_sf=False, lr_mult_sf=1.0, lr_mult_pv=1.0, lr_mult_act=1.0, train_act_head=True,
    # NOTE: non-ASCII comment removed.
    train_eval_data=None, test_eval_data=None,
    eval_every=10, early_patience=50, stage2_extra_epochs=0, stage2_patience=3,
    # NOTE: non-ASCII comment removed.
    accum=1, warmup_epochs=0, cos_final_lr_ratio=0.05, early_delta=1e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_workers=0, pin_memory=False,
):
    if train_data["s"].is_cuda and int(num_workers) != 0:
        logger.warning("Train tensors are on CUDA; forcing num_workers=0 to avoid multiprocessing CUDA errors.")
        num_workers = 0
        pin_memory = False

    ds = PsiPhiDataset(train_data)
    # NOTE: non-ASCII comment removed.
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory) and str(device).startswith("cuda"),
    )

    # NOTE: non-ASCII comment removed.
    pv_fixed_tensor = None
    if mode == "ft_sf_fix_pv":
        if pv_path is None:
            raise ValueError("ft_sf_fix_pv requires --pv_path")
        pv_fixed_tensor = load_pv_npy(pv_path, feature_dim=feature_dim, device="cpu")

    model = SF_PV_Model(feature_dim=feature_dim, pv_fixed=pv_fixed_tensor, tau=1.0).to(device)

    # NOTE: non-ASCII comment removed.
    if sf_path is not None:
        load_sf_state_dict(model, sf_path)

    # NOTE: non-ASCII comment removed.
    opt = build_optimizer(model, base_lr=lr,
                          lr_mult_sf=lr_mult_sf, lr_mult_pv=lr_mult_pv, lr_mult_act=lr_mult_act,
                          freeze_sf=(mode != "train_all" and freeze_sf),
                          freeze_pv=False,
                          train_act_head=train_act_head)

    logger.info(f"Mode={mode} | pv_mode={model.pv_mode()} | freeze_sf={freeze_sf} | train_act_head={train_act_head}")

    use_amp = device.startswith("cuda")
    amp_device = "cuda" if use_amp else "cpu"
    scaler = GradScaler(amp_device, enabled=use_amp)

    # Warmup + Cosine lr
    def _lr_lambda(epoch: int) -> float:
        warm = int(warmup_epochs)
        if warm > 0 and epoch < warm:
            return float(epoch + 1) / float(max(1, warm))
        if epochs <= warm:
            return 1.0
        prog = float(epoch - warm) / float(max(1, epochs - warm))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return float(cos_final_lr_ratio) + (1.0 - float(cos_final_lr_ratio)) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    def _clone_state_dict(m: nn.Module) -> dict:
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # NOTE: non-ASCII comment removed.
    best_stage1_acc = None
    best_epoch = -1
    no_improve = 0
    best_overall_acc = None
    best_overall_epoch = -1
    best_overall_state = None

    # NOTE: non-ASCII comment removed.
    use_early_stop = (test_eval_data is not None) and (test_eval_data["s"].shape[0] > 1)
    if not use_early_stop:
        logger.warning("Test split is empty or not provided; early stopping disabled.")

    stage = 1
    stage1_done_triggered = False
    stage1_extra_remaining = 0
    stage2_best_acc = None
    stage2_best_epoch = -1
    stage2_no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        total_bc, total_itd, total_err, total_acc_metric = 0.0, 0.0, 0.0, 0.0
        num_batches = 0

        opt.zero_grad(set_to_none=True)
        total_steps = len(dl)
        for step, batch in enumerate(tqdm(dl, desc=f"Epoch {ep:03d}", leave=False)):
            s      = batch["s"]
            a      = batch["a"]
            s_next = batch["s_next"]
            a_next = batch["a_next"]

            if s.device.type != device:
                s = s.to(device)
            if a.device.type != device:
                a = a.to(device)
            if s_next.device.type != device:
                s_next = s_next.to(device)
            if a_next.device.type != device:
                a_next = a_next.to(device)

            # NOTE: BC-Q negatives are sampled around a_t to match the action distribution and strengthen contrast.
            a_negs = sample_negative_actions_like(
                a, K_minus_1=K_neg, mode=neg_mode,
                uni_range=neg_uni_range, gauss_std=neg_gauss_std
            )

            with autocast(amp_device, enabled=use_amp):
                hs = model.encode_state(s)
                hs_next = model.encode_state(s_next)

                # NOTE: BC-Q positives use (s_t, a_t) to preserve policy semantics.
                loss_bc  = bc_q_loss_sampled_softmax_cached(model, hs, a, a_negs, temperature=model.tau)
                loss_itd = itd_loss_cached(model, hs, a, hs_next, a_next, gamma=gamma, stop_grad_next=True)
                curr_state = s[:, -1, :4]
                loss_err = acc_mse_loss_cached(model, hs, curr_state, a, a_next)
                acc_metric = acc_mse_metric_cached(model, hs, curr_state, a, a_next)

                if stage == 1:
                    # NOTE: Three-stage training: representation+ranking, then structural consistency, then ACC-dominant.
                    if ep < 50:
                        acc_weight = acc_coef * 0.1
                        bc_weight = bc_coef
                        itd_weight = 0.0
                    elif ep < 100:
                        acc_weight = acc_coef * 0.5
                        bc_weight = bc_coef
                        itd_weight = itd_coef
                    else:
                        acc_weight = acc_coef * 2.0
                        bc_weight = bc_coef * 0.2
                        itd_weight = itd_coef * 0.2
                else:
                    # NOTE: Stage 2 trains only act_head and is ACC-dominant.
                    acc_weight = acc_coef * 2.0
                    bc_weight = 0.0
                    itd_weight = 0.0

                loss = acc_weight * loss_err + bc_weight * loss_bc + itd_weight * loss_itd
                loss = loss / max(1, int(accum))

            scaler.scale(loss).backward()
            is_step = ((step + 1) % max(1, int(accum)) == 0) or ((step + 1) == total_steps)
            if is_step:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            total_bc  += float(loss_bc.item())
            total_itd += float(loss_itd.item())
            total_err += float(loss_err.item())
            total_acc_metric += float(acc_metric.item())
            num_batches += 1

        # NOTE: non-ASCII comment removed.
        if num_batches == 0:
            logger.warning("No training batches this epoch (check --batch vs train size).")
        else:
            logger.info(
                f"[Epoch {ep:03d}] BC-Q: {total_bc/num_batches:.4f} | ITD: {total_itd/num_batches:.4f} "
                f"| ACC(aux): {total_err/num_batches:.4f} | ACC(err): {total_acc_metric/num_batches:.4f} "
                f"| pv_mode={model.pv_mode()}"
            )

        # NOTE: non-ASCII comment removed.
        if use_early_stop and (ep % max(1, int(eval_every)) == 0):
            eval_test = evaluate_on_split(model, test_eval_data, idx=None,
                                          batch_size=max(1024, batch_size),
                                          gamma=gamma, K_neg=K_neg, neg_mode=neg_mode,
                                          neg_uni_range=neg_uni_range, neg_gauss_std=neg_gauss_std)
            eval_train = evaluate_on_split(model, train_eval_data, idx=None,
                                           batch_size=max(1024, batch_size),
                                           gamma=gamma, K_neg=K_neg, neg_mode=neg_mode,
                                           neg_uni_range=neg_uni_range, neg_gauss_std=neg_gauss_std)
            logger.info(f"[EVAL][Ep {ep:03d}][Train] BC-Q={eval_train['bc_q']:.4f} | ITD={eval_train['itd']:.4f} | ACC(aux)={eval_train['acc']:.4f} | count={eval_train['count']}")
            logger.info(f"[EVAL][Ep {ep:03d}][Test ] BC-Q={eval_test['bc_q']:.4f} | ITD={eval_test['itd']:.4f} | ACC(aux)={eval_test['acc']:.4f} | count={eval_test['count']}")

            if stage == 1:
                # NOTE: Stage 1 uses acc as the only early-stop metric.
                curr_acc = eval_test["acc"]
                if (best_overall_acc is None) or (curr_acc < best_overall_acc - float(early_delta)):
                    best_overall_acc = curr_acc
                    best_overall_epoch = ep
                    best_overall_state = _clone_state_dict(model)
                if (best_stage1_acc is None) or (curr_acc < best_stage1_acc - float(early_delta)):
                    best_stage1_acc = curr_acc
                    best_epoch = ep
                    no_improve = 0
                else:
                    no_improve += 1

                if (not stage1_done_triggered) and (no_improve >= int(early_patience)):
                    stage1_done_triggered = True
                    stage1_extra_remaining = max(0, int(stage2_extra_epochs))
                    logger.info(
                        f"Stage 1 early stop triggered at epoch {ep}: "
                        f"(ACC plateau for {early_patience} evals, acc={curr_acc:.6f})."
                    )
                    if stage1_extra_remaining > 0:
                        logger.info(f"Stage 1 will run {stage1_extra_remaining} extra epochs before Stage 2.")
            else:
                # NOTE: Stage 2 uses acc as the only early-stop metric.
                curr_acc = eval_test["acc"]
                if (best_overall_acc is None) or (curr_acc < best_overall_acc - float(early_delta)):
                    best_overall_acc = curr_acc
                    best_overall_epoch = ep
                    best_overall_state = _clone_state_dict(model)
                if (stage2_best_acc is None) or (curr_acc < stage2_best_acc - float(early_delta)):
                    stage2_best_acc = curr_acc
                    stage2_best_epoch = ep
                    stage2_no_improve = 0
                else:
                    stage2_no_improve += 1
                    if stage2_no_improve >= int(stage2_patience):
                        logger.info(
                            f"Stage 2 early stop at epoch {ep}: test ACC did not improve for {stage2_patience} evals "
                            f"(best acc={stage2_best_acc:.6f} @ epoch {stage2_best_epoch})."
                        )
                        break

        scheduler.step()

        if stage == 1 and stage1_done_triggered:
            if stage1_extra_remaining > 0:
                stage1_extra_remaining -= 1
            if stage1_extra_remaining == 0:
                stage = 2
                logger.info("Switching to Stage 2: freeze SF/PV and train act_head only.")
                opt = build_optimizer(model, base_lr=lr,
                                      lr_mult_sf=lr_mult_sf, lr_mult_pv=lr_mult_pv, lr_mult_act=lr_mult_act,
                                      freeze_sf=True, freeze_pv=True, train_act_head=True)
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
                stage2_no_improve = 0
                stage2_best_acc = None
                stage2_best_epoch = -1

    return model, {
        "stage1_best_acc": best_stage1_acc,
        "stage1_best_epoch": best_epoch,
        "stage2_best_acc": stage2_best_acc,
        "stage2_best_epoch": stage2_best_epoch,
        "best_overall_acc": best_overall_acc,
        "best_overall_epoch": best_overall_epoch,
        "best_overall_state": best_overall_state,
    }

@torch.no_grad()
def evaluate_on_split(model: SF_PV_Model, data: dict, idx: torch.Tensor | None,
                      batch_size=1024, gamma=0.97, K_neg=32, neg_mode="uniform",
                      neg_uni_range=3.0, neg_gauss_std=0.5):
    device = next(model.parameters()).device
    if data is None:
        return {"bc_q": float("nan"), "itd": float("nan"), "acc": float("nan"), "count": 0}

    if idx is None:
        s_sub = data["s"].contiguous()
        a_sub = data["a"].contiguous()
        s_next_sub = data["s_next"].contiguous()
        a_next_sub = data["a_next"].contiguous()
    else:
        if idx.numel() <= 1:
            return {"bc_q": float("nan"), "itd": float("nan"), "acc": float("nan"), "count": 0}
        s_sub = data["s"][idx].contiguous()
        a_sub = data["a"][idx].contiguous()
        s_next_sub = data["s_next"][idx].contiguous()
        a_next_sub = data["a_next"][idx].contiguous()
    ds = PsiPhiDataset({
        "s": s_sub,
        "a": a_sub,
        "s_next": s_next_sub,
        "a_next": a_next_sub,
    })
    n = len(ds)
    if n == 0:
        return {"bc_q": float("nan"), "itd": float("nan"), "acc": float("nan"), "count": 0}

    total_bc, total_itd, total_acc, total_cnt = 0.0, 0.0, 0.0, 0
    for i in range(0, n, batch_size):
        batch = [ds[j] for j in range(i, min(i + batch_size, n))]
        s      = torch.stack([b["s"] for b in batch]).to(device)
        a      = torch.stack([b["a"] for b in batch]).to(device)
        s_next = torch.stack([b["s_next"] for b in batch]).to(device)
        a_next = torch.stack([b["a_next"] for b in batch]).to(device)

        # NOTE: BC-Q negatives are sampled around a_t to match the action distribution.
        a_negs = sample_negative_actions_like(
            a, K_minus_1=K_neg, mode=neg_mode,
            uni_range=neg_uni_range, gauss_std=neg_gauss_std
        )
        hs = model.encode_state(s)
        hs_next = model.encode_state(s_next)
        # NOTE: BC-Q positives use (s_t, a_t).
        bc  = bc_q_loss_sampled_softmax_cached(model, hs, a, a_negs, temperature=getattr(model, "tau", 1.0))
        itd = itd_loss_cached(model, hs, a, hs_next, a_next, gamma=gamma, stop_grad_next=True)
        # NOTE: acc is an auxiliary metric for (s_t -> a_{t+1}); evaluation should not focus on it.
        curr_state = s[:, -1, :4]
        acc = acc_mse_loss_cached(model, hs, curr_state, a, a_next)

        total_bc  += float(bc.item())  * s.size(0)
        total_itd += float(itd.item()) * s.size(0)
        total_acc += float(acc.item()) * s.size(0)
        total_cnt += s.size(0)

    return {
        "bc_q": total_bc/max(1,total_cnt),
        "itd":  total_itd/max(1,total_cnt),
        "acc":  total_acc/max(1,total_cnt),
        "count": total_cnt
    }

# =========================
# Main
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, default=str(DEFAULT_TRAIN_FILE),
                        help="Path to a psiphi_*_train.pt file")
    parser.add_argument("--test_data", type=str, default=str(DEFAULT_TEST_FILE),
                        help="Path to a psiphi_*_test.pt file")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--kneg", type=int, default=32)
    # NOTE: Default to Gaussian negatives to avoid large-range uniform noise hurting BC-Q contrast.
    parser.add_argument("--neg_mode", type=str, default="gauss", choices=["uniform","gauss"])
    parser.add_argument("--neg_uni_range", type=float, default=3.0, help="Uniform negative sampling range.")
    parser.add_argument("--neg_gauss_std", type=float, default=0.35, help="Gaussian negative sampling std.")
    # NOTE: Default to a smaller acc loss weight to avoid act_head dominating representation learning.
    parser.add_argument("--acc_coef", type=float, default=0.1, help="weight for acceleration MSE loss")
    parser.add_argument("--bc_coef", type=float, default=0.05, help="weight for BC-Q regularizer")
    parser.add_argument("--itd_coef", type=float, default=0.01, help="weight for ITD regularizer")

    # NOTE: non-ASCII comment removed.
    parser.add_argument("--mode", type=str, default="train_all",
                        choices=["train_all", "ft_sf_fix_pv"],
                        help="train_all: train SF+PV from scratch; ft_sf_fix_pv: load SF and fine-tune, PV from npy and fixed")
    parser.add_argument("--sf_path", type=str, default=None,
                        help="When mode=ft_sf_fix_pv, provide sf_state_xxx.pth trained on a large source dataset")
    parser.add_argument("--pv_path", type=str, default=None,
                        help="When mode=ft_sf_fix_pv, provide pv_xxx.npy trained on a large source dataset")

    # NOTE: non-ASCII comment removed.
    parser.add_argument("--freeze_sf", type=int, default=0, help="Freeze SF (1/0); usually 0 for ft_sf_fix_pv fine-tuning")
    parser.add_argument("--lr_mult_sf", type=float, default=0.1, help="LR multiplier for SF parameter group (smaller for fine-tuning)")
    parser.add_argument("--lr_mult_pv", type=float, default=1.0, help="LR multiplier for PV parameter group (only in train_all)")
    parser.add_argument("--lr_mult_act", type=float, default=1.0, help="LR multiplier for act_head parameter group")
    parser.add_argument("--train_act_head", type=int, default=1, help="Train act_head (1/0)")

    # NOTE: non-ASCII comment removed.
    parser.add_argument("--train_coeff", type=float, default=1.0,
                        help="Train set fraction (0~1). Subsample within train; test set remains fixed.")

    # NOTE: non-ASCII comment removed.
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate on test every N epochs")
    parser.add_argument("--early_patience", type=int, default=5, help="Early stop if test ACC does not improve for N evals")
    parser.add_argument("--stage2_extra_epochs", type=int, default=0,
                        help="Extra epochs after Stage1 early stop before Stage2")
    parser.add_argument("--stage2_patience", type=int, default=3,
                        help="Stage2 early-stop patience based on acc")
    parser.add_argument("--early_delta", type=float, default=1e-5, help="early stop improvement threshold")
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--warmup", type=int, default=0, help="LR warmup epochs.")
    parser.add_argument("--cos_final_lr_ratio", type=float, default=0.05, help="Cosine LR final/base ratio.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--pin_memory", type=int, default=1, help="Enable DataLoader pin_memory (1/0)")
    parser.add_argument("--cache_gpu", type=int, default=0,
                        help="1: move all tensors to GPU to avoid CPU->GPU copy; requires enough VRAM")

    parser.add_argument("--save_dir", type=str, default="./outputs_sf_pv")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # logging
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = Path(args.save_dir) / f"psiphi_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, encoding="utf-8", enqueue=True)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32-1)
    # If launched by torchrun, bind each process to its own GPU and offset seed
    local_rank_env = os.environ.get("LOCAL_RANK")
    rank_env = os.environ.get("RANK")
    if local_rank_env is not None and torch.cuda.is_available():
        try:
            local_rank = int(local_rank_env)
            torch.cuda.set_device(local_rank)
        except ValueError:
            local_rank = None
    else:
        local_rank = None
    if rank_env is not None:
        try:
            rank = int(rank_env)
            seed = (seed + rank) % (2**32 - 1)
        except ValueError:
            pass
    set_seed(seed)
    logger.info(f"Using seed: {seed}")
    logger.info(f"Logging to: {log_file}")

    # data
    train_data_dict = torch.load(args.train_data, map_location="cpu")
    test_data_dict = torch.load(args.test_data, map_location="cpu")
    s_train_cpu = train_data_dict["s"]
    a_train_cpu = train_data_dict["a"]
    s_next_train_cpu = train_data_dict["s_next"]
    a_next_train_cpu = train_data_dict["a_next"]
    s_test_cpu = test_data_dict["s"]
    a_test_cpu = test_data_dict["a"]
    s_next_test_cpu = test_data_dict["s_next"]
    a_next_test_cpu = test_data_dict["a_next"]
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if local_rank is not None else "cuda"
    else:
        device = "cpu"
    logger.info(f"Process using device: {device}")
    if device.startswith("cuda"):
        logger.info("Caching full tensors on GPU (no per-batch CPU->GPU copy).")
        s_train_all = s_train_cpu.to(device)
        a_train_all = a_train_cpu.to(device)
        s_next_train_all = s_next_train_cpu.to(device)
        a_next_train_all = a_next_train_cpu.to(device)
        s_test_all = s_test_cpu.to(device)
        a_test_all = a_test_cpu.to(device)
        s_next_test_all = s_next_test_cpu.to(device)
        a_next_test_all = a_next_test_cpu.to(device)
        if int(args.num_workers) != 0:
            logger.warning("GPU cache enabled: forcing num_workers=0")
            args.num_workers = 0
            args.pin_memory = 0
    else:
        if int(args.cache_gpu) == 1:
            logger.warning("cache_gpu=1 requested but CUDA not available; using CPU tensors.")
        s_train_all = s_train_cpu
        a_train_all = a_train_cpu
        s_next_train_all = s_next_train_cpu
        a_next_train_all = a_next_train_cpu
        s_test_all = s_test_cpu
        a_test_all = a_test_cpu
        s_next_test_all = s_next_test_cpu
        a_next_test_all = a_next_test_cpu

    N_train = s_train_all.size(0)
    N_test = s_test_all.size(0)
    logger.info(
        f"Loaded train tensors: s={tuple(s_train_all.shape)}, a={tuple(a_train_all.shape)}, "
        f"s_next={tuple(s_next_train_all.shape)}, a_next={tuple(a_next_train_all.shape)} (N={N_train})"
    )
    logger.info(
        f"Loaded test tensors: s={tuple(s_test_all.shape)}, a={tuple(a_test_all.shape)}, "
        f"s_next={tuple(s_next_test_all.shape)}, a_next={tuple(a_next_test_all.shape)} (N={N_test})"
    )

    gen = torch.Generator(device=s_train_all.device)
    gen.manual_seed(seed)
    perm = torch.randperm(N_train, generator=gen, device=s_train_all.device)
    train_coeff = float(args.train_coeff)
    if not (0.0 < train_coeff <= 1.0):
        raise ValueError("--train_coeff must be in (0, 1].")
    pool_size = max(1, int(N_train * train_coeff))
    pool_idx = perm[:pool_size]
    train_idx = pool_idx
    pool_pct = (pool_idx.numel() / N_train) * 100.0
    train_pct = (train_idx.numel() / N_train) * 100.0
    logger.info(
        f"Train subset -> pool={pool_idx.numel()} ({pool_pct:.2f}%) | "
        f"train={train_idx.numel()} ({train_pct:.2f}%) | "
        f"fixed_test={N_test}"
    )

    train_data = {
        "s": s_train_all[train_idx],
        "a": a_train_all[train_idx],
        "s_next": s_next_train_all[train_idx],
        "a_next": a_next_train_all[train_idx],
    }
    test_data = {
        "s": s_test_all,
        "a": a_test_all,
        "s_next": s_next_test_all,
        "a_next": a_next_test_all,
    }

    # NOTE: non-ASCII comment removed.
    mode = args.mode
    if mode == "ft_sf_fix_pv":
        if args.sf_path is None or args.pv_path is None:
            raise ValueError("ft_sf_fix_pv requires both --sf_path and --pv_path")

    # NOTE: non-ASCII comment removed.
    model, es_info = train_sf_pv(
        train_data=train_data,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        feature_dim=args.feature_dim, gamma=args.gamma,
        K_neg=args.kneg, neg_mode=args.neg_mode,
        neg_uni_range=args.neg_uni_range, neg_gauss_std=args.neg_gauss_std,
        acc_coef=args.acc_coef, bc_coef=args.bc_coef, itd_coef=args.itd_coef,
        mode=mode, sf_path=args.sf_path, pv_path=args.pv_path,
        freeze_sf=bool(args.freeze_sf),
        lr_mult_sf=args.lr_mult_sf, lr_mult_pv=args.lr_mult_pv, lr_mult_act=args.lr_mult_act,
        train_act_head=bool(args.train_act_head),
        train_eval_data=train_data, test_eval_data=test_data,
        eval_every=args.eval_every, early_patience=args.early_patience,
        stage2_extra_epochs=args.stage2_extra_epochs, stage2_patience=args.stage2_patience,
        accum=args.accum, warmup_epochs=args.warmup,
        cos_final_lr_ratio=args.cos_final_lr_ratio, early_delta=args.early_delta,
        num_workers=args.num_workers, pin_memory=bool(args.pin_memory),
    )

    # NOTE: non-ASCII comment removed.
    eval_train = evaluate_on_split(model, train_data, idx=None,
                                   batch_size=max(1024, args.batch),
                                   gamma=args.gamma, K_neg=args.kneg, neg_mode=args.neg_mode,
                                   neg_uni_range=args.neg_uni_range, neg_gauss_std=args.neg_gauss_std)
    eval_test  = evaluate_on_split(model, test_data, idx=None,
                                   batch_size=max(1024, args.batch),
                                   gamma=args.gamma, K_neg=args.kneg, neg_mode=args.neg_mode,
                                   neg_uni_range=args.neg_uni_range, neg_gauss_std=args.neg_gauss_std)

    logger.info(f"[FINAL][Train] BC-Q={eval_train['bc_q']:.4f} | ITD={eval_train['itd']:.4f} | ACC(aux)={eval_train['acc']:.4f} | count={eval_train['count']}")
    logger.info(f"[FINAL][Test ] BC-Q={eval_test['bc_q']:.4f} | ITD={eval_test['itd']:.4f} | ACC(aux)={eval_test['acc']:.4f} | count={eval_test['count']}")

    # NOTE: non-ASCII comment removed.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: non-ASCII comment removed.
    model_path = save_dir / f"sf_pv_model_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model weights to: {model_path}")

    best_state = es_info.get("best_overall_state")
    best_epoch = es_info.get("best_overall_epoch", -1)
    best_acc = es_info.get("best_overall_acc")
    if best_state is not None:
        best_model_path = save_dir / f"sf_pv_model_best_ep{best_epoch:03d}_{timestamp}.pt"
        torch.save(best_state, best_model_path)
        logger.info(
            f"Saved best model weights to: {best_model_path} "
            f"(best_acc={best_acc:.6f} @ epoch {best_epoch})"
        )
    else:
        logger.info("Best model not saved (no eval results available).")

    # NOTE: non-ASCII comment removed.
    sf_state = extract_sf_state_dict(model)
    sf_path_out = save_dir / f"sf_state_{timestamp}.pth"
    torch.save(sf_state, sf_path_out)
    logger.info(f"Saved SF state_dict to: {sf_path_out}")

    # 3) PV
    if mode == "ft_sf_fix_pv" and args.pv_path is not None:
        pv_src_path = Path(args.pv_path)
        pv_out_path = save_dir / pv_src_path.name
        shutil.copy2(pv_src_path, pv_out_path)
        w = np.load(pv_src_path)
        pv_txt = pv_out_path.with_suffix(".txt")
    else:
        w = model.pv_value().detach().cpu().numpy()
        pv_out_path = save_dir / f"pv_{timestamp}.npy"
        np.save(pv_out_path, w)
        pv_txt = save_dir / f"pv_{timestamp}.txt"
    with open(pv_txt, "w") as f:
        f.write("pv_mode=" + model.pv_mode() + "\n")
        f.write("pv=" + np.array2string(w, precision=6, floatmode="fixed") + "\n")
        f.write(f"[FINAL][Train] BC-Q={eval_train['bc_q']:.6f}, ITD={eval_train['itd']:.6f}, ACC(aux)={eval_train['acc']:.6f}, count={eval_train['count']}\n")
        f.write(f"[FINAL][Test ] BC-Q={eval_test['bc_q']:.6f}, ITD={eval_test['itd']:.6f}, ACC(aux)={eval_test['acc']:.6f}, count={eval_test['count']}\n")
    logger.info(f"Saved PV to: {pv_out_path}")
    logger.info(f"Saved PV and metrics to: {pv_txt}")

    # 4) metrics
    # NOTE: Stage1/Stage2 both use acc for early stopping.
    metrics = {
        "seed": seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "feature_dim": args.feature_dim,
        "gamma": args.gamma,
        "kneg": args.kneg,
        "neg_mode": args.neg_mode,
        "neg_uni_range": float(args.neg_uni_range),
        "neg_gauss_std": float(args.neg_gauss_std),
        "acc_coef": args.acc_coef,
        "bc_coef": args.bc_coef,
        "itd_coef": args.itd_coef,
        "mode": args.mode,
        "sf_path_used": args.sf_path,
        "pv_path_used": args.pv_path,
        "freeze_sf": bool(args.freeze_sf),
        "lr_mult_sf": args.lr_mult_sf,
        "lr_mult_pv": args.lr_mult_pv,
        "lr_mult_act": args.lr_mult_act,
        "train_act_head": bool(args.train_act_head),
        "train_coeff": float(args.train_coeff),
        "eval_every": int(args.eval_every),
        "early_patience": int(args.early_patience),
        "stage2_extra_epochs": int(args.stage2_extra_epochs),
        "stage2_patience": int(args.stage2_patience),
        "early_delta": float(args.early_delta),
        "accum": int(args.accum),
        "warmup_epochs": int(args.warmup),
        "cos_final_lr_ratio": float(args.cos_final_lr_ratio),
        "stage1_best_acc": None if es_info["stage1_best_acc"] is None else float(es_info["stage1_best_acc"]),
        "stage1_best_epoch": int(es_info["stage1_best_epoch"]),
        "stage2_best_acc": None if es_info["stage2_best_acc"] is None else float(es_info["stage2_best_acc"]),
        "stage2_best_epoch": int(es_info["stage2_best_epoch"]),
        "best_overall_acc": None if es_info["best_overall_acc"] is None else float(es_info["best_overall_acc"]),
        "best_overall_epoch": int(es_info["best_overall_epoch"]),
        "final_train": eval_train,
        "final_test": eval_test
    }
    metrics_path = save_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics JSON to: {metrics_path}")

    logger.info("Saved PV, SF-only state, full model, metrics. Done.")

if __name__ == "__main__":
    main()
