import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class DeepTemporalModel(nn.Module):  # input [B, 12, 12], output [B, 2]
    def __init__(self, input_dim=12, hidden_dim=64, basis_dim=12, out_dim=2,
                 fixed_pv: torch.Tensor | None = None):
        """
        fixed_pv=None   -> pv is a trainable parameter
        fixed_pv=Tensor -> pv is a fixed buffer, shape [12] or [1,12]
        """
        super().__init__()
        self.out_dim = out_dim
        self.basis_dim = basis_dim

        # Temporal backbone
        self.rnn_0  = nn.RNN(input_size=input_dim, hidden_size=32, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.gru_3  = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # Basis expansion to [B, out_dim*basis_dim]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Linear(32, 16),
            nn.Linear(16, out_dim * basis_dim)
        )

        # cv is always trainable
        self.cv = nn.Parameter(torch.ones(1, basis_dim))  # [1, 12]

        # pv is set at init and does not switch later
        if fixed_pv is None:
            self.pv = nn.Parameter(torch.ones(1, basis_dim))  # trainable
            self._pv_is_fixed = False
        else:
            if fixed_pv.dim() == 1:
                fixed_pv = fixed_pv.unsqueeze(0)
            assert fixed_pv.shape == (1, basis_dim), f"fixed_pv must be [1,{basis_dim}] or [{basis_dim}]"
            self.register_buffer("pv", fixed_pv.detach().clone())  # fixed
            self._pv_is_fixed = True

    def forward(self, x):  # x: [B, 12, 12]
        x, _ = self.rnn_0(x)
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        x, _ = self.gru_3(x)

        h_last = x[:, -1, :]                                   # [B, 64]
        y = self.mlp(h_last).view(-1, self.out_dim, self.basis_dim)  # [B, 2, 12]

        scale = (self.cv * self.pv).unsqueeze(1)               # [1,1,12] -> broadcast to [B,2,12]
        y = y * scale
        return y.sum(dim=-1)                                   # [B, 2]

    # Read-only helpers (kept for logging/printing).
    def pv_mode(self) -> str:
        return "fixed" if self._pv_is_fixed else "trainable"

    def pv_value(self) -> torch.Tensor:
        """Return current pv (tensor, moves with device)."""
        return self.pv
