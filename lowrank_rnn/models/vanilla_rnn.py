from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

Nonlinearity = Literal["tanh", "relu"]


@dataclass
class VanillaRNNConfig:
    input_dim: int
    hidden_dim: int = 128
    output_dim: Optional[int] = None
    alpha: float = 0.2
    nonlinearity: Nonlinearity = "tanh"
    train_initial_state: bool = False
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class VanillaRNN(nn.Module):
    """Discrete-time leaky RNN with a fully trainable recurrent matrix.

    h_{t+1} = (1-alpha) h_t + alpha * phi(B x_t + J h_t)
    y_t    = R h_{t+1} + c
    """

    def __init__(self, cfg: VanillaRNNConfig):
        super().__init__()
        self.cfg = cfg
        out = cfg.output_dim if cfg.output_dim is not None else cfg.input_dim
        self.input = nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=True)
        self.recurrent = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.readout = nn.Linear(cfg.hidden_dim, out, bias=True)
        self.activation = torch.tanh if cfg.nonlinearity == "tanh" else torch.relu

        h0 = torch.zeros(cfg.hidden_dim, dtype=cfg.dtype)
        if cfg.train_initial_state:
            self.h0 = nn.Parameter(h0)
        else:
            self.register_buffer("h0", h0)

        self.to(device=torch.device(cfg.device), dtype=cfg.dtype)

    @property
    def J(self) -> torch.Tensor:
        return self.recurrent.weight

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, _ = x.shape
        h = self.h0[None, :].expand(B, -1) if h0 is None else h0
        alpha = self.cfg.alpha
        ys = []
        hs = [] if return_states else None
        for t in range(T):
            pre = self.input(x[:, t]) + self.recurrent(h)
            h_new = self.activation(pre)
            h = (1.0 - alpha) * h + alpha * h_new
            y = self.readout(h)
            ys.append(y[:, None, :])
            if return_states:
                hs.append(h[:, None, :])
        y_seq = torch.cat(ys, dim=1)
        h_seq = torch.cat(hs, dim=1) if return_states else None
        return y_seq, h_seq
