from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

@dataclass
class LowRankRNNConfig:
    N: int = 800
    g: float = 2.0
    b: float = 10.0
    m: float = 1.5
    tau: float = 1.0
    dt: float = 1e-3
    nonlinearity: str = "relu"
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    train_u: bool = False
    train_v: bool = False
    train_readout: bool = True
    out_dim: int = 1

class LowRankRNN(nn.Module):
    """ rank-1 low-rank RNN:
    dh/dt = (-h + J *phi(h) + I(t)) / tau
    J = gW - (b/N) 11^T + m u v^T
    W, real gaussian i.i.d. noise
    u, v: real unit vectors
    this class is designed to
    - share structure with dmft step, 
    - be trainable, 
    - simulate dynamics given I(t)
    """

    def __init__(self, config: LowRankRNNConfig):
        super().__init__()
        self.cfg = config
        device = torch.device(config.device)
        dtype = config.dtype
        N = config.N

        # random bulk connectivity
        W = torch.randn(N, N, device=device, dtype=dtype) / torch.sqrt(torch.tensor(N, dtype=dtype, device=device))
        self.register_buffer("W", W)

        # ones vector and low-rank vectors
        ones = torch.ones(N, device=device, dtype=dtype)
        self.register_buffer("ones", ones)

        u = torch.randn(N, device=device, dtype=dtype)
        v = torch.randn(N, device=device, dtype=dtype)

        u = u - u.mean()
        v = v - v.mean()
        u = u / (u.norm() + 1e-12)
        v = v / (v.norm() + 1e-12)

        if config.train_u:
            self.u = nn.Parameter(u)
        else:
            self.register_buffer("u", u)

        if config.train_v:
            self.v = nn.Parameter(v)
        else:
            self.register_buffer("v", v)

        # readout
        self.readout = nn.Linear(N, config.out_dim, bias=True)
        self.readout.weight.requires_grad = config.train_readout
        self.readout.bias.requires_grad = config.train_readout

        # nonlinearity
        if config.nonlinearity == "relu":
            self.phi = nn.ReLU()
        elif config.nonlinearity == "tanh":
            self.phi = nn.Tanh()
        else:
            raise ValueError(f"unknown nonlinearity: {config.nonlinearity}")

    @property
    def N(self) -> int:
        return self.cfg.N
        
    def _build_J(self) -> torch.Tensor:
        N = self.cfg.N
        g, b, m = self.cfg.g, self.cfg.b, self.cfg.m
        J = g * self.W
        J = J - (b / N) * torch.outer(self.ones, self.ones)
        J = J + m * torch.outer(self.u, self.v)
        return J
        
    def forward(self, I_t: torch.Tensor, 
    h0: Optional[torch.Tensor] = None, 
    return_states: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """simulate RNN given input time series I_t.
        parameters:
        - I_t: (T, ) or (T, 1) or (T, N) tensor
        - h0: (N,) tensor or None
        - return_states: bool (True, return full hidden state traj, False, return only outputs (T, out_dim))
        returns y_t (T, out_dim) and h_t (T, N) if return_states is True,
        """
        device = self.W.device
        dtype = self.W.dtype

        if I_t.dim() == 1:
            I_t = I_t[:, None] # (T, 1)
        if I_t.size(1) == 1:
            I_t = I_t * torch.ones(self.N, device=device, dtype=dtype)[None, :]
        elif I_t.size(1) != self.N:
            raise ValueError(f"I_t must be (T, ) or (T, 1) or (T, {self.N})")

        I_t = I_t.to(device=device, dtype=dtype)
        T = I_t.size(0)
        dt, tau = self.cfg.dt, self.cfg.tau

        if h0 is None:
            h = 0.1 * torch.randn(self.N, device=device, dtype=dtype)
        else:
            h = h0.to(device=device, dtype=dtype)
        
        J = self._build_J()

        states = []
        outputs = []

        for t in range(T):
            x = self.phi(h)
            # euler step
            dh = (-h + J @ x + I_t[t]) * (dt / tau)
            h = h + dh

            if return_states:
                states.append(h.unsqueeze(0))
            
            y = self.readout(h.unsqueeze(0)) # (1, out_dim)
            outputs.append(y)
        
        y_t = torch.cat(outputs, dim=0)
        h_t = torch.cat(states, dim=0) if return_states else None
        return y_t, h_t
