from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any
import math

import torch

GraphKind = Literal["curie_weiss", "lattice_2d", "block", "sk"]


@dataclass
class IsingConfig:
    n_spins: int = 64
    beta: float = 0.5
    field: float = 0.0
    graph_kind: GraphKind = "lattice_2d"
    coupling: float = 1.0
    n_blocks: int = 2
    j_in: float = 1.0
    j_out: float = 0.2
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


def _torch_device(cfg: IsingConfig) -> torch.device:
    return torch.device(cfg.device)


def make_couplings(cfg: IsingConfig) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return symmetric, zero-diagonal coupling matrix A and metadata dict."""
    n = cfg.n_spins
    device = _torch_device(cfg)
    dtype = cfg.dtype
    meta: Dict[str, Any] = {"graph_kind": cfg.graph_kind, "n_spins": n}

    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.seed)

    if cfg.graph_kind == "curie_weiss":
        A = (cfg.coupling / n) * torch.ones(n, n, dtype=dtype)
        A.fill_diagonal_(0.0)

    elif cfg.graph_kind == "lattice_2d":
        L = int(round(math.sqrt(n)))
        if L * L != n:
            raise ValueError(
                f"lattice_2d requires perfect-square n_spins, got n={n}"
            )
        A = torch.zeros(n, n, dtype=dtype)
        for r in range(L):
            for c in range(L):
                i = r * L + c
                rn = ((r + 1) % L) * L + c
                cn = r * L + ((c + 1) % L)
                A[i, rn] = cfg.coupling
                A[rn, i] = cfg.coupling
                A[i, cn] = cfg.coupling
                A[cn, i] = cfg.coupling
        A.fill_diagonal_(0.0)
        meta["lattice_shape"] = (L, L)

    elif cfg.graph_kind == "block":
        nb = max(1, cfg.n_blocks)
        block_labels = torch.arange(n) % nb
        same = block_labels[:, None] == block_labels[None, :]
        A = torch.where(
            same,
            torch.full((n, n), cfg.j_in / n, dtype=dtype),
            torch.full((n, n), cfg.j_out / n, dtype=dtype),
        )
        A.fill_diagonal_(0.0)
        meta["block_labels"] = block_labels.tolist()
        meta["n_blocks"] = nb

    elif cfg.graph_kind == "sk":
        G = torch.randn(n, n, generator=g) * (cfg.coupling / math.sqrt(float(n)))
        A = ((G + G.T) / 2.0).to(dtype=dtype)
        A.fill_diagonal_(0.0)

    else:
        raise ValueError(f"unknown graph_kind: {cfg.graph_kind}")

    A = A.to(device=device, dtype=dtype)
    return A, meta


def sample_initial_spins(
    batch_size: int,
    n_spins: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return spins in {-1, +1} of shape (batch_size, n_spins)."""
    device = torch.device(device) if isinstance(device, str) else device
    if generator is None:
        bits = torch.randint(0, 2, (batch_size, n_spins), device=device)
    else:
        bits = torch.randint(
            0, 2, (batch_size, n_spins), device=device, generator=generator
        )
    return (2 * bits - 1).to(dtype=dtype)


def synchronous_glauber_step(
    s: torch.Tensor,
    A: torch.Tensor,
    beta: float,
    h: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """One synchronous Glauber update of spins.

    Args:
        s: (batch, n_spins) tensor in {-1, +1}.
        A: (n_spins, n_spins) symmetric coupling matrix.
        beta: inverse temperature.
        h: optional external field broadcastable to s.
    """
    field = s @ A.T
    if h is not None:
        field = field + h
    p_plus = torch.sigmoid(2.0 * beta * field)
    if generator is None:
        u = torch.rand_like(p_plus)
    else:
        u = torch.rand(p_plus.shape, device=p_plus.device, dtype=p_plus.dtype, generator=generator)
    return torch.where(u < p_plus, torch.ones_like(s), -torch.ones_like(s))


def sample_ising_batch(
    cfg: IsingConfig,
    batch_size: int,
    seq_len: int,
    burn_in: int = 100,
    A: Optional[torch.Tensor] = None,
    meta: Optional[Dict[str, Any]] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Sample (states, A, meta) where states has shape (batch_size, seq_len+1, n_spins)."""
    device = _torch_device(cfg)
    dtype = cfg.dtype

    if A is None or meta is None:
        A, meta = make_couplings(cfg)

    h_field = None
    if cfg.field != 0.0:
        h_field = torch.full((cfg.n_spins,), cfg.field, device=device, dtype=dtype)

    s = sample_initial_spins(
        batch_size, cfg.n_spins, device=device, dtype=dtype, generator=generator
    )
    for _ in range(burn_in):
        s = synchronous_glauber_step(s, A, cfg.beta, h=h_field, generator=generator)

    states = torch.empty(batch_size, seq_len + 1, cfg.n_spins, device=device, dtype=dtype)
    states[:, 0] = s
    for t in range(seq_len):
        s = synchronous_glauber_step(s, A, cfg.beta, h=h_field, generator=generator)
        states[:, t + 1] = s

    return states, A, meta
