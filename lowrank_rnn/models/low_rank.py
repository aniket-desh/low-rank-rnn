# models/lowrank.py

from dataclasses import dataclass
import torch

@dataclass
class LowRankParams:
    g: float          # random bulk gain
    b: float          # balance parameter
    m: float          # low-rank strength
    seed: int | None = None

def sample_W(N: int, rng: torch.Generator) -> torch.Tensor:
    """real Ginibre bulk scaled by 1/sqrt(N)."""
    scale = 1.0 / torch.sqrt(torch.tensor(N, dtype=torch.float32))
    return torch.randn(N, N, generator=rng) * scale

def unit_vec_orth_to_ones(N: int, rng: torch.Generator) -> torch.Tensor:
    """unit vector orthogonal to 1/sqrt(N)."""
    v = torch.randn(N, generator=rng)
    v -= v.mean()
    v /= v.norm() + 1e-12
    return v

def build_rank1_uv(N: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """return u, v with u,v ⟂ 1 and v^T u = 1."""
    u = unit_vec_orth_to_ones(N, rng)
    v = unit_vec_orth_to_ones(N, rng)
    # normalize so v^T u = 1
    dot_uv = v.dot(u)
    v = v / (dot_uv + 1e-12)
    return u, v

def build_lowrank_J(N: int, params: LowRankParams) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    build J = g W - (b/N) 11^T + m u v^T and return (J, u, v).
    this is the 'ground-truth' low-rank structure for synthetic experiments.
    """
    rng = torch.Generator()
    rng.manual_seed(params.seed)
    W = sample_W(N, rng)
    u, v = build_rank1_uv(N, rng)
    J = params.g * W - (params.b / N) * torch.ones((N, N)) + params.m * torch.outer(u, v)
    return J, u, v

def subtract_balance(J: torch.Tensor, b: float) -> torch.Tensor:
    """remove the -b/N 11^T term if needed."""
    N = J.shape[0]
    return J + (b / N) * torch.ones((N, N))

def leading_mode(J: torch.Tensor, k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    svd-based leading singular directions of J.
    return (u1, v1) for rank-1 approximation J ≈ s1 u1 v1^T.
    """
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    u1 = U[:, 0]
    v1 = Vh[0, :]
    return u1, v1

def effective_m(J: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> float:
    """
    compute effective low-rank strength m_eff for a given (u,v):
    m_eff = v^T J u / (v^T u)
    (this is the rank-1 component along (u,v) in least-squares sense).
    """
    num = v.dot(J @ u)
    den = v.dot(u) + 1e-12
    return num / den

def largest_real_eig(A: torch.Tensor) -> complex:
    """Return eigenvalue with largest real part."""
    vals = torch.linalg.eigvals(A)
    idx = torch.argmax(vals.real)
    return vals[idx]

def jacobian_from_mask(J: torch.Tensor, Dbar: torch.Tensor) -> torch.Tensor:
    """
    A_avg = -I + J diag(Dbar) for ReLU networks,
    where Dbar is average phi'(h) mask.
    """
    N = J.shape[0]
    return -torch.eye(N, device=J.device, dtype=J.dtype) + J @ torch.diag(Dbar)


