from __future__ import annotations
from typing import Any, Dict, Optional

import torch


def _flatten_states(states: torch.Tensor) -> torch.Tensor:
    if states.dim() == 3:
        return states.reshape(-1, states.shape[-1])
    if states.dim() == 2:
        return states
    raise ValueError(f"expected 2D or 3D states, got {states.shape}")


def centered_cov(states: torch.Tensor) -> torch.Tensor:
    """Sample covariance (n, n) over the flattened batch/time dimension."""
    X = _flatten_states(states)
    X = X - X.mean(dim=0, keepdim=True)
    denom = max(X.shape[0] - 1, 1)
    return X.T @ X / denom


def lagged_cov(states: torch.Tensor, lag: int = 1) -> torch.Tensor:
    """Time-lagged covariance C_lag = E[(s_t - mu)(s_{t+lag} - mu)^T]."""
    if states.dim() != 3:
        raise ValueError(f"lagged_cov needs (batch, time, n), got {states.shape}")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    n = states.shape[-1]
    X = states[:, :-lag, :].reshape(-1, n)
    Y = states[:, lag:, :].reshape(-1, n)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    denom = max(X.shape[0] - 1, 1)
    return X.T @ Y / denom


def effective_rank(M: torch.Tensor) -> float:
    """Participation effective rank: (sum sigma)^2 / sum(sigma^2)."""
    s = torch.linalg.svdvals(M.detach().to(torch.float32))
    return float((s.sum() ** 2 / (s.pow(2).sum() + 1e-12)).item())


def energy_rank(M: torch.Tensor, eps: float = 0.05) -> int:
    """Smallest k such that top-k singular values capture (1-eps) of energy."""
    s2 = torch.linalg.svdvals(M.detach().to(torch.float32)).pow(2)
    frac = torch.cumsum(s2, dim=0) / (s2.sum() + 1e-12)
    idx = (frac >= 1.0 - eps).nonzero()
    if idx.numel() == 0:
        return int(s2.numel())
    return int(idx[0].item()) + 1


def _top_left_singular(M: torch.Tensor, k: int) -> torch.Tensor:
    M32 = M.detach().to(torch.float32)
    k = max(1, min(k, min(M32.shape)))
    U, _, _ = torch.linalg.svd(M32, full_matrices=False)
    return U[:, :k]


def subspace_alignment(M: torch.Tensor, N: torch.Tensor, k: int = 5) -> float:
    """1/k * || U_k(M)^T U_k(N) ||_F^2; in [0, 1]."""
    if M.shape[0] != N.shape[0]:
        raise ValueError(
            f"row dimensions must match for subspace alignment, got {M.shape} and {N.shape}"
        )
    UM = _top_left_singular(M, k)
    UN = _top_left_singular(N, k)
    return float((torch.linalg.norm(UM.T @ UN, ord="fro") ** 2 / k).item())


def henrici_departure(J: torch.Tensor) -> float:
    """||J^T J - J J^T||_F / ||J||_F^2; zero iff J is normal."""
    J32 = J.detach().to(torch.float32)
    num = torch.linalg.norm(J32.T @ J32 - J32 @ J32.T, ord="fro")
    den = torch.linalg.norm(J32, ord="fro").pow(2) + 1e-12
    return float((num / den).item())


def effective_spin_operator(model) -> torch.Tensor:
    """G_spin = R J B, mapping input spins to readout spins through the recurrent matrix."""
    B = model.input.weight        # (H, n)
    J = model.recurrent.weight    # (H, H)
    R = model.readout.weight      # (n, H)
    return R @ J @ B              # (n, n)


def _block_mean_pair(J: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    same_mask = labels[:, None] == labels[None, :]
    diag = torch.eye(J.shape[0], device=J.device, dtype=torch.bool)
    same_mask = same_mask & ~diag
    diff_mask = ~same_mask & ~diag
    same_mean = float(J[same_mask].mean().item()) if same_mask.any() else 0.0
    diff_mean = float(J[diff_mask].mean().item()) if diff_mask.any() else 0.0
    return same_mean, diff_mean


def geometry_report(
    model,
    J0: torch.Tensor,
    A: torch.Tensor,
    states: torch.Tensor,
    meta: Optional[Dict[str, Any]] = None,
    k: int = 5,
    lag: int = 1,
) -> Dict[str, Any]:
    """Compute the full diagnostic dict described in docs/experiment_spin_rnn.md."""
    meta = meta or {}
    J = model.J.detach()
    dJ = J - J0
    G = effective_spin_operator(model).detach()

    C = centered_cov(states.detach())
    if states.shape[1] > lag:
        C_lag = lagged_cov(states.detach(), lag=lag)
    else:
        C_lag = torch.zeros_like(C)

    out: Dict[str, Any] = {
        "eff_rank_J": effective_rank(J),
        "eff_rank_delta_J": effective_rank(dJ),
        "energy_rank_delta_J_95": energy_rank(dJ, eps=0.05),
        "eff_rank_G": effective_rank(G),
        "henrici_J": henrici_departure(J),
        "align_G_A": subspace_alignment(G, A, k=k),
        "align_G_C": subspace_alignment(G, C, k=k),
        "align_G_lag": subspace_alignment(G, C_lag, k=k),
    }

    # Random control: same shape as G, gaussian.
    torch_gen = torch.Generator(device=G.device if G.device.type == "cpu" else "cpu")
    torch_gen.manual_seed(0)
    R_rand = torch.randn(G.shape, generator=torch_gen).to(device=G.device, dtype=G.dtype)
    out["align_random_A"] = subspace_alignment(R_rand, A, k=k)
    out["align_random_C"] = subspace_alignment(R_rand, C, k=k)
    out["align_random_lag"] = subspace_alignment(R_rand, C_lag, k=k)

    # Singular spectrum of Delta J for later plotting.
    out["sv_delta_J_top10"] = torch.linalg.svdvals(dJ.to(torch.float32))[:10].tolist()

    # Block diagnostics if applicable.
    block_labels = meta.get("block_labels")
    if block_labels is not None and J.shape[0] == J.shape[1] == len(block_labels):
        labels = torch.tensor(block_labels, device=J.device)
        same, diff = _block_mean_pair(J, labels)
        out["same_block_mean"] = same
        out["diff_block_mean"] = diff
        if G.shape[0] == G.shape[1] == len(block_labels):
            g_same, g_diff = _block_mean_pair(G, labels)
            out["same_block_mean_G"] = g_same
            out["diff_block_mean_G"] = g_diff

    return out
