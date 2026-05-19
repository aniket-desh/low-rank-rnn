from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

import torch
import torch.nn as nn

from lowrank_rnn.data.ising import IsingConfig, sample_ising_batch, make_couplings
from lowrank_rnn.models.vanilla_rnn import VanillaRNN, VanillaRNNConfig
from lowrank_rnn.analysis.spin_geometry import geometry_report


@dataclass
class SpinTrainConfig:
    n_spins: int = 64
    hidden_dim: int = 64
    beta: float = 0.5
    field: float = 0.0
    graph_kind: str = "lattice_2d"
    coupling: float = 1.0
    n_blocks: int = 2
    j_in: float = 1.0
    j_out: float = 0.2
    seq_len: int = 100
    batch_size: int = 64
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    alpha: float = 0.2
    nonlinearity: str = "tanh"
    burn_in: int = 100
    eval_every: int = 25
    align_k: int = 5
    seed: int = 0
    device: str = "cpu"
    save_dir: str = "runs/spin_rnn_debug"
    refresh_couplings_every_epoch: bool = False
    # task variants — see docs/theory_spin_rnn.md "Candidate tasks" and HANDOFF §5.5
    # next_state:    x_t = s_t,                                 y_t = s_{t+1}      (default)
    # denoise:       x_t = M_t⊙s_t + (1-M_t)⊙ξ_t (Bernoulli ξ), y_t = s_t          (clean current)
    # partial:       x_t = P_Ω s_t (zero-fill unobserved coords), y_t = s_{t+1}
    task: str = "next_state"
    mask_frac: float = 0.3       # fraction CORRUPTED for denoise (1 - keep prob)
    obs_frac: float = 0.5        # fraction OBSERVED for partial


def _ising_config_from(cfg: SpinTrainConfig) -> IsingConfig:
    return IsingConfig(
        n_spins=cfg.n_spins,
        beta=cfg.beta,
        field=cfg.field,
        graph_kind=cfg.graph_kind,
        coupling=cfg.coupling,
        n_blocks=cfg.n_blocks,
        j_in=cfg.j_in,
        j_out=cfg.j_out,
        seed=cfg.seed,
        device=cfg.device,
    )


def _build_task_pair(
    states: torch.Tensor,
    cfg: SpinTrainConfig,
    obs_mask: Optional[torch.Tensor] = None,
):
    """Map (batch, seq+1, n) trajectories to (x, y) per the cfg.task.

    Returns (x, y_true) of equal seq length. `obs_mask` is the fixed (n,) bool
    tensor of observed indices for the `partial` task; ignored otherwise.
    """
    s_in = states[:, :-1, :]
    s_next = states[:, 1:, :]
    if cfg.task == "next_state":
        return s_in, s_next
    if cfg.task == "denoise":
        # Per-element keep-mask, then fill the dropped coords with iid ±1.
        keep = (torch.rand_like(s_in) > cfg.mask_frac).to(s_in.dtype)
        noise = (
            2.0 * torch.randint(0, 2, s_in.shape, device=s_in.device, dtype=torch.int64) - 1
        ).to(s_in.dtype)
        x = keep * s_in + (1.0 - keep) * noise
        return x, s_in  # target is the clean current state
    if cfg.task == "partial":
        if obs_mask is None:
            raise ValueError("partial task requires an obs_mask")
        # Zero-fill unobserved coords; predict full next state.
        mask = obs_mask.to(s_in.dtype).view(1, 1, -1)
        x = s_in * mask
        return x, s_next
    raise ValueError(f"unknown task {cfg.task!r}")


def train_spin_prediction(cfg: SpinTrainConfig) -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    ising_cfg = _ising_config_from(cfg)
    A, meta = make_couplings(ising_cfg)

    # Fixed observation mask for partial task: which coords are observed.
    obs_mask: Optional[torch.Tensor] = None
    if cfg.task == "partial":
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        n_obs = max(1, int(round(cfg.obs_frac * cfg.n_spins)))
        perm = torch.randperm(cfg.n_spins, generator=gen)
        obs_mask = torch.zeros(cfg.n_spins, dtype=torch.bool)
        obs_mask[perm[:n_obs]] = True
        obs_mask = obs_mask.to(device=device)

    model = VanillaRNN(
        VanillaRNNConfig(
            input_dim=cfg.n_spins,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.n_spins,
            alpha=cfg.alpha,
            nonlinearity=cfg.nonlinearity,
            device=cfg.device,
        )
    )

    J0 = model.J.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    losses: List[float] = []

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if cfg.refresh_couplings_every_epoch:
            A, meta = make_couplings(ising_cfg)
        states, _A, _meta = sample_ising_batch(
            ising_cfg,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            burn_in=cfg.burn_in,
            A=A,
            meta=meta,
        )
        x, y_true = _build_task_pair(states, cfg, obs_mask=obs_mask)

        y_pred, _ = model(x, return_states=False)
        loss = loss_fn(y_pred, y_true)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(float(loss.item()))

        if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                val_states, _, _ = sample_ising_batch(
                    ising_cfg,
                    batch_size=cfg.batch_size,
                    seq_len=cfg.seq_len,
                    burn_in=cfg.burn_in,
                    A=A,
                    meta=meta,
                )
                v_x, v_y = _build_task_pair(val_states, cfg, obs_mask=obs_mask)
                v_pred, _ = model(v_x, return_states=False)
                val_loss = float(loss_fn(v_pred, v_y).item())
                zero_loss = float(v_y.pow(2).mean().item())  # MSE of the zero predictor
                report = geometry_report(model, J0, A, val_states, meta, k=cfg.align_k)
            report["epoch"] = epoch
            report["loss"] = float(loss.item())
            report["val_loss"] = val_loss
            report["zero_loss"] = zero_loss
            report["delta_baseline"] = (zero_loss - val_loss) / max(zero_loss, 1e-12)
            report["wall_s"] = time.time() - t0
            history.append(report)
            print(
                f"[epoch {epoch:04d}] train_loss={loss.item():.4e} "
                f"val={val_loss:.4e} (zero={zero_loss:.3f}, Δ={report['delta_baseline']:+.3f}) "
                f"eff_rank_G={report['eff_rank_G']:.2f} "
                f"align[A,C,lag]={report['align_G_A']:.3f}/{report['align_G_C']:.3f}/{report['align_G_lag']:.3f} "
                f"(rand_A={report['align_random_A']:.3f})"
            )

    cfg_dict = asdict(cfg)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(save_dir / "losses.json", "w") as f:
        json.dump(losses, f)

    save_blob = {
        "model_state_dict": model.state_dict(),
        "cfg": cfg_dict,
        "history": history,
        "losses": losses,
        "J0": J0.cpu(),
        "A": A.cpu(),
        "meta": meta,
    }
    if obs_mask is not None:
        save_blob["obs_mask"] = obs_mask.cpu()
    torch.save(save_blob, save_dir / "final.pt")

    return {"model": model, "history": history, "losses": losses, "save_dir": str(save_dir)}
