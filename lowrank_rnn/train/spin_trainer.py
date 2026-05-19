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


def train_spin_prediction(cfg: SpinTrainConfig) -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    ising_cfg = _ising_config_from(cfg)
    A, meta = make_couplings(ising_cfg)

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
        x = states[:, :-1, :]
        y_true = states[:, 1:, :]

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
                report = geometry_report(model, J0, A, val_states, meta, k=cfg.align_k)
            report["epoch"] = epoch
            report["loss"] = float(loss.item())
            report["wall_s"] = time.time() - t0
            history.append(report)
            print(
                f"[epoch {epoch:04d}] loss={loss.item():.4e} "
                f"eff_rank_G={report['eff_rank_G']:.2f} "
                f"align_G_A={report['align_G_A']:.3f} "
                f"align_G_C={report['align_G_C']:.3f} "
                f"align_G_lag={report['align_G_lag']:.3f} "
                f"(rand_A={report['align_random_A']:.3f})"
            )

    cfg_dict = asdict(cfg)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(save_dir / "losses.json", "w") as f:
        json.dump(losses, f)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg_dict,
            "history": history,
            "losses": losses,
            "J0": J0.cpu(),
            "A": A.cpu(),
            "meta": meta,
        },
        save_dir / "final.pt",
    )

    return {"model": model, "history": history, "losses": losses, "save_dir": str(save_dir)}
