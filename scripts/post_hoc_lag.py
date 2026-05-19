#!/usr/bin/env python3
"""Post-hoc analysis: recompute align(G, C_τ) for additional lags τ ∈ {1,2,5,10}.

The trainer only stores τ=1 in history.json (sufficient for the headline
diagnostic). This script reloads each run's final.pt and validation-time
spin trajectories implicitly via fresh sampling, then computes lagged-covariance
alignments at multiple τ. Useful for testing whether the learned operator
aligns better with the *slow* modes of the dynamics than with the equilibrium
covariance.

Usage:
    python scripts/post_hoc_lag.py runs_h100/tier2_lattice_2d64_beta044_*
    python scripts/post_hoc_lag.py runs_h100/tier4_*

Writes a JSON sidecar `lag_alignment.json` into each run directory.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lowrank_rnn.data.ising import IsingConfig, sample_ising_batch
from lowrank_rnn.models.vanilla_rnn import VanillaRNN, VanillaRNNConfig
from lowrank_rnn.analysis.spin_geometry import (
    effective_spin_operator,
    subspace_alignment,
    lagged_cov,
    centered_cov,
)


def _rebuild_model(cfg: dict, state_dict) -> VanillaRNN:
    mcfg = VanillaRNNConfig(
        input_dim=cfg["n_spins"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["n_spins"],
        alpha=cfg["alpha"],
        nonlinearity=cfg.get("nonlinearity", "tanh"),
        device=cfg.get("device", "cpu"),
    )
    m = VanillaRNN(mcfg)
    m.load_state_dict(state_dict)
    return m


def analyze_run(run_dir: Path, lags=(1, 2, 5, 10), device="cuda") -> dict:
    cfg_path = run_dir / "config.json"
    fp = run_dir / "final.pt"
    if not (cfg_path.exists() and fp.exists()):
        return {"run": run_dir.name, "error": "missing config or final.pt"}
    cfg = json.loads(cfg_path.read_text())
    final = torch.load(fp, map_location=device, weights_only=False)

    model = _rebuild_model(cfg, final["model_state_dict"]).to(device)
    A = final["A"].to(device)

    # Sample fresh validation states for stable lagged-cov estimates.
    ising_cfg = IsingConfig(
        n_spins=cfg["n_spins"],
        beta=cfg["beta"],
        graph_kind=cfg["graph_kind"],
        coupling=cfg.get("coupling", 1.0),
        n_blocks=cfg.get("n_blocks", 2),
        j_in=cfg.get("j_in", 1.0),
        j_out=cfg.get("j_out", 0.2),
        seed=cfg["seed"],
        device=device,
    )
    states, _, _ = sample_ising_batch(
        ising_cfg,
        batch_size=cfg.get("batch_size", 64),
        seq_len=max(cfg.get("seq_len", 100), max(lags) + 50),
        burn_in=cfg.get("burn_in", 100),
        A=A,
    )

    with torch.no_grad():
        G = effective_spin_operator(model).detach()
        k = cfg.get("align_k", 5)
        out = {
            "run": run_dir.name,
            "n_spins": cfg["n_spins"],
            "beta": cfg["beta"],
            "graph_kind": cfg["graph_kind"],
            "seed": cfg["seed"],
            "align_k": k,
            "align_G_A": subspace_alignment(G, A, k=k),
            "align_G_C": subspace_alignment(G, centered_cov(states), k=k),
            "align_G_lag": {},
        }
        for tau in lags:
            C_tau = lagged_cov(states, lag=tau)
            out["align_G_lag"][str(tau)] = subspace_alignment(G, C_tau, k=k)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = ap.parse_args()
    for d in args.run_dirs:
        if not d.is_dir():
            continue
        res = analyze_run(d, lags=tuple(args.lags), device=args.device)
        (d / "lag_alignment.json").write_text(json.dumps(res, indent=2))
        print(f"[ok] {d}: align_G_A={res.get('align_G_A', 'NA'):.3f} lags={res.get('align_G_lag', {})}")


if __name__ == "__main__":
    main()
