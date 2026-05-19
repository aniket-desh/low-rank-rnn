#!/usr/bin/env python3
"""Plot diagnostics from a spin-RNN run directory.

Usage:
    python scripts/plot_spin_results.py runs/lattice_beta044
    python scripts/plot_spin_results.py runs/lattice_beta044 runs/curie_beta02 ...

Outputs go into <run_dir>/plots/.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lowrank_rnn.models.vanilla_rnn import VanillaRNN, VanillaRNNConfig
from lowrank_rnn.analysis.spin_geometry import effective_spin_operator


def _load(run_dir: Path):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    with open(run_dir / "history.json") as f:
        history = json.load(f)
    with open(run_dir / "losses.json") as f:
        losses = json.load(f)
    final = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    return cfg, history, losses, final


def _rebuild_model(cfg: dict, state_dict: dict) -> VanillaRNN:
    mcfg = VanillaRNNConfig(
        input_dim=cfg["n_spins"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["n_spins"],
        alpha=cfg["alpha"],
        nonlinearity=cfg.get("nonlinearity", "tanh"),
        device="cpu",
    )
    model = VanillaRNN(mcfg)
    model.load_state_dict(state_dict)
    return model


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_run(run_dir: Path) -> None:
    cfg, history, losses, final = _load(run_dir)
    out = run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]

    # 1. Loss curve + zero-predictor baseline.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, lw=1.0, color="tab:blue", label="train MSE")
    if all("val_loss" in h for h in history):
        ax.plot(epochs, [h["val_loss"] for h in history], "o-", color="tab:green", label="val MSE")
        zero = [h["zero_loss"] for h in history]
        ax.axhline(float(np.mean(zero)), color="tab:red", lw=1, ls="--", alpha=0.6,
                   label=f"zero predictor ≈ {np.mean(zero):.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"loss — {run_dir.name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out / "01_loss.png")

    # 1b. Relative improvement over zero predictor.
    if all("delta_baseline" in h for h in history):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, [h["delta_baseline"] for h in history], "o-", color="tab:purple")
        ax.axhline(0.0, color="k", lw=0.7, alpha=0.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$\Delta_{\mathrm{baseline}} = (L_{\rm zero} - L_{\rm val}) / L_{\rm zero}$")
        ax.set_title("relative improvement over zero predictor")
        ax.grid(True, alpha=0.3)
        _save(fig, out / "01b_delta_baseline.png")

    # 2. Singular spectrum of Delta J at first / last eval.
    fig, ax = plt.subplots(figsize=(6, 4))
    first = np.array(history[0]["sv_delta_J_top10"])
    last = np.array(history[-1]["sv_delta_J_top10"])
    idx = np.arange(1, len(first) + 1)
    ax.plot(idx, first, "o--", label=f"epoch {history[0]['epoch']}")
    ax.plot(idx, last, "o-", label=f"epoch {history[-1]['epoch']}")
    ax.set_xlabel("singular value index")
    ax.set_ylabel(r"$\sigma_i(\Delta J)$")
    ax.set_title(r"top singular values of $\Delta J$")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out / "02_delta_J_spectrum.png")

    # 3. Effective ranks vs epoch.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["eff_rank_J"] for h in history], "o-", label=r"$r_{\rm eff}(J)$")
    ax.plot(epochs, [h["eff_rank_delta_J"] for h in history], "s-", label=r"$r_{\rm eff}(\Delta J)$")
    ax.plot(epochs, [h["eff_rank_G"] for h in history], "^-", label=r"$r_{\rm eff}(G_{\rm spin})$")
    ax.set_xlabel("epoch")
    ax.set_ylabel("effective rank")
    ax.set_title("effective rank trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out / "03_effective_rank.png")

    # 4. Alignment trajectories with random baseline overlay.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["align_G_A"] for h in history], "o-", color="tab:blue", label=r"$G$ vs $A$")
    ax.plot(epochs, [h["align_G_C"] for h in history], "s-", color="tab:orange", label=r"$G$ vs $C$")
    ax.plot(epochs, [h["align_G_lag"] for h in history], "^-", color="tab:green", label=r"$G$ vs $C_{\tau=1}$")
    ax.axhline(history[-1]["align_random_A"], color="tab:blue", lw=1, ls="--", alpha=0.4, label="rand vs A")
    ax.axhline(history[-1]["align_random_C"], color="tab:orange", lw=1, ls="--", alpha=0.4, label="rand vs C")
    ax.axhline(history[-1]["align_random_lag"], color="tab:green", lw=1, ls="--", alpha=0.4, label="rand vs lag")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\mathrm{align}_k$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("subspace alignment vs random control")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _save(fig, out / "04_alignment.png")

    # 5. Heatmaps of A, G_spin, Delta J (requires hidden_dim==n_spins for Delta J vs A comparison).
    A = final["A"].numpy()
    J0 = final["J0"].numpy()
    model = _rebuild_model(cfg, final["model_state_dict"])
    with torch.no_grad():
        G = effective_spin_operator(model).cpu().numpy()
        dJ = (model.J.detach() - final["J0"]).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, M, title in zip(axes, [A, G, dJ], [r"$A$ (true coupling)", r"$G_{\rm spin}=RJB$", r"$\Delta J = J - J_0$"]):
        vmax = max(abs(M.min()), abs(M.max()), 1e-12)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out / "05_heatmaps.png")

    # 6. Block diagnostics if applicable.
    if "same_block_mean" in history[-1]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, [h["same_block_mean"] for h in history], "o-", color="tab:red", label=r"$\bar J_{\rm same}$")
        ax.plot(epochs, [h["diff_block_mean"] for h in history], "s-", color="tab:blue", label=r"$\bar J_{\rm diff}$")
        if "same_block_mean_G" in history[-1]:
            ax.plot(epochs, [h["same_block_mean_G"] for h in history], "o--", color="tab:red", alpha=0.6, label=r"$\bar G_{\rm same}$")
            ax.plot(epochs, [h["diff_block_mean_G"] for h in history], "s--", color="tab:blue", alpha=0.6, label=r"$\bar G_{\rm diff}$")
        ax.axhline(0.0, color="k", lw=0.7, alpha=0.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean weight")
        ax.set_title("block-structured mean weights")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save(fig, out / "06_block_means.png")

    print(f"[ok] wrote plots to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    args = ap.parse_args()
    for d in args.run_dirs:
        plot_run(d)


if __name__ == "__main__":
    main()
