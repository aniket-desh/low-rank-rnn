#!/usr/bin/env python3
"""Compute align(G, A) — and the C / C_τ variants — at multiple k values for
a set of runs. Writes a `k_sweep.json` sidecar and (optionally) renders a
panel-figure with one curve per (β, n).

Theory: align_k is the fraction of overlap between the top-k left singular
subspaces. At small k it picks out the dominant modes; at large k it asks
whether the full spectra coincide. Plotting align vs k separates these
two regimes.

Usage:
    python scripts/k_sweep.py runs_h100/tier1_lattice_2d_*
    python scripts/k_sweep.py --device cpu runs_h100/tier3_*
    python scripts/k_sweep.py --plot figures/summary/k_sweep.png runs_h100/tier1_lattice_2d_*
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lowrank_rnn.data.ising import IsingConfig, sample_ising_batch
from lowrank_rnn.models.vanilla_rnn import VanillaRNN, VanillaRNNConfig
from lowrank_rnn.analysis.spin_geometry import (
    effective_spin_operator,
    subspace_alignment,
    centered_cov,
    lagged_cov,
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


def analyze_run(run_dir: Path, ks, device="cuda"):
    cfg_path = run_dir / "config.json"
    fp = run_dir / "final.pt"
    if not (cfg_path.exists() and fp.exists()):
        return None
    cfg = json.loads(cfg_path.read_text())
    final = torch.load(fp, map_location=device, weights_only=False)
    model = _rebuild_model(cfg, final["model_state_dict"]).to(device)
    A = final["A"].to(device)

    ising_cfg = IsingConfig(
        n_spins=cfg["n_spins"], beta=cfg["beta"], graph_kind=cfg["graph_kind"],
        coupling=cfg.get("coupling", 1.0), n_blocks=cfg.get("n_blocks", 2),
        j_in=cfg.get("j_in", 1.0), j_out=cfg.get("j_out", 0.2),
        seed=cfg["seed"], device=device,
    )
    states, _, _ = sample_ising_batch(
        ising_cfg,
        batch_size=cfg.get("batch_size", 64),
        seq_len=cfg.get("seq_len", 100),
        burn_in=cfg.get("burn_in", 100),
        A=A,
    )
    with torch.no_grad():
        G = effective_spin_operator(model).detach()
        C = centered_cov(states.detach())
        C_lag = lagged_cov(states.detach(), lag=1)
        out = {
            "run": run_dir.name,
            "n_spins": cfg["n_spins"],
            "beta": cfg["beta"],
            "graph_kind": cfg["graph_kind"],
            "seed": cfg["seed"],
            "ks": list(ks),
            "align_G_A": [subspace_alignment(G, A, k=k) for k in ks],
            "align_G_C": [subspace_alignment(G, C, k=k) for k in ks],
            "align_G_lag": [subspace_alignment(G, C_lag, k=k) for k in ks],
        }
    return out, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--ks", type=int, nargs="+",
                     default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 50])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plot", type=Path,
                     help="if set, render figure to this path "
                          "(curves: one per (graph_kind, n, β); seed mean ± std)")
    args = ap.parse_args()

    all_results = []
    for d in args.run_dirs:
        if not d.is_dir():
            continue
        try:
            res = analyze_run(d, args.ks, device=args.device)
        except Exception as e:
            print(f"[skip] {d.name}: {e}")
            continue
        if res is None:
            continue
        out, _ = res
        (d / "k_sweep.json").write_text(json.dumps(out, indent=2))
        print(f"[ok] {d.name}: k={args.ks} align(A)={[f'{v:.2f}' for v in out['align_G_A']]}")
        all_results.append(out)

    if args.plot and all_results:
        by_group = defaultdict(list)
        for r in all_results:
            by_group[(r["graph_kind"], r["n_spins"], r["beta"])].append(r)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        metric_keys = ("align_G_A", "align_G_C", "align_G_lag")
        titles = (r"$\mathrm{align}(G, A)$",
                    r"$\mathrm{align}(G, C)$",
                    r"$\mathrm{align}(G, C_{\tau=1})$")
        for ax, key, title in zip(axes, metric_keys, titles):
            for (kind, n, beta), runs in sorted(by_group.items()):
                ks = runs[0]["ks"]
                vals = np.array([r[key] for r in runs])
                means = vals.mean(0); stds = vals.std(0)
                ax.errorbar(ks, means, yerr=stds, fmt="o-", capsize=3,
                             label=f"{kind} n={n} β={beta:g}")
            ax.set_xlabel("k")
            ax.set_ylabel(title.replace(r"$\mathrm{align}(G, A)$", "alignment"))
            ax.set_title(title, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=130)
        print(f"[ok] plot {args.plot}")


if __name__ == "__main__":
    main()
