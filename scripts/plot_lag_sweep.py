#!/usr/bin/env python3
"""Aggregate post_hoc_lag.py sidecars (lag_alignment.json) into a single figure.

For each (kind, beta, n) group found in `runs_h100/*/lag_alignment.json`,
plot $\\mathrm{align}(G, C_\\tau)$ vs lag $\\tau$, with seed-mean ± std bands.

Usage:
    python scripts/plot_lag_sweep.py
    python scripts/plot_lag_sweep.py --out figures/summary/lag_sweep.png
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs_h100"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                     default=REPO / "figures" / "summary" / "lag_sweep.png")
    args = ap.parse_args()

    by_group = defaultdict(list)  # (kind, beta, n) -> list of sidecar dicts
    for sc in sorted(RUNS.glob("*/lag_alignment.json")):
        try:
            d = json.loads(sc.read_text())
        except json.JSONDecodeError:
            continue
        key = (d["graph_kind"], float(d["beta"]), int(d["n_spins"]))
        by_group[key].append(d)

    if not by_group:
        print("(no lag_alignment.json sidecars yet)")
        return

    # one panel per kind, one curve per (beta, n) combination.
    kinds = sorted({k[0] for k in by_group})
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.0 * len(kinds), 4.0), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for (k, beta, n), runs in sorted(by_group.items()):
            if k != kind:
                continue
            taus = sorted(int(t) for t in runs[0]["align_G_lag"].keys())
            means = [np.mean([r["align_G_lag"][str(t)] for r in runs]) for t in taus]
            stds  = [np.std([r["align_G_lag"][str(t)] for r in runs]) for t in taus]
            ax.errorbar(taus, means, yerr=stds, fmt="o-", capsize=3,
                         label=f"n={n}, β={beta:g}")
            # also plot align(G, A) as a horizontal reference
            mean_A = np.mean([r["align_G_A"] for r in runs])
            ax.axhline(mean_A, ls=":", lw=0.7, alpha=0.5)
        ax.set_xlabel(r"lag $\tau$")
        ax.set_ylabel(r"$\mathrm{align}(G, C_\tau)$")
        ax.set_ylim(0, 1)
        ax.set_xscale("log")
        ax.set_title(kind, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[ok] {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
