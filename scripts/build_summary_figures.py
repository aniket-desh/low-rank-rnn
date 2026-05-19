#!/usr/bin/env python3
"""Build cross-run summary figures for summary.md.

Reads `history.json` / `config.json` from a list of run directories and
emits aggregate plots into figures/summary/. Each plot is one panel that
summarises a tier (smoke comparison, seed bands, phase diagram, scaling, etc.).

Usage:
    python scripts/build_summary_figures.py smoke
    python scripts/build_summary_figures.py tier1
    python scripts/build_summary_figures.py tier2
    python scripts/build_summary_figures.py tier3
    python scripts/build_summary_figures.py tier4
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs_h100"
OUT = REPO / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)


def _load_run(run_dir: Path):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    with open(run_dir / "history.json") as f:
        hist = json.load(f)
    return cfg, hist


def _last(hist):
    return hist[-1]


def _save(fig, name):
    p = OUT / name
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"[ok] {p.relative_to(REPO)}")


# -----------------------------------------------------------------------------
# smoke
# -----------------------------------------------------------------------------
def build_smoke():
    runs = [
        ("n=64",   RUNS / "_smoke_lattice64"),
        ("n=256",  RUNS / "_smoke_lattice256"),
        ("n=1024", RUNS / "_smoke_lattice1024"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for label, d in runs:
        if not d.exists():
            continue
        _, hist = _load_run(d)
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["val_loss"] for h in hist], "o-", label=label)
        axes[1].plot(ep, [h["eff_rank_G"] for h in hist], "o-", label=label)
        axes[2].plot(ep, [h["align_G_A"] for h in hist], "o-", label=label + r" $G$-vs-$A$")
        axes[2].plot(ep, [h["align_G_C"] for h in hist], "s--", alpha=0.5,
                     label=label + r" $G$-vs-$C$")
    axes[0].axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.4, label="zero predictor")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val MSE"); axes[0].legend(fontsize=8)
    axes[0].set_title("validation loss"); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel(r"$r_{\rm eff}(G_{\rm spin})$")
    axes[1].set_title("effective rank of $G_{\\rm spin}$"); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3); axes[1].set_yscale("log")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("alignment"); axes[2].set_ylim(0, 1)
    axes[2].set_title(r"alignment of $G_{\rm spin}$ with $A,C$")
    axes[2].legend(fontsize=7, ncol=1); axes[2].grid(True, alpha=0.3)
    _save(fig, "smoke_size_comparison.png")


# -----------------------------------------------------------------------------
# tier1
# -----------------------------------------------------------------------------
TIER1_NAME_RE = re.compile(r"tier1_(?P<kind>[a-z_2d]+?)_beta(?P<beta>[0-9]+)_seed(?P<seed>\d+)")


def build_tier1():
    rows = []
    for d in sorted(RUNS.glob("tier1_*")):
        m = TIER1_NAME_RE.search(d.name)
        if not m:
            continue
        try:
            cfg, hist = _load_run(d)
        except FileNotFoundError:
            continue
        last = _last(hist)
        rows.append({
            "kind": m["kind"],
            "beta": float(cfg["beta"]),
            "seed": int(m["seed"]),
            "delta_baseline": last["delta_baseline"],
            "eff_rank_G": last["eff_rank_G"],
            "align_G_A": last["align_G_A"],
            "align_G_C": last["align_G_C"],
            "align_G_lag": last["align_G_lag"],
            "align_random_A": last["align_random_A"],
        })
    if not rows:
        print("[skip] no tier1 runs"); return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    metric_titles = [
        ("delta_baseline", r"$\Delta_{\rm baseline}$"),
        ("eff_rank_G",     r"$r_{\rm eff}(G_{\rm spin})$"),
        ("align_G_C",      r"$\mathrm{align}(G,C)$"),
    ]
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[(r["kind"], r["beta"])].append(r)

    x_labels = sorted(by_kind.keys(), key=lambda k: (k[0], k[1]))
    xs = np.arange(len(x_labels))
    for ax, (key, title) in zip(axes, metric_titles):
        vals = [[r[key] for r in by_kind[k]] for k in x_labels]
        means = [np.mean(v) for v in vals]
        stds  = [np.std(v) for v in vals]
        ax.bar(xs, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{k[0]}\nβ={k[1]:g}" for k in x_labels], fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    _save(fig, "tier1_seed_bars.png")


# -----------------------------------------------------------------------------
# tier2: phase diagram
# -----------------------------------------------------------------------------
TIER2_NAME_RE = re.compile(r"tier2_(?P<kind>[a-z]+?)(?P<n>\d+)_beta(?P<beta>[0-9.]+)_seed(?P<seed>\d+)")


def build_tier2():
    by_family = defaultdict(lambda: defaultdict(list))
    for d in sorted(RUNS.glob("tier2_*")):
        m = TIER2_NAME_RE.search(d.name)
        if not m:
            continue
        try:
            cfg, hist = _load_run(d)
        except FileNotFoundError:
            continue
        family = m["kind"]
        beta = float(cfg["beta"])
        last = _last(hist)
        by_family[family][beta].append(last)

    if not by_family:
        print("[skip] no tier2 runs"); return

    metrics = [
        ("delta_baseline", r"$\Delta_{\rm baseline}$"),
        ("eff_rank_G",     r"$r_{\rm eff}(G)$"),
        ("align_G_A",      r"$\mathrm{align}(G,A)$"),
        ("align_G_C",      r"$\mathrm{align}(G,C)$"),
        ("align_G_lag",    r"$\mathrm{align}(G,C_{\tau=1})$"),
    ]
    n_families = len(by_family)
    fig, axes = plt.subplots(n_families, len(metrics),
                              figsize=(3.0 * len(metrics), 2.6 * n_families), squeeze=False)

    for row_i, (fam, data) in enumerate(sorted(by_family.items())):
        betas = sorted(data.keys())
        for col_i, (key, title) in enumerate(metrics):
            ax = axes[row_i, col_i]
            means = [np.mean([h[key] for h in data[b]]) for b in betas]
            stds  = [np.std([h[key] for h in data[b]]) for b in betas]
            ax.errorbar(betas, means, yerr=stds, fmt="o-", color="tab:blue", capsize=3)
            if key == "delta_baseline":
                ax.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.5)
            if key.startswith("align"):
                rand_key = "align_random_" + key.split("_")[-1]
                rand_vals = [np.mean([h[rand_key] for h in data[b]]) for b in betas]
                ax.plot(betas, rand_vals, "k--", lw=0.8, alpha=0.5, label="random")
                ax.set_ylim(0, 1)
            ax.set_title(f"{fam}  {title}", fontsize=9)
            ax.set_xlabel("β")
            ax.grid(True, alpha=0.3)
    _save(fig, "tier2_phase_diagram.png")


# -----------------------------------------------------------------------------
# tier3 / tier4: scaling
# -----------------------------------------------------------------------------
TIER34_NAME_RE = re.compile(
    r"tier(?P<tier>3|4)_(?:fss_)?lattice(?P<n>\d+)_beta(?P<beta>[0-9.]+)_seed(?P<seed>\d+)"
)


def _build_scaling(tier_tag: str):
    by_size = defaultdict(lambda: defaultdict(list))
    for d in sorted(RUNS.glob(f"tier{tier_tag}_*")):
        m = TIER34_NAME_RE.search(d.name)
        if not m or m["tier"] != tier_tag:
            continue
        try:
            cfg, hist = _load_run(d)
        except FileNotFoundError:
            continue
        n = int(m["n"])
        beta = float(cfg["beta"])
        last = _last(hist)
        by_size[n][beta].append(last)
    if not by_size:
        print(f"[skip] no tier{tier_tag} runs"); return

    metrics = [
        ("delta_baseline", r"$\Delta_{\rm baseline}$"),
        ("eff_rank_G",     r"$r_{\rm eff}(G)$"),
        ("align_G_A",      r"$\mathrm{align}(G,A)$"),
        ("align_G_C",      r"$\mathrm{align}(G,C)$"),
        ("align_G_lag",    r"$\mathrm{align}(G,C_{\tau=1})$"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.4))
    for ax, (key, title) in zip(axes, metrics):
        for n in sorted(by_size.keys()):
            betas = sorted(by_size[n].keys())
            means = [np.mean([h[key] for h in by_size[n][b]]) for b in betas]
            stds  = [np.std([h[key] for h in by_size[n][b]]) for b in betas]
            ax.errorbar(betas, means, yerr=stds, fmt="o-", label=f"n={n}", capsize=3)
        if key.startswith("align"):
            ax.set_ylim(0, 1)
        if key == "eff_rank_G":
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("β")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    _save(fig, f"tier{tier_tag}_scaling.png")


def build_tier3():
    _build_scaling("3")


def build_tier4():
    _build_scaling("4")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
BUILDERS = {
    "smoke": build_smoke,
    "tier1": build_tier1,
    "tier2": build_tier2,
    "tier3": build_tier3,
    "tier4": build_tier4,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=sorted(BUILDERS) + ["all"])
    args = ap.parse_args()
    which = sorted(BUILDERS) if "all" in args.which else args.which
    for w in which:
        BUILDERS[w]()


if __name__ == "__main__":
    main()
