#!/usr/bin/env python3
"""Build publication-quality figures for the blog/LessWrong writeup.

Reads completed runs under runs_h100/ and emits high-DPI PNG, SVG, and PDF
figures into figures/blog/. Tolerant of missing Tier 6 (skips that figure
gracefully).

Usage:
    python scripts/build_blog_figures.py --all
    python scripts/build_blog_figures.py --tier tier4
    python scripts/build_blog_figures.py --tier tier5
    python scripts/build_blog_figures.py --tier tier6
    python scripts/build_blog_figures.py --animations  # placeholder; animations
                                                       # live in build_animations.py
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lowrank_rnn.data.ising import IsingConfig, sample_ising_batch
from lowrank_rnn.models.vanilla_rnn import VanillaRNN, VanillaRNNConfig
from lowrank_rnn.analysis.spin_geometry import (
    effective_spin_operator,
    centered_cov,
    lagged_cov,
    subspace_alignment,
)

RUNS = REPO / "runs_h100"
OUT = REPO / "figures" / "blog"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "animations").mkdir(parents=True, exist_ok=True)

BETA_C_LATTICE = 0.4406867935097715  # 0.5 * ln(1 + sqrt(2))
BETA_C_CURIE = 1.0

# -----------------------------------------------------------------------------
# Aesthetic
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

N_PALETTE = {
    64:   "#1f77b4",
    256:  "#d62728",
    1024: "#2ca02c",
}
TASK_PALETTE = {
    "next_state": "#1f77b4",
    "denoise":    "#d62728",
    "partial":    "#2ca02c",
}
FAMILY_PALETTE = {
    "lattice_2d":  "#1f77b4",
    "curie_weiss": "#d62728",
    "block":       "#2ca02c",
}


def _save(fig, stem: str, also_pdf: bool = True):
    p_png = OUT / f"{stem}.png"
    p_svg = OUT / f"{stem}.svg"
    fig.savefig(p_png, dpi=300)
    fig.savefig(p_svg)
    if also_pdf:
        fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {p_png.relative_to(REPO)} (+ svg{' +pdf' if also_pdf else ''})")


# -----------------------------------------------------------------------------
# Run collection
# -----------------------------------------------------------------------------
TIER4_RE = re.compile(
    r"tier4_fss_lattice(?P<n>\d+)_beta(?P<beta>\d+)_seed(?P<seed>\d+)$"
)
TIER5_RE = re.compile(
    r"tier5_(?P<task>next_state|denoise|partial)_lattice_2d_beta(?P<beta>\d+)_seed(?P<seed>\d+)$"
)
TIER6_RE = re.compile(
    r"tier6_zoom_lattice(?P<n>\d+)_beta(?P<beta>\d+p\d+)_seed(?P<seed>\d+)$"
)
TIER2_RE = re.compile(
    r"tier2_(?P<kind>lattice_2d|curie_weiss|block)(?P<n>\d+)_beta(?P<beta>\d+)_seed(?P<seed>\d+)$"
)
TIER3_RE = re.compile(
    r"tier3_lattice(?P<n>\d+)_beta(?P<beta>\d+)_seed(?P<seed>\d+)$"
)
ANIM_RE = re.compile(
    r"animation_lattice(?P<n>\d+)_beta(?P<beta>\d+p\d+)_(?P<task>next_state|denoise|partial)$"
)


def _load(d: Path):
    cfg = json.loads((d / "config.json").read_text())
    hist = json.loads((d / "history.json").read_text())
    return cfg, hist


def collect_tier4() -> List[Dict]:
    rows = []
    for d in sorted(RUNS.glob("tier4_fss_lattice*_seed*")):
        if not d.is_dir() or not (d / "history.json").exists():
            continue
        m = TIER4_RE.search(d.name)
        if not m:
            continue
        cfg, hist = _load(d)
        last = hist[-1]
        rows.append({
            "n": int(m["n"]),
            "beta": float(cfg["beta"]),
            "seed": int(m["seed"]),
            "delta_baseline": last["delta_baseline"],
            "eff_rank_G": last["eff_rank_G"],
            "align_G_A": last["align_G_A"],
            "align_G_C": last["align_G_C"],
            "align_G_lag": last["align_G_lag"],
            "align_random_A": last["align_random_A"],
            "align_random_C": last["align_random_C"],
            "align_random_lag": last["align_random_lag"],
        })
    return rows


def collect_tier6() -> List[Dict]:
    rows = []
    for d in sorted(RUNS.glob("tier6_zoom_lattice*_seed*")):
        if not d.is_dir() or not (d / "history.json").exists():
            continue
        m = TIER6_RE.search(d.name)
        if not m:
            continue
        cfg, hist = _load(d)
        last = hist[-1]
        rows.append({
            "n": int(m["n"]),
            "beta": float(cfg["beta"]),
            "seed": int(m["seed"]),
            "delta_baseline": last["delta_baseline"],
            "eff_rank_G": last["eff_rank_G"],
            "align_G_A": last["align_G_A"],
            "align_G_C": last["align_G_C"],
            "align_G_lag": last["align_G_lag"],
            "align_random_A": last["align_random_A"],
            "align_random_C": last["align_random_C"],
            "align_random_lag": last["align_random_lag"],
        })
    return rows


def collect_tier5() -> List[Dict]:
    rows = []
    for d in sorted(RUNS.glob("tier5_*_seed*")):
        if not d.is_dir() or not (d / "history.json").exists():
            continue
        m = TIER5_RE.search(d.name)
        if not m:
            continue
        cfg, hist = _load(d)
        last = hist[-1]
        rows.append({
            "task": m["task"],
            "beta": float(cfg["beta"]),
            "seed": int(m["seed"]),
            "delta_baseline": last["delta_baseline"],
            "eff_rank_G": last["eff_rank_G"],
            "align_G_A": last["align_G_A"],
            "align_G_C": last["align_G_C"],
            "align_G_lag": last["align_G_lag"],
        })
    return rows


def collect_tier2() -> List[Dict]:
    rows = []
    for d in sorted(RUNS.glob("tier2_*_seed*")):
        if not d.is_dir() or not (d / "history.json").exists():
            continue
        m = TIER2_RE.search(d.name)
        if not m:
            continue
        cfg, hist = _load(d)
        last = hist[-1]
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
            "align_random_C": last["align_random_C"],
            "align_random_lag": last["align_random_lag"],
        })
    return rows


def _seed_agg(rows: List[Dict], key: Tuple[str, ...], metrics: List[str]):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = tuple(r[k] for k in key)
        for m in metrics:
            by[k][m].append(r[m])
    out = {}
    for k, d in by.items():
        out[k] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in d.items()}
    return out


# -----------------------------------------------------------------------------
# Figure 1: main result triptych
# -----------------------------------------------------------------------------
def fig_main_triptych(rows4: List[Dict]):
    if not rows4:
        print("[skip] fig1 — no tier4 rows"); return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    by_n_beta = defaultdict(lambda: defaultdict(list))
    for r in rows4:
        by_n_beta[r["n"]][r["beta"]].append(r)

    panels = [
        ("eff_rank_G", r"$r_{\rm eff}(G)$", False),
        ("delta_baseline", r"$\Delta_{\rm baseline}$", False),
        ("rel_align_A", r"$\mathrm{align}(G,A)\,/\,\mathrm{align}_{\rm rand}$", False),
    ]
    for ax, (key, ylab, ylog) in zip(axes, panels):
        for n in sorted(by_n_beta.keys()):
            betas = sorted(by_n_beta[n].keys())
            means = []
            stds = []
            for b in betas:
                runs = by_n_beta[n][b]
                if key == "rel_align_A":
                    vals = [r["align_G_A"] / max(r["align_random_A"], 1e-9) for r in runs]
                else:
                    vals = [r[key] for r in runs]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means = np.array(means); stds = np.array(stds)
            ax.plot(betas, means, "o-", lw=1.8, ms=5, color=N_PALETTE.get(n, None),
                    label=f"n={n}")
            ax.fill_between(betas, means - stds, means + stds, alpha=0.18,
                            color=N_PALETTE.get(n, None), edgecolor="none")
        ax.axvline(BETA_C_LATTICE, ls="--", color="#888", lw=0.8)
        ax.text(BETA_C_LATTICE, ax.get_ylim()[1] * 0.95 if not ylog else 1,
                r" $\beta_c$", color="#888", fontsize=8, va="top")
        if ylog:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(ylab)
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend(title="size", title_fontsize=8.5, loc="upper right")
    fig.suptitle("Tier 4 finite-size scaling near the lattice critical point",
                  y=1.02, fontsize=12)
    _save(fig, "main_result_triptych")


# -----------------------------------------------------------------------------
# Figure 2: effective-rank phase heatmap
# -----------------------------------------------------------------------------
def fig_effective_rank_heatmap(rows4: List[Dict], rows6: List[Dict]):
    rows = rows6 if rows6 else rows4
    if not rows:
        print("[skip] fig2 — no rows"); return
    tag = "tier6" if rows6 else "tier4"

    by = defaultdict(list)
    for r in rows:
        by[(r["n"], r["beta"])].append(r["eff_rank_G"])
    ns = sorted({k[0] for k in by})
    betas = sorted({k[1] for k in by})

    grid_raw = np.full((len(ns), len(betas)), np.nan)
    for i, n in enumerate(ns):
        for j, b in enumerate(betas):
            v = by.get((n, b))
            if v:
                grid_raw[i, j] = float(np.mean(v))

    variants = [
        ("effective_rank_raw_heatmap", grid_raw, r"$r_{\rm eff}(G)$", "viridis"),
        ("effective_rank_log_heatmap", np.log10(grid_raw),
            r"$\log_{10}\,r_{\rm eff}(G)$", "viridis"),
        ("effective_rank_heatmap", grid_raw / np.array(ns)[:, None],
            r"$r_{\rm eff}(G)/n$", "magma"),
    ]
    for stem, grid, cbar_label, cmap in variants:
        fig, ax = plt.subplots(figsize=(8.0, 3.0))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                       extent=[min(betas), max(betas), -0.5, len(ns) - 0.5])
        ax.set_yticks(range(len(ns))); ax.set_yticklabels([f"n={n}" for n in ns])
        ax.set_xlabel(r"$\beta$")
        ax.axvline(BETA_C_LATTICE, ls="--", color="white", lw=1.0, alpha=0.7)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(cbar_label)
        ax.set_title(f"effective rank phase map ({tag})")
        _save(fig, stem)


# -----------------------------------------------------------------------------
# Figure 3: scaling exponent α(β)
# -----------------------------------------------------------------------------
def fig_scaling_exponent(rows4: List[Dict], rows6: List[Dict]):
    rows = rows6 if rows6 else rows4
    if not rows:
        print("[skip] fig3 — no rows"); return

    by_beta_n = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_beta_n[r["beta"]][r["n"]].append(r["eff_rank_G"])

    betas = sorted(by_beta_n.keys())
    alpha_means, alpha_lo, alpha_hi, intercept_means = [], [], [], []
    rng = np.random.default_rng(0)
    for b in betas:
        ns = sorted(by_beta_n[b].keys())
        if len(ns) < 2:
            alpha_means.append(np.nan); alpha_lo.append(np.nan); alpha_hi.append(np.nan)
            intercept_means.append(np.nan)
            continue
        seed_vals = [by_beta_n[b][n] for n in ns]
        boot_alphas = []
        boot_intercepts = []
        for _ in range(1000):
            ys = []
            xs = []
            for n, vals in zip(ns, seed_vals):
                pick = rng.choice(vals, size=len(vals), replace=True)
                ys.append(float(np.mean(pick)))
                xs.append(n)
            logx = np.log(xs); logy = np.log(ys)
            slope, intercept = np.polyfit(logx, logy, 1)
            boot_alphas.append(slope)
            boot_intercepts.append(intercept)
        boot_alphas = np.array(boot_alphas)
        alpha_means.append(float(np.mean(boot_alphas)))
        alpha_lo.append(float(np.percentile(boot_alphas, 2.5)))
        alpha_hi.append(float(np.percentile(boot_alphas, 97.5)))
        intercept_means.append(float(np.mean(boot_intercepts)))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.errorbar(
        betas, alpha_means,
        yerr=[np.array(alpha_means) - np.array(alpha_lo),
              np.array(alpha_hi) - np.array(alpha_means)],
        fmt="o-", lw=1.6, ms=5, capsize=2.5,
        color="#1f77b4",
    )
    ax.axvline(BETA_C_LATTICE, ls="--", color="#888", lw=0.8)
    ax.text(BETA_C_LATTICE, ax.get_ylim()[1], r" $\beta_c$",
            color="#888", fontsize=8, va="top")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"scaling exponent $\alpha(\beta)$ in $r_{\rm eff}\sim n^\alpha$")
    ax.set_title("finite-size scaling exponent of the effective rank")
    _save(fig, "scaling_exponent_alpha")

    csv_path = OUT / "scaling_exponent_alpha.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["beta", "alpha_mean", "alpha_lo", "alpha_hi", "intercept_mean"])
        for b, am, lo, hi, im_ in zip(betas, alpha_means, alpha_lo, alpha_hi, intercept_means):
            w.writerow([b, am, lo, hi, im_])
    print(f"[ok] {csv_path.relative_to(REPO)}")


# -----------------------------------------------------------------------------
# Figure 4: geometry decomposition
# -----------------------------------------------------------------------------
def _rebuild_model(cfg: dict, state_dict) -> VanillaRNN:
    mcfg = VanillaRNNConfig(
        input_dim=cfg["n_spins"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["n_spins"],
        alpha=cfg["alpha"],
        nonlinearity=cfg.get("nonlinearity", "tanh"),
        device="cpu",
    )
    m = VanillaRNN(mcfg)
    m.load_state_dict(state_dict)
    return m


def _find_decomp_run(beta: float) -> Path | None:
    """Locate a representative completed run for the lattice n=64 at this β."""
    candidates = []
    # prefer animation source (deterministic, seed 0)
    btag = f"{beta:.3f}".replace(".", "p")
    p = RUNS / f"animation_lattice64_beta{btag}_next_state"
    if (p / "final.pt").exists():
        return p
    # tier 5 next_state
    btag_int = f"{beta}".replace(".", "")
    p = RUNS / f"tier5_next_state_lattice_2d_beta{btag_int}_seed0"
    if (p / "final.pt").exists():
        return p
    # tier 2 lattice
    p = RUNS / f"tier2_lattice_2d64_beta{btag_int}_seed0"
    if (p / "final.pt").exists():
        return p
    # tier 1 lattice (rougher)
    p = RUNS / f"tier1_lattice_2d_beta{btag_int}_seed0"
    if (p / "final.pt").exists():
        return p
    return None


def _gather_G_A_C(d: Path):
    cfg = json.loads((d / "config.json").read_text())
    final = torch.load(d / "final.pt", map_location="cpu", weights_only=False)
    model = _rebuild_model(cfg, final["model_state_dict"])
    with torch.no_grad():
        G = effective_spin_operator(model).cpu().float()
    A = final["A"].cpu().float()
    # Fresh val sample on CPU for C and C_tau
    ising_cfg = IsingConfig(
        n_spins=cfg["n_spins"], beta=cfg["beta"],
        graph_kind=cfg["graph_kind"], coupling=cfg.get("coupling", 1.0),
        n_blocks=cfg.get("n_blocks", 2), j_in=cfg.get("j_in", 1.0),
        j_out=cfg.get("j_out", 0.2), seed=cfg["seed"], device="cpu",
    )
    states, _, _ = sample_ising_batch(
        ising_cfg,
        batch_size=128,
        seq_len=100,
        burn_in=cfg.get("burn_in", 100),
        A=A,
    )
    C = centered_cov(states)
    C_lag = lagged_cov(states, lag=1)
    return G, A, C, C_lag, cfg


def fig_geometry_decomposition_lattice64():
    """Build three sub-figures for lattice n=64 across β ∈ {0.2, 0.44, 0.8}."""
    betas = [0.2, 0.44, 0.8]
    grabs = []
    for b in betas:
        d = _find_decomp_run(b)
        if d is None:
            print(f"[warn] no decomp source found for β={b}; skipping geometry fig")
            return
        try:
            grabs.append((b, _gather_G_A_C(d)))
        except Exception as e:
            print(f"[warn] decomp gather failed for β={b}: {e}")
            return

    # (a) Matrix heatmaps
    fig, axes = plt.subplots(len(betas), 4, figsize=(11.5, 8.5))
    col_titles = [r"$A$", r"$C$", r"$C_{\tau=1}$", r"$G=RJB$"]
    for row_i, (beta, (G, A, C, Cl, _)) in enumerate(grabs):
        mats = [A.numpy(), C.numpy(), Cl.numpy(), G.numpy()]
        for col_i, M in enumerate(mats):
            ax = axes[row_i, col_i]
            vmax = float(np.max(np.abs(M))) or 1e-12
            im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title(col_titles[col_i])
            if col_i == 0:
                ax.set_ylabel(rf"$\beta={beta}$", rotation=0, ha="right", va="center",
                              labelpad=12)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("matrix decomposition — lattice $n=64$, three temperatures", y=1.01)
    _save(fig, "matrix_heatmaps_lattice64")

    # (b) Singular values of G
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for beta, (G, *_rest) in grabs:
        s = torch.linalg.svdvals(G).numpy()
        ax.plot(np.arange(1, len(s) + 1), s, "o-", ms=3,
                label=rf"$\beta={beta}$")
    ax.set_yscale("log")
    ax.set_xlabel("singular index")
    ax.set_ylabel(r"$\sigma_i(G)$")
    ax.set_title("singular spectrum of $G$ — lattice $n=64$")
    ax.legend()
    _save(fig, "singular_values_lattice64")

    # (c) Top 4 left singular vectors reshaped to 8x8
    L = 8
    fig, axes = plt.subplots(len(betas), 4, figsize=(9.0, 6.5))
    for row_i, (beta, (G, *_rest)) in enumerate(grabs):
        U, _, _ = torch.linalg.svd(G, full_matrices=False)
        for col_i in range(4):
            ax = axes[row_i, col_i]
            v = U[:, col_i].numpy().reshape(L, L)
            vmax = float(np.max(np.abs(v))) or 1e-12
            im = ax.imshow(v, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title(f"$u_{col_i+1}$")
            if col_i == 0:
                ax.set_ylabel(rf"$\beta={beta}$", rotation=0, ha="right", va="center",
                              labelpad=12)
    fig.suptitle("top-4 left singular vectors of $G$ (reshaped to $8\\times 8$)", y=1.02)
    _save(fig, "top_modes_lattice64")


# -----------------------------------------------------------------------------
# Figure 5: task-induced coarse graining
# -----------------------------------------------------------------------------
def fig_task_induced(rows5: List[Dict]):
    if not rows5:
        print("[skip] fig5 — no tier5 rows"); return
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.0))

    by_task_beta = defaultdict(lambda: defaultdict(list))
    for r in rows5:
        by_task_beta[r["task"]][r["beta"]].append(r)

    tasks = sorted(by_task_beta.keys())

    # Panel 1: r_eff vs beta per task
    ax = axes[0]
    for task in tasks:
        betas = sorted(by_task_beta[task].keys())
        vals = [[r["eff_rank_G"] for r in by_task_beta[task][b]] for b in betas]
        means = [np.mean(v) for v in vals]; stds = [np.std(v) for v in vals]
        ax.errorbar(betas, means, yerr=stds, fmt="o-", capsize=2.5, lw=1.6,
                    color=TASK_PALETTE.get(task), label=task)
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$r_{\rm eff}(G)$")
    ax.set_yscale("log"); ax.set_title("rank vs temperature, by task")
    ax.legend()

    # Panel 2: delta vs beta per task
    ax = axes[1]
    for task in tasks:
        betas = sorted(by_task_beta[task].keys())
        vals = [[r["delta_baseline"] for r in by_task_beta[task][b]] for b in betas]
        means = [np.mean(v) for v in vals]; stds = [np.std(v) for v in vals]
        ax.errorbar(betas, means, yerr=stds, fmt="o-", capsize=2.5, lw=1.6,
                    color=TASK_PALETTE.get(task), label=task)
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\Delta_{\rm baseline}$")
    ax.set_title("prediction performance, by task")

    # Panel 3: alignments at β=0.44 per task
    ax = axes[2]
    metric_titles = [("align_G_A", r"$G\!\leftrightarrow\!A$"),
                      ("align_G_C", r"$G\!\leftrightarrow\!C$"),
                      ("align_G_lag", r"$G\!\leftrightarrow\!C_\tau$")]
    width = 0.25
    xs = np.arange(len(metric_titles))
    for i, task in enumerate(tasks):
        runs = by_task_beta[task].get(0.44, [])
        if not runs:
            continue
        means = []; stds = []
        for key, _ in metric_titles:
            v = [r[key] for r in runs]
            means.append(np.mean(v)); stds.append(np.std(v))
        ax.bar(xs + (i - 1) * width, means, width, yerr=stds,
               color=TASK_PALETTE.get(task), label=task,
               capsize=3, edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([t for _, t in metric_titles])
    ax.set_ylim(0, 1)
    ax.set_ylabel("alignment (k=5)")
    ax.set_title(r"alignments at $\beta=0.44$")

    # Panel 4: rank-vs-performance scatter
    ax = axes[3]
    for task in tasks:
        for b, runs in by_task_beta[task].items():
            for r in runs:
                ax.scatter(r["delta_baseline"], r["eff_rank_G"],
                           color=TASK_PALETTE.get(task), s=35, alpha=0.85,
                           edgecolor="white", linewidth=0.5)
    for task in tasks:
        # one labeled handle per task
        ax.scatter([], [], color=TASK_PALETTE.get(task), label=task,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel(r"$\Delta_{\rm baseline}$"); ax.set_ylabel(r"$r_{\rm eff}(G)$")
    ax.set_yscale("log"); ax.set_title("rank vs performance")
    ax.legend()

    fig.suptitle("Tier 5 — task-induced coarse graining at $n=64$ lattice", y=1.02,
                  fontsize=12)
    _save(fig, "task_induced_coarse_graining")


# -----------------------------------------------------------------------------
# Figure 6: graph-family comparison
# -----------------------------------------------------------------------------
def fig_graph_family(rows2: List[Dict]):
    if not rows2:
        print("[skip] fig6 — no tier2 rows"); return
    by_kind_beta = defaultdict(lambda: defaultdict(list))
    for r in rows2:
        by_kind_beta[r["kind"]][r["beta"]].append(r)
    kinds = sorted(by_kind_beta)
    metric_titles = [
        ("delta_baseline", r"$\Delta_{\rm baseline}$"),
        ("eff_rank_G",     r"$r_{\rm eff}(G)$"),
        ("align_G_A",      r"$\mathrm{align}(G,A)$"),
    ]
    fig, axes = plt.subplots(len(kinds), len(metric_titles),
                              figsize=(11.0, 8.5), squeeze=False)
    for row_i, kind in enumerate(kinds):
        data = by_kind_beta[kind]
        betas = sorted(data.keys())
        for col_i, (key, title) in enumerate(metric_titles):
            ax = axes[row_i, col_i]
            vals = [[r[key] for r in data[b]] for b in betas]
            means = [np.mean(v) for v in vals]; stds = [np.std(v) for v in vals]
            ax.errorbar(betas, means, yerr=stds, fmt="o-", capsize=2.5, lw=1.6,
                        color=FAMILY_PALETTE.get(kind, "#1f77b4"))
            ax.set_xlabel(r"$\beta$")
            ax.set_title(f"{kind}  —  {title}", fontsize=9.5)
            if key == "eff_rank_G":
                ax.set_yscale("log")
            if key.startswith("align"):
                ax.set_ylim(0, 1)
            if kind == "lattice_2d":
                ax.axvline(BETA_C_LATTICE, ls="--", color="#888", lw=0.7)
            elif kind == "curie_weiss":
                ax.axvline(BETA_C_CURIE, ls="--", color="#888", lw=0.7)
    fig.suptitle("Tier 2 — phase diagrams across graph families", y=1.01, fontsize=12)
    _save(fig, "graph_family_comparison")


# -----------------------------------------------------------------------------
# Figure 7: relative alignment
# -----------------------------------------------------------------------------
def fig_relative_alignment(rows4: List[Dict], rows6: List[Dict]):
    rows = rows6 if rows6 else rows4
    if not rows:
        print("[skip] fig7 — no rows"); return
    tag = "tier6" if rows6 else "tier4"

    by_n_beta = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_n_beta[r["n"]][r["beta"]].append(r)

    # main: just align(G,A) / rand
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for n in sorted(by_n_beta.keys()):
        betas = sorted(by_n_beta[n].keys())
        vals = [[r["align_G_A"] / max(r["align_random_A"], 1e-9) for r in by_n_beta[n][b]]
                for b in betas]
        means = np.array([np.mean(v) for v in vals]); stds = np.array([np.std(v) for v in vals])
        ax.plot(betas, means, "o-", lw=1.6, ms=5, color=N_PALETTE.get(n),
                 label=f"n={n}")
        ax.fill_between(betas, means - stds, means + stds, alpha=0.15,
                         color=N_PALETTE.get(n), edgecolor="none")
    ax.axvline(BETA_C_LATTICE, ls="--", color="#888", lw=0.8)
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\mathrm{align}(G,A)\,/\,\mathrm{align}_{\rm rand}$")
    ax.set_yscale("log")
    ax.set_title(f"phase diagram of *relative* alignment ({tag})")
    ax.legend(title="size", title_fontsize=8.5)
    _save(fig, "relative_alignment_phase_diagram")

    # subpanels for A, C, Cτ
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, (data_key, rand_key, title) in zip(axes, [
        ("align_G_A",   "align_random_A",   r"$G\!\leftrightarrow\!A$"),
        ("align_G_C",   "align_random_C",   r"$G\!\leftrightarrow\!C$"),
        ("align_G_lag", "align_random_lag", r"$G\!\leftrightarrow\!C_{\tau=1}$"),
    ]):
        for n in sorted(by_n_beta.keys()):
            betas = sorted(by_n_beta[n].keys())
            vals = [[r[data_key] / max(r[rand_key], 1e-9) for r in by_n_beta[n][b]] for b in betas]
            means = np.array([np.mean(v) for v in vals]); stds = np.array([np.std(v) for v in vals])
            ax.plot(betas, means, "o-", lw=1.5, ms=4, color=N_PALETTE.get(n),
                     label=f"n={n}")
            ax.fill_between(betas, means - stds, means + stds, alpha=0.15,
                             color=N_PALETTE.get(n), edgecolor="none")
        ax.axvline(BETA_C_LATTICE, ls="--", color="#888", lw=0.7)
        ax.set_xlabel(r"$\beta$"); ax.set_ylabel("relative alignment")
        ax.set_yscale("log"); ax.set_title(title, fontsize=10)
    axes[0].legend(title="size", title_fontsize=8.5)
    fig.suptitle("relative alignment — three operators", y=1.02)
    _save(fig, "relative_alignment_A_C_Ctau")


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tier", choices=["tier2", "tier3", "tier4", "tier5", "tier6"])
    ap.add_argument("--animations", action="store_true",
                     help="placeholder; animations are built by scripts/build_animations.py")
    args = ap.parse_args()
    if not (args.all or args.tier or args.animations):
        args.all = True

    # Lazy-load each tier
    cache = {}
    def get(t):
        if t not in cache:
            cache[t] = {
                "tier2": collect_tier2, "tier4": collect_tier4,
                "tier5": collect_tier5, "tier6": collect_tier6,
            }[t]()
        return cache[t]

    if args.all or args.tier == "tier4":
        fig_main_triptych(get("tier4"))
        fig_relative_alignment(get("tier4"), get("tier6"))
        fig_effective_rank_heatmap(get("tier4"), get("tier6"))
        fig_scaling_exponent(get("tier4"), get("tier6"))
    if args.all or args.tier == "tier6":
        fig_effective_rank_heatmap(get("tier4"), get("tier6"))
        fig_scaling_exponent(get("tier4"), get("tier6"))
        fig_relative_alignment(get("tier4"), get("tier6"))
    if args.all or args.tier == "tier5":
        fig_task_induced(get("tier5"))
    if args.all or args.tier == "tier2":
        fig_graph_family(get("tier2"))
    if args.all:
        fig_geometry_decomposition_lattice64()


if __name__ == "__main__":
    main()
