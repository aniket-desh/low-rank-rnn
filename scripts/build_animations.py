#!/usr/bin/env python3
"""Render MP4 + GIF animations from checkpointed runs.

The animation source runs are produced by `scripts/run_animation_sources.sh`
and store `checkpoints/epoch_NNNN.pt` snapshots under each run dir. This
script reloads those checkpoints in order and animates:

  1. G_heatmap_betaXX.{mp4,gif}            -- G = R J B as a heatmap evolving
  2. singular_spectrum_three_betas.mp4      -- side-by-side σ(G) for β=0.2/0.44/0.8
  3. top_modes_beta0p440.mp4                -- top-4 left singular vectors of G
  4. training_dashboard_three_betas.mp4     -- r_eff/Δ/align/spectrum dashboard

Usage:
    python scripts/build_animations.py [--out figures/blog/animations] [run_dir ...]

With no positional args, expects the three β-sweep animation runs to exist:
    runs_h100/animation_lattice64_beta0p200_next_state
    runs_h100/animation_lattice64_beta0p440_next_state
    runs_h100/animation_lattice64_beta0p800_next_state
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
# Locate ffmpeg via imageio-ffmpeg if it isn't on PATH (typical inside venvs).
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import matplotlib.animation as manim
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lowrank_rnn.models.vanilla_rnn import VanillaRNN, VanillaRNNConfig
from lowrank_rnn.analysis.spin_geometry import effective_spin_operator

RUNS = REPO / "runs_h100"
DEFAULT_OUT = REPO / "figures" / "blog" / "animations"

ANIM_RE = re.compile(r"animation_lattice(?P<n>\d+)_beta(?P<beta>\d+p\d+)_(?P<task>\w+)")

CKPT_RE = re.compile(r"epoch_(?P<e>\d+)\.pt$")


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


def _list_checkpoints(run_dir: Path) -> List[Path]:
    cdir = run_dir / "checkpoints"
    if not cdir.is_dir():
        return []
    out = sorted([p for p in cdir.glob("epoch_*.pt") if CKPT_RE.search(p.name)],
                  key=lambda p: int(CKPT_RE.search(p.name)["e"]))
    return out


def _G_from_ckpt(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = _rebuild_model(cfg, ck["model_state_dict"])
    with torch.no_grad():
        G = effective_spin_operator(model).cpu().float()
    return G, ck


def _ffmpeg_available() -> bool:
    if shutil.which("ffmpeg") is not None:
        return True
    rc = matplotlib.rcParams.get("animation.ffmpeg_path", "")
    return bool(rc) and Path(rc).exists()


def _writer_mp4(fps: int = 30):
    if _ffmpeg_available():
        # `-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"` pads the frame to even
        # dimensions so libx264 (which requires multiples of 2) doesn't bail.
        return manim.FFMpegWriter(
            fps=fps, codec="libx264", bitrate=4000,
            extra_args=[
                "-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            ],
        )
    return None


def _save_gif(anim: manim.FuncAnimation, path: Path, fps: int = 10):
    try:
        anim.save(path, writer=manim.PillowWriter(fps=fps))
        print(f"[ok] {path.relative_to(REPO)}")
    except Exception as e:
        print(f"[warn] gif {path}: {e}")


# -----------------------------------------------------------------------------
# Animation 1: G heatmap per run
# -----------------------------------------------------------------------------
def anim_G_heatmap(run_dir: Path, out_dir: Path):
    ckpts = _list_checkpoints(run_dir)
    if not ckpts:
        print(f"[skip] no checkpoints in {run_dir.name}")
        return
    m = ANIM_RE.search(run_dir.name)
    if m:
        # Always include the task to avoid overwriting siblings at same β.
        btag = f"{m['beta']}_{m['task']}"
    else:
        btag = "unknown"

    # Pre-compute global color range
    Gs = []
    metas = []
    for c in ckpts:
        G, ck = _G_from_ckpt(c)
        Gs.append(G.numpy())
        # last logged metrics so far
        hist = ck["history_so_far"]
        last_eval = hist[-1] if hist else None
        metas.append({
            "epoch": ck["epoch"],
            "val_loss": last_eval["val_loss"] if last_eval else None,
            "r_eff": last_eval["eff_rank_G"] if last_eval else None,
            "align_A": last_eval["align_G_A"] if last_eval else None,
            "align_C": last_eval["align_G_C"] if last_eval else None,
        })
    vmax = float(np.max([np.max(np.abs(G)) for G in Gs])) or 1e-12

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    im = ax.imshow(Gs[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$G_{ij}$")
    title = ax.set_title("")
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        im.set_data(Gs[i])
        meta = metas[i]
        text = f"epoch {meta['epoch']:>4d}"
        if meta["val_loss"] is not None:
            text += f"   val={meta['val_loss']:.3f}   r_eff(G)={meta['r_eff']:.2f}"
            text += f"\nalign(G,A)={meta['align_A']:.2f}   align(G,C)={meta['align_C']:.2f}"
        title.set_text(text)
        return im, title

    anim = manim.FuncAnimation(fig, update, frames=len(Gs), interval=80, blit=False)
    mp4 = out_dir / f"G_heatmap_beta{btag}.mp4"
    gif = out_dir / f"G_heatmap_beta{btag}.gif"
    w = _writer_mp4(fps=20)
    if w:
        anim.save(mp4, writer=w, dpi=160)
        print(f"[ok] {mp4.relative_to(REPO)}")
    _save_gif(anim, gif, fps=10)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Animation 2: singular spectrum (three betas side-by-side)
# -----------------------------------------------------------------------------
def anim_singular_spectrum_triptych(beta_runs: List[Path], out_dir: Path):
    valid = [(p, _list_checkpoints(p)) for p in beta_runs if (p / "checkpoints").is_dir()]
    valid = [(p, c) for p, c in valid if c]
    if len(valid) < 2:
        print("[skip] singular_spectrum_triptych: need at least 2 source runs with checkpoints")
        return

    # Align epoch counts
    n_frames = min(len(c) for _, c in valid)

    fig, axes = plt.subplots(1, len(valid), figsize=(5.0 * len(valid), 4.0), sharey=True)
    if len(valid) == 1:
        axes = [axes]

    # Pre-compute per-frame singular values
    series = []
    titles = []
    s_max_global = 0
    for p, ckpts in valid:
        m = ANIM_RE.search(p.name)
        beta = float(m["beta"].replace("p", "."))
        titles.append(rf"$\beta={beta:g}$")
        run_series = []
        for c in ckpts[:n_frames]:
            G, _ = _G_from_ckpt(c)
            s = torch.linalg.svdvals(G).numpy()
            run_series.append(s)
            s_max_global = max(s_max_global, float(s.max()))
        series.append(run_series)

    lines = []
    text_handles = []
    for ax, t, run_series in zip(axes, titles, series):
        s0 = run_series[0]
        ln, = ax.semilogy(np.arange(1, len(s0) + 1), s0, "o-", ms=4, lw=1.5, color="#1f77b4")
        ax.set_xlabel("singular index")
        ax.set_ylabel(r"$\sigma_i(G)$")
        ax.set_ylim(1e-3, max(1e0, s_max_global * 1.2))
        ax.set_title(t, fontsize=11)
        text_handles.append(ax.text(0.97, 0.93, "", transform=ax.transAxes,
                                     ha="right", va="top", fontsize=9))
        lines.append(ln)

    def update(i):
        for ln, run_series, txt in zip(lines, series, text_handles):
            s = run_series[i]
            ln.set_data(np.arange(1, len(s) + 1), s)
            r_eff = float((s.sum() ** 2) / (np.sum(s ** 2) + 1e-12))
            txt.set_text(f"epoch {25 * (i + 1)}\n$r_{{\\rm eff}}={r_eff:.2f}$")
        return lines + text_handles

    anim = manim.FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    mp4 = out_dir / "singular_spectrum_three_betas.mp4"
    w = _writer_mp4(fps=20)
    if w:
        anim.save(mp4, writer=w, dpi=160)
        print(f"[ok] {mp4.relative_to(REPO)}")
    _save_gif(anim, out_dir / "singular_spectrum_three_betas.gif", fps=10)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Animation 3: top 4 modes at β=0.44
# -----------------------------------------------------------------------------
def anim_top_modes(run_dir: Path, out_dir: Path):
    ckpts = _list_checkpoints(run_dir)
    if not ckpts:
        print(f"[skip] top_modes: no ckpts in {run_dir.name}")
        return
    L = 8  # n=64 lattice
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))
    ims = []
    titles = []
    modes_series = []  # list of 4-mode arrays per frame
    epoch_series = []
    for c in ckpts:
        G, ck = _G_from_ckpt(c)
        U, _, _ = torch.linalg.svd(G, full_matrices=False)
        modes = [U[:, k].numpy().reshape(L, L) for k in range(4)]
        modes_series.append(modes)
        epoch_series.append(ck["epoch"])

    vmax = max(float(np.max(np.abs(m))) for run_modes in modes_series for m in run_modes) or 1e-12
    fig.suptitle("")
    suptitle = fig.suptitle("", fontsize=11)
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(modes_series[0][i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"$u_{i+1}$", fontsize=10)
        ims.append(im)

    def update(idx):
        modes = modes_series[idx]
        for im, m in zip(ims, modes):
            im.set_data(m)
        s = torch.linalg.svdvals(torch.tensor(np.stack([m.flatten() for m in modes]))).numpy()
        # use a coarse r_eff proxy from the first 4 modes only for the suptitle:
        suptitle.set_text(f"top-4 modes — epoch {epoch_series[idx]}")
        return ims + [suptitle]

    anim = manim.FuncAnimation(fig, update, frames=len(modes_series), interval=80, blit=False)
    mp4 = out_dir / "top_modes_beta0p440.mp4"
    w = _writer_mp4(fps=20)
    if w:
        anim.save(mp4, writer=w, dpi=160)
        print(f"[ok] {mp4.relative_to(REPO)}")
    _save_gif(anim, out_dir / "top_modes_beta0p440.gif", fps=10)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Animation 4: training dashboard (three betas)
# -----------------------------------------------------------------------------
def anim_dashboard(beta_runs: List[Path], out_dir: Path):
    valid = []
    for p in beta_runs:
        c = _list_checkpoints(p)
        if c:
            valid.append((p, c))
    if len(valid) < 2:
        print("[skip] dashboard: need at least 2 sources")
        return
    n_frames = min(len(c) for _, c in valid)

    # Build cumulative metric series per run.
    series = []
    titles = []
    spectra_series = []
    for p, ckpts in valid:
        m = ANIM_RE.search(p.name)
        beta = float(m["beta"].replace("p", "."))
        titles.append(rf"$\beta={beta:g}$")
        r_effs, deltas, aA, aC, aL = [], [], [], [], []
        spectra = []
        for c in ckpts[:n_frames]:
            G, ck = _G_from_ckpt(c)
            hist = ck["history_so_far"]
            last = hist[-1] if hist else None
            r_effs.append(last["eff_rank_G"] if last else np.nan)
            deltas.append(last["delta_baseline"] if last else np.nan)
            aA.append(last["align_G_A"] if last else np.nan)
            aC.append(last["align_G_C"] if last else np.nan)
            aL.append(last["align_G_lag"] if last else np.nan)
            spectra.append(torch.linalg.svdvals(G).numpy())
        series.append({
            "epochs": [ck for ck in [int(c.name[6:10]) for c in ckpts[:n_frames]]],
            "r": np.array(r_effs), "d": np.array(deltas),
            "aA": np.array(aA), "aC": np.array(aC), "aL": np.array(aL),
            "spec": spectra,
        })
        spectra_series.append(spectra)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    ax_r, ax_d, ax_a, ax_s = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    lines_r, lines_d, lines_a, lines_s = [], [], [], []
    for k, (s, title) in enumerate(zip(series, titles)):
        color = palette[k % len(palette)]
        l1, = ax_r.plot([], [], "-", lw=1.8, color=color, label=title)
        l2, = ax_d.plot([], [], "-", lw=1.8, color=color, label=title)
        l3a, = ax_a.plot([], [], "-", lw=1.5, color=color, label=f"{title} A")
        l3c, = ax_a.plot([], [], "--", lw=1.0, color=color, alpha=0.7, label=f"{title} C")
        l3l, = ax_a.plot([], [], ":", lw=1.0, color=color, alpha=0.7, label=f"{title} Cτ")
        l4, = ax_s.plot([], [], "o-", ms=3, color=color, label=title)
        lines_r.append(l1); lines_d.append(l2); lines_a.append((l3a, l3c, l3l)); lines_s.append(l4)

    ax_r.set_xlabel("epoch"); ax_r.set_ylabel(r"$r_{\rm eff}(G)$")
    ax_r.set_yscale("log"); ax_r.set_title("effective rank")
    ax_r.legend(fontsize=8)
    ax_d.set_xlabel("epoch"); ax_d.set_ylabel(r"$\Delta_{\rm baseline}$")
    ax_d.set_title("prediction performance")
    ax_d.set_ylim(-0.05, 1.05)
    ax_a.set_xlabel("epoch"); ax_a.set_ylabel("alignment (k=5)")
    ax_a.set_ylim(0, 1.05); ax_a.set_title("alignment with $A$, $C$, $C_\\tau$")
    ax_a.legend(fontsize=7, ncol=3)
    ax_s.set_xlabel("singular index"); ax_s.set_ylabel(r"$\sigma_i(G)$")
    ax_s.set_yscale("log"); ax_s.set_title("singular spectrum (current epoch)")
    s_max = max(float(np.max(s)) for ser in spectra_series for s in ser)
    s_min = max(1e-4, min(
        float(np.min(s[s > 0])) for ser in spectra_series for s in ser
        if np.any(s > 0)
    ))
    spec_len = max(len(s) for ser in spectra_series for s in ser)
    ax_s.set_xlim(0.5, spec_len + 0.5)
    ax_s.set_ylim(s_min * 0.5, s_max * 1.5)
    ax_s.legend(fontsize=8, loc="upper right")

    # Find x-limits
    max_epoch = max((max(s["epochs"]) for s in series), default=1000)
    for ax in (ax_r, ax_d, ax_a):
        ax.set_xlim(0, max_epoch + 25)

    def update(i):
        artists = []
        for k, s in enumerate(series):
            eps = s["epochs"][: i + 1]
            lines_r[k].set_data(eps, s["r"][: i + 1]); artists.append(lines_r[k])
            lines_d[k].set_data(eps, s["d"][: i + 1]); artists.append(lines_d[k])
            l3a, l3c, l3l = lines_a[k]
            l3a.set_data(eps, s["aA"][: i + 1])
            l3c.set_data(eps, s["aC"][: i + 1])
            l3l.set_data(eps, s["aL"][: i + 1])
            artists += [l3a, l3c, l3l]
            spec = s["spec"][i]
            lines_s[k].set_data(np.arange(1, len(spec) + 1), spec)
            artists.append(lines_s[k])
        return artists

    anim = manim.FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    mp4 = out_dir / "training_dashboard_three_betas.mp4"
    w = _writer_mp4(fps=20)
    if w:
        anim.save(mp4, writer=w, dpi=150)
        print(f"[ok] {mp4.relative_to(REPO)}")
    _save_gif(anim, out_dir / "training_dashboard_three_betas.gif", fps=10)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Render MP4 + GIF animations from checkpointed runs."
    )
    ap.add_argument("run_dirs", nargs="*", type=Path,
                     help="optional run dirs to drive a per-run G_heatmap animation; "
                          "if empty, defaults to the three β-sweep animation runs.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.run_dirs:
        provided = list(args.run_dirs)
    else:
        provided = [
            RUNS / "animation_lattice64_beta0p200_next_state",
            RUNS / "animation_lattice64_beta0p440_next_state",
            RUNS / "animation_lattice64_beta0p800_next_state",
        ]
    provided = [p for p in provided if p.is_dir()]
    if not provided:
        print("No checkpoints found. Run scripts/run_animation_sources.sh first.")
        return

    # Resolve to absolute paths so name lookups against RUNS/... succeed.
    sources_with_ckpts = [p.resolve() for p in provided if _list_checkpoints(p)]
    if not sources_with_ckpts:
        print("No checkpoints found. Run scripts/run_animation_sources.sh first.")
        return
    by_name = {p.name: p for p in sources_with_ckpts}

    # 1. per-run G heatmap
    for p in sources_with_ckpts:
        anim_G_heatmap(p, args.out)

    # 2. singular spectrum triptych (need the three β-runs)
    beta_runs = []
    for beta in (0.200, 0.440, 0.800):
        btag = f"{beta:.3f}".replace(".", "p")
        name = f"animation_lattice64_beta{btag}_next_state"
        if name in by_name:
            beta_runs.append(by_name[name])
    if len(beta_runs) >= 2:
        anim_singular_spectrum_triptych(beta_runs, args.out)
        anim_dashboard(beta_runs, args.out)

    # 3. top modes at β=0.44
    name44 = "animation_lattice64_beta0p440_next_state"
    if name44 in by_name:
        anim_top_modes(by_name[name44], args.out)


if __name__ == "__main__":
    main()
