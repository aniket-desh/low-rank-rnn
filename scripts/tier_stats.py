#!/usr/bin/env python3
"""Aggregate per-tier statistics from runs_h100/ into a markdown table.

Usage:
    python scripts/tier_stats.py tier1   # 4 regimes × 5 seeds at n=64
    python scripts/tier_stats.py tier2   # β-sweep
    python scripts/tier_stats.py tier3   # scale sweep
    python scripts/tier_stats.py tier4   # FSS

Outputs a markdown table to stdout. Pipe into summary.md as needed.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs_h100"

METRICS = ("delta_baseline", "eff_rank_G", "align_G_A", "align_G_C", "align_G_lag")


def _load(d: Path):
    cfg = json.loads((d / "config.json").read_text())
    hist = json.loads((d / "history.json").read_text())
    return cfg, hist


def _fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def _mean_std(vals):
    if not vals:
        return float("nan"), float("nan")
    import statistics

    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def tier1():
    pat = re.compile(r"tier1_(?P<kind>[a-z_2d]+?)_beta(?P<beta>\d+)_seed(?P<seed>\d+)")
    by_regime = defaultdict(list)
    for d in sorted(RUNS.glob("tier1_*")):
        m = pat.search(d.name)
        if not m or not (d / "history.json").exists():
            continue
        cfg, hist = _load(d)
        last = hist[-1]
        by_regime[(m["kind"], float(cfg["beta"]))].append(last)

    if not by_regime:
        print("(no tier1 results yet)"); return
    print("| regime | β | n seeds | Δ_baseline | r_eff(G) | align(G,A) | align(G,C) | align(G,Cτ) |")
    print("|---|---|---|---|---|---|---|---|")
    for (kind, beta), rows in sorted(by_regime.items()):
        cells = [f"{kind}", f"{beta:g}", f"{len(rows)}"]
        for k in METRICS:
            mean, std = _mean_std([r[k] for r in rows])
            cells.append(_fmt(mean, std))
        print("| " + " | ".join(cells) + " |")


def tier2():
    pat = re.compile(r"tier2_(?P<kind>[a-z]+?)(?P<n>\d+)_beta(?P<beta>[0-9.]+)_seed(?P<seed>\d+)")
    by_family = defaultdict(lambda: defaultdict(list))
    for d in sorted(RUNS.glob("tier2_*")):
        m = pat.search(d.name)
        if not m or not (d / "history.json").exists():
            continue
        cfg, hist = _load(d)
        by_family[m["kind"]][float(cfg["beta"])].append(hist[-1])

    if not by_family:
        print("(no tier2 results yet)"); return
    for fam, data in sorted(by_family.items()):
        print(f"\n### {fam}\n")
        print("| β | n seeds | Δ_baseline | r_eff(G) | align(G,A) | align(G,C) | align(G,Cτ) |")
        print("|---|---|---|---|---|---|---|")
        for beta in sorted(data.keys()):
            rows = data[beta]
            cells = [f"{beta:g}", f"{len(rows)}"]
            for k in METRICS:
                mean, std = _mean_std([r[k] for r in rows])
                cells.append(_fmt(mean, std))
            print("| " + " | ".join(cells) + " |")


def _scaling(tier_tag: str):
    pat = re.compile(
        r"tier(?P<tier>3|4)_(?:fss_)?lattice(?P<n>\d+)_beta(?P<beta>[0-9.]+)_seed(?P<seed>\d+)"
    )
    by_n = defaultdict(lambda: defaultdict(list))
    for d in sorted(RUNS.glob(f"tier{tier_tag}_*")):
        m = pat.search(d.name)
        if not m or m["tier"] != tier_tag or not (d / "history.json").exists():
            continue
        cfg, hist = _load(d)
        by_n[int(m["n"])][float(cfg["beta"])].append(hist[-1])
    if not by_n:
        print(f"(no tier{tier_tag} results yet)"); return
    for n in sorted(by_n.keys()):
        print(f"\n### n={n}\n")
        print("| β | n seeds | Δ_baseline | r_eff(G) | align(G,A) | align(G,C) | align(G,Cτ) |")
        print("|---|---|---|---|---|---|---|")
        data = by_n[n]
        for beta in sorted(data.keys()):
            rows = data[beta]
            cells = [f"{beta:g}", f"{len(rows)}"]
            for k in METRICS:
                mean, std = _mean_std([r[k] for r in rows])
                cells.append(_fmt(mean, std))
            print("| " + " | ".join(cells) + " |")


def tier3():
    _scaling("3")


def tier4():
    _scaling("4")


def tier5():
    """tier5_{task}_lattice_2d_beta{btag}_seed{s}"""
    pat = re.compile(
        r"tier5_(?P<task>next_state|denoise|partial)_lattice_2d_beta(?P<beta>\d+)_seed(?P<seed>\d+)"
    )
    by_task = defaultdict(lambda: defaultdict(list))
    for d in sorted(RUNS.glob("tier5_*")):
        m = pat.search(d.name)
        if not m or not (d / "history.json").exists():
            continue
        cfg, hist = _load(d)
        by_task[m["task"]][float(cfg["beta"])].append(hist[-1])
    if not by_task:
        print("(no tier5 results yet)"); return
    for task, data in sorted(by_task.items()):
        print(f"\n### task = {task}\n")
        print("| β | n seeds | Δ_baseline | r_eff(G) | align(G,A) | align(G,C) | align(G,Cτ) |")
        print("|---|---|---|---|---|---|---|")
        for beta in sorted(data.keys()):
            rows = data[beta]
            cells = [f"{beta:g}", f"{len(rows)}"]
            for k in METRICS:
                mean, std = _mean_std([r[k] for r in rows])
                cells.append(_fmt(mean, std))
            print("| " + " | ".join(cells) + " |")


BUILDERS = {"tier1": tier1, "tier2": tier2, "tier3": tier3, "tier4": tier4, "tier5": tier5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=sorted(BUILDERS))
    args = ap.parse_args()
    BUILDERS[args.which]()


if __name__ == "__main__":
    main()
