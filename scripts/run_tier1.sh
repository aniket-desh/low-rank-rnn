#!/usr/bin/env bash
# Tier 1 — n=64 seed replication. 20 runs (5 seeds × 4 regimes).
# Each run is ~140 s at 1000 epochs on H100, so the full sweep takes ~45 min
# sequentially. We run 4 in parallel per (seed, regime) batch to amortize the
# CUDA-context startup cost.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="runs_h100"
mkdir -p "$OUT_ROOT"

run_one() {
    local kind="$1" beta="$2" seed="$3"
    local beta_tag="${beta/./}"
    local save="$OUT_ROOT/tier1_${kind}_beta${beta_tag}_seed${seed}"
    if [ -f "$save/history.json" ]; then
        echo "[skip] $save already has history.json"
        return
    fi
    python3 scripts/run_spin_experiment.py \
        --graph-kind "$kind" --beta "$beta" \
        --n-spins 64 --hidden-dim 64 \
        --epochs 1000 --seq-len 200 --batch-size 256 \
        --eval-every 50 --seed "$seed" --device cuda \
        --save-dir "$save" \
        > "$save.log" 2>&1
    echo "[done] $save"
}

specs=(
    "lattice_2d 0.2"
    "lattice_2d 0.44"
    "curie_weiss 0.2"
    "block 0.5"
)

# 4 regimes × 5 seeds = 20 jobs. We launch 4 at a time (1 regime concurrently
# across seeds is the easiest to read at the end).
for spec in "${specs[@]}"; do
    read -r kind beta <<<"$spec"
    for seed in 0 1 2 3 4; do
        # Pre-create the dir so the .log file lands beside it.
        beta_tag="${beta/./}"
        mkdir -p "$OUT_ROOT/tier1_${kind}_beta${beta_tag}_seed${seed}"
    done
    echo "[launch] $kind β=$beta — 5 seeds in parallel"
    for seed in 0 1 2 3 4; do
        run_one "$kind" "$beta" "$seed" &
    done
    wait
    echo "[regime done] $kind β=$beta"
done

echo "[tier1 done]"
