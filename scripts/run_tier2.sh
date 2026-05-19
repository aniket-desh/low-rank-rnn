#!/usr/bin/env bash
# Tier 2 — temperature phase diagram at n=64. 69 runs total:
#   lattice_2d: 10 βs × 3 seeds = 30
#   curie_weiss: 7 βs × 3 seeds = 21
#   block:       6 βs × 3 seeds = 18
# All at n=hidden=64, seq_len=200, batch=256, 1000 epochs, 5-way parallel.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="runs_h100"
PARALLEL="${PARALLEL:-5}"

beta_tag() { echo "${1//./}"; }

run_one() {
    local kind="$1" beta="$2" seed="$3"
    local btag; btag="$(beta_tag "$beta")"
    local save="$OUT_ROOT/tier2_${kind}${4}_beta${btag}_seed${seed}"
    if [ -f "$save/history.json" ]; then
        echo "[skip] $save"
        return
    fi
    mkdir -p "$save"
    python3 scripts/run_spin_experiment.py \
        --graph-kind "$kind" --beta "$beta" \
        --n-spins 64 --hidden-dim 64 \
        --epochs 1000 --seq-len 200 --batch-size 256 \
        --eval-every 50 --seed "$seed" --device cuda \
        --save-dir "$save" \
        > "$save.log" 2>&1
    echo "[done] $save"
}

# Submit jobs to a bounded pool of $PARALLEL workers.
declare -a JOBS=()
submit() {
    local kind="$1" beta="$2" seed="$3" tag="$4"
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
        wait -n
    done
    run_one "$kind" "$beta" "$seed" "$tag" &
}

# Lattice (10 β × 3 seeds = 30). Tag with the n.
for beta in 0.1 0.2 0.3 0.38 0.42 0.44 0.46 0.5 0.6 0.8; do
    for seed in 0 1 2; do
        submit "lattice_2d" "$beta" "$seed" "64"
    done
done

# Curie–Weiss (7 β × 3 seeds = 21). Couplings are J/n, so β needs to be large.
for beta in 0.2 0.5 0.8 1.0 1.2 1.5 2.0; do
    for seed in 0 1 2; do
        submit "curie_weiss" "$beta" "$seed" "64"
    done
done

# Block (6 β × 3 seeds = 18).
for beta in 0.5 0.8 1.0 1.2 1.5 2.0; do
    for seed in 0 1 2; do
        submit "block" "$beta" "$seed" "64"
    done
done

wait
echo "[tier2 done]"
