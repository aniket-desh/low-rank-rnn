#!/usr/bin/env bash
# Tier 4 — finite-size scaling near criticality. 63 runs:
#   n ∈ {64, 256, 1024} × β ∈ {0.36, 0.40, 0.42, 0.44, 0.46, 0.48, 0.52} × 3 seeds.
# This is the headline FSS sweep (HANDOFF §5.4).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="runs_h100"
BETAS="0.36 0.40 0.42 0.44 0.46 0.48 0.52"

run_one() {
    local n="$1" beta="$2" seed="$3" batch="$4"
    local btag="${beta//./}"
    local save="$OUT_ROOT/tier4_fss_lattice${n}_beta${btag}_seed${seed}"
    if [ -f "$save/history.json" ]; then
        echo "[skip] $save"
        return
    fi
    mkdir -p "$save"
    python3 scripts/run_spin_experiment.py \
        --graph-kind lattice_2d --beta "$beta" \
        --n-spins "$n" --hidden-dim "$n" \
        --epochs 1000 --seq-len 200 --batch-size "$batch" \
        --eval-every 100 --align-k 10 \
        --seed "$seed" --device cuda \
        --save-dir "$save" \
        > "$save.log" 2>&1
    echo "[done] $save"
}

batch_pool() {
    local pool="$1"; shift
    local args=("$@")
    local n_args=${#args[@]}
    local pos=0
    while [ "$pos" -lt "$n_args" ]; do
        local launched=0
        while [ "$launched" -lt "$pool" ] && [ "$pos" -lt "$n_args" ]; do
            eval "${args[$pos]}" &
            pos=$((pos + 1))
            launched=$((launched + 1))
        done
        wait
    done
}

# n=64: 5 parallel
JOBS=()
for beta in $BETAS; do
    for seed in 0 1 2; do
        JOBS+=("run_one 64 $beta $seed 256")
    done
done
batch_pool 5 "${JOBS[@]}"

# n=256: 3 parallel
JOBS=()
for beta in $BETAS; do
    for seed in 0 1 2; do
        JOBS+=("run_one 256 $beta $seed 128")
    done
done
batch_pool 3 "${JOBS[@]}"

# n=1024: serial
JOBS=()
for beta in $BETAS; do
    for seed in 0 1 2; do
        JOBS+=("run_one 1024 $beta $seed 64")
    done
done
batch_pool 1 "${JOBS[@]}"

echo "[tier4 done]"
