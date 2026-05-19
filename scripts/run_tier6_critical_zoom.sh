#!/usr/bin/env bash
# Tier 6 — dense lattice β-zoom around β_c ≈ 0.4406868.
#   n_spins ∈ {64, 256, 1024}
#   β       ∈ {0.400, 0.405, ..., 0.500} (21 values)
#   seeds   ∈ {0, 1, 2}
# → 189 runs total. Resolves the critical region at much finer grain than Tier 4.
#
# Parallelism per scale (conservative — H100 already saturates around these
# concurrency levels for the model size in question):
#   n=64   : 6 concurrent jobs
#   n=256  : 4 concurrent jobs
#   n=1024 : 2 concurrent jobs
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="runs_h100"
LOG_ROOT="logs/tier6"
mkdir -p "$LOG_ROOT"

BETAS=(0.400 0.405 0.410 0.415 0.420 0.425 0.430 0.435 0.440 0.445 0.450 \
       0.455 0.460 0.465 0.470 0.475 0.480 0.485 0.490 0.495 0.500)
SEEDS=(0 1 2)

beta_tag() {
    # 0.440 -> 0p440 so directory names stay glob-friendly
    echo "${1//./p}"
}

run_one() {
    local n="$1" beta="$2" seed="$3" batch="$4"
    local btag
    btag="$(beta_tag "$beta")"
    local save="$OUT_ROOT/tier6_zoom_lattice${n}_beta${btag}_seed${seed}"
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
        > "$LOG_ROOT/$(basename "$save").log" 2>&1
    echo "[done] $save"
}

batch_pool() {
    local pool="$1"; shift
    local jobs=("$@")
    local n_jobs=${#jobs[@]}
    local pos=0
    while [ "$pos" -lt "$n_jobs" ]; do
        local launched=0
        while [ "$launched" -lt "$pool" ] && [ "$pos" -lt "$n_jobs" ]; do
            eval "${jobs[$pos]}" &
            pos=$((pos + 1))
            launched=$((launched + 1))
        done
        wait
    done
}

# Build per-scale job arrays
JOBS_64=()
JOBS_256=()
JOBS_1024=()
for beta in "${BETAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS_64+=("run_one 64 $beta $seed 128")
        JOBS_256+=("run_one 256 $beta $seed 128")
        JOBS_1024+=("run_one 1024 $beta $seed 64")
    done
done

echo "[tier6] launching ${#JOBS_64[@]} jobs at n=64 (pool=6)"
batch_pool 6 "${JOBS_64[@]}"

echo "[tier6] launching ${#JOBS_256[@]} jobs at n=256 (pool=4)"
batch_pool 4 "${JOBS_256[@]}"

echo "[tier6] launching ${#JOBS_1024[@]} jobs at n=1024 (pool=2)"
batch_pool 2 "${JOBS_1024[@]}"

echo "[tier6 done]"

# Post hooks — build blog figures with the new Tier 6 data and dump a stats table.
python3 scripts/build_blog_figures.py --tier tier6 || true
python3 scripts/tier_stats.py tier6 > figures/blog/tier6_zoom_stats.md 2>/dev/null || true
