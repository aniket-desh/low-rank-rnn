#!/usr/bin/env bash
# Tier 5 — task comparison (HANDOFF §5.5). 27 runs:
#   tasks ∈ {next_state, denoise, partial}
#   βs    ∈ {0.2, 0.44, 0.8}   (lattice_2d)
#   seeds ∈ {0, 1, 2}
# All at n=hidden=64, seq_len=200, batch=256, 1000 epochs, 5-way parallel.
#
# Predicted by docs/theory_spin_rnn.md §"Candidate tasks":
#  - denoise/partial should shift align(G,*) from A toward C and C_τ.
#  - For lattice ferromagnet near criticality this contrast should be largest.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="runs_h100"
PARALLEL="${PARALLEL:-5}"

run_one() {
    local task="$1" beta="$2" seed="$3"
    local btag="${beta//./}"
    local save="$OUT_ROOT/tier5_${task}_lattice_2d_beta${btag}_seed${seed}"
    if [ -f "$save/history.json" ]; then
        echo "[skip] $save"
        return
    fi
    mkdir -p "$save"
    local extra=()
    case "$task" in
        denoise) extra+=(--task denoise --mask-frac 0.3) ;;
        partial) extra+=(--task partial --obs-frac 0.5) ;;
        next_state) extra+=(--task next_state) ;;
        *) echo "unknown task $task"; exit 1 ;;
    esac
    python3 scripts/run_spin_experiment.py \
        --graph-kind lattice_2d --beta "$beta" \
        --n-spins 64 --hidden-dim 64 \
        --epochs 1000 --seq-len 200 --batch-size 256 \
        --eval-every 50 --seed "$seed" --device cuda \
        --save-dir "$save" \
        "${extra[@]}" \
        > "$save.log" 2>&1
    echo "[done] $save"
}

submit() {
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
        wait -n
    done
    run_one "$@" &
}

for task in next_state denoise partial; do
    for beta in 0.2 0.44 0.8; do
        for seed in 0 1 2; do
            submit "$task" "$beta" "$seed"
        done
    done
done

wait
echo "[tier5 done]"
